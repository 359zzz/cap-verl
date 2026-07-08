# Cap-X + VeRL PPO/OPD：第一阶段合成数据采集设计文档

> **版本**: v1  
> **目标**: 为 `franka_lift_code_env` 和 `franka_nut_assembly_code_env` 生成 SFT 训练/验证数据  
> **数据策略**: Prompt 用真实 Visual API（SAM3/Molmo/GraspNet）生成，代码用 Privileged State 生成  
> **执行约束**: 本文档为设计阶段，不修改任何代码；待审计通过后进入实现阶段  

### 与总体计划文档的关系

本文档只参考 `capx_verl_ppo_opd_engineering_plan_v2.md` 中的**整体架构设计**（Phase 0→SFT→Bootstrap→PPO→OPD 的流程、硬件预算等），**不参考其中的任何代码实现**。该计划文档中的代码基于对 cap-x API 的想象，与本机实际接口不符，因此本文档中的所有实现细节均基于本机 cap-x/verl 代码重新设计。

---

## 1. 总体设计

### 1.1 核心原则

1. **单入口脚本**: 所有合成功能统一到一个 Python 脚本 `scripts/generate_synthetic_data.py`。
2. **命令行驱动**: 通过 `python scripts/generate_synthetic_data.py --task franka_lift_code_env --train-size 100 ...` 调用。
3. **Visual Prompt + Privileged Code**: 使用 Visual API 生成真实感知 prompt，使用 privileged state 生成 100% 成功的代码。
4. **轨迹格式**: 所有数据（单轮/多轮）统一保存为 `messages` 对话轨迹。
5. **全量验证**: 每个样本都必须在仿真中执行验证，只保留成功样本。

### 1.2 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         合成数据生成流程                              │
├─────────────────────────────────────────────────────────────────────┤
│  1. 读取命令行参数（任务、数据量、轮数策略、输出目录等）              │
│                          ↓                                          │
│  2. 加载任务 YAML 配置（Visual API + privileged 双环境）              │
│                          ↓                                          │
│  3. 启动感知服务（SAM3 / Molmo / Contact GraspNet / PyRoKi）          │
│                          ↓                                          │
│  4. 循环 N 个 episode：                                              │
│       a. env.reset(seed) 随机化场景                                  │
│       b. Visual Env 获取观测，构造第一轮 user prompt                 │
│       c. Privileged Env 读取真实状态，生成成功代码                    │
│       d. 按轮数策略拆分为单轮/多轮/错误-修正对                        │
│       e. 执行验证（ privileged 模式快速执行）                         │
│       f. 保存 trajectory 到内存                                      │
│                          ↓                                          │
│  5. 训练/验证集划分（按设定比例或固定数量）                           │
│                          ↓                                          │
│  6. 输出 parquet（SFT 格式）                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 第一阶段范围

### 2.1 选定任务

| 任务 | 分类 | Train | Val | 默认注册 API | Visual API |
|---|---|---|---|---|---|
| `franka_lift_code_env` | 单轮任务 | 100 | 20 | `FrankaControlPrivilegedApi` | `FrankaControlApi` |
| `franka_nut_assembly_code_env` | 多轮简单任务 | 500 | 50 | `FrankaControlNutAssemblyPrivilegedApi` | `FrankaControlNutAssemblyVisualApi` |

### 2.2 任务语义与轮数策略

#### `franka_lift_code_env`（单轮任务）

- **目标**: Pick up the red cube and lift it.
- **Oracle 行为**: 一次完整执行即可成功。
- **数据混合**:
  - 80% 单轮成功轨迹
  - 20% 错误/修正对（2 轮轨迹）
- **轮数分布**:
  - 1 轮: 80%
  - 2 轮: 20%

#### `franka_nut_assembly_code_env`（多轮简单任务）

- **目标**: Grasp the brown square nut by its handle and insert it onto the peg.
- **Oracle 阶段**:
  1. 抓取 nut handle
  2. 计算 handle-to-center transform
  3. 将 nut 对准 peg
  4. 插入并释放
- **建议中心轮数**: 3 轮
- **数据混合**:
  - 70% 多轮成功轨迹
  - 30% 错误/修正对
- **轮数分布**:
  - 2 轮: 10%
  - 3 轮: 70%
  - 4 轮: 20%

---

## 3. 扩展性设计

为了让后续任务能方便接入，脚本采用**插件式任务注册**。核心脚本不硬编码任务细节，而是通过 `TASK_REGISTRY` 查找任务配置。

### 3.1 TaskConfig 抽象

```python
@dataclass
class TaskConfig:
    task_name: str
    visual_yaml: Path          # Visual API 的 YAML 配置路径
    privileged_yaml: Path      # Privileged API 的 YAML 配置路径
    category: str              # "single_turn" / "multi_turn_simple" / "multi_turn_complex"
    default_turn_distribution: dict[int, float]
    code_templates: dict       # 代码模板集合
    perturbation_specs: list   # 扰动策略规格
```

### 3.2 TASK_REGISTRY 示例

```python
TASK_REGISTRY: dict[str, TaskConfig] = {
    "franka_lift_code_env": TaskConfig(
        task_name="franka_lift_code_env",
        visual_yaml=Path("env_configs/synthetic/franka_lift_visual.yaml"),
        privileged_yaml=Path("env_configs/synthetic/franka_lift_privileged.yaml"),
        category="single_turn",
        default_turn_distribution={1: 0.8, 2: 0.2},
        code_templates={...},
        perturbation_specs=[...],
    ),
    "franka_nut_assembly_code_env": TaskConfig(
        task_name="franka_nut_assembly_code_env",
        visual_yaml=Path("env_configs/synthetic/franka_nut_assembly_visual.yaml"),
        privileged_yaml=Path("env_configs/synthetic/franka_nut_assembly_privileged.yaml"),
        category="multi_turn_simple",
        default_turn_distribution={2: 0.1, 3: 0.7, 4: 0.2},
        code_templates={...},
        perturbation_specs=[...],
    ),
}
```

### 3.3 接入新任务的方式

新增任务时，只需：

1. 新增 Visual/Privileged YAML 配置文件；
2. 在 `TASK_REGISTRY` 中注册 `TaskConfig`；
3. 提供该任务的代码模板和扰动策略规格。

命令行自动支持新任务，无需修改主脚本逻辑：

```bash
python scripts/generate_synthetic_data.py --task franka_spill_wipe_code_env --train-size 1000
```

---

## 4. Visual API 配置

### 4.1 需要新增的 YAML 配置

本机默认配置使用 privileged API，为了生成 visual prompt，需要新增两个 YAML 文件：

**`env_configs/synthetic/franka_lift_visual.yaml`**

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_lift.FrankaLiftCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cube_lift_low_level
    privileged: false
    enable_render: true
    apis:
      - FrankaControlApi
```

**`env_configs/synthetic/franka_nut_assembly_visual.yaml`**

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_nut_assembly.FrankaNutAssemblyCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_nut_assembly_low_level
    privileged: false
    enable_render: true
    apis:
      - FrankaControlNutAssemblyVisualApi
```

### 4.2 Prompt API 对比

| API | `franka_lift_visual.yaml` | `franka_nut_assembly_visual.yaml` |
|---|---|---|
| `get_object_pose` | ✅ | ✅ |
| `sample_grasp_pose` | ✅ | ✅ |
| `goto_pose` | ✅ | ✅ |
| `open_gripper` | ✅ | ✅ |
| `close_gripper` | ✅ | ✅ |
| `home_pose` | ✅ | ❌ |
| `goto_home_joint_position` | ❌ | ✅ |

---

## 5. 双环境设计

### 5.1 为什么需要两个 Env

| Env 实例 | 作用 | 模式 |
|---|---|---|
| **Visual Env** | 生成 prompt（包含真实感知结果） | `privileged: false`，启用 SAM3/Molmo |
| **Privileged Env** | 读取 ground truth 生成代码，并执行验证 | `privileged: true`，禁用视觉模型 |

### 5.2 双环境状态同步

1. 两个 Env 使用相同的 `low_level` 类，但不同的 `apis`。
2. 每个 episode 用相同 `seed` 分别 reset 两个 Env，确保底层场景一致。
3. Visual Env 仅用于构造 prompt，不执行模型生成的代码。
4. Privileged Env 用于：
   - 读取真实物体位姿
   - 生成成功代码
   - 执行验证

### 5.3 感知服务启动

Visual API 内部会自动连接感知服务器。脚本需要保证以下服务在运行：

| 服务 | 默认端口 | 用途 |
|---|---|---|
| SAM3 server | 8114 | 图像分割 |
| Contact GraspNet server | 8115 | 抓取姿态规划 |
| PyRoKi IK server | 8116 | 逆运动学求解 |
| Molmo（在 nut_assembly visual 中 init） | - | 语言到点提示 |

脚本可以选择：
- 自动检测服务是否在运行，若未运行则报错提示用户启动；
- 或提供 `--auto-start-servers` 参数由脚本启动（可选功能，第一阶段建议手动启动）。

---

## 6. 代码生成策略

### 6.1 代码来源

代码不由 LLM 生成，而由本地 Python 脚本基于 privileged state 动态生成。

对于每个任务，脚本维护一个**代码模板集合**（不是硬编码坐标，而是参数化模板）。每次 reset 后：

1. 从 Privileged Env 读取真实位姿。
2. 将真实数值填入模板。
3. 随机选择控制参数（`z_approach`、`lift_height`、`retreat_height` 等）和代码风格（变量名、注释等）。
4. 输出可执行代码字符串。

### 6.2 代码模板示例

#### `franka_lift_code_env` 单轮成功模板

```python
import numpy as np

# Get grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Open gripper before grasping
open_gripper()

# Approach the grasp pose from above
goto_pose(grasp_pos, grasp_quat, z_approach={z_approach:.3f})

# Grasp the cube
close_gripper()

# Lift the cube to a safe height
lift_pos = grasp_pos + np.array([0.0, 0.0, {lift_height:.3f}])
goto_pose(lift_pos, grasp_quat)
```

#### `franka_nut_assembly_code_env` 多轮成功模板（3 轮拆分）

**轮 1：抓取 nut handle**

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Sample grasp pose for the nut handle
handle_pos, handle_quat = sample_grasp_pose("extruded handle of the brown square nut")

# Open gripper and approach
open_gripper()
goto_pose(handle_pos, handle_quat, z_approach={z_approach:.3f})

# Grasp the handle
close_gripper()
```

**轮 2：对准 peg**

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Get current nut center and peg poses
nut_pos, _ = get_object_pose("white hollow center of the brown square nut")
peg_pos, peg_quat = get_object_pose("square block")

# Compute handle-to-center transform
T_handle = pose_to_matrix(handle_pos, handle_quat)
T_nut = pose_to_matrix(nut_pos, handle_quat)
T_handle_to_center = np.linalg.inv(T_nut) @ T_handle

# Compute desired handle pose for insertion
T_peg = pose_to_matrix(peg_pos, peg_quat)
T_desired_handle = T_peg @ T_handle_to_center
desired_handle_pos, desired_handle_quat = matrix_to_pose(T_desired_handle)

# Move above peg
goto_pose(desired_handle_pos, desired_handle_quat, z_approach={z_approach:.3f})
```

**轮 3：插入并释放**

```python
# Final insertion
final_pos = desired_handle_pos + np.array([0.0, 0.0, -0.02])
goto_pose(final_pos, desired_handle_quat, z_approach=0.0)

# Release
open_gripper()
```

---

## 7. 错误/修正对生成

### 7.1 生成方法

对于每个成功代码，按一定概率生成对应的"错误前置状态 + 修正代码"样本：

1. **参数扰动法**（主要）：修改成功代码中的一个参数，使其失败，然后下一轮给出修正代码。
2. **阶段截断法**（多轮任务）：只执行成功代码的前半段，然后让模型继续完成。
3. **注入错误观测法**（可选）：构造一个假的失败观测作为 user message，target 为修正代码。

### 7.2 扰动策略自定义

每个任务的扰动类型和分布可以通过配置文件或命令行参数自定义。

#### 配置文件示例

```yaml
# configs/synthetic/franka_lift_perturbations.yaml
perturbations:
  missing_open_gripper:
    weight: 0.3
    description: "Omit open_gripper() before grasping"
    parameter: null
  z_approach_zero:
    weight: 0.3
    description: "Set z_approach to 0.0"
    parameter: "z_approach"
    error_value: 0.0
    correct_value: 0.05
  lift_height_too_low:
    weight: 0.4
    description: "Set lift height to 0.05"
    parameter: "lift_height"
    error_value: 0.05
    correct_value: 0.10
```

#### 命令行覆盖

```bash
python scripts/generate_synthetic_data.py \
  --task franka_lift_code_env \
  --train-size 100 \
  --perturbation-config configs/synthetic/franka_lift_perturbations.yaml
```

若未提供配置文件，则使用 `TaskConfig` 中的默认扰动分布。

### 7.3 具体扰动策略

#### `franka_lift_code_env`

| 扰动类型 | 错误代码 | 修正代码 |
|---|---|---|
| 缺少 open_gripper | 直接 goto_pose + close_gripper | 先 open_gripper |
| z_approach=0.0 | 直接撞击 cube | z_approach=0.05 |
| lift_height 太小 | 0.05 | 0.10 |

#### `franka_nut_assembly_code_env`

| 扰动类型 | 错误代码 | 修正代码 |
|---|---|---|
| 抓取位置不准 | z_approach=0.0 直接下压 | z_approach=0.05 缓慢接近 |
| 未回 home | 抓取后直接插 peg | 先 goto_home_joint_position |
| 插入深度不够 | final_pos 不下降 | final_pos = desired_handle_pos + [0,0,-0.02] |

### 7.4 轨迹格式

错误/修正对的轨迹格式：

```json
[
  {"role": "system", "content": "You are a helpful assistant..."},
  {"role": "user", "content": "Task: ...\nAPIs: ..."},
  {"role": "assistant", "content": "# 错误代码 v1\n..."},
  {"role": "user", "content": "The previous code failed: grasp missed the cube. Observation: ...\nPlease regenerate or finish."},
  {"role": "assistant", "content": "# 修正代码 v2\n..."}
]
```

---

## 8. 数据格式

### 8.1 SFT Parquet Schema

| 字段 | 类型 | 说明 |
|---|---|---|
| `data_source` | string | 环境名，如 `franka_lift_code_env` |
| `messages` | list[dict] | 完整对话轨迹 |
| `ability` | string | `"agent"` |
| `reward_model` | dict | `{"style": "sim_code", "ground_truth": {"program": "..."}}` |
| `extra_info` | dict | `{"split": "train/val", "index": int, "seed": int, "n_turns": int, "has_error": bool, "source": "synthetic"}` |

### 8.2 messages 结构

**单轮成功**：

```json
[
  {"role": "system", "content": "You are a helpful assistant that generates Python code to directly solve the task."},
  {"role": "user", "content": "Task: pick up the red cube and lift it.\nAPIs:\n..."},
  {"role": "assistant", "content": "import numpy as np\n..."}
]
```

**多轮成功**（以 nut_assembly 3 轮为例）：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Task: ...\nAPIs: ..."},
  {"role": "assistant", "content": "# Round 1: grasp handle\n..."},
  {"role": "user", "content": "Code executed successfully. Current observation: handle grasped. Please continue."},
  {"role": "assistant", "content": "# Round 2: align with peg\n..."},
  {"role": "user", "content": "Code executed successfully. Current observation: aligned with peg. Please continue."},
  {"role": "assistant", "content": "# Round 3: insert and release\n..."}
]
```

### 8.3 SFT 训练入口

verl 的 SFT 入口是 `verl/trainer/sft_trainer.py`（不是计划文档中写的 `main_sft.py`，本机 verl 没有该文件）。

#### 单卡训练

```bash
python -m verl.trainer.sft_trainer \
  data.train_files=data/processed/franka_lift_code_env/train.parquet \
  data.val_files=data/processed/franka_lift_code_env/val.parquet \
  data.max_length=2048 \
  model.path=Qwen/Qwen2.5-Coder-7B-Instruct \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=3 \
  trainer.project_name=capx-sft \
  trainer.experiment_name=franka_lift_code_env
```

#### 双卡训练（H200）

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m verl.trainer.sft_trainer \
  data.train_files=data/processed/franka_lift_code_env/train.parquet \
  data.val_files=data/processed/franka_lift_code_env/val.parquet \
  data.max_length=2048 \
  model.path=Qwen/Qwen2.5-Coder-7B-Instruct \
  trainer.n_gpus_per_node=2 \
  trainer.total_epochs=3 \
  trainer.project_name=capx-sft \
  trainer.experiment_name=franka_lift_code_env
```

#### 关键配置说明

- `data.messages_key=messages`：默认已配置，读取 `messages` 字段。
- `data.max_length`：建议设置为 2048，因为代码较长。
- `model.path`：使用 `Qwen/Qwen2.5-Coder-7B-Instruct`。
- 训练位置：`xys@192.168.110.102`（双卡 H200）。

---

## 9. 命令行接口设计

### 9.1 主脚本

```bash
python scripts/generate_synthetic_data.py \
  --task franka_lift_code_env \
  --train-size 100 \
  --val-size 20 \
  --output-dir data/processed \
  --seed 42 \
  --num-workers 4
```

### 9.2 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--task` | str | 必填 | 任务名，如 `franka_lift_code_env` |
| `--train-size` | int | 必填 | 训练集目标数量 |
| `--val-size` | int | 0 | 验证集目标数量 |
| `--output-dir` | Path | `data/processed` | 输出目录 |
| `--seed` | int | 42 | 随机种子 |
| `--num-workers` | int | 1 | 并行 worker 数 |
| `--visual-config-dir` | Path | `env_configs/synthetic` | visual YAML 配置目录 |
| `--single-turn-ratio` | float | 按任务默认 | 单轮成功样本比例 |
| `--multi-turn-ratio` | float | 按任务默认 | 多轮成功样本比例 |
| `--error-correction-ratio` | float | 按任务默认 | 错误/修正对比例 |
| `--max-turns` | int | 按任务默认 | 最大轮数 |
| `--verify` | bool | true | 是否执行全量验证 |
| `--save-failed` | bool | false | 是否保存失败样本（调试用） |
| `--auto-start-servers` | bool | false | 是否自动启动感知服务器 |
| `--server-ports` | dict | 默认端口 | 感知服务器端口 |

### 9.3 示例命令

```bash
# lift 任务
python scripts/generate_synthetic_data.py \
  --task franka_lift_code_env \
  --train-size 100 \
  --val-size 20 \
  --output-dir data/processed \
  --seed 42

# nut_assembly 任务
python scripts/generate_synthetic_data.py \
  --task franka_nut_assembly_code_env \
  --train-size 500 \
  --val-size 50 \
  --output-dir data/processed \
  --seed 42 \
  --num-workers 2
```

---

## 10. 输出文件结构

```
data/processed/
├── franka_lift_code_env/
│   ├── train.parquet
│   ├── val.parquet
│   ├── manifest.json
│   └── failed_samples.jsonl  (可选，调试用)
└── franka_nut_assembly_code_env/
    ├── train.parquet
    ├── val.parquet
    ├── manifest.json
    └── failed_samples.jsonl  (可选，调试用)
```

### 10.1 manifest.json 内容

```json
{
  "task": "franka_lift_code_env",
  "train": 100,
  "val": 20,
  "data_source": "franka_lift_code_env",
  "seed": 42,
  "single_turn_count": 80,
  "multi_turn_count": 0,
  "error_correction_count": 20,
  "success_rate": 1.0,
  "avg_turns": 1.2
}
```

---

## 11. 验证流程

### 11.1 验证要求

1. 每个 trajectory 的每一段 assistant code 都必须在 Privileged Env 中执行。
2. 只有最终 reward == 1.0 的 trajectory 才保留。
3. 对于多轮轨迹，每轮代码执行后都要检查中间状态是否合理（可选，第一阶段可只检查最终 reward）。

### 11.2 验证步骤

```python
# 对单轮成功样本
env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(code)
assert reward == 1.0

# 对多轮成功样本
env.reset(seed=seed)
for turn_code in turn_codes:
    obs, reward, terminated, truncated, info = env.step(turn_code)
    # 中间轮次可以不要求 reward==1.0，但要求无异常
assert reward == 1.0

# 对错误/修正对
env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(error_code)
assert reward < 1.0  # 确认确实是错误代码
env.reset(seed=seed)  # 重新 reset 到初始状态
obs, reward, terminated, truncated, info = env.step(correction_code)
assert reward == 1.0
```

---

## 12. 依赖和环境准备

### 12.1 代码依赖

- cap-x 环境已安装（`uv sync --extra robosuite`）
- verl 已安装
- 感知模型权重已下载（SAM3、Molmo、Contact GraspNet）

### 12.2 启动感知服务

第一阶段建议手动启动服务：

```bash
# SAM3
python -m capx.serving.launch_sam3_server --port 8114 --host 127.0.0.1

# Contact GraspNet
python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1

# PyRoKi
python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target_link panda_hand

# Molmo vLLM server（仅 nut_assembly 需要）
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate molmo
python -m vllm.entrypoints.openai.api_server \
  --model /data/cap-Model/Molmo2-8B/ \
  --port 8122 \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.4
```

### 12.3 Molmo 环境现状

根据对 `xys@192.168.110.102` 的实测：

- Molmo 权重路径存在：`/data/cap-Model/Molmo2-8B/`
- Conda 环境路径：`/home/xys/.conda/envs/molmo`
- 当前 `huggingface-hub==1.22.0` 与 `transformers` 要求的 `>=0.34.0,<1.0` 冲突
- 需要在服务器新配 venv 时修复该依赖冲突

---

## 13. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| Visual API 感知失败 | 生成的 prompt 无有效观测，样本无法生成 | 跳过该 seed，记录失败原因 |
| 多轮拆分后中间状态异常 | 多轮 trajectory 不连续 | 在中间轮次加入状态检查，异常则丢弃 |
| 感知服务启动复杂 | 影响使用体验 | 第一阶段手动启动，后续可考虑自动检测 |
| 数据产出速度慢 | Visual API 调用慢 | 控制第一阶段数据量，后续可并行优化 |
| Privileged/Prompt 不一致 | 模型学到错误的输入输出映射 | 严格保证 Visual Env 和 Privileged Env 用相同 seed reset |

---

## 14. 待审计点与初步回复

请重点审计以下内容。对于每个问题，下方给出了设计者的初步判断，供审计时参考：

### 14.1 双环境设计是否合理？

**初步判断**：合理，但成本高。

替代方案：
- 单 Visual 环境：简单，但感知错误会导致代码失败，数据产出率低。
- 单 Privileged 环境：快且 100% 成功，但 prompt 不真实，和部署不一致。

双环境（Visual prompt + Privileged code）是当前最稳妥的方案。如果后续发现 Visual API 感知错误率很低，可以考虑降级为单 Visual 环境简化实现。

### 14.2 Visual API 配置是否正确？

**初步判断**：本机代码已确认正确。

- `franka_lift_code_env` visual API 为 `FrankaControlApi`，包含 `home_pose`。
- `franka_nut_assembly_code_env` visual API 为 `FrankaControlNutAssemblyVisualApi`，包含 `goto_home_joint_position`，不含 `home_pose`。

需要注意：不同任务的 oracle_code 不能跨任务复用，因为 API 集合不同。

### 14.3 错误/修正对生成策略是否过于简化？

**初步判断**：第一阶段够用，但偏简单。

当前扰动主要是单参数错误（如 `z_approach` 大小、`lift_height` 太小）。更复杂的失败模式（如 API 顺序错误、高度公式错误）可以在第二阶段扩展。第一阶段先验证 pipeline。

### 14.4 数据量 100/500 是否合理？

**初步判断**：作为第一阶段验证 pipeline 合理，作为最终训练数据可能偏少。

建议：
- 第一阶段：lift 100 条 + nut_assembly 500 条，验证 pipeline。
- 第二阶段：根据 SFT 验证 loss 和仿真成功率，扩大到每个任务 1000-3000 条。

### 14.5 是否需要同时生成 PPO prompt 数据集？

**初步判断**：不需要单独生成。

PPO 是在线算法，只需要 prompt 集合。SFT parquet 中的 `messages` 字段前两条（system + user）可以直接作为 PPO prompt 使用，无需第一阶段额外生成。

### 14.6 多轮反馈的 user message 如何设计？

**初步判断**：第一阶段用文本反馈。

反馈内容包含：
- 上一轮代码执行结果（成功/失败）
- 当前观测摘要（如物体位置、夹爪状态）
- 继续生成或 FINISH 的指令

格式示例：
```
The previous code failed: grasp missed the cube. The cube is still at [0.45, 0.02, -0.08]. Please regenerate the code or type FINISH if the task is complete.
```

第一阶段不引入视觉差分信息，降低复杂度。后续可考虑加入 VDM 图像/视频描述。

---

## 15. 下一阶段工作

审计通过后，进入实现阶段：

1. 新增 `env_configs/synthetic/` 下的两个 YAML。
2. 实现 `scripts/generate_synthetic_data.py`。
3. 实现任务特定的代码模板和扰动策略。
4. 跑通 `franka_lift_code_env` 的 100 条数据生成。
5. 跑通 `franka_nut_assembly_code_env` 的 500 条数据生成。
6. 验证 parquet 能被 `verl.trainer.sft_trainer` 正确读取。
