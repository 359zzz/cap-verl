# Cap-X + VeRL PPO/OPD

基于 cap-x 仿真环境和 verl RL 框架的机器人操作策略训练项目。

## 项目结构

```
cap-verl/
├── cap-x/          # cap-x 代码库（独立 git 仓库）
├── verl/           # verl 代码库（独立 git 仓库）
├── configs/        # 训练配置文件
├── docs/           # 设计文档
├── scripts/        # 项目级脚本
├── run.sh          # 一键运行入口
└── requirements.txt
```

## 环境准备

```bash
# 1. clone 本仓库
git clone https://github.com/359zzz/cap-verl.git
cd cap-verl

# 2. clone cap-x 和 verl
git clone https://github.com/capgym/cap-x.git
git clone https://github.com/359zzz/verl.git

# 3. 安装依赖
# (详见后续安装文档)
```

## 主要文档

- [总体工程计划](docs/capx_verl_ppo_opd_engineering_plan_v2.md)
- [第一阶段合成数据设计](docs/capx_verl_ppo_opd_synthetic_data_design_v1.md)
