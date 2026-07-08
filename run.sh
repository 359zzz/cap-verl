#!/usr/bin/env bash
# Cap-X + VeRL PPO/OPD 一键运行入口
# 待实现

set -euo pipefail

echo "Cap-X + VeRL PPO/OPD pipeline"
echo "Usage: bash run.sh [phase]"
echo ""
echo "Phases:"
echo "  generate    - 生成合成数据"
echo "  sft         - SFT 训练"
echo "  bootstrap   - 自举数据增强"
echo "  ppo         - PPO 专家训练"
echo "  opd         - OPD 蒸馏合并"
