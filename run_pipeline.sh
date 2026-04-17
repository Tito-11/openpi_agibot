#!/bin/bash
set -e

# 跳过 Git LFS 大文件下载，防止测试集对象缺失导致 uv install 崩溃
export GIT_LFS_SKIP_SMUDGE=1
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=1

# 同步并安装依赖，触发 pyproject.toml 里面新的国内代理链接
uv sync

echo "============================================="
echo " 第1步：转换原始数据为 LeRobot HuggingFace 格式"
echo "============================================="
uv run agi_bot/convert_agibot_data_to_lerobot.py --data_dir agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2

echo ""
echo "============================================="
echo " 第2步：计算 Dataset 归一化参数 (Norm Stats)"
echo "============================================="
uv run scripts/compute_norm_stats.py --config-name pi0_agibot

echo ""
echo "============================================="
echo " 第3步：按配置启动 30,000 步 LoRA 预训练"
echo " (支持 WandB 在线监控 Loss 曲线)"
echo "============================================="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_agibot --exp-name=agibot_routeB_lora_tuning --overwrite
