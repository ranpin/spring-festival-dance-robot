#!/bin/bash
# 一键准备 Mac viser 演示环境并启动。幂等：可重复运行。
# 用法: ./scripts/bootstrap.sh [--port N] [--checks N]
set -eo pipefail

ENV_NAME="v2rviz"
PORT="${PORT:-8789}"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V2R="$ROOT/video2robot"
PROJECT_DIR="$V2R/data/video_001"

command -v conda >/dev/null 2>&1 || { echo "错误: 找不到 conda，请先安装 miniconda/anaconda" >&2; exit 1; }

CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "===> [1/4] 检查/创建 conda 环境 $ENV_NAME"
if [ ! -x "$PY" ]; then
  conda create -y -n "$ENV_NAME" python=3.11 -c conda-forge --override-channels
else
  echo "    环境已存在，跳过创建"
fi

echo "===> [2/4] 安装可视化依赖（torch CPU + viser + trimesh + opencv）"
"$PY" -m pip install --quiet torch numpy scipy joblib trimesh opencv-python viser==0.2.23

echo "===> [3/4] 校验演示产物"
for f in results.pkl original.mp4 robot_motion.pkl; do
  [ -f "$PROJECT_DIR/$f" ] || { echo "错误: 缺少 $PROJECT_DIR/$f —— 请先从 Kaggle 拉取产物" >&2; echo "  参见 docs/replication-notes.md 第 3.2/3.3 节" >&2; exit 1; }
done
TRACK_COUNT=$(ls "$PROJECT_DIR"/robot_motion_track_*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "    产物齐全，检测到 $TRACK_COUNT 条轨迹"

echo "===> [4/4] 启动演示（端口 $PORT）"
cd "$V2R"
if [ "${#EXTRA_ARGS[@]}" -eq 0 ]; then
  PORT="$PORT" ./run_demo.sh
else
  PORT="$PORT" ./run_demo.sh "${EXTRA_ARGS[@]}"
fi
