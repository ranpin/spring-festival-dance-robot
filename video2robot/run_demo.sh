#!/bin/bash
# 春晚舞蹈机器人 —— 交互式 viser 3D 演示（Mac 本地，CPU）
# 用法:
#   ./run_demo.sh                 # 展示 data/video_001 全部轨迹
#   ./run_demo.sh --tracks 1      # 只展示第 1 条轨迹
#   PORT=9000 ./run_demo.sh       # 换端口
set -euo pipefail

V2R="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/ranpin/miniconda3/envs/v2rviz/bin/python"
PROJECT="${PROJECT:-data/video_001}"
PORT="${PORT:-8789}"

if [ ! -x "$PY" ]; then
  echo "错误: 找不到可视化环境 $PY" >&2; exit 1
fi
if [ ! -f "$V2R/$PROJECT/results.pkl" ] || [ ! -f "$V2R/$PROJECT/original.mp4" ]; then
  echo "错误: $PROJECT 缺少 results.pkl 或 original.mp4（需先从 Kaggle 拉回产物）" >&2; exit 1
fi
ls "$V2R/$PROJECT"/robot_motion*.pkl >/dev/null 2>&1 || {
  echo "错误: $PROJECT 缺少 robot_motion*.pkl" >&2; exit 1; }

cd "$V2R"
echo "===================================================="
echo " 春晚舞蹈机器人 · viser 3D 交互演示"
echo " 项目: $PROJECT   端口: $PORT"
echo " 浏览器打开:  http://localhost:$PORT"
echo " (同网段其他设备:  http://<本机IP>:$PORT)"
echo " Ctrl+C 退出"
echo "===================================================="
# 无参数时默认：全部轨迹 + 抽帧2（1180帧视频降负载，保证流畅）
if [ "$#" -eq 0 ]; then
  exec "$PY" video2robot/visualization/robot_viser.py \
    --project "$PROJECT" --all-tracks --subsample 2 \
    --host 0.0.0.0 --port "$PORT"
else
  exec "$PY" video2robot/visualization/robot_viser.py \
    --project "$PROJECT" "$@" \
    --host 0.0.0.0 --port "$PORT"
fi
