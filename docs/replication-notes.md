# 复现笔记（Replication Notes）

本仓库复刻自 [Datawhale `every-embodied` 教程](https://github.com/datawhalechina/every-embodied)。本文记录从零到新机器复现的**完整步骤、踩坑与排查**，作为 README 的补充细节版。

> 最终状态：`dance_1.mp4`（春晚群舞 39s，1180 帧）在 Kaggle P100 上完整跑通，9 个舞者全部重定向为 unitree_g1 机器人动作。

---

## 1. 流水线总览

```
舞蹈视频 ──[PromptHMR]──> SMPL-X 人体姿态 ──[GMR]──> unitree_g1 关节动作 ──[robot_viser]──> 浏览器交互式 3D
```

| 阶段 | 工具 | 运行环境 | 产物 |
|---|---|---|---|
| 人体姿态提取 | PromptHMR | Kaggle GPU（CUDA，重）| `results.pkl` / `smplx*.npz` |
| 动作重定向 | GMR（mink IK）| Kaggle（CPU）| `robot_motion*.pkl` |
| 可视化 | `robot_viser.py` | Mac（CPU，轻）| 浏览器 viser 场景 |

**分工原则**：Mac 无 NVIDIA GPU → 重推理必须在 Kaggle；Mac 只做轻量可视化。

---

## 2. 环境准备

### 2.1 Mac（可视化端）
```bash
# 可视化只需 CPU 轻量环境
conda create -y -n v2rviz python=3.11 -c conda-forge --override-channels
conda run -n v2rviz python -m pip install torch numpy scipy joblib trimesh opencv-python viser==0.2.23
```
- `robot_viser.py` 自己用纯 torch 实现 MJCF 运动学解析器，**不需要** detectron2/sam2/MuJoCo。
- GMR 网格资产挂在 `video2robot/third_party/GMR/assets/unitree_g1/`（vendor 时已裁掉其他机器人）。

### 2.2 Kaggle（推理端）
Kaggle kernel 由 `kaggle/main.py` 自动搭建环境，无需手动配置。关键约束：

| 约束 | 说明 | 应对 |
|---|---|---|
| Kaggle **无 conda** | 系统 Python 3.12，无 conda | 装 miniconda 到大 overlay |
| `/kaggle/working` 仅 20GB | 产物出口，装不下权重+环境 | 权重/env/repos 放 `/root/v2r_scratch`（overlay ~1.1T）|
| 新版 conda 需 ToS | 建 env 前要接受 Anaconda ToS | `conda create ... -c conda-forge --override-channels` |
| GPU 是 P100/T4 | 不是 4090 | cp311 wheels 已验证覆盖 sm_60/sm_75 |

---

## 3. 完整复现步骤（新机器）

### 3.1 克隆仓库
```bash
git clone https://github.com/ranpin/spring-festival-dance-robot.git
cd spring-festival-dance-robot
```
仓库已 **vendor** 三方源（无需再单独 clone）：
- `video2robot/` ← datawhalechina/every-embodied main HEAD
- `video2robot/third_party/GMR/` ← `taeyoun811/GMR@069b4fd` + `gmr.patch`（assets 裁到 unitree_g1）
- `video2robot/third_party/PromptHMR/` ← `taeyoun811/PromptHMR@4f8915c` + `prompthmr.patch`

### 3.2 Kaggle 推理（产出动作文件）
```bash
cd kaggle
kaggle kernels push                              # 提交 kernel（需 ~/.kaggle/kaggle.json 凭证）
kaggle kernels status <user>/v2r-main            # 监控（P100 ~1.5h）
kaggle kernels output <user>/v2r-main -p /tmp/v2r_out
ls /tmp/v2r_out/output/                          # results.pkl + robot_motion*.pkl + smplx*.npz + original.mp4
```

### 3.3 Mac 演示
```bash
mkdir -p video2robot/data/video_001
cp /tmp/v2r_out/output/* video2robot/data/video_001/
cd video2robot && ./run_demo.sh
# 浏览器打开 http://localhost:8789（局域网 http://<Mac IP>:8789）
```

---

## 4. Kaggle Kernel 详解（`kaggle/main.py` 18 步）

| STEP | 动作 | 关键点 |
|---|---|---|
| 0 | 选 scratch 目录 | 优先 `/root/v2r_scratch`（大 overlay），避免 `/kaggle/working` |
| 1 | apt 装 libgl1/ffmpeg | 非致命 |
| 2 | 装 miniconda | 装到 scratch，别装系统目录 |
| 3 | 建 env v2r（py3.11.9）| `-c conda-forge --override-channels` 绕 ToS |
| 4 | torch 2.4.0 cu121 | `--index-url https://download.pytorch.org/whl/cu121` |
| 5 | torch-scatter | 从 pyg wheel 装，对应 torch-2.4.0+cu121 |
| 6 | clone 本仓库 | 现已 vendor，无需再 clone 上游 |
| 7 | 验证 patches 已 baked-in | 断言 sam2 import / droidcalib / `--max_seconds` 存在 |
| 8 | PromptHMR requirements + chumpy + xformers | chumpy 用 `--no-build-isolation` |
| 9 | 下载 HF 权重 | `Datawhale/spring-festival-wushu-robot-replication-model`，含 `examples/dance_1.mp4` |
| 10 | 软链权重 + `/code` | HF `data/` → `third_party/PromptHMR/data/`；`ln -sfn $PH /code`（PromptHMR 硬编码 `/code/...` 路径）|
| 11 | 装 4 个 cp311 wheels | detectron2/droid_backends_intr/lietorch/sam2，来自 HF `data/wheels/` |
| 12 | 装 GMR 依赖 | mujoco/mink/smplx 等 |
| 13 | smoke test | 验证 CUDA + 各库 import |
| 14 | 短片段验证（boxing）| 快速跑通全链路，早发现 sm_60 kernel 问题 |
| 15 | extract_pose（dance_1）| `--static-camera`，~1h10m |
| 16 | convert_to_robot | `--all-tracks`，9 条轨迹 |
| 17 | MuJoCo 离屏渲染 | `MUJOCO_GL=egl`，非致命 |
| 18 | 收集产物 + results.pkl 瘦身 | drop masks（11GB→~100MB），copy 到 `/kaggle/working/output` |

---

## 5. 关键踩坑与修复（必看）

### 5.1 权重路径
- HF 权重仓库 `data/` 软链到 `third_party/PromptHMR/data/` 即可（config.py 已约定路径）。
- PromptHMR `cam_calib.py` **硬编码** `/code/data/pretrain/camcalib_sa_biplied_l2.ckpt` → 必须 `ln -sfn <PromptHMR根> /code`（kernel STEP 10 已处理）。
- HF wheels（detectron2/lietorch/sam2/droid_backends_intr/gloss）是 **cp311 linux_x86_64** → env 必须 **Python 3.11**，**不能用**上游 `envs/phmr.yml`（py3.10 且 detectron2/lietorch 不在 PyPI，必失败）。
- `yolo11x.pt` 不必有：检测器硬编码 `detectron2`，权重在 `sam2_ckpts/keypoint_rcnn_5ad38f.pkl`。

### 5.2 Pipeline
- `--static-camera` 跳过 DROID-SLAM → **绕开权重仓库缺失的 `droid.pth`**。春晚舞台机位基本固定，static 完全 OK。
- `results.pkl` 含 masks（1180 帧 × N 人 × 896² bool）会膨胀到 **~11GB** → kernel STEP 18 用 joblib 自动瘦身（drop masks，保 people + camera_world），缩到 ~100MB。`robot_viser.py` 只需 people + camera_world，不需 masks。
- 1180 帧在 P100 上：extract_pose ~1h10m，convert_to_robot ~5m，整轮 ~1.5h。

### 5.3 Kaggle 运行
- Kaggle kernel 是纯 web notebook，**无 SSH** → 用 `kaggle kernels push/output` CLI 交互。
- 单 GPU session 最长 9h；P100 推理 1.5h 在限内。
- Mac 端 DNS 偶尔抽风导致状态轮询超时 → kernel 本身不受影响，重连即可。

---

## 6. 故障排查表

| 现象 | 根因 | 修复 |
|---|---|---|
| kernel 报 `WORK: unbound variable` | bash 脚本未定义 `WORK` | kernel 顶部加 `WORK=/kaggle/working` |
| extract_pose 报找不到 `camcalib_sa_biplied_l2.ckpt` | PromptHMR 硬编码 `/code/...` | `ln -sfn $PH /code` |
| pip 装 detectron2/lietorch 失败 | 用了上游 phmr.yml（py3.10）| 改用 py3.11 + HF cp311 wheels |
| results.pkl 传不回 Mac（11GB）| masks 膨胀 | STEP 18 joblib 瘦身 |
| `conda create` 卡 ToS | 新版 conda 要接受 ToS | `-c conda-forge --override-channels` |
| MuJoCo 渲染报 X11/GLFW | Kaggle 无显示器 | `MUJOCO_GL=egl` 离屏渲染 |
| 演示黑屏/不动 | 产物不全 | 确认 `data/video_001/` 有 results.pkl + robot_motion*.pkl + original.mp4 |

---

## 7. 已知限制

- 仅 vendor 了 `unitree_g1` 机器人资产（其他机器人未保留）；如需其他机器人，从 GMR 上游重新 vendor。
- 权重未入仓（14.8G），复现时需 Kaggle 自动从 HF 下载。
- 输入视频用权重仓库自带的 `examples/dance_1.mp4`；如需其他视频，替换 `data/video_001/original.mp4` 后重跑 kernel。
