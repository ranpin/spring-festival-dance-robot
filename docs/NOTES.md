# spring-festival-dance-robot 复刻笔记：工作边界、vendor 基线与关键踩坑

[README](../README.md) 的补充文档：README 讲核心结果与快速开始，本文讲工作边界（诚实边界）、三方 vendor 基线与复现/迭代时的踩坑。

## 工作边界（诚实边界）

本项目是 Datawhale [`every-embodied` 教程](https://github.com/datawhalechina/every-embodied/blob/main/07-%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%93%8D%E4%BD%9C%E3%80%81%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6/Locomotion/01%E6%98%A5%E6%99%9A%E8%88%9E%E8%B9%88%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%A4%8D%E5%88%BB.md)的**复刻**，核心算法（PromptHMR / GMR）均来自上游。本人的工作是：

- 把三仓 vendor 进一个**可复现**仓库（冻结在已知好 commit + patch，见下表）；
- 适配 Kaggle kernel（`kaggle/main.py` 18 步全流程 GPU 推理，clone 本仓库即可跑、不再依赖 every-embodied）；
- 打 patch 修通环境（sam2 import / droidcalib / `.type→.scalar_type` / 去 SMPL 依赖等）；
- 新增 Mac 本地 viser 交互演示（`video2robot/run_demo.sh`，纯 CPU）。

上游署名见 README[「许可与致谢」](../README.md#许可与致谢)。

## Vendor 基线（Baseline & Patches）

| 源 | commit/状态 | patches |
|---|---|---|
| `video2robot/` | datawhalechina/every-embodied main HEAD | 上游 `main.patch` 已合并到 HEAD |
| `third_party/GMR/` | `taeyoun811/GMR@069b4fd` | `patches/gmr.patch` 已应用（添加 `--once/--max_seconds`）；assets 裁到只剩 unitree_g1 |
| `third_party/PromptHMR/` | `taeyoun811/PromptHMR@4f8915c` | `patches/prompthmr.patch` 已应用（sam2 import / droidcalib / `.type→.scalar_type` / 去 SMPL 依赖）|

升级三方源后，patches 可能需要重新 rebase；本仓库冻结在已知好状态以便迭代。

## 关键踩坑（复现/迭代时必看）

### Kaggle 环境
- Kaggle **无 conda** → 装 miniconda 到大 overlay（`/root/...` 或 `/tmp/...`），别用 `/kaggle/working`（仅 20GB）。
- 新版 conda 建 env 前要接受 Anaconda ToS → 用 `-c conda-forge --override-channels` 绕开。
- Kaggle GPU 是 **P100/T4**（不是 4090），但 cp311 wheels 在 P100(sm_60) 上验证可用，T4(sm_75) 也覆盖。
- `/kaggle/working` 是产物出口（20GB 上限），weights/env/repos 都放大 overlay。

### 权重路径
- HF `Datawhale/spring-festival-wushu-robot-replication-model`（~14.8G：pretrain + body_models + examples + wheels，**包含春晚舞蹈视频** `examples/dance_1.mp4`）。
- HF weights 的 `data/` 软链到 `third_party/PromptHMR/data/` 即可（config.py 已约定路径）。
- PromptHMR **硬编码 `/code/data/pretrain/camcalib_sa_biased_l2.ckpt`** → 必须 `ln -sfn $PH /code`（kernel STEP 10 已处理）。
- HF wheels（detectron2 / lietorch / sam2 / droid_backends_intr）是 **cp311 linux_x86_64** → phmr env 必须 **Python 3.11**，不能用上游 yml 的 py3.10（wheels 里另有 cp38-abi3 的 gloss，kernel 未安装）。
- `yolo11x.pt` 不必有，检测器硬编码 `detectron2`（权重在 `sam2_ckpts/keypoint_rcnn_5ad38f.pkl`）。

### Pipeline
- `--static-camera` 跳过 DROID-SLAM → **绕开缺失的 `droid.pth`**。春晚舞台机位基本固定，static 完全 OK。
- `extract_pose` 后 `results.pkl` 含 masks（1180 帧 × N 人 × 896² bool）会膨胀到 ~11GB → kernel STEP 18 自动 joblib 瘦身（drop masks，保 people + camera_world），缩到 ~100MB。
- 1180 帧在 P100 上 extract_pose ~1h10m，convert_to_robot ~5m。整轮 ~1.5h。

### Mac 演示
- robot_viser.py **依赖很轻**：torch CPU + viser + trimesh + opencv + joblib + scipy + numpy。
- 不需要 detectron2/sam2/MuJoCo（它自己用纯 torch 实现 MJCF 运动学解析器）。
- GMR 网格资产挂在 `third_party/GMR/assets/unitree_g1/`（vendor 时已裁掉其他机器人，只留这一个）。
