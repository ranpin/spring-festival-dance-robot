# spring-festival-dance-robot

复刻 [Datawhale `every-embodied` 教程](../../datawhalechina/every-embodied/blob/main/07-%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%93%8D%E4%BD%9C%E3%80%81%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6/Locomotion/01%E6%98%A5%E6%99%9A%E8%88%9E%E8%B9%88%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%A4%8D%E5%88%BB.md) 的「春晚舞蹈机器人」项目：

```
文字/视频 → PromptHMR(SMPL-X) → GMR → unitree_g1 动作 → viser 交互式 3D
```

**当前状态**：dance_1.mp4（春晚群舞 39s，1180 帧）已在 Kaggle P100 上完整跑通，9 个舞者全部重定向为 unitree_g1 机器人动作，Mac 上 viser 演示运行中。

---

## Repo 结构

```
spring-festival-dance-robot/
├── video2robot/                  # vendored from datawhalechina/every-embodied (main HEAD)
│   ├── scripts/                  # extract_pose / convert_to_robot / visualize / run_pipeline
│   ├── video2robot/              # python package
│   ├── web/                      # FastAPI + viser web UI
│   ├── configs/                  # default.yaml 等
│   ├── data/                     # video_001..005 占位（产物不入仓）
│   ├── envs/                     # 上游 phmr/gmr conda yamls（参考）
│   ├── patches/                  # 上游 patches（已应用到 third_party 内）
│   ├── run_demo.sh               # Mac 交互式演示启动脚本（本仓库新增）
│   └── third_party/
│       ├── GMR/                  # vendored from taeyoun811/GMR@069b4fd + gmr.patch；assets 裁到 unitree_g1
│       └── PromptHMR/            # vendored from taeyoun811/PromptHMR@4f8915c + prompthmr.patch
└── kaggle/                       # Kaggle GPU kernel（本仓库的核心）
    ├── main.py                   # 18 步全流程：env → weights → extract_pose → convert_to_robot
    ├── kernel-metadata.json      # Kaggle kernel 元数据（GPU + Internet）
    └── probe.py                  # GPU/CUDA/conda 探针
```

产物（`results.pkl` / `robot_motion*.pkl` / `smplx*.npz`）**不入仓**——体积大且可重生。需要时跑 Kaggle kernel 自动产出，或从 SCM 存档拉取（见下文）。

---

## 跑法

### 1. Kaggle GPU 推理（产出动作文件）
```bash
cd kaggle
kaggle kernels push                              # 提交 kernel
kaggle kernels status mamihlapinatapai/v2r-main  # 监控（P100 上 ~1.5h）
kaggle kernels output mamihlapinatapai/v2r-main -p /tmp/v2r_out
ls /tmp/v2r_out/output/                          # results.pkl + robot_motion*.pkl + smplx*.npz + original.mp4
```

### 2. Mac 本地演示（viser 交互式 3D）
```bash
# 一次性准备 v2rviz conda env（torch CPU + viser + trimesh + opencv）
conda create -y -n v2rviz python=3.11 -c conda-forge --override-channels
conda run -n v2rviz python -m pip install torch numpy scipy joblib trimesh opencv-python viser==0.2.23

# 把 Kaggle 产物放进 video2robot/data/video_001/
mkdir -p video2robot/data/video_001
cp /tmp/v2r_out/output/* video2robot/data/video_001/

# 启动演示
cd video2robot && ./run_demo.sh
# 浏览器打开 http://localhost:8789
# 局域网: http://<Mac IP>:8789
```

`run_demo.sh` 用法：
```bash
./run_demo.sh                          # 全部 9 个机器人 + subsample 2
./run_demo.sh --tracks 1 2 3           # 只看某几个
PORT=9000 ./run_demo.sh                # 换端口
```

---

## Baseline & Patches（vendor 状态）

| 源 | commit/状态 | patches |
|---|---|---|
| `video2robot/` | datawhalechina/every-embodied main HEAD | 上游 `main.patch` 已合并到 HEAD |
| `third_party/GMR/` | `taeyoun811/GMR@069b4fd` | `patches/gmr.patch` 已应用（添加 `--once/--max_seconds`）|
| `third_party/PromptHMR/` | `taeyoun811/PromptHMR@4f8915c` | `patches/prompthmr.patch` 已应用（sam2 import / droidcalib / `.type→.scalar_type` / 去 SMPL 依赖）|

升级三方源后，patches 可能需要重新 rebase；本仓库冻结在已知好状态以便迭代。

---

## 关键踩坑（复现/迭代时必看）

### Kaggle 环境
- Kaggle **无 conda** → 装 miniconda 到大 overlay（`/root/...` 或 `/tmp/...`），别用 `/kaggle/working`（仅 20GB）。
- 新版 conda 建 env 前要接受 Anaconda ToS → 用 `-c conda-forge --override-channels` 绕开。
- Kaggle GPU 是 **P100/T4**（不是 4090），但 cp311 wheels 在 P100(sm_60) 上验证可用，T4(sm_75) 也覆盖。
- `/kaggle/working` 是产物出口（20GB 上限），weights/env/repos 都放大 overlay。

### 权重路径
- HF `Datawhale/spring-festival-wushu-robot-replication-model`（~14.8G：pretrain + body_models + examples + wheels，**包含春晚舞蹈视频** `examples/dance_1.mp4`）。
- HF weights 的 `data/` 软链到 `third_party/PromptHMR/data/` 即可（config.py 已约定路径）。
- PromptHMR **硬编码 `/code/data/pretrain/camcalib_sa_biplied_l2.ckpt`** → 必须 `ln -sfn $PH /code`（kernel STEP 10 已处理）。
- HF wheels（detectron2 / lietorch / sam2 / droid_backends_intr / gloss）是 **cp311 linux_x86_64** → phmr env 必须 **Python 3.11**，不能用上游 yml 的 py3.10。
- `yolo11x.pt` 不必有，检测器硬编码 `detectron2`（权重在 `sam2_ckpts/keypoint_rcnn_5ad38f.pkl`）。

### Pipeline
- `--static-camera` 跳过 DROID-SLAM → **绕开缺失的 `droid.pth`**。春晚舞台机位基本固定，static 完全 OK。
- `extract_pose` 后 `results.pkl` 含 masks（1180 帧 × N 人 × 896² bool）会膨胀到 ~11GB → kernel STEP 18 自动 joblib 瘦身（drop masks，保 people + camera_world），缩到 ~100MB。
- 1180 帧在 P100 上 extract_pose ~1h10m，convert_to_robot ~5m。整轮 ~1.5h。

### Mac 演示
- robot_viser.py **依赖很轻**：torch CPU + viser + trimesh + opencv + joblib + scipy + numpy。
- 不需要 detectron2/sam2/MuJoCo（它自己用纯 torch 实现 MJCF 运动学解析器）。
- GMR 网格资产挂在 `third_party/GMR/assets/unitree_g1/`（vendor 时已裁掉其他机器人，只留这一个）。

---

## Acknowledgement

- Datawhale [`every-embodied`](https://github.com/datawhalechina/every-embodied) — 教程源文档
- [`taeyoun811/PromptHMR`](https://github.com/taeyoun811/PromptHMR) — SMPL-X 人体姿态提取
- [`taeyoun811/GMR`](https://github.com/taeyoun811/GMR) — 通用动作重定向
- HF: [`Datawhale/spring-festival-wushu-robot-replication-model`](https://huggingface.co/Datawhale/spring-festival-wushu-robot-replication-model) — 权重 + cp311 wheels
