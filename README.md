# spring-festival-dance-robot

> 复刻 [Datawhale `every-embodied` 教程](https://github.com/datawhalechina/every-embodied/blob/main/07-%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%93%8D%E4%BD%9C%E3%80%81%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6/Locomotion/01%E6%98%A5%E6%99%9A%E8%88%9E%E8%B9%88%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%A4%8D%E5%88%BB.md)的「春晚舞蹈机器人」：一段春晚群舞视频，变成 9 台 Unitree G1 机器人的齐舞动作，Mac 上可交互 3D 演示。

> 🎬 **在线 Demo**：<https://ranpin.github.io/spring-festival-dance-robot/> —— 「人类原舞 vs 9 台 G1 重定向」逐帧对齐双视频对照 + 五阶段管线与工程要点。

## 核心结果

- dance_1.mp4（春晚群舞，39s / 1180 帧 @30fps / 检出 9 位舞者）在 Kaggle GPU（P100，全流程 ~90–100 分钟）完整跑通：PromptHMR 提取 SMPL-X → GMR 重定向到 unitree_g1（29 DoF），9 个机器人动作全部产出。
- Mac 本地 viser 交互式 3D 演示：纯 CPU、轻依赖（torch CPU + viser + trimesh + opencv），`run_demo.sh` 一条命令启动。
- demo 页有「人类原舞 vs 9 台 G1 重定向」逐帧对齐的双视频对照（前 24s / 720 帧）。

管线（五阶段，重推理在 Kaggle GPU、3D 演示在 Mac CPU）：

```
视频 → PromptHMR(SMPL-X) → GMR 重定向(G1 29-DoF) → 机器人动作 → viser 交互式 3D
```

本仓做的工作：把三仓 vendor 进一个**可复现**仓库（冻结在已知好 commit + patch）、适配 18 步全流程的 Kaggle GPU kernel、新增 Mac 本地 viser 交互演示。核心算法（PromptHMR / GMR）来自上游（见「许可与致谢」）；工作边界、vendor 基线与复现踩坑见 [docs/NOTES.md](docs/NOTES.md)。

## 快速开始

### 1. Kaggle GPU 推理（产出动作文件）

```bash
cd kaggle
kaggle kernels push                              # 提交 kernel
kaggle kernels status mamihlapinatapai/v2r-main  # 监控（P100 上 ~90–100 分钟）
kaggle kernels output mamihlapinatapai/v2r-main -p /tmp/v2r_out
ls /tmp/v2r_out/output/                          # results.pkl + robot_motion*.pkl + smplx*.npz + original.mp4
```

### 2. Mac 本地演示（viser 交互式 3D）

```bash
# 一次性准备 v2rviz conda env（torch CPU + viser + trimesh + opencv）
# 注：run_demo.sh 内 PY 硬编码为 /Users/ranpin/miniconda3/envs/v2rviz/bin/python，其他机器请改成自己的环境路径
conda create -y -n v2rviz python=3.11 -c conda-forge --override-channels
conda run -n v2rviz python -m pip install torch numpy scipy joblib trimesh opencv-python viser==0.2.23

# 把 Kaggle 产物放进 video2robot/data/video_001/
mkdir -p video2robot/data/video_001
cp /tmp/v2r_out/output/* video2robot/data/video_001/

# 启动演示
cd video2robot && ./run_demo.sh
# 浏览器打开 http://localhost:8789；局域网: http://<Mac IP>:8789
```

`run_demo.sh` 用法：`./run_demo.sh`（全部 9 个机器人 + subsample 2）、`--tracks 1 2 3`（只看某几个）、`PORT=9000 ./run_demo.sh`（换端口）。

产物（`results.pkl` / `robot_motion*.pkl` / `smplx*.npz`）**不入仓**——体积大且可重生，需要时跑 Kaggle kernel 自动产出。

⚠️ 复现前必看：权重路径（HF ~14.8G、PromptHMR 硬编码 `/code` 软链）、`--static-camera` 绕开缺失的 `droid.pth`、`results.pkl` masks 瘦身（~11GB→~100MB）、Kaggle 磁盘配额等——见 [docs/NOTES.md](docs/NOTES.md)。

## 仓库结构与文档导航

```
spring-festival-dance-robot/
├── video2robot/                  # vendored from datawhalechina/every-embodied (main HEAD)
│   ├── scripts/                  # extract_pose / convert_to_robot / visualize / run_pipeline
│   ├── video2robot/              # python package
│   ├── web/                      # FastAPI + viser web UI
│   ├── configs/                  # default.yaml 等
│   ├── data/                     # 仅 .gitkeep 占位（video_001..005 运行时自建，产物不入仓）
│   ├── envs/                     # 上游 phmr/gmr conda yamls（参考）
│   ├── patches/                  # 上游 patches（已应用到 third_party 内）
│   ├── run_demo.sh               # Mac 交互式演示启动脚本（本仓库新增）
│   └── third_party/
│       ├── GMR/                  # vendored from taeyoun811/GMR@069b4fd + gmr.patch；assets 裁到 unitree_g1
│       └── PromptHMR/            # vendored from taeyoun811/PromptHMR@4f8915c + prompthmr.patch
├── kaggle/                       # Kaggle GPU kernel（本仓库的核心）
│   ├── main.py                   # 18 步全流程：env → weights → extract_pose → convert_to_robot
│   ├── kernel-metadata.json      # Kaggle kernel 元数据（GPU + Internet）
│   └── probe.py                  # GPU/CUDA/conda 探针
├── docs/                         # 演示页（index.html + 双视频，Pages 部署）+ NOTES.md（工作边界/vendor 基线/踩坑）
└── LICENSE                       # MIT
```

## 许可与致谢

MIT——见 [`LICENSE`](LICENSE)。

- Datawhale [`every-embodied`](https://github.com/datawhalechina/every-embodied) — 教程源文档
- [`taeyoun811/PromptHMR`](https://github.com/taeyoun811/PromptHMR) — SMPL-X 人体姿态提取
- [`taeyoun811/GMR`](https://github.com/taeyoun811/GMR) — 通用动作重定向
- HF: [`Datawhale/spring-festival-wushu-robot-replication-model`](https://huggingface.co/Datawhale/spring-festival-wushu-robot-replication-model) — 权重 + cp311 wheels（含春晚舞蹈视频 `dance_1.mp4`）
