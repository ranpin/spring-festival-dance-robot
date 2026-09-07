import os, subprocess, time

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
LOG = os.path.join(WORK, "v2r_main.log")

BASH = r'''
set -uo pipefail
die(){ echo "FATAL: $1"; exit 1; }

echo "############ STEP 0: scratch dir ############"
SCRATCH=""
for cand in /root/v2r_scratch "$HOME/v2r_scratch" /tmp/v2r_scratch; do
  if mkdir -p "$cand" 2>/dev/null && touch "$cand/.w" 2>/dev/null; then
    rm -f "$cand/.w"; SCRATCH="$cand"; break
  fi
done
[ -n "$SCRATCH" ] || die "no writable scratch dir found"
echo "SCRATCH=$SCRATCH"
df -h "$SCRATCH" || true
whoami || true

CONDA="$SCRATCH/miniconda3"
PY="$CONDA/envs/v2r/bin/python"
PIP="$CONDA/envs/v2r/bin/pip"
ACT="source $CONDA/etc/profile.d/conda.sh && conda activate v2r"
V2R="$SCRATCH/spring-festival-dance-robot/video2robot"
PH="$V2R/third_party/PromptHMR"
GMRD="$V2R/third_party/GMR"
HFWE="$SCRATCH/hf_weights"
WORK=/kaggle/working

echo "############ STEP 1: system libs (non-fatal) ############"
(apt-get update -y && apt-get install -y libgl1 libglib2.0-0 ffmpeg) || echo "WARN: apt install failed (non-fatal)"

echo "############ STEP 2: miniconda ############"
if [ ! -x "$CONDA/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/miniconda.sh" || die "miniconda download"
  bash "$SCRATCH/miniconda.sh" -b -p "$CONDA" || die "miniconda install"
fi
"$CONDA/bin/conda" --version || die "conda missing"
# conda>=24.9 requires accepting ToS for default anaconda channels; accept them (non-fatal)
"$CONDA/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "############ STEP 3: create env v2r (py3.11) ############"
"$CONDA/bin/conda" create -y -n v2r python=3.11.9 -c conda-forge --override-channels || die "conda create"
"$PY" --version || die "env python missing"

echo "############ STEP 4: torch 2.4 cu121 ############"
"$PIP" install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 || die "torch install"
"$PIP" install --upgrade setuptools pip || die "setuptools"

echo "############ STEP 5: torch-scatter ############"
"$PIP" install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html || echo "WARN: torch-scatter failed (non-fatal)"

echo "############ STEP 6: clone repo (vendored) ############"
cd "$SCRATCH" || die "cd scratch"
if [ ! -d "$SCRATCH/spring-festival-dance-robot/.git" ]; then
  git clone https://github.com/ranpin/spring-festival-dance-robot.git || die "clone repo"
fi
[ -d "$V2R" ] || die "video2robot dir missing"
[ -d "$PH" ]  || die "PromptHMR not vendored at $PH"
[ -d "$GMRD" ] || die "GMR not vendored at $GMRD"

echo "############ STEP 7: verify patches baked-in ############"
cd "$V2R" || die "cd V2R"
# Vendor 时已在 GMR/PromptHMR 上应用过 patch；这里只做断言（防止后续误改）
[ -f "$PH/pipeline/detector/sam2_video_predictor.py" ] || die "prompthmr.patch artifact missing"
grep -q "scalar_type" "$PH/pipeline/droidcalib/src/altcorr_kernel.cu" 2>/dev/null || die "droidcalib .type→.scalar_type patch missing"
grep -qE -- "(--max_seconds|--once)" "$GMRD/scripts/vis_robot_motion.py" || die "gmr.patch (--once/--max_seconds) missing"
echo "patches verified (vendor 状态正确)"

echo "############ STEP 8: PromptHMR requirements + chumpy + xformers ############"
"$PIP" install -r "$PH/requirements.txt" || die "PromptHMR requirements"
mkdir -p "$PH/python_libs"
if [ ! -d "$PH/python_libs/chumpy" ]; then
  git clone https://github.com/Arthur151/chumpy "$PH/python_libs/chumpy" || echo "WARN: chumpy clone failed"
fi
"$PIP" install -e "$PH/python_libs/chumpy" --no-build-isolation || echo "WARN: chumpy install failed (non-fatal)"
"$PIP" install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps || echo "WARN: xformers failed (non-fatal)"

echo "############ STEP 9: download HF weights ############"
"$PIP" install -U huggingface_hub || die "huggingface_hub"
export HF_HUB_DISABLE_XET=1
"$PY" - <<PYEOF || die "HF weights download"
import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="Datawhale/spring-festival-wushu-robot-replication-model",
    local_dir="$HFWE",
    ignore_patterns=["data/annotations/*"],
)
print("HF download done ->", p)
PYEOF
du -sh "$HFWE" || true

echo "############ STEP 10: link weights into PromptHMR/data ############"
rm -rf "$PH/data"
ln -s "$HFWE/data" "$PH/data" || die "symlink data"
ls -la "$PH/data" || true
ls -la "$PH/data/pretrain" || true
# PromptHMR has hardcoded absolute paths like /code/data/pretrain/camcalib_sa_biased_l2.ckpt
# (author's dev machine root was /code). Symlink /code -> PromptHMR root so they resolve.
rm -rf /code
ln -sfn "$PH" /code || echo "WARN: cannot create /code symlink"
echo "--- /code/data/pretrain check ---"
ls -la /code/data/pretrain/camcalib_sa_biased_l2.ckpt 2>/dev/null || echo "WARN: camcalib ckpt not resolvable via /code"

echo "############ STEP 11: install cp311 wheels ############"
"$PIP" install "$HFWE/data/wheels/detectron2-0.8-cp311-cp311-linux_x86_64.whl" || die "detectron2 wheel"
"$PIP" install "$HFWE/data/wheels/droid_backends_intr-0.3-cp311-cp311-linux_x86_64.whl" || die "droid_backends wheel"
"$PIP" install "$HFWE/data/wheels/lietorch-0.3-cp311-cp311-linux_x86_64.whl" || die "lietorch wheel"
"$PIP" install "$HFWE/data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl" || die "sam2 wheel"

echo "############ STEP 12: GMR deps + editable installs ############"
"$PIP" install mujoco mink loop-rate-limiters "imageio[ffmpeg]" rich || die "gmr deps"
"$PIP" install -e "$V2R" || die "install video2robot"
"$PIP" install -e "$GMRD" || echo "WARN: pip install -e GMR failed (will rely on sys.path)"

echo "############ STEP 13: smoke test imports + CUDA ############"
"$PY" - <<'PYEOF' | tee /kaggle/working/setup_report.txt
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(512,512, device="cuda"); y = x@x; torch.cuda.synchronize()
    print("cuda matmul OK", float(y.sum()))
    print("gpu", torch.cuda.get_device_name(0))
for m in ["detectron2","lietorch","sam2","droid_backends_intr","mujoco","mink","smplx","general_motion_retargeting","viser"]:
    try:
        mod = __import__(m)
        print("IMPORT OK:", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("IMPORT FAIL:", m, repr(e))
PYEOF

echo "############ STEP 14: VALIDATION on short clip (boxing.mp4) ############"
mkdir -p "$V2R/data/video_val"
cp "$HFWE/data/examples/boxing.mp4" "$V2R/data/video_val/original.mp4" || die "copy val video"
bash -c "$ACT && cd '$V2R' && python scripts/extract_pose.py --project data/video_val --static-camera" || die "VAL extract_pose"
bash -c "$ACT && cd '$V2R' && python scripts/convert_to_robot.py --project data/video_val --all-tracks" || die "VAL convert_to_robot"
echo "=== VAL outputs ==="; ls -la "$V2R/data/video_val"

echo "############ STEP 15: extract_pose on dance_1 (video_001) ############"
mkdir -p "$V2R/data/video_001"
cp "$HFWE/data/examples/dance_1.mp4" "$V2R/data/video_001/original.mp4" || die "copy input video"
bash -c "$ACT && cd '$V2R' && python scripts/extract_pose.py --project data/video_001 --static-camera" || die "extract_pose"
ls -la "$V2R/data/video_001"

echo "############ STEP 16: convert_to_robot (dance_1, all tracks) ############"
bash -c "$ACT && cd '$V2R' && python scripts/convert_to_robot.py --project data/video_001 --all-tracks" || die "convert_to_robot"
ls -la "$V2R/data/video_001"

echo "############ STEP 17: optional MuJoCo render (non-fatal) ############"
bash -c "$ACT && cd '$GMRD' && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/vis_robot_motion.py --robot unitree_g1 --robot_motion_path '$V2R/data/video_001/robot_motion.pkl' --record_video --video_path '$V2R/data/video_001/mujoco_robot.mp4' --once" || echo "WARN: mujoco render failed (non-fatal)"

echo "############ STEP 18: collect outputs ############"
mkdir -p "$WORK/output"
PROJ="$V2R/data/video_001"
cd "$PROJ"

# Slim results.pkl: drop heavy masks (~11GB) but keep people + camera_world for viser
PROJ="$PROJ" OUT="$WORK/output" "$PY" - <<'PYEOF' || echo "WARN: results slim failed"
import joblib, os
src = os.path.join(os.environ["PROJ"], "results.pkl")
out = os.path.join(os.environ["OUT"], "results.pkl")
res = joblib.load(src)
res.pop("masks", None)
for k, v in (res.get("people") or {}).items():
    if isinstance(v, dict):
        v.pop("masks", None); v.pop("mask", None)
joblib.dump(res, out, compress=3)
print("SLIM results.pkl saved, size MB: %.1f" % (os.path.getsize(out)/1e6))
PYEOF

cp -v robot_motion*.pkl "$WORK/output/" 2>/dev/null || true
cp -v smplx*.npz "$WORK/output/" 2>/dev/null || true
cp -v smplx_tracks.json "$WORK/output/" 2>/dev/null || true
cp -v original.mp4 "$WORK/output/" 2>/dev/null || true
cp -v mujoco_robot.mp4 "$WORK/output/" 2>/dev/null || true
echo "=== OUTPUT DIR ==="
ls -la "$WORK/output"
echo "############ ALL DONE ############"
'''

with open(os.path.join(WORK, "run_all.sh"), "w") as f:
    f.write(BASH)

t0 = time.time()
with open(LOG, "a") as lf:
    lf.write(f"===== run start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    lf.flush()
    p = subprocess.run(["bash", os.path.join(WORK, "run_all.sh")],
                       stdout=lf, stderr=subprocess.STDOUT)
with open(LOG, "a") as lf:
    lf.write(f"===== run end rc={p.returncode} elapsed={time.time()-t0:.0f}s =====\n")

# always surface the tail of the log
try:
    tail = subprocess.run(["tail", "-60", LOG], capture_output=True, text=True).stdout
    print(tail)
except Exception as e:
    print("tail err", e)

print("MAIN_RC =", p.returncode)
