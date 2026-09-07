import sys, os, subprocess, platform

L = []
def log(s=""):
    print(s, flush=True)
    L.append(str(s))

def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR: {e}"

log("=== PYTHON ==="); log(sys.version); log("exe: " + sys.executable)
log("=== PLATFORM ==="); log(platform.platform() + " " + platform.machine())
log("=== NVIDIA-SMI ==="); log(sh("nvidia-smi"))
log("=== NVCC ==="); log(sh("nvcc --version"))
log("=== TORCH ===")
try:
    import torch
    log("torch %s cuda_avail=%s torch_cuda=%s" % (torch.__version__, torch.cuda.is_available(), torch.version.cuda))
    if torch.cuda.is_available():
        log("gpu0 %s mem_GB=%.1f" % (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9))
except Exception as e:
    log("torch import ERR: %r" % e)
log("=== DISK ==="); log(sh("df -h / /kaggle/working /tmp 2>/dev/null"))
log("=== RAM ==="); log(sh("free -h"))
log("=== NPROC ==="); log(sh("nproc"))
log("=== CONDA ==="); log("which: " + sh("which conda")); log("/opt/conda exists: %s" % os.path.exists("/opt/conda")); log(sh("/opt/conda/bin/conda --version 2>/dev/null"))
log("=== GIT-LFS ==="); log(sh("git lfs version 2>/dev/null; which git-lfs"))
log("=== INTERNET ===")
log("huggingface: " + sh("curl -sI -m 10 https://huggingface.co 2>/dev/null | head -1"))
log("hf-mirror: " + sh("curl -sI -m 10 https://hf-mirror.com 2>/dev/null | head -1"))
log("github: " + sh("curl -sI -m 10 https://github.com 2>/dev/null | head -1"))
log("=== KEY PIP PKGS ==="); log(sh("pip list 2>/dev/null | grep -iE 'torch|detectron|lietorch|sam2|sam-2|xformers|numpy|mujoco|smplx|viser'"))
log("=== DONE ===")

try:
    os.makedirs("/kaggle/working", exist_ok=True)
    with open("/kaggle/working/probe_report.txt", "w") as f:
        f.write("\n".join(L))
    print("WROTE /kaggle/working/probe_report.txt")
except Exception as e:
    print("write report ERR:", e)
