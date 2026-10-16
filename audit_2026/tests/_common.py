"""Shared helpers for GATE-IR audit smoke tests. Adds the student repo to sys.path explicitly."""
import sys, os, json, time, platform
REPO = os.environ.get("GATEIR_STUDENT_REPO", "/home/mpcrlab/gateir_audit/repos/student_repo")
ZREPO = os.environ.get("GATEIR_ZER0_REPO", "/home/mpcrlab/gateir_audit/repos/zer0_repo")
RESULTS = os.environ.get("GATEIR_RESULTS", "/home/mpcrlab/gateir_audit/results")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.makedirs(RESULTS, exist_ok=True)

def env_info():
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import ultralytics; info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics"] = f"ERR {e}"
    try:
        import subprocess
        info["student_repo_commit"] = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
        info["zer0_repo_commit"] = subprocess.check_output(["git", "-C", ZREPO, "rev-parse", "HEAD"]).decode().strip()
    except Exception as e:
        info["repo_commit"] = f"ERR {e}"
    return info

def save(name, obj):
    p = os.path.join(RESULTS, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {p}")
