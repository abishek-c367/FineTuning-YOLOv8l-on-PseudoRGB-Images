"""T05b: cache writer/loader path consistency on the PATCHED code (branch audit/2026-10, after patch 0002/0003).
Counterpart of T05, which characterises the original code and fails by design once the patches are applied.
Checks: recursive listing returns relative paths; resume/skip matches files under 'data/'; writer creates sub-dirs;
loader resolves the writer's path; cache miss raises FileNotFoundError; no-cache + no thermal raises ValueError;
no-cache + thermal runs the live-generator fallback and returns (B,3,H,W) in [0,1]."""
import _common, os, sys, tempfile, json, torch
from _common import env_info, save
sys.path.insert(0, os.path.join(_common.REPO, "Cycle_gan"))
import Distillation as D
from cache_pseudo_rgb import list_cached_pngs
res = {"env": env_info(), "branch": _common.git_rev() if hasattr(_common, "git_rev") else None}
tmp = tempfile.mkdtemp(prefix="gateir_cache_b_")
CACHE = os.path.join(tmp, "pseudo_rgb_cache"); os.makedirs(CACHE)
fname = "data/video-abc-frame-000001-xyz.jpg"
cache_name = os.path.splitext(fname)[0] + ".png"
save_path = os.path.join(CACHE, cache_name)
res["skip_before_write"] = cache_name in list_cached_pngs(CACHE)          # expect False
from PIL import Image
os.makedirs(os.path.dirname(save_path), exist_ok=True)                     # writer line 204 (patch 0002)
Image.new("RGB", (4, 4)).save(save_path)
listed = list_cached_pngs(CACHE)
res["listing_is_relative_with_subdir"] = cache_name in listed                 # expect True
res["skip_after_write"] = cache_name in listed                                # expect True (resume works)
tr = D.DistillationTrainer.__new__(D.DistillationTrainer)
tr.pseudo_rgb_cache_dir = CACHE; tr.device = torch.device("cpu"); tr.use_cache = True
t = tr._load_pseudo_rgb_from_cache([fname], CACHE, tr.device)
res["loader_finds_writer_file"] = list(t.shape)                                # expect [1,3,4,4]
try:
    tr._generate_pseudo_rgb(["data/missing.jpg"]); res["cache_miss_no_thermal"] = "returned"
except Exception as e:
    res["cache_miss_no_thermal"] = type(e).__name__                            # expect FileNotFoundError
tr.use_cache = False
try:
    tr._generate_pseudo_rgb(["data/x.jpg"]); res["no_cache_no_thermal"] = "returned"
except Exception as e:
    res["no_cache_no_thermal"] = type(e).__name__                              # expect ValueError
tr.cyclegan_generator = lambda x: torch.tanh(x)                                # stand-in generator, [-1,1] output
out = tr._generate_pseudo_rgb(["data/x.jpg"], thermal=torch.rand(2, 1, 8, 8))
res["no_cache_with_thermal_fallback"] = {"shape": list(out.shape), "min": float(out.min()), "max": float(out.max())}
src = open(os.path.join(_common.REPO, "Distillation.py")).read()
res["validate_step_passes_thermal_tensor"] = "_generate_pseudo_rgb(None, thermal_batch)" in src   # patch 0003: validate_step passes the tensor positionally
res["all_pass"] = (res["skip_before_write"] is False and res["listing_is_relative_with_subdir"] and res["skip_after_write"]
                   and res["loader_finds_writer_file"] == [1, 3, 4, 4] and res["cache_miss_no_thermal"] == "FileNotFoundError"
                   and res["no_cache_no_thermal"] == "ValueError" and res["no_cache_with_thermal_fallback"]["shape"] == [2, 3, 8, 8]
                   and 0.0 <= res["no_cache_with_thermal_fallback"]["min"] and res["no_cache_with_thermal_fallback"]["max"] <= 1.0)
save("T05b_cache_paths_patched.json", res); print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1, default=str))
