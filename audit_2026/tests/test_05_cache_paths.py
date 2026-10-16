"""T05: Pseudo-RGB cache writer/loader path consistency, using FLIR-style file names ('data/<name>.jpg') and
a fake dataset. Checks: where the writer saves, where the loader looks, whether resume/skip logic works,
whether the writer creates the 'data/' subfolder, and what happens on cache miss / no cache (fallback)."""
import _common, os, tempfile, json, types, torch
from _common import env_info, save
import Distillation as D
res = {"env": env_info()}
tmp = tempfile.mkdtemp(prefix="gateir_cache_")
CACHE = os.path.join(tmp, "pseudo_rgb_cache")
os.makedirs(CACHE)
fname = "data/video-abc-frame-000001-xyz.jpg"      # FLIR ADAS v2 coco.json file_name style
# --- writer naming logic (Cycle_gan/cache_pseudo_rgb.py lines 173-206) ---
cache_name = os.path.splitext(fname)[0] + ".png"
save_path = os.path.join(CACHE, cache_name)
res["writer_save_path_rel"] = os.path.relpath(save_path, tmp)
existing_files = set(os.listdir(CACHE))
res["writer_skip_check_would_match_existing"] = cache_name in existing_files  # resume logic compares 'data/x.png' to listdir entries
try:
    from PIL import Image
    Image.new("RGB", (4, 4)).save(save_path)
    res["writer_save_without_mkdir_data"] = "OK"
except Exception as e:
    res["writer_save_without_mkdir_data"] = f"FAIL {type(e).__name__}: {e}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True); Image.new("RGB", (4, 4)).save(save_path)
existing_files = set(os.listdir(CACHE))
res["writer_resume_skip_detects_saved_file"] = cache_name in existing_files
# --- loader logic (Distillation.py __init__ lines 336-351 and _load_pseudo_rgb_from_cache) ---
tr = D.DistillationTrainer.__new__(D.DistillationTrainer)
tr.pseudo_rgb_cache_dir = CACHE; tr.device = torch.device("cpu")
root_dir = os.path.join(CACHE, "data")
tr.use_cache = os.path.isdir(CACHE)
tr._cache_files = sorted(f for f in os.listdir(root_dir) if f.endswith(".png"))
res["loader_lists_files_from"] = os.path.relpath(root_dir, tmp); res["loader_n_listed"] = len(tr._cache_files)
try:
    t = tr._load_pseudo_rgb_from_cache([fname], CACHE, tr.device)
    res["loader_finds_file_written_by_writer"] = {"ok": True, "shape": list(t.shape), "resolved": os.path.relpath(os.path.join(CACHE, os.path.splitext(fname)[0] + ".png"), tmp)}
except Exception as e:
    res["loader_finds_file_written_by_writer"] = f"FAIL {e!r}"
# --- same, but if coco file_name had NO 'data/' prefix (e.g. a different dataset) ---
try:
    tr._load_pseudo_rgb_from_cache(["video-abc-frame-000001-xyz.jpg"], CACHE, tr.device); res["loader_without_data_prefix"] = "OK"
except Exception as e:
    res["loader_without_data_prefix"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
# --- cache miss and no-cache fallback ---
try:
    tr._load_pseudo_rgb_from_cache(["data/missing.jpg"], CACHE, tr.device); res["cache_miss"] = "returned"
except Exception as e:
    res["cache_miss"] = f"{type(e).__name__}"
tr.use_cache = False
out = tr._generate_pseudo_rgb(["data/x.jpg"])
res["no_cache_fallback_returns"] = repr(out)
# validate_step passes a tensor instead of filenames:
try:
    tr.use_cache = True; tr._generate_pseudo_rgb(torch.zeros(1, 1, 8, 8)); res["validate_step_tensor_arg"] = "OK"
except Exception as e:
    res["validate_step_tensor_arg"] = f"{type(e).__name__}: {str(e)[:80]}"
save("T05_cache_paths.json", res); print(json.dumps({k: v for k, v in res.items() if k != 'env'}, indent=1, default=str))
