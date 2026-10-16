"""T09: factor-isolation unit tests for audit_tools/preproc_controls.py on synthetic 14-bit counts.
No detector, no data. Establishes that U, V, W, Z differ by exactly one declared operation each, that float32
carries no information beyond the counts, that the 8-bit full-range container collapses a scene band, that the
causal per-video affine uses no future frame, and that the constant affine folds exactly into a first convolution."""
import _common, os, sys, json
import numpy as np
from _common import env_info, save
sys.path.insert(0, os.path.join(os.path.dirname(_common.__file__), "..", "audit_tools"))
import preproc_controls as P

res = {"env": env_info()}
rng = np.random.default_rng(0)

def scene(h=64, w=96, lo=1200, hi=3000, hot_frac=0.01, rng=rng):
    x = rng.integers(lo, hi, size=(h, w))
    hot = rng.random((h, w)) < hot_frac
    x[hot] = rng.integers(9000, P.FULL + 1, size=hot.sum())       # hot outliers above the dataset band
    return x.astype(np.uint16)

train = [scene() for _ in range(20)]
st = P.fit_dataset_stats(train, n_pixels=2000, seed=0)
x = scene(rng=np.random.default_rng(7))
U, V, W, Z = P.arm_A0u(x, st), P.arm_A0(x, st), P.arm_A0_8(x, st), P.arm_R_8(x)

# 1. each step is exactly one declared operation
res["V_equals_clip_U"] = bool(np.array_equal(V, np.clip(U, 0, 1).astype(np.float32)))
res["W_equals_requant_V"] = bool(np.array_equal(W, (np.round(V * 255) / 255).astype(np.float32)))
res["Z_equals_requant_fullrange"] = bool(np.array_equal(Z, (np.round(np.clip(x / P.FULL, 0, 1) * 255) / 255).astype(np.float32)))
res["requant8_refuses_unclipped"] = False
try:
    P.requant8(U)
except ValueError:
    res["requant8_refuses_unclipped"] = True

# 2. clipping intervention is visible and measurable; V == U wherever U in [0,1]
inside = (U >= 0) & (U <= 1)
res["clipped_fraction_U"] = P.clipped_fraction(U)
res["V_equals_U_inside_band"] = bool(np.array_equal(V[inside], U[inside]))
res["V_differs_from_U_only_on_clipped"] = bool(np.all((V != U) == ~inside))

# 3. rounding intervention bounded by half a step; 8-bit has <= 256 levels; float keeps count resolution
mx, mean = P.rounding_error(V, W)
res["rounding_error_max"] = mx; res["rounding_error_bound"] = P.HALF_STEP8
res["rounding_within_half_step"] = bool(mx <= P.HALF_STEP8 + 1e-6)
res["levels"] = {"x": P.n_levels(x), "U": P.n_levels(U), "V": P.n_levels(V), "W": P.n_levels(W), "Z": P.n_levels(Z)}
res["float_adds_no_levels"] = bool(P.n_levels(U) <= P.n_levels(x))
res["W_at_most_256_levels"] = bool(P.n_levels(W) <= 256)
res["V_keeps_more_levels_than_W"] = bool(P.n_levels(V) > P.n_levels(W))

# 4. range allocation: the full-range 8-bit container collapses the scene band into few levels
band = int(np.percentile(x, 99) - np.percentile(x, 1))
res["scene_band_counts"] = band
res["Z_levels_in_band_le_band_over_64_plus_2"] = P.n_levels(Z) <= band // 64 + 2 + int(np.sum(x > np.percentile(x, 99)) > 0) * 60
res["Z_fewer_levels_than_W"] = bool(P.n_levels(Z) < P.n_levels(W))

# 5. the decomposition identity for an arbitrary scalar functional (levels count), with named terms
terms = P.decomposition_terms({k: float(P.n_levels(a)) for k, a in zip("UVWZ", (U, V, W, Z))})
res["decomposition_terms_levels"] = terms

# 6. monotonicity of every map on the full count grid
grid = np.arange(0, P.FULL + 1, dtype=np.uint16)
res["monotone"] = {"A0u": P.is_monotone(lambda g: P.arm_A0u(g, st), grid), "A0": P.is_monotone(lambda g: P.arm_A0(g, st), grid),
                   "A0_8": P.is_monotone(lambda g: P.arm_A0_8(g, st), grid), "R_8": P.is_monotone(P.arm_R_8, grid)}

# 7. A2 per-frame affine: its own 1st/99th percentiles land on 0 and 1
A2 = P.arm_A2(x)
res["A2_percentiles"] = [float(np.percentile(A2, 1)), float(np.percentile(A2, 99))]
res["A2_p1_p99_at_0_1"] = bool(abs(np.percentile(A2, 1)) < 1e-3 and abs(np.percentile(A2, 99) - 1) < 1e-3)

# 8. A1 causal vs offline: perturbing a FUTURE frame changes the offline map of frame t but not the causal one
video = [scene(rng=np.random.default_rng(100 + i)) for i in range(6)]
video_pert = [f.copy() for f in video]; video_pert[5] = np.clip(video_pert[5].astype(np.int64) * 3, 0, P.FULL).astype(np.uint16)
caus_a, caus_b = P.arm_A1_causal(video, n_pixels=500), P.arm_A1_causal(video_pert, n_pixels=500)
off_a, off_b = P.arm_A1_offline(video, n_pixels=500), P.arm_A1_offline(video_pert, n_pixels=500)
res["A1_causal_frame2_unchanged_by_future"] = bool(np.array_equal(caus_a[2], caus_b[2]))
res["A1_offline_frame2_changed_by_future"] = not bool(np.array_equal(off_a[2], off_b[2]))

# 9. fold of the constant affine into conv1 (interior pixels)
wgt = np.random.default_rng(3).normal(size=(4, 1, 3, 3)).astype(np.float32); b = np.random.default_rng(4).normal(size=4).astype(np.float32)
w2, b2 = P.fold_affine_into_conv(wgt, b, st.q1, st.span)
y_u = P.conv2d_valid(U.astype(np.float64), wgt, b); y_x = P.conv2d_valid(x.astype(np.float64), w2, b2)
res["fold_max_abs_diff"] = float(np.max(np.abs(y_u - y_x)))
res["fold_exact_interior"] = bool(res["fold_max_abs_diff"] < 1e-3)
# the fold does NOT hold for the clipped arm V where clipping is active (documented, not a defect)
y_v = P.conv2d_valid(V.astype(np.float64), wgt, b)
res["fold_fails_for_clipped_V_when_clipping_active"] = float(np.max(np.abs(y_v - y_x))) > 1e-3 if res["clipped_fraction_U"] > 0 else "n/a"

res["dtypes_float32"] = bool(all(a.dtype == np.float32 for a in (U, V, W, Z, A2)))
keys = ["V_equals_clip_U", "W_equals_requant_V", "Z_equals_requant_fullrange", "requant8_refuses_unclipped", "V_equals_U_inside_band",
        "V_differs_from_U_only_on_clipped", "rounding_within_half_step", "float_adds_no_levels", "W_at_most_256_levels",
        "V_keeps_more_levels_than_W", "Z_fewer_levels_than_W", "A2_p1_p99_at_0_1", "A1_causal_frame2_unchanged_by_future",
        "A1_offline_frame2_changed_by_future", "fold_exact_interior", "dtypes_float32"]
res["all_pass"] = all(res[k] is True for k in keys) and all(res["monotone"].values())
save("T09_preproc_factor_isolation.json", res); print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1, default=str))
