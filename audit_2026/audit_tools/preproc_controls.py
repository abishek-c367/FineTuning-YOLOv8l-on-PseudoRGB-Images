"""Preprocessing factor-isolation transforms for the RAW-penalty study.

Written 16 Oct 2026 in response to the independent method review (P0: "fix the mechanism attribution").
Each function applies exactly ONE declared operation, so that any two arms differ by a named intervention:

    x  : integer counts (uint16 array, 14-bit container assumed; FULL = 2**14 - 1)
    U  = affine(x, q1_D, s_D)            dataset-wide affine, float32, UNCLIPPED           (A0u)
    V  = clip01(U)                       clip to [0,1] only                                 (A0, float cell)
    W  = requant8(V)                     8-bit rounding only                                (A0, 8-bit cell)
    Z  = requant8(clip01(fullrange(x)))  full-range scaling, clip (a no-op) and 8-bit       (R, 8-bit cell)

    m(U) - m(V)  : clipping intervention
    m(V) - m(W)  : requantization (rounding) intervention
    m(W) - m(Z)  : range allocation / mapping at fixed 8-bit output (bundles any saturation difference)

Also provided: per-frame affine (A2), per-video affine in an OFFLINE (non-causal) and a CAUSAL running form (A1),
dataset-statistics fitting, label-free covariates (clipped fraction, rounding error, dynamic range, level counts)
and the exact fold of a constant affine into a first convolution (interior pixels).

No detector is run here. Everything is numpy; torch is not required.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

BITS = 14
FULL = 2 ** BITS - 1          # 16383
LEVELS8 = 255
HALF_STEP8 = 0.5 / LEVELS8    # max rounding error of requant8 on [0,1]


# ----------------------------------------------------------------------------- dataset statistics
@dataclass(frozen=True)
class DatasetStats:
    q1: float
    q99: float

    @property
    def span(self) -> float:
        return self.q99 - self.q1


def sample_pixels(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    flat = x.reshape(-1)
    if flat.size <= n:
        return flat.astype(np.float64)
    idx = rng.choice(flat.size, size=n, replace=False)
    return flat[idx].astype(np.float64)


def fit_dataset_stats(frames: Iterable[np.ndarray], n_pixels: int = 10_000, seed: int = 0,
                      lo: float = 1.0, hi: float = 99.0) -> DatasetStats:
    """Pooled percentiles over a fixed random sample of n_pixels per TRAIN frame (seed recorded by the caller)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([sample_pixels(f, n_pixels, rng) for f in frames])
    q1, q99 = np.percentile(pooled, [lo, hi])
    if q99 <= q1:
        raise ValueError("degenerate dataset span")
    return DatasetStats(float(q1), float(q99))


# ----------------------------------------------------------------------------- the four declared operations
def _check_counts(x: np.ndarray) -> None:
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"counts must be an integer array, got {x.dtype}")
    if x.min() < 0 or x.max() > FULL:
        raise ValueError(f"counts outside the {BITS}-bit container [0, {FULL}]: min {x.min()} max {x.max()}")


def affine(x: np.ndarray, q1: float, span: float) -> np.ndarray:
    """U = (x - q1) / span, float32, no clipping. One operation."""
    _check_counts(x)
    return ((x.astype(np.float32) - np.float32(q1)) / np.float32(span)).astype(np.float32)


def clip01(u: np.ndarray) -> np.ndarray:
    """V = clip(u, 0, 1). One operation."""
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def requant8(v: np.ndarray) -> np.ndarray:
    """W = round(255 v) / 255. One operation. Requires v in [0,1] (clipping is NOT hidden here)."""
    if v.min() < 0.0 or v.max() > 1.0:
        raise ValueError("requant8 expects input already in [0,1]; apply clip01 explicitly first")
    return (np.round(v.astype(np.float32) * LEVELS8) / LEVELS8).astype(np.float32)


def fullrange(x: np.ndarray) -> np.ndarray:
    """R = x / (2^14 - 1), float32. TCNet's RAW row scaling. One operation."""
    _check_counts(x)
    return (x.astype(np.float32) / np.float32(FULL)).astype(np.float32)


# ----------------------------------------------------------------------------- named arms
def arm_A0u(x, st: DatasetStats):  return affine(x, st.q1, st.span)                       # U
def arm_A0(x, st: DatasetStats):   return clip01(arm_A0u(x, st))                          # V
def arm_A0_8(x, st: DatasetStats): return requant8(arm_A0(x, st))                         # W
def arm_R(x):                      return clip01(fullrange(x))                            # clip is a no-op for valid counts
def arm_R_8(x):                    return requant8(arm_R(x))                              # Z


def arm_A2(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Per-frame affine: (x - q1(x)) / (q99(x) - q1(x)), clipped to [0,1]. Two per-frame parameters."""
    _check_counts(x)
    q1, q99 = np.percentile(x.astype(np.float64), [lo, hi])
    if q99 <= q1:
        q99 = q1 + 1.0
    return clip01(((x.astype(np.float32) - np.float32(q1)) / np.float32(q99 - q1)))


def per_video_stats_offline(frames: Sequence[np.ndarray], n_pixels: int = 10_000, seed: int = 0,
                            lo: float = 1.0, hi: float = 99.0) -> DatasetStats:
    """NON-CAUSAL: uses every frame of the video, including future frames. Upper-bound setting."""
    return fit_dataset_stats(frames, n_pixels, seed, lo, hi)


def arm_A1_offline(frames: Sequence[np.ndarray], **kw) -> List[np.ndarray]:
    st = per_video_stats_offline(frames, **kw)
    return [clip01(affine(f, st.q1, st.span)) for f in frames]


def arm_A1_causal(frames: Sequence[np.ndarray], n_pixels: int = 10_000, seed: int = 0,
                  lo: float = 1.0, hi: float = 99.0, warmup: int = 1) -> List[np.ndarray]:
    """CAUSAL per-video affine: frame t is mapped with percentiles of the pooled samples of frames 0..t only
    (expanding window). The first `warmup` frames use their own statistics. Deployable; no future frames."""
    rng = np.random.default_rng(seed)
    pooled: List[np.ndarray] = []
    out = []
    for t, f in enumerate(frames):
        pooled.append(sample_pixels(f, n_pixels, rng))
        q1, q99 = np.percentile(np.concatenate(pooled[: max(t + 1, warmup)]), [lo, hi])
        if q99 <= q1:
            q99 = q1 + 1.0
        out.append(clip01(affine(f, float(q1), float(q99 - q1))))
    return out


# ----------------------------------------------------------------------------- covariates and checks
def clipped_fraction(u: np.ndarray) -> float:
    """Fraction of pixels that clip01 would alter."""
    return float(np.mean((u < 0.0) | (u > 1.0)))


def rounding_error(v: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """(max, mean) |w - v|; must be <= HALF_STEP8 (+ float eps) for w = requant8(v)."""
    d = np.abs(w.astype(np.float64) - v.astype(np.float64))
    return float(d.max()), float(d.mean())


def n_levels(a: np.ndarray) -> int:
    return int(np.unique(a).size)


def dynamic_range(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> float:
    q1, q99 = np.percentile(x.astype(np.float64), [lo, hi])
    return float(q99 - q1)


def is_monotone(f, x_grid: np.ndarray) -> bool:
    y = f(x_grid)
    return bool(np.all(np.diff(y.astype(np.float64)) >= 0))


def decomposition_terms(m: dict) -> dict:
    """Given m = {'U':..,'V':..,'W':..,'Z':..} (any scalar functional), return the three named terms and check the identity."""
    clip_term = m['U'] - m['V']
    rounding_term = m['V'] - m['W']
    range_alloc_term = m['W'] - m['Z']
    total = m['U'] - m['Z']
    assert abs((clip_term + rounding_term + range_alloc_term) - total) < 1e-9
    return {'clipping': clip_term, 'requantization': rounding_term, 'range_allocation_8bit': range_alloc_term, 'total': total}


# ----------------------------------------------------------------------------- fold of a constant affine into conv1
def fold_affine_into_conv(weight: np.ndarray, bias: np.ndarray, q1: float, span: float) -> Tuple[np.ndarray, np.ndarray]:
    """conv(U) == conv'(x) on interior pixels, with U = (x - q1)/span, weight (O, 1, k, k), bias (O,).
    w' = w / span ; b' = b - q1/span * sum(w). Exact up to float32 rounding; does not hold on zero-padded borders
    unless the pad value is also mapped (the protocol pads with 0 in the MAPPED domain for this reason)."""
    w2 = (weight / span).astype(np.float32)
    b2 = (bias - (q1 / span) * weight.sum(axis=(1, 2, 3))).astype(np.float32)
    return w2, b2


def conv2d_valid(img: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Minimal 'valid' cross-correlation for tests: img (H, W) single channel, weight (O, 1, k, k)."""
    O, _, k, _ = weight.shape
    H, W = img.shape
    out = np.zeros((O, H - k + 1, W - k + 1), dtype=np.float64)
    for o in range(O):
        for i in range(k):
            for j in range(k):
                out[o] += weight[o, 0, i, j] * img[i:i + H - k + 1, j:j + W - k + 1]
        out[o] += bias[o]
    return out
