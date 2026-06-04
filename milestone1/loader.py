"""
loader.py
=========

Loader layer for the 3DGS alignment engine.

Responsibilities (per Architecture Elements):
    * parse point cloud / 3DGS  (.ply files)
    * create an analysis cloud   (aggressive downsampling, iGPU-friendly cap)
    * compute cloud statistics   (bbox, centroid, scale)

This module is CPU-only and does not import torch — keeping the import-time
footprint minimal so the GUI launches fast on integrated-graphics machines.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from plyfile import PlyData


# Spherical harmonics 0th-order band coefficient (Y00).
# 3DGS .ply files store base colour as `f_dc_0..2`; renderable RGB is
# 0.5 + SH_C0 * f_dc_i  (per the original 3DGS reference implementation).
SH_C0 = 0.28209479177387814

# iGPU-safe upper bound on points kept after downsampling.
DEFAULT_POINT_CAP = 50_000


@dataclass
class CloudStats:
    """Spatial statistics on the *analysis* (decimated) cloud."""
    bbox_min: np.ndarray         # shape (3,)
    bbox_max: np.ndarray         # shape (3,)
    centroid: np.ndarray         # shape (3,)  — arithmetic mean of points
    extent: np.ndarray           # shape (3,)  — bbox_max - bbox_min
    scale: float                 # longest extent along any axis
    point_count: int             # number of points in the analysis cloud

    def as_dict(self) -> dict:
        return {
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "centroid": self.centroid.tolist(),
            "extent": self.extent.tolist(),
            "scale": float(self.scale),
            "point_count": int(self.point_count),
        }


@dataclass
class PointCloud:
    """
    Lightweight in-memory point cloud.

    `positions` is float32 (N,3); `colors` is float32 (N,3) in [0,1].
    `stats` is computed on this same decimated cloud — never on the full
    raw input — so downstream consumers (alignment, renderer) reason about
    exactly the data they will see.
    """
    positions: np.ndarray
    colors: np.ndarray
    stats: CloudStats
    source_path: str = ""
    raw_point_count: int = 0
    used_voxel_size: Optional[float] = None
    is_gaussian_splat: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


# ---------------------------------------------------------------------------
# PLY parsing
# ---------------------------------------------------------------------------

def _read_ply_vertex(path: str) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
    """
    Read positions + colors from a PLY 'vertex' element.

    Returns
    -------
    positions : (N,3) float32
    colors    : (N,3) float32 in [0,1]  — synthesised if file has no colour
    is_gs     : bool — True if the file carries 3DGS-style f_dc_* properties
    extras    : dict — passes through any extra named properties we kept
                       (opacity for now) so future stages can use them
    """
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY file {path!r} has no 'vertex' element")

    v = ply["vertex"]
    names = set(v.data.dtype.names or [])

    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(
            f"PLY file {path!r} is missing x/y/z vertex properties "
            f"(found: {sorted(names)})"
        )

    pos = np.stack(
        [np.asarray(v["x"], dtype=np.float32),
         np.asarray(v["y"], dtype=np.float32),
         np.asarray(v["z"], dtype=np.float32)],
        axis=1,
    )

    is_gs = {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names)
    colors: np.ndarray

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack(
            [np.asarray(v["red"], dtype=np.float32),
             np.asarray(v["green"], dtype=np.float32),
             np.asarray(v["blue"], dtype=np.float32)],
            axis=1,
        )
        # PLY colour channels can be uint8 (0..255) or float (0..1)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        colors = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    elif is_gs:
        # 3DGS PLY: SH DC coefficients → linear RGB ≈ 0.5 + SH_C0 * f_dc
        fdc = np.stack(
            [np.asarray(v["f_dc_0"], dtype=np.float32),
             np.asarray(v["f_dc_1"], dtype=np.float32),
             np.asarray(v["f_dc_2"], dtype=np.float32)],
            axis=1,
        )
        colors = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0).astype(np.float32)

    else:
        # No colour info — make a neutral grey so the cloud is still visible.
        colors = np.full(pos.shape, 0.75, dtype=np.float32)

    extras: dict = {}
    if "opacity" in names:
        # 3DGS opacity is logit-space → sigmoid; useful for renderer alpha later
        op = np.asarray(v["opacity"], dtype=np.float32)
        extras["opacity"] = 1.0 / (1.0 + np.exp(-op))

    return pos, colors, bool(is_gs), extras


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def voxel_downsample(
    pts: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxel-grid decimation.

    Buckets points by integer voxel index and averages position + colour per
    occupied voxel.  O(N) plus one sort/unique step.

    Parameters
    ----------
    pts        : (N,3) float32 positions
    colors     : (N,3) float32 colours in [0,1]
    voxel_size : edge length of one voxel cell in world units
    """
    if voxel_size <= 0:
        return pts, colors

    keys = np.floor(pts / voxel_size).astype(np.int64)
    # Encode (i,j,k) into a single int64 via a stable scheme:
    # shift each axis into a roughly-balanced bit range. We avoid OR'ing
    # tightly so we don't lose negative-coord precision.
    mn = keys.min(axis=0)
    shifted = keys - mn                               # all non-negative
    span = shifted.max(axis=0) + 1
    flat = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]

    # Bucket by flat key, take mean per group.
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    boundary = np.concatenate(
        ([True], flat_s[1:] != flat_s[:-1], [True])
    )
    starts = np.flatnonzero(boundary[:-1])
    ends = np.flatnonzero(boundary[1:]) + 1

    out_pos = np.empty((starts.size, 3), dtype=np.float32)
    out_col = np.empty((starts.size, 3), dtype=np.float32)
    pts_s = pts[order]
    col_s = colors[order]

    # np.add.reduceat lets us sum contiguous slices in one vectorised pass.
    sums_p = np.add.reduceat(pts_s, starts, axis=0)
    sums_c = np.add.reduceat(col_s, starts, axis=0)
    counts = (ends - starts).astype(np.float32)[:, None]
    out_pos = (sums_p / counts).astype(np.float32)
    out_col = (sums_c / counts).astype(np.float32)
    return out_pos, out_col


def random_downsample(
    pts: np.ndarray,
    colors: np.ndarray,
    target: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random subsample to exactly `target` points."""
    if pts.shape[0] <= target:
        return pts, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=target, replace=False)
    return pts[idx], colors[idx]


def downsample_adaptive(
    pts: np.ndarray,
    colors: np.ndarray,
    cap: int = DEFAULT_POINT_CAP,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Hybrid decimation: voxel grid first (preserves spatial coverage),
    then a final random pass if still over the cap.

    The starting voxel size is chosen from the cloud's bounding-box diagonal
    so we converge in O(log) doublings regardless of input scale.
    """
    if pts.shape[0] <= cap:
        return pts, colors, None

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    diag = float(np.linalg.norm(mx - mn)) or 1.0

    # Two-pass strategy:
    #   pass 1 — grow voxel until we're under the cap   (handles "too dense")
    #   pass 2 — shrink (binary search) toward the cap  (avoid over-decimating)
    # Real captures cluster tightly inside their bbox, so the analytic
    # uniform-density estimate tends to overshoot; the second pass recovers
    # most of the resolution we lost on the first.
    initial = diag / (max(cap, 1) ** (1.0 / 3.0))
    voxel = initial

    grow_low: Optional[float] = None     # smaller-than-this fits over cap
    grow_high: Optional[float] = None    # this fits under cap

    out_pts = pts
    out_col = colors
    out_voxel = voxel

    for _ in range(16):
        dp, dc = voxel_downsample(pts, colors, voxel)
        if dp.shape[0] <= cap:
            grow_high = voxel
            out_pts, out_col, out_voxel = dp, dc, voxel
            break
        grow_low = voxel
        voxel *= 1.5

    # Binary refine — push voxel size *down* to recover resolution while
    # staying under the cap.  If we never overshot in the growth phase we
    # still binary-search between (initial/4, initial) to claim back density.
    if grow_high is not None:
        if grow_low is None:
            grow_low = grow_high / 4.0
        lo, hi = grow_low, grow_high
        for _ in range(8):
            mid = (lo + hi) * 0.5
            dp, dc = voxel_downsample(pts, colors, mid)
            if dp.shape[0] <= cap:
                hi = mid
                out_pts, out_col, out_voxel = dp, dc, mid
            else:
                lo = mid

    if out_pts.shape[0] > cap:
        # Voxelisation never converged — last-ditch random thinning.
        out_pts, out_col = random_downsample(out_pts, out_col, cap)

    return out_pts, out_col, out_voxel


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(pts: np.ndarray) -> CloudStats:
    if pts.shape[0] == 0:
        z = np.zeros(3, dtype=np.float32)
        return CloudStats(z, z, z, z, 0.0, 0)
    mn = pts.min(axis=0).astype(np.float32)
    mx = pts.max(axis=0).astype(np.float32)
    centroid = pts.mean(axis=0).astype(np.float32)
    extent = (mx - mn).astype(np.float32)
    scale = float(extent.max())
    return CloudStats(mn, mx, centroid, extent, scale, int(pts.shape[0]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_point_cloud(
    path: str,
    point_cap: int = DEFAULT_POINT_CAP,
    seed: int = 0,
) -> PointCloud:
    """
    Load a `.ply` file and return a downsampled analysis cloud.

    Parameters
    ----------
    path      : path to a PLY (standard, or 3DGS-style with f_dc/opacity)
    point_cap : hard upper bound on the returned point count
    seed      : RNG seed used only if random fallback fires

    The returned PointCloud's `stats` are computed on the *decimated* points,
    consistent with what alignment and the renderer will work on.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    raw_pos, raw_col, is_gs, extras = _read_ply_vertex(path)
    raw_n = int(raw_pos.shape[0])
    if raw_n == 0:
        raise ValueError(f"PLY file {path!r} contains zero vertices")

    pos, col, voxel_used = downsample_adaptive(raw_pos, raw_col, cap=point_cap)
    # Trim opacity to match if we sub-sampled randomly; otherwise drop it,
    # since voxel-bucketing scrambled per-point identity.
    kept_extras: dict = {}
    if "opacity" in extras and extras["opacity"].shape[0] == pos.shape[0]:
        kept_extras["opacity"] = extras["opacity"]

    stats = compute_stats(pos)

    return PointCloud(
        positions=pos.astype(np.float32),
        colors=col.astype(np.float32),
        stats=stats,
        source_path=path,
        raw_point_count=raw_n,
        used_voxel_size=voxel_used,
        is_gaussian_splat=is_gs,
        extras=kept_extras,
    )


# ---------------------------------------------------------------------------
# CLI for quick sanity checks: `python loader.py path.ply`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Load + downsample a PLY cloud.")
    ap.add_argument("ply", help="path to .ply")
    ap.add_argument("--cap", type=int, default=DEFAULT_POINT_CAP)
    args = ap.parse_args()

    pc = load_point_cloud(args.ply, point_cap=args.cap)
    info = {
        "source": pc.source_path,
        "raw_points": pc.raw_point_count,
        "kept_points": pc.n,
        "is_gaussian_splat": pc.is_gaussian_splat,
        "voxel_size_used": pc.used_voxel_size,
        "stats": pc.stats.as_dict(),
    }
    print(json.dumps(info, indent=2))
