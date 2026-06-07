"""
visualize_pipeline.py
======================

Render every Milestone-1 pipeline stage to a single report-ready figure per
cloud, so each deliverable can be *seen* rather than just asserted.

For each input cloud the script produces a 3×3 contact sheet
(`visualizations/<name>__pipeline.png`) with one panel per step:

    row 1 : raw cloud         -> downsampled analysis cloud -> outliers removed
    row 2 : scene/object      -> geometry alignment BEFORE  -> alignment AFTER
            classification        (tilted)                     (upright)
    row 3 : manual Rotate 90° -> manual Flip Front/Back     -> real renderer
                                                               output + gizmo

It drives the *actual* pipeline submodules (`OutlierCleaner`,
`ContentTypeClassifier`, `GeometryAxisEstimator`) and the real
`VulkanRenderer`, so the figure reflects exactly what the app does.

Usage
-----
    python visualize_pipeline.py                 # the curated default set
    python visualize_pipeline.py data/bunny.ply  # one or more explicit files

World up is +Y; every 3-D panel is drawn with world-Y vertical and the X/Y/Z
gizmo triad overlaid (same convention as the app's axis gizmo).
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless: save PNGs, never a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401  (registers 3d proj)

from loader import load_point_cloud, _read_points, DEFAULT_POINT_CAP
from alignment import (
    OutlierCleaner, ContentTypeClassifier, GeometryAxisEstimator,
    AlignmentEngine, _ransac_plane, UP_AXIS_WORLD,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "visualizations")
DATA = os.path.join(HERE, "data")

AXIS_COLORS = {"X": "#dc5050", "Y": "#50c850", "Z": "#5096e6"}
PLOT_CAP = 6000                              # points actually scattered per panel


# ---------------------------------------------------------------------------
# Small 3-D plotting helpers (world-Y vertical)
# ---------------------------------------------------------------------------

def _remap(P: np.ndarray) -> np.ndarray:
    """World (x,y,z) -> matplotlib (x, z, y) so world-up +Y is the vertical axis."""
    return P[:, [0, 2, 1]]


def _thin(P, C, cap=PLOT_CAP, seed=0):
    if P.shape[0] <= cap:
        return P, C
    idx = np.random.default_rng(seed).choice(P.shape[0], cap, replace=False)
    return P[idx], (C[idx] if C is not None else None)


def _equal_aspect(ax, P):
    Q = _remap(P)
    mn, mx = Q.min(axis=0), Q.max(axis=0)
    c = (mn + mx) / 2.0
    r = float((mx - mn).max()) * 0.5 + 1e-6
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _scatter(ax, P, C, s=2.0, alpha=0.7):
    Q = _remap(P)
    if C is None:
        C = np.full((P.shape[0], 3), 0.7, dtype=np.float32)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=np.clip(C, 0, 1),
               s=s, alpha=alpha, linewidths=0, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=16, azim=-62)


def _draw_gizmo(ax, P, frac=0.45):
    """Overlay the X/Y/Z world-axis triad (same colours as the app gizmo)."""
    Q = _remap(P)
    span = float((Q.max(axis=0) - Q.min(axis=0)).max())
    L = span * frac * 0.5
    o = _remap(P.mean(axis=0, keepdims=True))[0]
    for axis_name, vec in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1])):
        d = _remap(np.array([vec], dtype=float))[0] * L
        ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]], [o[2], o[2] + d[2]],
                color=AXIS_COLORS[axis_name], lw=2.2)
        ax.text(o[0] + d[0], o[1] + d[1], o[2] + d[2], axis_name,
                color=AXIS_COLORS[axis_name], fontsize=8, fontweight="bold")


def _apply_T(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((P.shape[0], 1), dtype=P.dtype)
    return (np.concatenate([P, ones], axis=1) @ T.T)[:, :3]


def _floor_tilt_deg(P: np.ndarray) -> float:
    """Angle of the dominant-plane normal from +Y (0° == perfectly upright)."""
    n, _, _ = _ransac_plane(P.astype(np.float64), iterations=300, inlier_thresh=0.02)
    if n[1] < 0:
        n = -n
    return float(np.degrees(np.arccos(min(1.0, abs(n[1])))))


# ---------------------------------------------------------------------------
# One contact sheet per cloud
# ---------------------------------------------------------------------------

def visualize(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"  [{name}] loading…", flush=True)

    # --- pull both full-res and analysis clouds -------------------------------
    raw_pos, raw_col, is_gs, _ = _read_points(path)
    pc = load_point_cloud(path, point_cap=DEFAULT_POINT_CAP)
    P = pc.positions
    C = pc.colors

    # --- run the pipeline submodules, capturing every intermediate ------------
    cleaner = OutlierCleaner()
    cleaned, mask = cleaner(P)
    if cleaned.shape[0] < 16:
        cleaned, mask = P, np.ones(P.shape[0], bool)

    clf = ContentTypeClassifier()
    mode, cls_conf, diag = clf(cleaned)

    est = GeometryAxisEstimator()
    T, est_conf, flags = est(cleaned, mode, diag)

    result = AlignmentEngine().align(P)      # the official combined result

    aligned = _apply_T(cleaned, T)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        f"{name}{'  (3DGS)' if is_gs else ''}    "
        f"mode={result.mode}   confidence={result.confidence:.2f}   "
        f"flags={result.ambiguity_flags or '—'}",
        fontsize=15, fontweight="bold", y=0.985,
    )

    def ax3d(i):
        return fig.add_subplot(3, 3, i, projection="3d")

    # ----- 1. raw cloud -------------------------------------------------------
    a = ax3d(1)
    rp, rc = _thin(raw_pos, raw_col)
    _scatter(a, rp, rc, s=1.5, alpha=0.5)
    _equal_aspect(a, rp)
    a.set_title(f"1 · Raw cloud\n{pc.raw_point_count:,} points", fontsize=10)

    # ----- 2. downsampled analysis cloud -------------------------------------
    a = ax3d(2)
    dp, dc = _thin(P, C)
    _scatter(a, dp, dc, s=2.0, alpha=0.7)
    _equal_aspect(a, dp)
    vox = f"voxel={pc.used_voxel_size:.4f}" if pc.used_voxel_size else "no decimation"
    a.set_title(f"2 · Preprocess / downsample\n{pc.n:,} points  ({vox})", fontsize=10)

    # ----- 3. outlier cleaning -----------------------------------------------
    a = ax3d(3)
    removed = P[~mask]
    kp, kc = _thin(P[mask], C[mask])
    _scatter(a, kp, kc, s=2.0, alpha=0.55)
    if removed.shape[0]:
        rr, _ = _thin(removed, None, cap=3000)
        rr_m = _remap(rr)
        a.scatter(rr_m[:, 0], rr_m[:, 1], rr_m[:, 2], c="#ff2d2d",
                  s=10, alpha=0.9, marker="x")
    _equal_aspect(a, P)
    a.set_title(f"3 · Outlier cleaning\n{int((~mask).sum()):,} stragglers removed "
                f"(red ✕)", fontsize=10)

    # ----- 4. scene / object classification ----------------------------------
    a = ax3d(4)
    _draw_classification(a, cleaned, diag)
    _equal_aspect(a, cleaned)
    pr = diag.get("plane_inlier_ratio", 0.0)
    wr = diag.get("wall_inlier_ratio", 0.0)
    wa = diag.get("wall_angle_deg", 0.0)
    a.set_title(f"4 · Classify → {mode.upper()}  (conf {cls_conf:.2f})\n"
                f"floor {pr*100:.0f}% · wall {wr*100:.0f}% @ {wa:.0f}°", fontsize=10)

    # ----- 5. alignment BEFORE (tilted) --------------------------------------
    a = ax3d(5)
    cp, cc = _thin(cleaned, C[mask] if C[mask].shape[0] == cleaned.shape[0] else None)
    _scatter(a, cp, cc, s=2.0, alpha=0.6)
    _draw_gizmo(a, cleaned)
    _equal_aspect(a, cleaned)
    a.set_title(f"5 · Alignment INPUT\nfloor/up tilt ≈ {_floor_tilt_deg(cleaned):.0f}° "
                f"from +Y", fontsize=10)

    # ----- 6. alignment AFTER (upright) --------------------------------------
    a = ax3d(6)
    ap, ac = _thin(aligned, C[mask] if C[mask].shape[0] == cleaned.shape[0] else None)
    _scatter(a, ap, ac, s=2.0, alpha=0.6)
    _draw_gizmo(a, aligned)
    _equal_aspect(a, aligned)
    tilt_after = _floor_tilt_deg(aligned)
    a.set_title(f"6 · Geometry alignment APPLIED\nfloor/up tilt ≈ {tilt_after:.0f}° "
                f"from +Y", fontsize=10)

    # ----- 7. manual Rotate 90° ----------------------------------------------
    a = ax3d(7)
    rot = _apply_T(cleaned, AlignmentEngine.rotate_about_up(T, 90.0))
    rp2, rc2 = _thin(rot, C[mask] if C[mask].shape[0] == cleaned.shape[0] else None)
    _scatter(a, rp2, rc2, s=2.0, alpha=0.6)
    _draw_gizmo(a, rot)
    _equal_aspect(a, rot)
    a.set_title("7 · Manual tweak · Rotate 90° (yaw)", fontsize=10)

    # ----- 8. manual Flip Front/Back -----------------------------------------
    a = ax3d(8)
    flip = _apply_T(cleaned, AlignmentEngine.flip_front_back(T))
    fp, fc = _thin(flip, C[mask] if C[mask].shape[0] == cleaned.shape[0] else None)
    _scatter(a, fp, fc, s=2.0, alpha=0.6)
    _draw_gizmo(a, flip)
    _equal_aspect(a, flip)
    a.set_title("8 · Manual tweak · Flip Front/Back (180°)", fontsize=10)

    # ----- 9. real renderer output + gizmo -----------------------------------
    a = fig.add_subplot(3, 3, 9)
    _draw_renderer_panel(a, pc, T)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{name}__pipeline.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  [{name}] mode={result.mode} conf={result.confidence:.2f} "
          f"flags={result.ambiguity_flags} → {os.path.relpath(out, HERE)}", flush=True)
    return out


def _draw_classification(ax, P, diag):
    """Colour points by their role in the indoor/object decision."""
    extent = float((P.max(axis=0) - P.min(axis=0)).max())
    tol = max(0.015 * extent, 1e-6)
    n = np.array([diag.get("plane_normal_x", 0.0),
                  diag.get("plane_normal_y", 1.0),
                  diag.get("plane_normal_z", 0.0)])
    p = np.array([diag.get("plane_point_x", 0.0),
                  diag.get("plane_point_y", 0.0),
                  diag.get("plane_point_z", 0.0)])
    floor = np.abs((P - p) @ n) < tol
    rest = P[~floor]
    wall = np.zeros(rest.shape[0], bool)
    if rest.shape[0] > 200:
        n2, p2, _ = _ransac_plane(rest, iterations=180, inlier_thresh=0.015)
        wall = np.abs((rest - p2) @ n2) < tol

    # grey rest, green floor, blue wall — same semantics as the classifier
    Pf, _ = _thin(P[floor], None, cap=3000, seed=1)
    Pr_rest = rest[~wall]; Pw = rest[wall]
    Pr, _ = _thin(Pr_rest, None, cap=3000, seed=2)
    for sub, col, sz, al in ((Pr, "#b8b8b8", 2, 0.35),
                             (Pf, "#28a745", 4, 0.8),
                             (Pw, "#2d6fe6", 4, 0.8)):
        if sub.shape[0]:
            Q = _remap(sub)
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=col, s=sz, alpha=al,
                       linewidths=0, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=16, azim=-62)


def _draw_renderer_panel(ax, pc, T):
    """Render the cloud through the real VulkanRenderer (aligned) and imshow it."""
    ax.axis("off")
    try:
        from renderer import VulkanRenderer
        r = VulkanRenderer(width=480, height=420)
        r.set_points(pc.positions, pc.colors)
        r.set_scene_root_transform(T.astype(np.float32))
        r.set_view(yaw_deg=35.0, pitch_deg=18.0)
        img = r.render()
        info = r.info()
        r.close()
        ax.imshow(img)
        eng = "Vulkan" if info.get("is_vulkan") else "CPU"
        ax.set_title(f"9 · Renderer output (aligned)\n{eng} · "
                     f"{info.get('device_type','?')}", fontsize=10)
    except Exception as exc:                                   # pragma: no cover
        ax.text(0.5, 0.5, f"renderer unavailable:\n{exc}",
                ha="center", va="center", fontsize=9)
        ax.set_title("9 · Renderer output", fontsize=10)


# ---------------------------------------------------------------------------
# Default curated set + entry point
# ---------------------------------------------------------------------------

DEFAULT_SET = [
    "indoor.ply", "object.ply",            # synthetic (always available)
    "indoor_real.ply",                     # real indoor scene
    "buddha.ply", "bunny.ply",             # obvious + ambiguous objects
    "luigi_3dgs.ply",                      # real 3DGS object (.ply)
    "bonsai_3dgs.splat",                   # real 3DGS scene (.splat)
]


def _ensure_synth():
    need = [f for f in ("indoor.ply", "object.ply")
            if not os.path.isfile(os.path.join(DATA, f))]
    if need:
        sys.path.insert(0, os.path.join(HERE, "tests"))
        from gen_synth_ply import main as gen
        os.makedirs(DATA, exist_ok=True)
        gen(DATA)


def main(argv):
    os.makedirs(OUTDIR, exist_ok=True)
    if argv:
        targets = argv
    else:
        _ensure_synth()
        targets = [os.path.join(DATA, f) for f in DEFAULT_SET
                   if os.path.isfile(os.path.join(DATA, f))]

    print(f"== visualizing {len(targets)} cloud(s) -> {os.path.relpath(OUTDIR, HERE)}/ ==")
    made = []
    for path in targets:
        if not os.path.isfile(path):
            print(f"  SKIP {path} (not found)")
            continue
        try:
            made.append(visualize(path))
        except Exception as exc:                              # pragma: no cover
            import traceback
            print(f"  FAIL {path}: {exc}")
            traceback.print_exc()
    print(f"\nwrote {len(made)} figure(s) to {os.path.relpath(OUTDIR, HERE)}/")
    return 0


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    # The vulkan cffi wrapper can segfault during interpreter teardown; the
    # figures are already on disk, so exit hard to keep the run clean.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
