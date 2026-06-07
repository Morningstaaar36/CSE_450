"""
test_pipeline.py
================

Autonomous end-to-end test for Milestone 1.

Runs without a display:

    python tests/test_pipeline.py

Checks:
    1. Loader can parse a synthetic .ply and downsample under the cap.
    2. Alignment engine returns a valid 4×4 matrix, correct class label,
       and a finite confidence on both indoor and object inputs.
    3. After applying the transform, the cloud's *known* up-axis (the
       ground-truth direction the synthetic generator tilted away from +Y)
       lands back on world +Y within tolerance.  This is a genuine recovery
       test — it does NOT measure the smallest-PCA axis, which the alignment
       forces onto +Y by construction and so always "passes".
    4. Renderer.render() returns an (H,W,4) uint8 image — using Vulkan
       when available, the software fallback otherwise.
    5. The 4x4 transform is applied via the renderer's scene root (a
       transform change is reflected in the next render call, not requiring
       a re-upload of vertex data).
    6. Real-data robustness (when the files are present): a large real
       capture (data/big.ply) loads + downsamples + aligns + renders, and a
       free-form real object (data/bunny.ply, the Stanford Bunny) is
       correctly reported as up-ambiguous rather than confidently mis-aligned.

The script prints PASS / FAIL with timings and exits non-zero on failure.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from loader import load_point_cloud, DEFAULT_POINT_CAP
from alignment import AlignmentEngine, _ransac_plane
from renderer import VulkanRenderer
from tests.gen_synth_ply import main as gen_synth


DATA_DIR = os.path.join(ROOT, "data")


def _ensure_synth():
    if not (os.path.isfile(os.path.join(DATA_DIR, "indoor.ply"))
            and os.path.isfile(os.path.join(DATA_DIR, "object.ply"))):
        os.makedirs(DATA_DIR, exist_ok=True)
        gen_synth(DATA_DIR)


def _rx(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Ground-truth up direction for each synthetic cloud: the generator starts
# with up = +Y and applies (Rx then Rz) — see tests/gen_synth_ply.py — so the
# true up in the file is Rz @ Rx @ +Y.  Keep these in sync with the generator.
TRUE_UP = {
    "indoor.ply": _rz(18) @ _rx(25) @ np.array([0.0, 1.0, 0.0]),
    "object.ply": _rz(35) @ _rx(20) @ np.array([0.0, 1.0, 0.0]),
}


def _true_up_error_deg(R: np.ndarray, up_true: np.ndarray) -> float:
    """Angle (deg) between the aligned ground-truth up and world +Y.

    Non-circular: `up_true` is the physical up baked into the cloud by the
    generator, independent of any PCA the estimator runs.
    """
    up_after = R @ up_true
    up_after = up_after / (np.linalg.norm(up_after) + 1e-12)
    return float(np.degrees(np.arccos(min(1.0, abs(up_after[1])))))


def _apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return (np.concatenate([points, ones], axis=1) @ T.T)[:, :3]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_loader(ply_path: str):
    t0 = time.perf_counter()
    pc = load_point_cloud(ply_path, point_cap=DEFAULT_POINT_CAP)
    dt = time.perf_counter() - t0
    assert pc.n > 0, "loader returned empty cloud"
    assert pc.n <= DEFAULT_POINT_CAP, f"cloud not capped: {pc.n}"
    assert pc.positions.shape == (pc.n, 3)
    assert pc.colors.shape == (pc.n, 3)
    assert np.isfinite(pc.positions).all(), "non-finite positions"
    assert pc.stats.scale > 0, "degenerate stats"
    return pc, dt


def check_alignment(pc, expected_mode: str, axis_tol_deg: float, up_true):
    eng = AlignmentEngine()
    t0 = time.perf_counter()
    res = eng.align(pc.positions)
    dt = time.perf_counter() - t0
    T = res.transform
    assert T.shape == (4, 4), f"bad transform shape {T.shape}"
    assert np.isfinite(T).all(), "non-finite transform"
    # The result should be a rigid transform — rotation block orthonormal.
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-3), "rotation block not orthogonal"
    assert abs(np.linalg.det(R) - 1.0) < 1e-3, "rotation determinant != 1"

    assert res.mode == expected_mode, \
        f"expected mode={expected_mode!r}, got {res.mode!r}"
    assert 0.0 <= res.confidence <= 1.0, f"bad confidence {res.confidence}"

    # Genuine recovery check against the known physical up direction.
    ang = _true_up_error_deg(R, up_true)
    assert ang <= axis_tol_deg, \
        f"true up-axis error after alignment = {ang:.2f}° > tol {axis_tol_deg}°"
    return res, dt, ang


def check_renderer(pc, transform: np.ndarray):
    """Render a small frame and confirm the scene-root transform takes effect."""
    r = VulkanRenderer(width=256, height=192)
    info = r.info()
    r.set_points(pc.positions, pc.colors)

    # Frame 1: identity transform.
    r.set_scene_root_transform(np.eye(4, dtype=np.float32))
    t0 = time.perf_counter()
    img_identity = r.render()
    t1 = time.perf_counter() - t0

    # Frame 2: aligned transform.  We must see a different pixel buffer.
    r.set_scene_root_transform(transform)
    t0 = time.perf_counter()
    img_aligned = r.render()
    t2 = time.perf_counter() - t0

    assert img_identity.shape == (192, 256, 4), \
        f"bad shape {img_identity.shape}"
    assert img_identity.dtype == np.uint8
    # The image should not be entirely the clear colour — at least some
    # pixels are non-background.
    nonbg_id = int(np.any(img_identity[..., :3] > 30))
    nonbg_al = int(np.any(img_aligned[..., :3] > 30))
    assert nonbg_id == 1, "identity render is entirely background"
    assert nonbg_al == 1, "aligned render is entirely background"
    # Transform applied at scene root → pixels should differ (unless the
    # transform happens to be the identity, which it isn't here).
    diff = int(np.sum(img_identity != img_aligned))
    assert diff > 0, "scene-root transform produced identical pixels"

    info2 = r.info()
    r.close()
    return info2, t1, t2, diff


# Real-world test files and what each is meant to exercise.  Files are loaded
# only when present (see fetch_real_data.py to download them); the suite still
# passes on a fresh checkout with just the synthetic clouds.
#
#   role "scene"     — real indoor scan, must classify indoor & sit upright
#   role "gs_object" — real 3DGS .ply, exercises the SH-DC colour/opacity path
#                      and must NOT be misread as a scene (the flat-object trap)
#   role "obvious"   — clearly-oriented object, geometry must recover its up
#   role "ambiguous" — free-form object, up is semantic → must be flagged
#   role "big"       — large capture, must downsample under the cap & render
REAL_FILES = [
    ("indoor_real.ply", "indoor", "scene"),       # Redwood living-room fragment
    ("luigi_3dgs.ply",  "object", "gs_object"),   # real 3D Gaussian Splatting figurine
    ("buddha.ply",      "object", "obvious"),      # Stanford Happy Buddha (tall)
    ("bunny.ply",       "object", "ambiguous"),    # Stanford Bunny
    ("armadillo.ply",   "object", "ambiguous"),    # Stanford Armadillo (binary big-endian)
    ("big.ply",         "object", "big"),          # 250k-point capture
]


def _rand_tilt(seed: int) -> np.ndarray:
    """A repeatable moderate rotation (≤~50° per axis) to tilt an object by."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-50, 50, size=3)
    return _rz(ang[2]) @ _rx(ang[0]) @ _rz(ang[1])


def _floor_tilt_after_deg(points: np.ndarray, T: np.ndarray) -> float:
    """Tilt (deg) of the dominant plane's normal from +Y after applying T.
    For a scene the dominant plane is the floor, so this is the residual
    upright error — a non-circular, physically-meaningful scene metric."""
    aligned = _apply_transform(points, T).astype(np.float64)
    n, p, _ = _ransac_plane(aligned, iterations=400, inlier_thresh=0.02)
    if n[1] < 0:
        n = -n
    return float(np.degrees(np.arccos(min(1.0, abs(n[1])))))


def check_real(ply_path: str, expected_mode: str, role: str):
    """Load + classify + align a real file and run role-specific assertions.
    Returns (pc, res, note) where note is a short human-readable result."""
    pc, _ = check_loader(ply_path)
    eng = AlignmentEngine()
    res = eng.align(pc.positions)

    R = res.transform[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-3), "rotation not orthogonal"
    assert abs(np.linalg.det(R) - 1.0) < 1e-3, "rotation determinant != 1"
    assert res.mode == expected_mode, \
        f"expected mode={expected_mode!r}, got {res.mode!r}"

    note = ""
    if role == "scene":
        tilt = _floor_tilt_after_deg(pc.positions, res.transform)
        assert tilt <= 10.0, f"scene floor not upright after align: {tilt:.1f}°"
        note = f"floor upright to {tilt:.1f}°"
    elif role == "gs_object":
        assert pc.is_gaussian_splat, "3DGS file not detected as gaussian splat"
        assert pc.colors.min() >= 0.0 and pc.colors.max() <= 1.0, \
            "SH-DC colours out of [0,1]"
        note = "3DGS SH-DC colours OK; not misclassified as scene"
    elif role == "obvious":
        # Tilt the (centred) object by a known rotation and require the
        # estimator to bring its up axis back close to +Y.
        P = pc.positions.astype(np.float64)
        P = P - P.mean(axis=0)
        up_true = np.array([0.0, 1.0, 0.0])
        # The object's own up at rest is what the estimator calls up; verify
        # that same physical axis is recovered after an arbitrary tilt.
        rest_R = eng.align(P.astype(np.float32)).transform[:3, :3]
        phys_up = rest_R.T @ up_true
        worst = 0.0
        for s in range(3):
            Q = _rand_tilt(s)
            r2 = eng.align((P @ Q.T).astype(np.float32))
            rec = r2.transform[:3, :3] @ (Q @ phys_up)
            rec = rec / (np.linalg.norm(rec) + 1e-12)
            worst = max(worst, np.degrees(np.arccos(min(1.0, abs(rec[1])))))
        assert worst <= 15.0, f"obvious object up unstable under tilt: {worst:.1f}°"
        note = f"up recovered to ≤{worst:.1f}° under tilt"
    elif role == "ambiguous":
        assert "up_ambiguous" in res.ambiguity_flags, \
            f"free-form object not flagged up_ambiguous: {res.ambiguity_flags}"
        assert res.confidence < 0.75, \
            f"ambiguous object over-confident: {res.confidence:.2f}"
        note = "correctly flagged up_ambiguous"
    elif role == "big":
        assert pc.raw_point_count > DEFAULT_POINT_CAP, "did not exercise downsampling"
        assert pc.n <= DEFAULT_POINT_CAP, "not capped"
        _info, _, _, diff = check_renderer(pc, res.transform)
        assert diff > 0, "scene-root transform produced identical pixels"
        note = f"downsampled {pc.raw_point_count:,}→{pc.n:,}, renders"
    return pc, res, note


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("== generating synthetic data ==")
    _ensure_synth()
    print("ok")

    failures = []
    summary = {}

    for name, expected_mode, axis_tol in [
        ("indoor.ply", "indoor", 8.0),
        ("object.ply", "object", 10.0),
    ]:
        ply = os.path.join(DATA_DIR, name)
        print(f"\n== {name} ==")
        try:
            pc, dt_load = check_loader(ply)
            print(f"loader: raw={pc.raw_point_count:,}  kept={pc.n:,}  "
                  f"voxel={pc.used_voxel_size}  time={dt_load*1000:.1f} ms")
        except AssertionError as exc:
            failures.append(f"{name} loader: {exc}")
            print(f"FAIL loader: {exc}")
            continue

        try:
            result, dt_align, ang = check_alignment(
                pc, expected_mode, axis_tol, TRUE_UP[name])
            print(f"alignment: mode={result.mode}  conf={result.confidence:.2f}  "
                  f"true-up err = {ang:.2f}°  flags={result.ambiguity_flags}  "
                  f"time={dt_align*1000:.1f} ms")
        except AssertionError as exc:
            failures.append(f"{name} alignment: {exc}")
            print(f"FAIL alignment: {exc}")
            continue

        try:
            info, t_id, t_al, diff = check_renderer(pc, result.transform)
            engine = "Vulkan" if info["is_vulkan"] else "CPU-fallback"
            print(f"renderer: {engine} on {info['device_name']} "
                  f"({info['device_type']})  "
                  f"id={t_id*1000:.1f}ms  al={t_al*1000:.1f}ms  "
                  f"diff_pixels={diff}")
        except AssertionError as exc:
            failures.append(f"{name} renderer: {exc}")
            print(f"FAIL renderer: {exc}")
            continue

        summary[name] = {
            "mode": result.mode,
            "confidence": result.confidence,
            "up_axis_deg": ang,
            "renderer": engine,
            "device": info["device_name"],
            "render_ms_identity": t_id * 1000,
            "render_ms_aligned": t_al * 1000,
        }

    # ---- real-data robustness (run only when the files are present) -------
    print("\n== real-world data ==")
    n_real = 0
    for name, expected_mode, role in REAL_FILES:
        ply = os.path.join(DATA_DIR, name)
        if not os.path.isfile(ply):
            print(f"  {name:18s} SKIP (not present — run fetch_real_data.py)")
            continue
        n_real += 1
        try:
            pc, res, note = check_real(ply, expected_mode, role)
            print(f"  {name:18s} mode={res.mode:6s} conf={res.confidence:.2f}  "
                  f"[{role}] {note}")
            summary[name] = {"mode": res.mode, "confidence": res.confidence,
                             "role": role, "flags": res.ambiguity_flags,
                             "raw_points": pc.raw_point_count, "kept": pc.n}
        except AssertionError as exc:
            failures.append(f"{name} ({role}): {exc}")
            print(f"  {name:18s} FAIL [{role}]: {exc}")
    if n_real == 0:
        print("  (no real files found — synthetic checks only)")

    print("\n== summary ==")
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    rc = main()
    # vulkan-python wrapper segfaults at interpreter exit on Linux; sidestep.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
