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
    3. After applying the transform, the cloud's smallest-PCA axis lines
       up with the world up-axis (+Y) within tolerance.
    4. Renderer.render() returns an (H,W,4) uint8 image — using Vulkan
       when available, the software fallback otherwise.
    5. The 4x4 transform is applied via the renderer's scene root (a
       transform change is reflected in the next render call, not requiring
       a re-upload of vertex data).

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
from alignment import AlignmentEngine
from renderer import VulkanRenderer
from tests.gen_synth_ply import main as gen_synth


DATA_DIR = os.path.join(ROOT, "data")


def _ensure_synth():
    if not (os.path.isfile(os.path.join(DATA_DIR, "indoor.ply"))
            and os.path.isfile(os.path.join(DATA_DIR, "object.ply"))):
        os.makedirs(DATA_DIR, exist_ok=True)
        gen_synth(DATA_DIR)


def _angle_to_up(points: np.ndarray) -> float:
    """Angle (degrees) between the cloud's smallest-PCA axis and +Y."""
    c = points - points.mean(axis=0)
    cov = c.T @ c / max(c.shape[0] - 1, 1)
    _, V = np.linalg.eigh(cov)
    smallest = V[:, 0]
    return float(np.degrees(np.arccos(min(1.0, abs(smallest[1])))))


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


def check_alignment(pc, expected_mode: str, axis_tol_deg: float):
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

    aligned = _apply_transform(pc.positions, T)
    ang = _angle_to_up(aligned)
    assert ang <= axis_tol_deg, \
        f"up-axis angle after alignment = {ang:.2f}° > tol {axis_tol_deg}°"
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
            result, dt_align, ang = check_alignment(pc, expected_mode, axis_tol)
            print(f"alignment: mode={result.mode}  conf={result.confidence:.2f}  "
                  f"up-axis after = {ang:.2f}°  time={dt_align*1000:.1f} ms")
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
