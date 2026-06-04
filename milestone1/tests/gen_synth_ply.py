"""
gen_synth_ply.py
================

Generate small synthetic .ply files for testing loader + alignment.

Two synthetic scenes:
    indoor.ply  — a coarse "room" with floor, four walls, a couple of boxy
                  fixtures.  Up-axis intentionally rotated so the alignment
                  engine has to recover it.
    object.ply  — an asymmetric "vase" (radial flare with a flat base),
                  intentionally tilted off-axis.

Both files use plain XYZ + RGB uchar PLY (the standard format produced by
common 3DGS exporters) so loader.py exercises its non-3DGS code path too.
"""

import os
import sys
import argparse

import numpy as np
from plyfile import PlyData, PlyElement


def _make_room(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []

    # Floor (heavy density — this is the spike the classifier should find).
    n_floor = 5000
    xz = rng.uniform(-2.0, 2.0, size=(n_floor, 2))
    floor = np.stack([xz[:, 0],
                      np.zeros(n_floor) + rng.normal(0, 0.005, n_floor),
                      xz[:, 1]], axis=1)
    pts.append(floor)

    # Walls (sparser).
    n_wall = 800
    for sign, axis in [(-2.0, 0), (2.0, 0), (-2.0, 2), (2.0, 2)]:
        if axis == 0:
            yz = rng.uniform([0, -2], [2.5, 2], size=(n_wall, 2))
            wall = np.stack([np.full(n_wall, sign), yz[:, 0], yz[:, 1]], axis=1)
        else:
            xy = rng.uniform([-2, 0], [2, 2.5], size=(n_wall, 2))
            wall = np.stack([xy[:, 0], xy[:, 1], np.full(n_wall, sign)], axis=1)
        pts.append(wall + rng.normal(0, 0.005, wall.shape))

    # Ceiling — lighter than the floor.
    n_ceil = 1500
    xz = rng.uniform(-2.0, 2.0, size=(n_ceil, 2))
    ceil = np.stack([xz[:, 0], np.full(n_ceil, 2.5), xz[:, 1]], axis=1)
    pts.append(ceil + rng.normal(0, 0.01, ceil.shape))

    # Two boxy fixtures.
    for cx, cz in [(-1.0, -1.0), (1.0, 1.2)]:
        n_box = 600
        box = rng.uniform([-0.3, 0, -0.3], [0.3, 0.8, 0.3], size=(n_box, 3))
        box[:, 0] += cx; box[:, 2] += cz
        pts.append(box + rng.normal(0, 0.005, box.shape))

    cloud = np.concatenate(pts, axis=0)

    # Rotate so the alignment engine has work to do: 25° about X, 18° about Z.
    cx, sx = np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))
    cz, sz = np.cos(np.deg2rad(18)), np.sin(np.deg2rad(18))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (cloud @ Rx.T @ Rz.T).astype(np.float32)


def _make_object(seed: int = 0) -> np.ndarray:
    """Asymmetric vase: thin base, broad shoulder."""
    rng = np.random.default_rng(seed)
    pts = []
    n_layers = 40
    per_layer = 200
    for i in range(n_layers):
        h = i / (n_layers - 1)
        r = 0.25 + 0.25 * np.sin(np.pi * h) + 0.05 * np.cos(4 * np.pi * h)
        theta = rng.uniform(0, 2 * np.pi, per_layer)
        layer = np.stack([r * np.cos(theta),
                          np.full(per_layer, h * 1.4),
                          r * np.sin(theta)], axis=1)
        pts.append(layer + rng.normal(0, 0.008, layer.shape))
    # Flat base disc.
    n_base = 1500
    theta = rng.uniform(0, 2 * np.pi, n_base)
    rad = np.sqrt(rng.uniform(0, 1, n_base)) * 0.30
    base = np.stack([rad * np.cos(theta),
                      np.zeros(n_base) + rng.normal(0, 0.003, n_base),
                      rad * np.sin(theta)], axis=1)
    pts.append(base)

    cloud = np.concatenate(pts, axis=0)

    # Tilt the vase: 35° about Z, 20° about X.
    cx, sx = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))
    cz, sz = np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (cloud @ Rx.T @ Rz.T).astype(np.float32)


def _write_ply(path: str, pts: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    colors = (rng.random(pts.shape) * 255).astype(np.uint8)
    structured = np.empty(pts.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    structured["x"] = pts[:, 0]
    structured["y"] = pts[:, 1]
    structured["z"] = pts[:, 2]
    structured["red"] = colors[:, 0]
    structured["green"] = colors[:, 1]
    structured["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(structured, "vertex")], text=False).write(path)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    room_path = os.path.join(out_dir, "indoor.ply")
    obj_path = os.path.join(out_dir, "object.ply")
    _write_ply(room_path, _make_room())
    _write_ply(obj_path, _make_object())
    print(f"wrote {room_path}")
    print(f"wrote {obj_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    main(args.out)
