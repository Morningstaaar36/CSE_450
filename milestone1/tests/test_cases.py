"""
test_cases.py
=============

Case-matrix coverage for the Milestone-1 classifier + geometry alignment.

Where `test_pipeline.py` proves the end-to-end pipeline on a handful of real
clouds, this module hammers the *decision space* with many controlled
synthetic shapes — one per behaviour the engine is supposed to exhibit — and
asserts the right `mode`, ambiguity `flags`, and (where a geometric up exists)
recovery of a known tilted up-axis.

    python tests/test_cases.py            # run the matrix → PASS/FAIL
    python tests/test_cases.py --report   # just print what each case does
    python tests/test_cases.py --viz      # also write visualizations/case_matrix.png

Every shape is built up=+Y by construction; cases marked `tilt` are rotated by
a known rotation before alignment so the up-recovery check is non-circular
(it measures the *known* physical up, not the PCA axis the estimator forces
onto +Y).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from alignment import AlignmentEngine, UP_AXIS_WORLD


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rx(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _tilt(P, rx=22.0, rz=31.0):
    """Apply a fixed, known tilt (used to make up-recovery non-circular)."""
    R = _rz(rz) @ _rx(rx)
    return (P @ R.T).astype(np.float32), R


# ---------------------------------------------------------------------------
# Shape builders — all are up=+Y by construction, roughly centred
# ---------------------------------------------------------------------------

def _box_filled(rng, sx, sy, sz, n):
    return np.stack([rng.uniform(-sx, sx, n),
                     rng.uniform(-sy, sy, n),
                     rng.uniform(-sz, sz, n)], axis=1).astype(np.float32)


def _cyl_filled(rng, radius, half_h, n):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th),
                     rng.uniform(-half_h, half_h, n),
                     r * np.sin(th)], axis=1).astype(np.float32)


def _sphere_filled(rng, radius, n):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    r = radius * np.cbrt(rng.uniform(0, 1, n))
    return (v * r[:, None]).astype(np.float32)


def case_room(seed=0):
    """Floor + 4 walls + ceiling — the classic indoor room."""
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(-2, 2, 5000), np.full(5000, -1.0),
                      rng.uniform(-2, 2, 5000)], 1)
    parts = [floor]
    for sign, axis in [(-2, 0), (2, 0), (-2, 2), (2, 2)]:
        yz = rng.uniform([-1, -2], [1.5, 2], size=(800, 2))
        if axis == 0:
            parts.append(np.stack([np.full(800, sign), yz[:, 0], yz[:, 1]], 1))
        else:
            parts.append(np.stack([yz[:, 1], yz[:, 0], np.full(800, sign)], 1))
    ceil = np.stack([rng.uniform(-2, 2, 1500), np.full(1500, 1.5),
                     rng.uniform(-2, 2, 1500)], 1)
    parts.append(ceil)
    return np.concatenate(parts).astype(np.float32)


def case_corner(seed=0):
    """Floor + one orthogonal wall (an L) — minimal indoor."""
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(-2, 2, 4000), np.full(4000, -1.0),
                      rng.uniform(-2, 2, 4000)], 1)
    wall = np.stack([np.full(3000, -2.0), rng.uniform(-1, 1.5, 3000),
                     rng.uniform(-2, 2, 3000)], 1)
    clutter = _box_filled(rng, 0.4, 0.4, 0.4, 800) + np.array([0.5, -0.6, 0.5])
    return np.concatenate([floor, wall, clutter]).astype(np.float32)


def case_ground(seed=0):
    """One overwhelmingly dominant floor + sparse clutter (no walls)."""
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(-3, 3, 9000), np.full(9000, 0.0),
                      rng.uniform(-3, 3, 9000)], 1)
    clutter = _sphere_filled(rng, 0.3, 700) + np.array([1.0, 0.3, -0.5])
    return np.concatenate([floor, clutter]).astype(np.float32)


def case_prolate_box(seed=0):
    """Elongated box with a *non-square* cross-section — long axis up, and the
    heading is defined (the two short axes differ), so NO yaw ambiguity."""
    return _box_filled(np.random.default_rng(seed), 0.30, 1.4, 0.18, 6000)


def case_pillar(seed=0):
    """Very tall thin column of revolution — long axis up, heading undefined."""
    return _cyl_filled(np.random.default_rng(seed), 0.18, 1.5, 6000)


def case_oblate_slab(seed=0):
    """Flat slab with a *non-square* footprint — thin axis up, heading defined,
    so NO yaw ambiguity (contrast with the round puck below)."""
    return _box_filled(np.random.default_rng(seed), 0.70, 0.13, 0.45, 6000)


def case_oblate_puck(seed=0):
    """Round thick disc / hockey-puck — thin axis up, heading undefined."""
    return _cyl_filled(np.random.default_rng(seed), 0.6, 0.16, 6000)


def case_isotropic_blob(seed=0):
    """Filled sphere — no geometric up; must be flagged up_ambiguous."""
    return _sphere_filled(np.random.default_rng(seed), 0.6, 6000)


def case_flat_relief(seed=0):
    """A thin 'relief panel' — two near-parallel faces, like a figurine shot
    front-on.  The flat-object trap: must stay OBJECT, not become a scene."""
    rng = np.random.default_rng(seed)
    faces = []
    for z in (-0.18, 0.18):
        faces.append(np.stack([rng.uniform(-0.6, 0.6, 2600),
                               rng.uniform(-0.8, 0.8, 2600),
                               np.full(2600, z) + rng.normal(0, 0.05, 2600)], 1))
    edges = _box_filled(rng, 0.6, 0.8, 0.18, 1800)   # filled interior/edges
    return np.concatenate(faces + [edges]).astype(np.float32)


# ---------------------------------------------------------------------------
# Case registry: (builder, tilt?, expect_mode, expect_flags, check_up, tol, note)
# ---------------------------------------------------------------------------

class Case:
    def __init__(self, name, build, tilt, mode, flags, check_up, tol, note):
        self.name, self.build, self.tilt = name, build, tilt
        self.mode, self.flags = mode, set(flags)
        self.check_up, self.tol, self.note = check_up, tol, note


CASES = [
    # name                build                  tilt  mode      flags                check_up tol  note
    Case("room",          case_room,             True, "indoor", [],                  True, 8.0,  "floor+walls+ceiling"),
    Case("corner_L",      case_corner,           True, "indoor", [],                  True, 8.0,  "floor + one orthogonal wall"),
    Case("ground_plane",  case_ground,           True, "indoor", [],                  True, 8.0,  "one huge dominant floor"),
    Case("prolate_box",   case_prolate_box,      True, "object", [],                  True, 15.0, "elongated, yaw-defined → long axis up"),
    Case("pillar",        case_pillar,           True, "object", ["yaw_ambiguous"],   True, 15.0, "column of revolution → yaw undefined"),
    Case("oblate_slab",   case_oblate_slab,      True, "object", [],                  True, 15.0, "flat, yaw-defined → thin axis up"),
    Case("oblate_puck",   case_oblate_puck,      True, "object", ["yaw_ambiguous"],   True, 15.0, "round disc → thin axis up, yaw undefined"),
    Case("isotropic_blob",case_isotropic_blob,   True, "object", ["up_ambiguous"],    False, 0.0, "sphere → up undefined (semantic)"),
    Case("flat_relief",   case_flat_relief,      True, "object", [],                  False, 0.0, "flat-object trap → stays object"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _up_error_deg(R, up_true):
    up_after = R @ up_true
    up_after /= (np.linalg.norm(up_after) + 1e-12)
    return float(np.degrees(np.arccos(min(1.0, abs(up_after[1])))))


def run_case(case: Case, engine: AlignmentEngine):
    P = case.build()
    up_true = np.array([0.0, 1.0, 0.0])
    if case.tilt:
        P, Rt = _tilt(P)
        up_true = Rt @ up_true                  # physical up after the known tilt
    res = engine.align(P)
    R = res.transform[:3, :3]
    up_err = _up_error_deg(R, up_true) if case.check_up else None
    return res, up_err, P


def evaluate(report_only=False):
    engine = AlignmentEngine()
    rows, failures = [], []
    for c in CASES:
        res, up_err, _ = run_case(c, engine)
        ok = True
        why = []
        # rotation sanity always checked
        R = res.transform[:3, :3]
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-3)
                and abs(np.linalg.det(R) - 1.0) < 1e-3):
            ok = False; why.append("non-rigid R")
        if not report_only:
            if res.mode != c.mode:
                ok = False; why.append(f"mode={res.mode}≠{c.mode}")
            missing = c.flags - set(res.ambiguity_flags)
            if missing:
                ok = False; why.append(f"missing flags {missing}")
            if c.check_up and up_err is not None and up_err > c.tol:
                ok = False; why.append(f"up {up_err:.1f}°>{c.tol:.0f}°")
        rows.append((c, res, up_err, ok, why))
        if not ok:
            failures.append((c.name, why))
    return rows, failures


def _fmt(rows):
    print(f"\n{'case':16s} {'mode':7s} {'conf':5s} {'up_err':7s} flags")
    print("-" * 64)
    for c, res, up_err, ok, why in rows:
        ue = f"{up_err:5.1f}°" if up_err is not None else "   —  "
        mark = "ok " if ok else "FAIL"
        fl = ",".join(res.ambiguity_flags) or "—"
        print(f"{mark} {c.name:14s} {res.mode:7s} {res.confidence:4.2f} {ue:7s} {fl}")
        if why:
            print(f"      ↳ {'; '.join(why)}   ({c.note})")


def visualize(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D       # noqa: F401

    n = len(rows)
    fig = plt.figure(figsize=(4 * n, 8))
    fig.suptitle("Case matrix — alignment INPUT (top) vs APPLIED (bottom)",
                 fontsize=14, fontweight="bold")
    eng = AlignmentEngine()
    for i, (c, res, up_err, ok, why) in enumerate(rows):
        P = c.build()
        if c.tilt:
            P, _ = _tilt(P)
        ones = np.ones((P.shape[0], 1), np.float32)
        A = (np.concatenate([P, ones], 1) @ res.transform.T)[:, :3]
        for row, Q, lab in ((0, P, "input"), (1, A, "applied")):
            ax = fig.add_subplot(2, n, row * n + i + 1, projection="3d")
            S = Q[np.random.default_rng(0).choice(Q.shape[0],
                  min(3500, Q.shape[0]), replace=False)]
            M = S[:, [0, 2, 1]]
            col = "#28a745" if res.mode == "indoor" else "#2d6fe6"
            ax.scatter(M[:, 0], M[:, 1], M[:, 2], s=2, c=col, alpha=0.5,
                       linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=16, azim=-62)
            mn, mx = M.min(0), M.max(0); ctr = (mn + mx) / 2
            r = float((mx - mn).max()) * 0.5 + 1e-6
            ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
            ax.set_zlim(ctr[2]-r, ctr[2]+r)
            try: ax.set_box_aspect((1, 1, 1))
            except Exception: pass
            if row == 0:
                fl = ",".join(res.ambiguity_flags) or "—"
                ax.set_title(f"{c.name}\n{res.mode} {res.confidence:.2f} [{fl}]",
                             fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print only, no asserts")
    ap.add_argument("--viz", action="store_true", help="write case_matrix.png")
    args = ap.parse_args()

    print("== Milestone-1 case matrix ==")
    rows, failures = evaluate(report_only=args.report)
    _fmt(rows)

    if args.viz:
        visualize(rows, os.path.join(ROOT, "visualizations", "case_matrix.png"))

    if args.report:
        print("\n(report mode — no assertions)")
        return 0
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for name, why in failures:
            print(f"  - {name}: {'; '.join(why)}")
        return 1
    print(f"\nALL {len(rows)} CASES PASS")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
