"""
check_success.py
================

Measure the **Milestone-1 success criteria** directly and report PASS/FAIL with
the numbers behind each verdict.  Where `tests/test_pipeline.py` asserts the
pipeline is wired correctly, this script answers the project's headline
questions:

    1. Indoor scenes align upright reliably
    2. Obvious objects get roughly aligned
    3. No meaningful FPS drop after load

Usage
-----
    python check_success.py                 # human-readable table + verdicts
    python check_success.py --json          # machine-readable JSON only
    python check_success.py --frames 200    # longer FPS measurement window

Real files (data/indoor_real.ply, data/buddha.ply, data/big.ply) are used when
present — run `python fetch_real_data.py` first.  The script always falls back
to the synthetic clouds so it works on a fresh checkout, and exits non-zero if
any criterion fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loader import load_point_cloud, DEFAULT_POINT_CAP
from alignment import AlignmentEngine, _ransac_plane
from renderer import VulkanRenderer

DATA = os.path.join(HERE, "data")

# Thresholds defining each criterion's pass condition.
INDOOR_UPRIGHT_TOL_DEG = 10.0     # floor tilt from +Y after alignment
OBJECT_UP_TOL_DEG = 15.0          # up-axis recovery under random tilt
MIN_INTERACTIVE_FPS = 30.0        # "no meaningful FPS drop" floor
MAX_MEANINGFUL_DROP = 60.0        # % drop vs empty scene we still call "fine"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_synth():
    need = [f for f in ("indoor.ply", "object.ply")
            if not os.path.isfile(os.path.join(DATA, f))]
    if need:
        sys.path.insert(0, os.path.join(HERE, "tests"))
        from gen_synth_ply import main as gen
        os.makedirs(DATA, exist_ok=True)
        gen(DATA)


def _rx(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rand_tilt(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-50, 50, size=3)
    return _rz(a[2]) @ _rx(a[0]) @ _rz(a[1])


def _apply(P, T):
    ones = np.ones((P.shape[0], 1), dtype=P.dtype)
    return (np.concatenate([P, ones], 1) @ T.T)[:, :3]


def _floor_tilt_after(points, T):
    """Tilt (deg) of the dominant plane normal from +Y after applying T."""
    aligned = _apply(points, T).astype(np.float64)
    n, _, _ = _ransac_plane(aligned, iterations=400, inlier_thresh=0.02)
    if n[1] < 0:
        n = -n
    return float(np.degrees(np.arccos(min(1.0, abs(n[1])))))


# ---------------------------------------------------------------------------
# Criterion 1 — indoor scenes align upright reliably
# ---------------------------------------------------------------------------

def criterion_indoor_upright():
    eng = AlignmentEngine()
    files = [("indoor.ply", "synthetic"), ("indoor_real.ply", "real")]
    rows, ok = [], True
    for name, kind in files:
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            continue
        pc = load_point_cloud(path)
        res = eng.align(pc.positions)
        tilt = _floor_tilt_after(pc.positions, res.transform)
        passed = (res.mode == "indoor") and (tilt <= INDOOR_UPRIGHT_TOL_DEG)
        ok = ok and passed
        rows.append({"file": name, "kind": kind, "mode": res.mode,
                     "floor_tilt_deg": round(tilt, 2),
                     "confidence": round(res.confidence, 2),
                     "pass": passed})
    return {"name": "Indoor scenes align upright reliably",
            "tol_deg": INDOOR_UPRIGHT_TOL_DEG, "pass": ok and bool(rows),
            "detail": rows}


# ---------------------------------------------------------------------------
# Criterion 2 — obvious objects get roughly aligned
# ---------------------------------------------------------------------------

def criterion_object_upright():
    eng = AlignmentEngine()
    files = [("object.ply", "synthetic"), ("buddha.ply", "real")]
    rows, ok = [], True
    for name, kind in files:
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            continue
        pc = load_point_cloud(path)
        P = pc.positions.astype(np.float64)
        P = P - P.mean(axis=0)
        # The estimator's notion of this object's up at rest.
        rest_R = eng.align(P.astype(np.float32)).transform[:3, :3]
        phys_up = rest_R.T @ np.array([0.0, 1.0, 0.0])
        worst = 0.0
        for s in range(3):
            Q = _rand_tilt(s)
            r2 = eng.align((P @ Q.T).astype(np.float32))
            rec = r2.transform[:3, :3] @ (Q @ phys_up)
            rec = rec / (np.linalg.norm(rec) + 1e-12)
            worst = max(worst, float(np.degrees(np.arccos(min(1.0, abs(rec[1]))))))
        passed = worst <= OBJECT_UP_TOL_DEG
        ok = ok and passed
        rows.append({"file": name, "kind": kind,
                     "worst_up_err_deg": round(worst, 2), "pass": passed})
    return {"name": "Obvious objects get roughly aligned",
            "tol_deg": OBJECT_UP_TOL_DEG, "pass": ok and bool(rows),
            "detail": rows}


# ---------------------------------------------------------------------------
# Criterion 3 — no meaningful FPS drop after load
# ---------------------------------------------------------------------------

def _measure_fps(renderer, frames):
    renderer.render()                      # warm up
    t0 = time.perf_counter()
    for _ in range(frames):
        renderer.render()
    dt = time.perf_counter() - t0
    return float(frames / dt) if dt > 0 else 0.0


def criterion_fps_drop(frames=120, width=900, height=700):
    # pick the heaviest cloud available as the "after load" worst case
    candidates = ["big.ply", "bicycle_3dgs.splat", "bonsai_3dgs.splat",
                  "indoor_real.ply", "object.ply"]
    path = next((os.path.join(DATA, c) for c in candidates
                 if os.path.isfile(os.path.join(DATA, c))), None)
    r = VulkanRenderer(width, height)
    info = r.info()

    baseline = _measure_fps(r, frames)     # empty scene
    loaded_fps, n_pts = 0.0, 0
    if path is not None:
        pc = load_point_cloud(path)
        n_pts = pc.n
        r.set_points(pc.positions, pc.colors)
        r.set_view(yaw_deg=35.0, pitch_deg=18.0)
        loaded_fps = _measure_fps(r, frames)
    r.close()

    drop = max(0.0, (baseline - loaded_fps) / baseline * 100.0) if baseline > 0 else 0.0
    passed = (loaded_fps >= MIN_INTERACTIVE_FPS) and (drop <= MAX_MEANINGFUL_DROP)
    return {"name": "No meaningful FPS drop after load",
            "engine": "Vulkan" if info["is_vulkan"] else "CPU-fallback",
            "device": info["device_name"], "resolution": f"{width}x{height}",
            "cloud": os.path.basename(path) if path else None,
            "points": n_pts,
            "baseline_fps_empty": round(baseline, 1),
            "loaded_fps": round(loaded_fps, 1),
            "drop_pct": round(drop, 1),
            "min_interactive_fps": MIN_INTERACTIVE_FPS,
            "max_meaningful_drop_pct": MAX_MEANINGFUL_DROP,
            "pass": passed}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(frames=120):
    _ensure_synth()
    return [criterion_indoor_upright(),
            criterion_object_upright(),
            criterion_fps_drop(frames=frames)]


def _print_human(results):
    print("== Milestone-1 success criteria ==\n")
    for c in results:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"[{mark}] {c['name']}")
        if c["name"].startswith("No meaningful"):
            print(f"        {c['engine']} · {c['device']} · {c['resolution']} · "
                  f"{c['cloud']} ({c['points']:,} pts)")
            print(f"        empty {c['baseline_fps_empty']} fps → loaded "
                  f"{c['loaded_fps']} fps  (drop {c['drop_pct']}%; "
                  f"need ≥{c['min_interactive_fps']:.0f} & ≤{c['max_meaningful_drop_pct']:.0f}%)")
        else:
            for d in c["detail"]:
                if "floor_tilt_deg" in d:
                    print(f"        {d['file']:20s} [{d['kind']:9s}] mode={d['mode']:6s} "
                          f"floor_tilt={d['floor_tilt_deg']:5.2f}°  "
                          f"{'ok' if d['pass'] else 'FAIL'}")
                else:
                    print(f"        {d['file']:20s} [{d['kind']:9s}] "
                          f"worst_up_err={d['worst_up_err_deg']:5.2f}°  "
                          f"{'ok' if d['pass'] else 'FAIL'}")
        print()
    allok = all(c["pass"] for c in results)
    print("ALL CRITERIA PASS" if allok else "SOME CRITERIA FAILED")
    return allok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--frames", type=int, default=120,
                    help="frames per FPS measurement window")
    args = ap.parse_args()

    results = run(frames=args.frames)
    if args.json:
        print(json.dumps({"criteria": results,
                          "all_pass": all(c["pass"] for c in results)}, indent=2))
        allok = all(c["pass"] for c in results)
    else:
        allok = _print_human(results)
    return 0 if allok else 1


if __name__ == "__main__":
    rc = main()
    # vulkan cffi wrapper can segfault at interpreter teardown; sidestep.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
