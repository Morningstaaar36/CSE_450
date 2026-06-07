# Milestone 1 — How to Use & How to Test

This guide shows how to **run** the baseline app and how to **test every
deliverable** — including scene-vs-object classification — against both
synthetic and **real-world** data (real indoor scan, real 3D Gaussian
Splatting object, Stanford scans).

> TL;DR
> ```bash
> python fetch_real_data.py          # download real scenes + objects into data/
> python tests/test_pipeline.py      # full automated suite → "ALL PASS"
> python app.py data/indoor_real.ply # interactive app on a real scene
> ```

---

## 1. Environment

All dependencies (numpy, scipy, scikit-learn, torch-CPU, plyfile, PyQt5,
vulkan, PyOpenGL, Pillow) are available in the project's Python env. On this
machine that is the **anaconda base** interpreter:

```bash
~/anaconda3/bin/python --version        # 3.13
```

Quick import check:

```bash
~/anaconda3/bin/python - <<'PY'
import numpy, scipy, sklearn, torch, plyfile, PyQt5, vulkan
print("env OK — torch device:", torch.zeros(1).device)
PY
```

Vulkan is used when present (this machine: Intel Iris Xe iGPU). If Vulkan init
fails the renderer transparently falls back to a NumPy software rasteriser, so
all tests still run.

---

## 2. Get real data

`big.ply` ships with the repo. Everything else is downloaded by a stdlib-only
script:

```bash
python fetch_real_data.py            # skips files already present
python fetch_real_data.py --force    # re-download everything
```

| File | What it is | Source |
|---|---|---|
| `data/indoor_real.ply` | Real **indoor scene** (Redwood living-room fragment, 196k pts, has floor+wall) | open3d_downloads |
| `data/luigi_3dgs.ply` | Real **3D Gaussian Splatting** figurine (`f_dc_*`, `opacity`, `scale_*`, `rot_*`) | HuggingFace `dylanebert/3dgs` |
| `data/buddha.ply` | **Obvious object** — Stanford Happy Buddha (tall statue) | Stanford 3D repo |
| `data/bunny.ply` | **Ambiguous object** — Stanford Bunny (up is semantic) | Stanford 3D repo |
| `data/armadillo.ply` | Stanford Armadillo — **binary big-endian** PLY (byte-order test) | Stanford 3D repo |
| `data/big.ply` | 250k-point capture (isotropic blob) | ships with repo |

---

## 3. Run the full automated suite

```bash
python tests/test_pipeline.py
```

This generates the synthetic clouds, then runs loader → classifier →
alignment → renderer end-to-end. It exits non-zero on any failure. Expected
tail:

```
== real-world data ==
  indoor_real.ply    mode=indoor conf=0.77  [scene] floor upright to 3.3°
  luigi_3dgs.ply     mode=object conf=0.56  [gs_object] 3DGS SH-DC colours OK; not misclassified as scene
  buddha.ply         mode=object conf=0.90  [obvious] up recovered to ≤0.0° under tilt
  bunny.ply          mode=object conf=0.60  [ambiguous] correctly flagged up_ambiguous
  armadillo.ply      mode=object conf=0.64  [ambiguous] correctly flagged up_ambiguous
  big.ply            mode=object conf=0.49  [big] downsampled 250,000→49,537, renders
...
ALL PASS
```

### What each check proves

| Check | Deliverable / criterion |
|---|---|
| synthetic `indoor.ply` true-up error ≤ 8° (gets **0.04°**) | geometry alignment recovers a *known* tilt (non-circular) |
| synthetic `object.ply` true-up error ≤ 10° (gets **0.44°**) | object long-axis up recovery |
| `indoor_real` classified **indoor**, floor upright ≤ 10° (gets **3.3°**) | "indoor scenes align upright reliably" on real data |
| `luigi_3dgs` 3DGS colours decoded, classified **object** | 3DGS loader path + flat-object classification |
| `buddha` up recovered ≤ 15° under random tilts (gets **0°**) | "obvious objects get roughly aligned" on real data |
| `bunny`/`armadillo` flagged `up_ambiguous`, conf < 0.75 | honest uncertainty instead of confident wrong answer |
| `big` downsamples 250k→≤50k and renders | preprocessing/downsampling at scale |

---

## 4. Test scene-vs-object classification

### 4.1 How the classifier decides

Classification is **rotation-invariant** (works on a tilted cloud) and based on
plane structure, not size:

1. **Dominant plane** — RANSAC finds the biggest planar surface and measures
   what fraction of points lie on it (`plane_inlier_ratio`). A floor covers a
   lot; a sculpted object covers little.
2. **Second plane (wall test)** — the dominant plane's inliers are removed and
   RANSAC runs again on the rest. A real room has a **second large plane
   roughly orthogonal** to the first (a wall meeting the floor, ~90°). A
   flat object (e.g. a figurine captured front-on) has a second plane that is
   nearly **parallel** (its back face) — that does *not* count as a scene.
3. **Decision** — `indoor` if there is a floor **plus** an orthogonal wall, or
   a single overwhelmingly dominant floor; otherwise `object`.

> This second-plane rule is why the real 3DGS figurine `luigi_3dgs.ply`
> (one big flat face, second face parallel at ~10°) is correctly an **object**
> and not mistaken for a room.

### 4.2 Inspect the classifier on any file

```bash
python alignment.py data/indoor_real.ply | python -m json.tool
```

Look at `diagnostics`:

| Field | Meaning |
|---|---|
| `plane_inlier_ratio` | floor coverage (0–1) |
| `wall_inlier_ratio` | second-plane coverage |
| `wall_angle_deg` | angle between the two planes (≈90° ⇒ a real wall) |
| `orthogonal_wall` | 1.0 if a wall was found |
| `score_indoor` | soft indoor score; `mode = indoor if ≥ 0.5` |

### 4.3 One-liner classification sweep

```bash
~/anaconda3/bin/python - <<'PY'
from loader import load_point_cloud
from alignment import AlignmentEngine
eng = AlignmentEngine()
for f in ["indoor_real","luigi_3dgs","buddha","bunny","armadillo","big"]:
    r = eng.align(load_point_cloud(f"data/{f}.ply").positions)
    print(f"{f:14s} -> {r.mode:6s} conf={r.confidence:.2f} flags={r.ambiguity_flags}")
PY
```

Expected: `indoor_real → indoor`, everything else `→ object`.

---

## 5. Test alignment quality

Three honest, **non-circular** metrics are used (a naive "is the smallest PCA
axis at +Y?" test always passes because alignment forces exactly that):

* **Synthetic clouds** — the generator tilts a known +Y-up cloud by a known
  rotation; the test applies the alignment and measures the residual angle of
  that *known* up direction from +Y.
* **Real scene** — after alignment, RANSAC the dominant (floor) plane and
  measure its normal's tilt from +Y. `indoor_real` goes from **89° tilted →
  3.3° upright**.
* **Real obvious object** — tilt the object by random rotations, re-align, and
  check the recovered up axis is stable and near +Y (`buddha` ⇒ 0°).

Run alignment on one file directly:

```bash
python alignment.py data/buddha.ply | python -m json.tool   # transform, mode, confidence, flags
```

Ambiguity flags you may see: `up_ambiguous` (geometry can't pin the up axis —
a free-form object; resolved by the M2 ML module) and `yaw_ambiguous`
(rotationally symmetric about up, e.g. the vase).

---

## 6. Test the interactive app (UX deliverables)

```bash
python app.py data/indoor_real.ply      # or any file in data/
```

| Action | What to look for |
|---|---|
| **Auto Align** | scene snaps upright; status shows `mode=indoor conf=…` |
| **Align Up Only** | keeps the up axis, zeroes heading |
| **Rotate 90° / Flip Front/Back** | instant re-orient (push-constant transform) |
| **Reset** | back to the raw orientation |
| **Save Alignment** | writes `<name>_alignment.json` (4×4 matrix + mode + confidence) |
| **Axis gizmo** (bottom-right) | the X/Y/Z triad rotates to follow the transform |
| **HUD** (top-left) | device + live FPS; should stay ≫ 30 fps |

These same actions are verified headlessly — see
`docs`/CI by running the snippet in §7.

---

## 7. Headless UX + performance smoke test

Drives the real `MainWindow` actions without a display:

```bash
QT_QPA_PLATFORM=offscreen ~/anaconda3/bin/python - <<'PY'
import numpy as np, os, json, tempfile
from PyQt5 import QtWidgets
import app as A
from loader import load_point_cloud
q = QtWidgets.QApplication([]); w = A.MainWindow()
pc = load_point_cloud("data/indoor_real.ply")
w._cloud = pc; w._loaded_path = os.path.abspath("data/indoor_real.ply"); w._last_alignment=None
w.viewport.set_point_cloud(pc); w.viewport.set_scene_transform(np.eye(4, dtype=np.float32))
I = np.eye(4, dtype=np.float32); chg = lambda a,b: float(np.abs(a-b).max())
w.action_auto_align();      assert chg(w.viewport.scene_transform(), I) > 1e-3
w.action_rotate_90();       w.action_flip_front_back(); w.action_align_up_only()
w.action_reset();           assert chg(w.viewport.scene_transform(), I) < 1e-6
tmp = tempfile.mktemp(suffix=".json")
QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a,**k:(tmp,""))
w.action_auto_align(); w.action_save_alignment()
assert os.path.isfile(tmp), "save failed"
print("UX actions OK; saved", sorted(json.load(open(tmp)).keys()))
w.close()
PY
```

FPS sanity (offscreen render of the 49k-point real capture):

```bash
QT_QPA_PLATFORM=offscreen ~/anaconda3/bin/python - <<'PY'
import time, numpy as np
from loader import load_point_cloud; from renderer import VulkanRenderer
pc = load_point_cloud("data/big.ply"); r = VulkanRenderer(900,700)
r.set_points(pc.positions, pc.colors); r.render()
t=time.perf_counter()
for _ in range(60): r.render()
dt=(time.perf_counter()-t)/60
print(f"{pc.n} pts @900x700: {dt*1000:.2f} ms/frame -> {1/dt:.0f} FPS"); r.close()
PY
# observed: ~386 FPS  (criterion: ≥ 30)
```

---

## 8. Visual evidence

Before/after screenshots are generated under `data/screenshots/`
(left = raw, right = auto-aligned):

* `scene_indoor_real.png` — room settles onto a horizontal floor
* `object_buddha.png` — statue stays upright on its base
* `object_luigi_3dgs.png` — real 3DGS figurine stands upright

---

## 9. Milestone-1 success-criteria checklist (measured)

| Criterion | Result | Evidence |
|---|---|---|
| Indoor scenes align upright reliably | ✅ | real scene **89° → 3.3°**; synthetic **0.04°** |
| Obvious objects get roughly aligned | ✅ | Buddha **≤0°** up recovery under tilt; vase **0.44°** |
| Ambiguous objects handled honestly | ✅ | bunny/armadillo flagged `up_ambiguous`, low confidence |
| Scene vs object classification | ✅ | all 8 clouds classified correctly (incl. real 3DGS) |
| Preprocessing / downsampling | ✅ | `big.ply` 250k → 49,537 under the 50k cap |
| Preview / apply / reset / manual tweak | ✅ | all toolbar actions verified (§6, §7) |
| Axis gizmo in renderer | ✅ | gizmo tracks the scene-root transform |
| No meaningful FPS drop after load | ✅ | **~386 FPS** at 49k pts, 900×700 |
| 3DGS input format | ✅ | `luigi_3dgs.ply` SH-DC colour + opacity decoded |
