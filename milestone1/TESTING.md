# Milestone 1 — How to Use & How to Test

This guide shows how to **run** the baseline app and how to **test every
deliverable** — including scene-vs-object classification — against both
synthetic and **real-world** data (real indoor scan, real 3D Gaussian
Splatting object, Stanford scans).

> TL;DR — **test everything**, in order:
> ```bash
> python fetch_real_data.py          # 1. download real scenes + objects into data/
> python tests/test_pipeline.py      # 2. end-to-end pipeline suite      → "ALL PASS"
> python tests/test_cases.py         # 3. classifier/geometry case matrix → "ALL 9 CASES PASS"
> python check_success.py            # 4. the 3 success criteria, measured → "ALL CRITERIA PASS"
> python visualize_pipeline.py       # 5. per-step contact sheets → visualizations/
> python tests/test_cases.py --viz   #    + decision-space matrix → visualizations/case_matrix.png
> python app.py data/indoor_real.ply # 6. interactive app (Preview → Apply → Save) on a real scene
> ```
> Use `~/anaconda3/bin/python` if `python` isn't the anaconda base env (see §1).

---

## 1. Environment

All dependencies (numpy, scipy, scikit-learn, torch-CPU, plyfile, PyQt5,
vulkan, PyOpenGL, matplotlib, imageio, Pillow, pyyaml) are captured in
[`environment.yml`](environment.yml) — recreate the env in one command:

```bash
conda env create -f environment.yml && conda activate m1-aligner
# …or, pip:  pip install -r requirements.txt
```

On this machine the verified interpreter is the **anaconda base** env:

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
| `data/luigi_3dgs.ply` | Real **3DGS object** `.ply` (`f_dc_*`, `opacity`, `scale_*`, `rot_*`) | HuggingFace `dylanebert/3dgs` |
| `data/bonsai_3dgs.splat` | Real **3DGS indoor scene** `.splat` (Mip-NeRF360, 273k Gaussians) | HuggingFace `dylanebert/3dgs` |
| `data/bicycle_3dgs.splat` | Real **3DGS outdoor scene** `.splat` (Mip-NeRF360, 568k Gaussians) | HuggingFace `dylanebert/3dgs` |
| `data/buddha.ply` | **Obvious object** — Stanford Happy Buddha (tall statue) | Stanford 3D repo |
| `data/bunny.ply` | **Ambiguous object** — Stanford Bunny (up is semantic) | Stanford 3D repo |
| `data/armadillo.ply` | Stanford Armadillo — **binary big-endian** PLY (byte-order test) | Stanford 3D repo |
| `data/big.ply` | 250k-point capture (isotropic blob) | ships with repo |

The two `.splat` files exercise the loader's antimatter15 32-byte-Gaussian
path. Mip-NeRF360 captures are object-centric 360° orbits (no dominant floor),
so they classify as `object` — see §4.

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

The alignment flow is **preview → apply** (nothing is committed until you say
so), with manual tweaks and a full **change history** recorded along the way:

| Action | What to look for |
|---|---|
| **Preview Align** | computes auto-alignment and shows it *staged* — an orange **“PREVIEW — not applied”** badge appears top-centre; `committed` transform is untouched |
| **Apply** | commits the preview; badge clears; status shows `applied — alignment committed` |
| **Align Up Only** | keeps the up axis, zeroes heading (auto-previews first if nothing is staged) |
| **Rotate 90° / Flip Front/Back** | instant re-orient; edits the preview if one is staged, else the committed transform |
| **Reset** | back to the raw orientation (identity) |
| **Save Alignment + History** | writes `<name>_alignment.json`: committed 4×4 **plus the full ordered change log** (every preview/apply/tweak/reset with before/after matrices and timestamps) |
| **changes: N (last: …)** (side panel) | live count of recorded changes — this is the *“store what was changed”* deliverable |
| **Axis gizmo** (bottom-right) | the X/Y/Z triad rotates to follow the current transform |
| **HUD** (top-left) | device + live FPS; should stay ≫ 30 fps. The diagnostics panel shows the **FPS drop after load** vs the empty-scene baseline |

A saved JSON looks like:

```jsonc
{
  "committed_transform_4x4": [[...]],
  "has_pending_preview": false,
  "n_changes": 4,
  "history": [
    {"seq": 1, "action": "auto_preview", "iso_time": "…",
     "before": [[...]], "after": [[...]],
     "note": "mode=indoor conf=0.77",
     "meta": {"mode": "indoor", "confidence": 0.77, "flags": [], "wall_yawed": true}},
    {"seq": 2, "action": "apply",     "before": [[...]], "after": [[...]]},
    {"seq": 3, "action": "rotate_90", "before": [[...]], "after": [[...]]},
    {"seq": 4, "action": "reset",     "before": [[...]], "after": [[...]]}
  ],
  "last_auto_align": { "transform": [[...]], "mode": "indoor", "confidence": 0.77, … },
  "source": { "source_ply": "…", "raw_points": 196133, "kept_points": 49537,
              "baseline_fps": 243.2, "loaded_fps": 186.1 }
}
```

These same actions are verified headlessly by the snippet in §7.

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
w._cloud = pc; w._loaded_path = os.path.abspath("data/indoor_real.ply")
w._session = A.AlignmentSession()
w.viewport.set_point_cloud(pc); w._sync_viewport()
w._loaded_fps = w.viewport.probe_fps(40)
I = np.eye(4, dtype=np.float32); chg = lambda a,b: float(np.abs(a-b).max())

# preview stages a transform but does NOT commit it
w.action_preview_align()
assert w._session.has_preview(), "preview not staged"
assert chg(w.viewport.scene_transform(), I) > 1e-3, "preview not shown"
assert chg(w._session.committed, I) < 1e-6, "preview wrongly committed"
# apply commits; manual tweaks + reset
w.action_apply();           assert chg(w._session.committed, I) > 1e-3
w.action_rotate_90();       w.action_flip_front_back(); w.action_align_up_only()
w.action_reset();           assert chg(w.viewport.scene_transform(), I) < 1e-6

# save committed transform + FULL change history
tmp = tempfile.mktemp(suffix=".json")
QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a,**k:(tmp,""))
w.action_preview_align(); w.action_apply(); w.action_save_alignment()
data = json.load(open(tmp)); assert os.path.isfile(tmp), "save failed"
print("UX OK · baseline=%.0f loaded=%.0f fps · %d changes: %s"
      % (w._baseline_fps, w._loaded_fps, data["n_changes"],
         [h["action"] for h in data["history"]]))
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
# observed: ~186 FPS at 49k pts  (criterion: ≥ 30)
```

---

## 7b. Success-criteria checker (one command, all three criteria)

The headline question — *“are the three Milestone-1 success criteria met?”* —
has its own checker that **measures** each one and prints PASS/FAIL with the
numbers behind the verdict, including the **load-time FPS drop** (empty-scene
baseline vs the heaviest loaded cloud):

```bash
python check_success.py            # human-readable table
python check_success.py --json     # machine-readable, for CI
python check_success.py --frames 200   # longer FPS window
```

Expected output:

```
== Milestone-1 success criteria ==

[PASS] Indoor scenes align upright reliably
        indoor.ply           [synthetic] mode=indoor floor_tilt= 0.05°  ok
        indoor_real.ply      [real     ] mode=indoor floor_tilt= 0.62°  ok

[PASS] Obvious objects get roughly aligned
        object.ply           [synthetic] worst_up_err= 0.00°  ok
        buddha.ply           [real     ] worst_up_err= 0.00°  ok

[PASS] No meaningful FPS drop after load
        Vulkan · Intel(R) Iris(R) Xe Graphics (TGL GT2) · 900x700 · big.ply (49,537 pts)
        empty 243.2 fps → loaded 186.1 fps  (drop 23.5%; need ≥30 & ≤60%)

ALL CRITERIA PASS
```

What each criterion measures, and how it passes:

| Criterion | Metric | Pass condition |
|---|---|---|
| Indoor scenes align upright | floor-plane tilt from +Y **after** alignment | classified `indoor` **and** tilt ≤ 10° |
| Obvious objects roughly aligned | up-axis error after re-tilting by 3 random rotations | worst-case ≤ 15° |
| No meaningful FPS drop after load | empty-scene FPS vs heaviest-cloud FPS | loaded ≥ 30 FPS **and** drop ≤ 60% |

The checker falls back to the synthetic clouds when real files are absent, so it
runs on a fresh checkout, and exits non-zero if any criterion fails.

---

## 8. Visual evidence — visualize every pipeline step

Each Milestone-1 step is *seen*, not just asserted. One command renders a
9-panel contact sheet per cloud (driving the **real** submodules + the real
`VulkanRenderer`):

```bash
python visualize_pipeline.py                  # curated default set → visualizations/
python visualize_pipeline.py data/buddha.ply  # one or more explicit files
python tests/test_cases.py --viz              # decision-space matrix → case_matrix.png
```

`visualizations/<name>__pipeline.png` lays out, panel by panel:

| Panel | Step |
|---|---|
| 1 | Raw cloud |
| 2 | Preprocess / **downsample** (shows voxel size + kept-point count) |
| 3 | **Outlier cleaning** (removed stragglers in red ✕) |
| 4 | **Scene/object classification** (floor green, wall blue, → mode + conf) |
| 5 | Alignment **INPUT** (tilted; this is what *Preview* sees) |
| 6 | Geometry alignment **APPLIED** (upright; floor tilt ≈ 0°) |
| 7–8 | Manual tweaks — **Rotate 90°**, **Flip Front/Back** |
| 9 | Real **renderer** output (aligned) + **axis gizmo** |

Generated set: `indoor`, `object`, `indoor_real`, `buddha`, `bunny`,
`luigi_3dgs` (real 3DGS object), `bonsai_3dgs` (real 3DGS scene), plus
`case_matrix.png` (9 controlled shapes, input vs applied).

---

## 9. Milestone-1 success-criteria checklist (measured)

Re-measure any time with `python check_success.py` (§7b).

| Criterion | Result | Evidence |
|---|---|---|
| Indoor scenes align upright reliably | ✅ | real scene **89° → 0.62°** (wall-squared); synthetic **0.05°** |
| Obvious objects get roughly aligned | ✅ | Buddha **0°** up recovery under 3 random tilts; synthetic **0°** |
| Ambiguous objects handled honestly | ✅ | bunny/armadillo flagged `up_ambiguous`, conf < 0.75 |
| Scene vs object classification | ✅ | all 8 clouds classified correctly (incl. real 3DGS) |
| Preprocessing / downsampling | ✅ | `big.ply` 250k → 49,537 under the 50k cap |
| **Preview** / apply / reset / manual tweak | ✅ | `Preview Align` stages (orange badge), `Apply` commits — verified headless (§7) |
| **Change history** ("store what was changed") | ✅ | every preview/apply/tweak/reset logged with before/after matrices + timestamps; saved in `<name>_alignment.json` |
| Geometry alignment (heading) | ✅ | scenes squared to a wall (`wall_yawed`), not just upright |
| Axis gizmo in renderer | ✅ | gizmo tracks the scene-root transform |
| No meaningful FPS drop after load | ✅ | empty **243** → loaded **186 FPS** (49k pts, 900×700), **drop 23.5%**, ≥30 ✔ |
| 3DGS input format | ✅ | `luigi_3dgs.ply` SH-DC colour + opacity; `.splat` 360° captures decoded |
