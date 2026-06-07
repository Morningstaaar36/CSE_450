# 3DGS Alignment Engine — Milestone 1

Working baseline app: load a 3D Gaussian Splatting (`.ply`) point cloud, run a
purely geometric alignment pass, and display it in a Vulkan viewport with an
axis gizmo and manual nudge buttons.

The whole pipeline is tuned for laptops **with no dedicated GPU** — the
renderer prefers the integrated GPU (and falls back to a software rasteriser
if Vulkan is unavailable), the analysis cloud is downsampled to ≤ 50 000
points, and every PyTorch op runs on CPU.

> **Testing this milestone?** See **[TESTING.md](TESTING.md)** for the full
> how-to-use / how-to-test guide, including scene-vs-object classification and
> real-world data (real indoor scan, real 3DGS object, Stanford scans):
> ```bash
> python fetch_real_data.py        # download real scenes + objects into data/
> python tests/test_pipeline.py    # full suite → "ALL PASS"
> ```

| Module | Responsibility |
| --- | --- |
| `loader.py` | Parse `.ply` (standard + 3DGS SH-DC) **and `.splat`** (antimatter15 32-byte Gaussians), aggressive voxel-grid + random downsampling, bbox / centroid / extent stats |
| `alignment.py` | `OutlierCleaner`, `ContentTypeClassifier`, `GeometryAxisEstimator` → `(4×4 transform, confidence, mode, ambiguity flags)` on CPU |
| `renderer.py` | Offscreen Vulkan point renderer (`VK_PRIMITIVE_TOPOLOGY_POINT_LIST`), iGPU-preferred device pick, scene-root transform applied per frame, software fallback |
| `app.py` | PyQt5 single-window app: viewport, axis gizmo, Auto Align / Align Up Only / Rotate 90° / Flip Front-Back / Save Alignment |
| `visualize_pipeline.py` | Render every pipeline stage to a report-ready contact sheet per cloud (`visualizations/`) |
| `shaders/*` | GLSL sources + compiled SPIR-V (`points.vert.spv`, `points.frag.spv`) |
| `tests/test_pipeline.py` | End-to-end pipeline check on synthetic + real clouds |
| `tests/test_cases.py` | Many-case classifier / alignment decision-matrix check (+ `--viz`) |
| `environment.yml` / `requirements.txt` | Reproducible CPU-only conda env / pip fallback |

---

## 1. Environment Setup

The project targets **Python 3.10–3.13** and a **CPU-only** PyTorch build.

### Conda (recommended) — one command

The pinned, reproducible env lives in [`environment.yml`](environment.yml)
(CPU-only PyTorch via the `cpuonly` package — never a CUDA build):

```bash
conda env create -f environment.yml
conda activate m1-aligner
```

This installs everything the engine, GUI, tests, **and** the visualization
tooling need: numpy / scipy / scikit-learn / plyfile, CPU PyTorch, PyQt5 +
vulkan, and matplotlib / imageio / pillow / pyyaml.

### pip alternative

If conda isn't available, [`requirements.txt`](requirements.txt) pulls the same
stack (CPU torch from the PyTorch extra index):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<details><summary>Manual conda install (if you prefer explicit steps)</summary>

```bash
conda create -n m1-aligner python=3.11 -y
conda activate m1-aligner
conda install -c pytorch pytorch=2.* cpuonly -y
conda install -c conda-forge numpy scipy scikit-learn plyfile \
                             matplotlib imageio pillow pyyaml -y
pip install PyQt5 vulkan PyOpenGL
```
</details>

### Sanity check

```bash
python - <<'PY'
import torch, numpy, plyfile, vulkan, PyQt5
print("torch     ", torch.__version__, "device =", torch.zeros(1).device)
print("numpy     ", numpy.__version__)
print("plyfile   ", plyfile.__version__)
print("PyQt5     OK")
print("vulkan    bindings OK")
PY
```

You should see `device = cpu` even if a discrete GPU exists — the engine
explicitly targets `device='cpu'` (see `alignment.py`).

---

## 2. Device Setup — verifying Vulkan visibility

The renderer prefers an integrated GPU, then a software rasteriser
(LLVMpipe / Lavapipe), then a discrete GPU. To verify your Vulkan loader sees
**any** ICD:

### Linux

```bash
# Driver libraries on Debian/Ubuntu
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan1
vulkaninfo --summary           # should list at least one physicalDevice
```

If `vulkaninfo` isn't available, the project also ships its own probe:

```bash
python renderer.py --width 320 --height 240
# expected: "is_vulkan": true and a deviceName string
```

A typical Intel-iGPU machine prints
`device_name: "Intel(R) Iris(R) Xe Graphics (TGL GT2)"` and
`device_type: "integrated"`.

### Windows

Install the GPU vendor's regular driver (Intel HD/Iris, AMD Radeon, NVIDIA
Studio); the Vulkan ICD ships with it. Run `vulkaninfo.exe` from the LunarG
SDK to confirm.

### macOS

Vulkan on macOS requires **MoltenVK** (Vulkan on top of Metal):

```bash
brew install vulkan-headers vulkan-loader molten-vk
```

If MoltenVK initialisation fails, the renderer transparently falls back to
its NumPy software rasteriser. The rest of the app stays fully functional.

---

## 3. Dataset Handling

The app accepts:

* standard `.ply` with `x, y, z` (+ optional `red/green/blue`),
* 3DGS `.ply` with spherical-harmonic colour (`f_dc_0..2`, `opacity`),
* **3DGS `.splat`** files (antimatter15 32-byte Gaussians — the format most
  web 3DGS viewers export).

Missing colour data renders neutral grey.

### Pre-vetted lightweight datasets (auto-downloaded)

`python fetch_real_data.py` pulls these real clouds into `data/` (stdlib only,
no extra deps). Each targets a specific part of the pipeline:

| Cloud | Type | Exercises |
| --- | --- | --- |
| `indoor_real.ply` (Redwood living-room fragment) | **indoor scene** | classifier + "indoor aligns upright" (89° → 3.3°) |
| `luigi_3dgs.ply` (HuggingFace `dylanebert/3dgs`) | **real 3DGS object** (`.ply`) | loader SH-DC colour/opacity path; flat-object classification |
| `bonsai_3dgs.splat` (Mip-NeRF360 via `dylanebert/3dgs`) | **real 3DGS indoor scene** (`.splat`) | `.splat` loader at scale (272k → ≤50k), object-centric-scene classification |
| `bicycle_3dgs.splat` (Mip-NeRF360 via `dylanebert/3dgs`) | **real 3DGS outdoor scene** (`.splat`) | `.splat` loader on a 568k-Gaussian outdoor capture |
| `buddha.ply` (Stanford Happy Buddha) | **obvious object** | "obvious objects get roughly aligned" |
| `bunny.ply` (Stanford Bunny) | ambiguous object | honest `up_ambiguous` flagging |
| `armadillo.ply` (Stanford Armadillo) | object (binary **big-endian**) | loader byte-order robustness |

`data/big.ply` (250k-pt capture) ships with the repo.

> **Note on 3DGS 360° captures.** Mip-NeRF360 scenes (`bonsai`, `bicycle`) are
> *object-centric* — the camera orbits a central subject, so there is no single
> dominant floor plane. The classifier therefore (correctly) returns `object`
> rather than hallucinating a room. The canonical "indoor → upright" demo is
> the Redwood `indoor_real.ply`, which is a partial room scan with a real
> floor + wall.

### Pointing the app at a `.ply`

```bash
# Auto-load a file from the CLI:
python app.py /path/to/cloud.ply

# Or open via the File menu / "Open .ply…" button in the side panel.
```

Programmatic use (the engine is importable without the GUI):

```python
from loader import load_point_cloud
from alignment import AlignmentEngine

pc = load_point_cloud("data/indoor.ply", point_cap=50_000)
result = AlignmentEngine().align(pc.positions)
print(result.mode, result.confidence, result.ambiguity_flags)
print(result.transform)      # (4, 4) float32 numpy array
```

### Synthetic data (no download required)

For development without a network connection:

```bash
python tests/gen_synth_ply.py --out data
# wrote data/indoor.ply  (rotated synthetic room, ~11 k points)
# wrote data/object.ply  (tilted synthetic vase, ~9.5 k points)
```

---

## 4. Running the App

```bash
python app.py                 # opens an empty viewport
python app.py data/indoor.ply # auto-loads on launch
```

Controls:

| Action | How |
| --- | --- |
| Orbit | left-mouse drag |
| Zoom | mouse wheel |
| Auto Align | side-panel button / re-runs `AlignmentEngine.align` |
| Align Up Only | preserves the auto up-axis, zeros yaw |
| Rotate 90° | premultiplies a 90° turn around world up |
| Flip Front/Back | premultiplies a 180° turn around world up |
| Reset | restores identity transform |
| Save Alignment | writes a JSON sidecar (transform + mode + confidence) |

---

## 5. Running the autonomous test

The end-to-end check generates synthetic clouds, runs them through loader +
alignment, exercises Vulkan-vs-software paths in the renderer, and asserts
the recovered up-axis lies within tolerance of world +Y.

```bash
python fetch_real_data.py          # once, to pull the real clouds
python tests/test_pipeline.py
# == indoor.ply ==
# alignment: mode=indoor  conf=0.95  true-up err = 0.04°  flags=[]
# == object.ply ==
# alignment: mode=object  conf=0.82  true-up err = 0.44°  flags=['yaw_ambiguous']
# == real-world data ==
#   indoor_real.ply  mode=indoor conf=0.77  [scene] floor upright to 3.3°
#   luigi_3dgs.ply   mode=object conf=0.56  [gs_object] 3DGS SH-DC colours OK
#   buddha.ply       mode=object conf=0.90  [obvious] up recovered to ≤0.0° under tilt
#   bunny.ply        mode=object conf=0.60  [ambiguous] correctly flagged up_ambiguous
#   ...
# ALL PASS
```

The suite uses **non-circular** metrics (recovery of a known up direction,
floor-tilt after alignment, stability under random tilts) and exits non-zero
on any assertion failure. Real files are skipped gracefully if not downloaded.
See **[TESTING.md](TESTING.md)** for the full breakdown.

---

## 6. Visualizing each pipeline stage

Every Milestone-1 step can be *seen*, not just asserted. `visualize_pipeline.py`
drives a cloud through the real submodules and the Vulkan renderer and writes a
3×3 contact sheet per cloud to `visualizations/`:

```bash
python visualize_pipeline.py                 # the curated default set
python visualize_pipeline.py data/bunny.ply  # one or more explicit files
```

Each sheet lays out:

| col 1 | col 2 | col 3 |
| --- | --- | --- |
| 1 · raw cloud | 2 · downsampled analysis cloud | 3 · outliers removed (red ✕) |
| 4 · classify (green floor, blue wall) | 5 · alignment **input** (tilted) | 6 · alignment **applied** (upright) |
| 7 · manual Rotate 90° | 8 · manual Flip Front/Back | 9 · real renderer output + gizmo |

The classifier/alignment **decision matrix** (scenes vs prolate / oblate /
isotropic / flat-object) renders to `visualizations/case_matrix.png`:

```bash
python tests/test_cases.py --viz
```

---

## 7. Architecture notes

* **Scene-root transform, not splat baking.** The 4×4 matrix is uploaded to
  the vertex shader as a push constant and combined with the camera matrix
  to form MVP. Manual nudges are O(1) and require no vertex re-upload.
  ([renderer.py — `_record_command_buffer`])
* **Outlier cleaner first**, so RANSAC and PCA aren't biased by stragglers.
* **Plane-coverage classifier.** Indoor/object is decided primarily from the
  fraction of points lying on the best RANSAC plane — a rotation-invariant
  signal that doesn't depend on the cloud being aligned to start with.
* **Inline `[M2 HOOK]` markers** show where the Milestone 2 ML up-axis
  predictor will slot in.

---

## 8. Success Criteria Checklist

Tick these against your own captures once the environment is up.

* [ ] Indoor scenes align upright reliably using the geometry estimator.
* [ ] Obvious objects get roughly aligned.
* [ ] No meaningful FPS drop after initial load and downsampling.
* [ ] The 4x4 transform successfully applies to the scene root.

### How to evaluate each criterion

1. **Indoor alignment.** Load an indoor scan (Redwood frame, NeRF/3DGS room).
   Click *Auto Align*. The floor should sit horizontally and the ceiling
   should be above the camera target. The status bar should say
   `mode=indoor` with `conf ≥ 0.6`.

2. **Object alignment.** Load a single-object cloud (Stanford Bunny / your
   own captured vase). Click *Auto Align*. The detected base of the object
   should be at the bottom of the view. Status bar shows `mode=object`.

3. **FPS sanity.** Watch the FPS counter in the top-left HUD. After load it
   should remain above ~30 fps while you orbit (Intel Iris Xe: ~120 fps at
   50 k points on the offscreen path).

4. **Scene-root transform.** Click *Auto Align*, then *Rotate 90°*. The
   cloud should re-orient instantly; the axis gizmo should follow. Saving
   alignment and inspecting the JSON should reveal the same matrix in
   `transform_4x4`.

---

## 9. Known limitations & follow-up

* **Point formats:** standard `.ply` (XYZ + RGB), 3DGS `.ply` (SH-DC colour),
  and antimatter15 `.splat` are all loaded. Per-Gaussian ellipsoids
  (`scale_*`, `rot_*`) are parsed/decoded but rendered as points, not splats,
  for now.
* **Very flat lone objects** (a bare plate/coin whose single face covers ≥45 %
  of points) can still read as a scene via the dominant-floor path — geometry
  alone cannot tell a floor from a plate. Real figurines (`luigi_3dgs`) and
  relief panels stay `object` because their dominant face is < 45 %.
* **Vulkan cleanup on Linux** can segfault at interpreter shutdown — a known
  issue with the `vulkan-python` cffi bindings. The app calls `os._exit(0)`
  on quit to sidestep it; no data is lost.
* **Object up-axis is shape-based.** Elongated objects (long axis up) and
  flat/disc objects (thin axis up) are aligned from geometry. Free-form
  objects whose up is *semantic* (e.g. the Stanford Bunny) cannot be oriented
  by geometry alone — these are flagged `up_ambiguous` with low confidence
  rather than confidently mis-aligned, and are the target of the Milestone-2
  ML up-axis predictor.
* **Yaw ambiguity** on symmetric objects (e.g. a vase) is flagged
  `yaw_ambiguous` but not yet resolved (handled by the Milestone-2 ML module).
