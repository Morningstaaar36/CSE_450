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
| `loader.py` | Parse `.ply` (standard + 3DGS), aggressive voxel-grid + random downsampling, bbox / centroid / extent stats |
| `alignment.py` | `OutlierCleaner`, `ContentTypeClassifier`, `GeometryAxisEstimator` → `(4×4 transform, confidence, mode, ambiguity flags)` on CPU |
| `renderer.py` | Offscreen Vulkan point renderer (`VK_PRIMITIVE_TOPOLOGY_POINT_LIST`), iGPU-preferred device pick, scene-root transform applied per frame, software fallback |
| `app.py` | PyQt5 single-window app: viewport, axis gizmo, Auto Align / Align Up Only / Rotate 90° / Flip Front-Back / Save Alignment |
| `shaders/*` | GLSL sources + compiled SPIR-V (`points.vert.spv`, `points.frag.spv`) |
| `tests/` | Synthetic `.ply` generator + end-to-end pipeline check |

---

## 1. Environment Setup

The project targets **Python 3.10–3.13** and a **CPU-only** PyTorch build.

### Conda (recommended)

```bash
# 1. create the env
conda create -n m1-aligner python=3.11 -y
conda activate m1-aligner

# 2. CPU-only PyTorch (no CUDA wheels — see hardware constraint)
conda install -c pytorch pytorch=2.* cpuonly -y

# 3. numerical / loading deps from conda-forge
conda install -c conda-forge numpy scipy scikit-learn plyfile -y

# 4. GUI + Vulkan Python bindings (pip — these aren't on conda-forge yet)
pip install PyQt5 vulkan PyOpenGL
```

### One-liner pip alternative

If conda isn't available:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-learn plyfile \
            torch --index-url https://download.pytorch.org/whl/cpu \
            PyQt5 vulkan PyOpenGL
```

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

The app accepts any standard `.ply` file with `x, y, z` vertex properties.
RGB (`red, green, blue` channels) or 3DGS-style spherical-harmonic colour
(`f_dc_0..2`) are both consumed; missing colour data renders neutral grey.

### Pre-vetted lightweight datasets (auto-downloaded)

`python fetch_real_data.py` pulls these real clouds into `data/` (stdlib only,
no extra deps). Each targets a specific part of the pipeline:

| Cloud | Type | Exercises |
| --- | --- | --- |
| `indoor_real.ply` (Redwood living-room fragment) | **indoor scene** | classifier + "indoor aligns upright" (89° → 3.3°) |
| `luigi_3dgs.ply` (HuggingFace `dylanebert/3dgs`) | **real 3DGS object** | loader SH-DC colour/opacity path; flat-object classification |
| `buddha.ply` (Stanford Happy Buddha) | **obvious object** | "obvious objects get roughly aligned" |
| `bunny.ply` (Stanford Bunny) | ambiguous object | honest `up_ambiguous` flagging |
| `armadillo.ply` (Stanford Armadillo) | object (binary **big-endian**) | loader byte-order robustness |

`data/big.ply` (250k-pt capture) ships with the repo.

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

## 6. Architecture notes

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

## 7. Success Criteria Checklist

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

## 8. Known limitations & follow-up

* **PLY format only** for now (standard XYZ + RGB, or 3DGS SH-DC). Splat
  ellipsoids (`scale_*`, `rot_*`) are parsed but not rendered yet.
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
