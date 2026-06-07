# Codebase Analysis — 3DGS Alignment Engine (Milestone 1)

> Generated: 2026-06-07
> Branch: `milestone1_shahraz`
> Working directory: `milestone1/`

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [How the Files Are Organised](#2-how-the-files-are-organised)
3. [How Data Flows Through the System](#3-how-data-flows-through-the-system)
4. [What Each File Does](#4-what-each-file-does)
   - [loader.py](#41-loaderpy)
   - [alignment.py](#42-alignmentpy)
   - [renderer.py](#43-rendererpy)
   - [app.py](#44-apppy)
   - [fetch_real_data.py](#45-fetch_real_datapy)
   - [tests/gen_synth_ply.py](#46-testsgen_synth_plypy)
   - [tests/test_pipeline.py](#47-teststest_pipelinepy)
   - [shaders/](#48-shaders)
5. [The Dataset Library](#5-the-dataset-library)
6. [How the Architecture Spec Maps to the Code](#6-how-the-architecture-spec-maps-to-the-code)
7. [Milestone 1 Deliverables Checklist](#7-milestone-1-deliverables-checklist)
8. [Success Criteria Assessment](#8-success-criteria-assessment)
9. [Hooks Left for Milestone 2](#9-hooks-left-for-milestone-2)
10. [Known Gaps and Risks](#10-known-gaps-and-risks)

---

## 1. What This Project Does

This project is a tool that takes a 3D point cloud file and figures out how it should be oriented — which direction is "up" — so that it stands upright when rendered. Think of it like a tool that automatically rotates a crooked 3D scan so that the floor is flat and objects look natural.

The input is a `.ply` file, which is a standard format for storing 3D point clouds. These files come from two sources: ordinary scanners that capture X/Y/Z positions and RGB colours, and 3D Gaussian Splatting (3DGS) captures that store extra data per point for photorealistic rendering. The system handles both.

The output is a 4×4 transformation matrix — a mathematical description of how to rotate and translate the scene so it appears correctly oriented. This matrix is applied live to the renderer so the user sees the result immediately.

The project also includes a graphical viewer where users can open a file, trigger automatic alignment, manually fine-tune the result with a few buttons, and save the final orientation.

---

## 2. How the Files Are Organised

```
milestone1/
│
├── loader.py               ← reads .ply files and prepares them for processing
├── alignment.py            ← figures out the correct orientation of the cloud
├── renderer.py             ← draws the point cloud on screen
├── app.py                  ← the graphical window the user interacts with
├── fetch_real_data.py      ← downloads real-world test datasets from the internet
│
├── shaders/                ← GPU drawing programs for the Vulkan renderer
│   ├── points.vert / .spv  ← positions each point on screen
│   └── points.frag / .spv  ← colours each point as a round circle
│
├── tests/
│   ├── gen_synth_ply.py    ← generates synthetic test data (no internet needed)
│   └── test_pipeline.py    ← automatically tests the whole pipeline end-to-end
│
├── data/                   ← all point cloud files used for testing
│   └── screenshots/        ← visual proof screenshots saved during testing
│
├── Project Milestones.md   ← describes what to build in each phase (M1–M4)
├── Architecture Elements.md← describes how the modules should be designed
├── README.md               ← quick-start instructions
├── TESTING.md              ← how to run the tests
└── 2306.11406v1.pdf        ← the 3DGS research paper (Kerbl et al. 2023)
```

The four main Python files follow a layered design: `loader.py` feeds `alignment.py`, both feed `renderer.py`, and `app.py` wraps all three into a usable application. The test files sit in `tests/` and exercise the full chain without needing a screen.

---

## 3. How Data Flows Through the System

The path a `.ply` file takes from disk to screen is:

1. **Loading** — `loader.py` reads the raw file. It extracts the XYZ positions and colour data for each point. If the file is a 3DGS capture, it converts the proprietary colour encoding to plain RGB. If the file has more points than the GPU can comfortably handle (more than 50,000), it shrinks it down using a spatial grid so that the reduced set still faithfully represents the shape.

2. **Alignment** — `alignment.py` receives the loaded point positions and runs a three-stage pipeline. First it removes stray noise points. Then it classifies whether the scene looks like an indoor room or an isolated object. Finally, based on that classification, it estimates which direction is "up" and builds the rotation matrix that corrects the orientation.

3. **Rendering** — `renderer.py` receives the (possibly downsampled) points and draws them using Vulkan. The alignment matrix is applied at the moment of drawing — the point positions themselves never change. When the user clicks "Auto Align", only the matrix is updated, and the next frame renders the corrected view.

4. **Interaction** — `app.py` provides the window, buttons, and file dialogs. It calls the loader when the user opens a file, calls the alignment engine when the user clicks a button, and tells the renderer which matrix to use. It also shows information about the loaded cloud (point count, scale, type) and the alignment result (confidence, detected mode, any warnings).

---

## 4. What Each File Does

### 4.1 `loader.py`

`loader.py` is the gateway between a `.ply` file on disk and the rest of the system. Its job is to produce a clean, usable representation of the point cloud regardless of what format the file happens to use.

**Reading the file**

When it opens a PLY file, it looks for colour data in three ways, in this priority order. First it looks for standard red/green/blue columns — these are present in most ordinary point cloud files. Second, if those columns are absent, it looks for the special `f_dc` columns that 3DGS files use to encode colour via spherical harmonics — it converts these to regular RGB so the rest of the system doesn't need to know about 3DGS internals. Third, if no colour data exists at all, it assigns a neutral grey to every point so the cloud is still visible.

It also reads the opacity data that 3DGS files store in logit form, converts it to a 0–1 range, and keeps it alongside the point data in case downstream code needs it.

**Downsampling**

Raw scans can contain hundreds of thousands or millions of points. Rendering that many on an integrated GPU (the target hardware) would be slow. `loader.py` enforces a hard cap of 50,000 points.

To reduce the cloud without losing its shape, it uses a technique called voxel downsampling: it divides space into a 3D grid and replaces all points inside each grid cell with a single averaged point. This preserves the overall structure while dramatically reducing the count. It starts with a grid spacing estimated from the cloud's size and refines it iteratively until the point count lands just under the cap. As a last resort if the grid approach fails, it falls back to random sampling.

The result of `loader.py` is a `PointCloud` object that holds the position array, the colour array, basic statistics (bounding box, centroid, scale), and metadata like the original point count before downsampling and whether the file was a 3DGS capture.

---

### 4.2 `alignment.py`

`alignment.py` is the analytical core. It takes the loaded point positions and produces an `AlignmentResult` that describes how to orient the scene correctly. Internally it is structured as four cooperating components.

**OutlierCleaner**

Before any analysis, the cloud is cleaned. The cleaner looks at how far each point is from its nearest neighbours. Points that are much more isolated than the average — statistical outliers — are removed. This prevents a handful of stray scan artifacts from skewing the geometry analysis that follows.

**ContentTypeClassifier**

Before computing an orientation, the system needs to know what kind of scene it is looking at, because rooms and objects need different alignment strategies.

The classifier works by searching for flat surfaces. It uses a technique called RANSAC (Random Sample Consensus): it repeatedly picks three random points, fits a plane through them, and counts how many other points lie close to that plane. It does this many times and keeps the best-fitting plane found.

An indoor room has a floor that typically accounts for a large fraction of the total point cloud. If a dominant plane covers more than 22% of the cloud, the classifier considers the scene to be indoors. It then checks for a second plane (a wall) that is perpendicular to the first. Finding a perpendicular wall increases the confidence that this is genuinely a room — not, for example, a flat disc-shaped object whose front and back faces happen to be parallel.

If no dominant plane is found, or the plane coverage is low, the cloud is classified as an isolated object.

The result is a classification (indoor or object) and a confidence score. It also saves the detected floor plane geometry so the next stage can reuse it without running RANSAC again.

**GeometryAxisEstimator**

This component builds the actual rotation matrix based on the classification result.

For indoor scenes it uses the floor plane that the classifier already found. The floor's surface normal points perpendicular to the floor. The estimator rotates the scene so that normal aligns with the world's up direction (+Y). It also recentres the scene at the origin. The result is a transform that makes the floor lie flat and the scene sit centred.

For isolated objects it uses a mathematical technique called Principal Component Analysis (PCA), which finds the axes of greatest spread in the point cloud. If an object is elongated — like a statue or a bottle — its longest axis is a good candidate for up. If it is flat — like a plate or a book — its thinnest axis (the one with the least spread) points up. The estimator decides which case applies and picks the appropriate axis.

It also handles ambiguity honestly. Some objects like the Stanford Bunny have no geometrically obvious up direction — the up axis is determined by what the object looks like, not by its shape. When the geometry is ambiguous, the estimator sets a flag rather than guessing confidently, so downstream code can treat the result with appropriate scepticism.

**AlignmentEngine**

The engine is the top-level coordinator. It runs the three stages in order (clean → classify → estimate), combines the confidence scores from each stage into a single overall confidence, and packages everything into an `AlignmentResult`.

It also provides the manual correction tools used by the GUI buttons: rotating the current transform by 90 degrees around the up axis, flipping the front-to-back orientation, and stripping out all horizontal rotation while keeping the up-axis correction.

---

### 4.3 `renderer.py`

`renderer.py` draws the point cloud. It uses Vulkan, a modern low-level graphics API, to render directly on the GPU. Because the project targets integrated graphics hardware (iGPU), the code specifically prefers iGPUs when choosing which GPU to use, rather than any discrete card.

Each point is drawn as a small circle rather than a square pixel — the fragment shader discards fragments outside a circular radius, giving the cloud a cleaner appearance.

The alignment matrix is passed to the GPU as a push constant, which is a small piece of data that the GPU receives just before each draw call. This means changing the alignment — for example when the user clicks "Auto Align" — just requires sending a new matrix. The point data itself is not re-uploaded. This makes alignment updates essentially instant.

If Vulkan is unavailable (for example on a CI server without GPU drivers), the renderer falls back to a pure NumPy software implementation. This fallback projects all points mathematically, sorts them by depth so nearer points appear on top, and composites them into an image array. It produces the same visual result and is used in the test suite so tests can run headlessly.

The renderer also supports live resizing: when the window is resized, it recreates its GPU resources to match the new dimensions.

---

### 4.4 `app.py`

`app.py` is the user-facing application. It builds a PyQt5 window with a viewport on the left and a control panel on the right.

**The viewport** is the 3D view. It redraws up to 30 times per second, but only when something has changed — this avoids wasting CPU cycles when the scene is static. The user can rotate the view by dragging the mouse and zoom by scrolling the wheel. In the corner of the view is an axis gizmo — three coloured arrows showing the X, Y, and Z directions of the current scene orientation. This updates live as alignment is applied, so the user can immediately see whether the axes are correct. A small heads-up display shows the current point count, rendering backend (Vulkan or CPU), and the live frame rate.

**The control panel** has buttons that map directly to alignment operations:

- *Open .ply* loads a file and shows it with no transformation applied.
- *Auto Align* runs the full alignment pipeline and applies the result to the view.
- *Align Up Only* keeps the up-axis correction from auto-align but removes any horizontal rotation, useful when the object is upright but facing the wrong way.
- *Rotate 90°* turns the scene 90 degrees horizontally, for fine-tuning facing direction.
- *Flip Front/Back* reverses the horizontal facing by 180 degrees.
- *Reset* removes all transformations and returns to the raw loaded orientation.
- *Save Alignment* writes a JSON file containing the final transform matrix and metadata about the loaded file. If auto-align was run, the automatic result is also saved alongside whatever manual adjustments the user applied on top.

The status bar at the bottom shows what action was last taken and which rendering backend is active. A diagnostics label shows the alignment engine's output: what mode was detected, the confidence score, and any warning flags like "up direction is ambiguous".

When the window is closed, the application explicitly tears down the Vulkan resources before exiting. This is necessary because the Vulkan wrapper library does not handle cleanup safely if left to Python's garbage collector.

---

### 4.5 `fetch_real_data.py`

`fetch_real_data.py` downloads real-world point cloud files from the internet to populate the `data/` directory. It uses only Python's standard library — no third-party packages required.

It downloads five datasets, each testing a different aspect of the pipeline:

- A real indoor scan from the Open3D project (Redwood living room fragment) — for testing that the indoor classifier and floor-alignment work on actual scan data.
- A 3DGS figurine file of Luigi from Hugging Face — for testing that 3DGS files with their special colour encoding are handled correctly, and that the flat-object trap is avoided.
- Three Stanford models (Buddha, Bunny, Armadillo) — for testing object alignment on well-known benchmark geometry.

The script is smart about not re-downloading files that already exist, and it accepts a `--force` flag to re-download everything. If any download fails, it exits with an error code so CI pipelines can detect the failure.

---

### 4.6 `tests/gen_synth_ply.py`

`gen_synth_ply.py` creates two synthetic test datasets that do not require an internet connection. These are used as the primary automated test fixtures.

The first is a synthetic indoor room: a floor, four walls, a ceiling, and two box-shaped objects sitting on the floor. The room is deliberately tilted — rotated by two angles at once — so that the ground truth "up" direction is known precisely. The test suite uses this known-correct up direction to verify that the alignment engine recovers it accurately.

The second is a synthetic vase: a rotationally symmetric object with a bulging profile, also tilted by a known amount. This tests the object alignment path.

Both files are written as standard PLY files (not 3DGS format), so they exercise the standard colour-reading path in the loader.

Because the tilts and point placements are fully deterministic (seeded random generation), running the generator twice always produces identical files. This makes the test results reproducible.

---

### 4.7 `tests/test_pipeline.py`

`test_pipeline.py` runs the full pipeline automatically, without any GUI or display. It produces a pass/fail verdict for each test.

The tests cover several layers:

**Loader tests** verify that a file can be opened, that the output has sensible shapes and values (no NaN, no negative scale, within the point cap), and that the cloud statistics are correctly computed.

**Alignment tests** verify that the transform output is mathematically valid (a proper rotation matrix, not a shear or reflection), that the correct mode was detected, and that the estimated up direction is close to the known ground-truth up direction. The angular error tolerance is 8 degrees for indoor scenes and 10 degrees for objects. The ground-truth direction is the physical tilt baked in by the generator — it is entirely independent of what the alignment engine computes, so the test is a genuine check rather than a circular one.

**Renderer tests** verify that the renderer produces an image with actual content (not a blank frame), and that applying the alignment transform visibly changes the rendered image. This confirms that the scene-root matrix mechanism works end-to-end.

**Real-data tests** verify specific properties of the downloaded datasets. The indoor scan is checked by aligning it and then running a second RANSAC on the aligned cloud to measure how flat the floor is — the residual tilt must be under 10 degrees. The 3DGS figurine is checked to confirm it was recognised as a 3DGS file and not misclassified as an indoor scene. The Buddha is tilted randomly three times and must recover to within 15 degrees each time. The Bunny and Armadillo must be honestly flagged as ambiguous and given low confidence rather than confidently mis-aligned.

Real-data tests skip gracefully if the file was not downloaded, so the test suite still runs offline with only the synthetic fixtures.

---

### 4.8 `shaders/`

The `shaders/` folder contains the GPU programs that tell Vulkan how to draw the points.

The vertex shader receives each point's position and colour from the GPU buffer. It transforms the position using the combined matrix (projection × camera view × alignment transform) to place it correctly on screen. It also sets the size of each point sprite.

The fragment shader runs once for each pixel covered by a point sprite. Instead of filling the full square area that the GPU uses for point sprites, it discards any pixel that falls outside the inscribed circle. This makes each point appear as a round dot rather than a square — visually cleaner, especially at larger point sizes.

The `.spv` files are pre-compiled binary versions of the shaders in SPIR-V format, which is what Vulkan actually loads at runtime. Both the GLSL source and compiled SPIR-V are stored so the shaders can be inspected and recompiled if needed.

---

## 5. The Dataset Library

The `data/` directory holds all point cloud files used for development and testing. They fall into three categories.

### Synthetic datasets (generated locally)

These are created by running `gen_synth_ply.py` and do not require internet access. They have precisely known ground-truth orientations, which makes them ideal for automated correctness testing.

| File | What it represents | Why it exists |
|------|--------------------|---------------|
| `indoor.ply` | A room with floor, walls, ceiling, and two box objects, tilted 25° + 18° | Primary test for indoor classification and floor alignment |
| `object.ply` | A vase shape with a bulging profile, tilted 20° + 35° | Primary test for object classification and PCA alignment |

### Repository datasets (shipped with the code)

| File | What it represents | Why it exists |
|------|--------------------|---------------|
| `big.ply` | A dense point cloud with over 250,000 points | Tests that the downsampling pipeline handles large files correctly and that the renderer stays fast after the reduction |

### Real-world datasets (downloaded by `fetch_real_data.py`)

These come from public research datasets and capture real scan conditions — noise, partial coverage, irregular density — that the synthetic files cannot fully simulate.

| File | Source | What it is | What it tests |
|------|--------|------------|---------------|
| `indoor_real.ply` | Open3D / Redwood RGB-D dataset | A fragment of a real living-room scan | Whether the floor detection works on actual scan noise and partial coverage — this is the primary M1 success criterion for indoor scenes |
| `luigi_3dgs.ply` | Hugging Face (dylanebert/3dgs) | A 3DGS capture of a Luigi figurine | Whether the 3DGS colour path (`f_dc` → RGB) works, and whether the classifier avoids the flat-object trap on an object with two nearly parallel large faces |
| `buddha.ply` | Stanford 3D Scanning Repository | The Stanford Happy Buddha (~540k points, reduced to cap) | A tall, elongated statue with a clear geometric up direction — ideal for testing that obvious objects align well even after large random tilts |
| `bunny.ply` | Stanford 3D Scanning Repository | The Stanford Bunny (~35k points) | A freeform, roughly isotropic shape where "up" is semantic (the bunny has ears, not a dominant axis) — tests that the engine flags ambiguity honestly instead of guessing |
| `armadillo.ply` | Stanford 3D Scanning Repository | The Stanford Armadillo (~172k points) | A large file stored in big-endian binary PLY format — tests that the loader handles byte-order variation in PLY files correctly |

### Screenshots

`data/screenshots/` contains PNG images captured during manual testing of the application, showing the before and after of alignment for each major dataset. These serve as visual evidence that the pipeline is working as expected.

---

## 6. How the Architecture Spec Maps to the Code

The `Architecture Elements.md` document describes the target module structure. Here is how the code implements it.

### Loader layer

The loader is responsible for getting point cloud data into a consistent internal form. `loader.py` handles all three parts: parsing the PLY file (including both standard and 3DGS colour formats), creating a spatially reduced analysis cloud that fits in GPU memory, and computing bounding box statistics used by the rest of the system.

### Alignment engine

The alignment engine is the most architecturally significant module. `alignment.py` implements three of the four planned submodules:

- **OutlierCleaner** removes statistical outliers before analysis begins.
- **ContentTypeClassifier** determines whether the scene is an indoor environment or an isolated object using RANSAC plane detection.
- **GeometryAxisEstimator** computes the orientation transform using floor normal analysis for indoor scenes and principal component analysis for objects.
- **MLAxisEstimator** is the fourth submodule — it is not implemented in Milestone 1. Integration points are marked with `[M2 HOOK]` comments at the exact lines where a machine-learning predictor would be blended in.
- **AlignmentFusion** (combining multiple confidence scores) is implemented as a single line inside `AlignmentEngine.align` rather than a separate class. This is acceptable for Milestone 1; it may need to grow into its own class when ML blending is added in Milestone 2.
- **ConfidenceScorer** is embedded within each submodule's output — every stage computes its own confidence score and the engine combines them via a geometric mean.

### Renderer integration

The alignment matrix is applied as a scene-root transform at render time, not baked into the vertex data. This is the preferred approach for the iterative development phase (M1–M2) because it allows instantaneous matrix updates. Baking into vertex data is deferred to Milestone 4.

### UX layer

All planned UI controls are implemented in `app.py`: Auto Align, Align Up Only, Rotate 90°, Flip Front/Back, and Save Alignment. A Reset button was added beyond the spec as a convenience.

---

## 7. Milestone 1 Deliverables Checklist

| Deliverable | Status | What was built |
|-------------|--------|----------------|
| Preprocessing / downsampling | Complete | Hybrid voxel-grid downsampling with binary-search refinement, hard cap at 50,000 points, handles files of 250k+ points |
| Scene vs object classification | Complete | RANSAC dominant-plane detection plus orthogonal wall test; correct on all synthetic and real-world test cases |
| Geometry-based alignment | Complete | Floor-normal alignment for indoor scenes, PCA prolate/oblate heuristic for objects; angular error under tolerance on all tested files |
| Preview | Complete | Live Vulkan renderer at up to 30 fps with orbit camera; falls back to CPU renderer when Vulkan is unavailable |
| Apply | Complete | "Auto Align" button applies the computed transform immediately as a GPU push constant |
| Reset | Complete | "Reset" button returns the scene to its unmodified loaded orientation |
| Manual tweak | Complete | Three composable operations (Align Up Only, Rotate 90°, Flip Front/Back) stack on top of any existing transform |
| Axis gizmo | Complete | Three-axis orientation indicator in the bottom-right corner of the viewport, updates every frame |

All 7 Milestone 1 deliverables are complete.

---

## 8. Success Criteria Assessment

| Criterion | Status | How it is verified |
|-----------|--------|--------------------|
| Indoor scenes align upright reliably | Met | After aligning `indoor_real.ply`, a second RANSAC pass on the aligned cloud measures how flat the floor is. The residual tilt must be under 10 degrees. This is a non-circular test — it measures the physical floor angle in the output, not just whether the engine's own plane estimate ended up on +Y. |
| Obvious objects get roughly aligned | Met | The Stanford Buddha is tilted by three different random amounts (up to ±50° per axis) and the engine must recover the up direction to within 15 degrees each time. Free-form objects (Bunny, Armadillo) are accepted as long as they are honestly flagged as ambiguous rather than given a confident wrong answer. |
| No meaningful FPS drop after load | Met | The 50,000-point hard cap ensures a constant GPU vertex budget regardless of input file size. Transform updates cost only a matrix push constant, not a vertex re-upload. The 30 Hz timer prevents busy-looping. The HUD shows live FPS for manual verification. |

All 3 success criteria are met.

---

## 9. Hooks Left for Milestone 2

The code was written with Milestone 2's machine learning extension in mind. Three specific places in `alignment.py` are marked with `[M2 HOOK]` comments, each corresponding to a place where an ML predictor's output would be blended with the current geometry-only result.

The first hook is in the classifier — an ML model's scene-type prediction would be blended with the geometry-based plane-coverage score. The second and third hooks are in the axis estimator — for both indoor and object paths, the ML model's predicted up direction would be weighted against the geometric estimate, with the blend ratio determined by each side's confidence.

The `up_ambiguous` flag is the primary signal that M2 should act on. When the geometry cannot determine up (isotropic shapes, symmetric objects), the flag is set and the confidence is heavily discounted. These are exactly the cases where an ML model trained on visual features would add value that pure geometry cannot.

---

## 10. Known Gaps and Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| ML axis estimator not implemented | Expected | This is a Milestone 2 deliverable. All three integration points are marked in the code. |
| AlignmentFusion not a standalone class | Low | Currently it is a single line in AlignmentEngine. Will likely need to grow into its own class when ML blending logic is added. |
| Yaw disambiguation for objects | Medium | The estimator correctly finds up but picks an arbitrary facing direction based on PCA axes. The correct front face is a semantic concept that geometry alone cannot determine. Planned fix is the M2 ML predictor. |
| Baking transform into vertex data | Deferred | Scene-root matrix approach is used throughout. Baking is a Milestone 4 task. |
| Alignment not auto-loaded on reopen | Partial | Save Alignment writes a JSON file, but there is no mechanism to reload that alignment when the same file is opened again. Planned as a Milestone 4 deliverable. |
| Vulkan driver dependency | Low | The NumPy software fallback is fully tested and produces correct output. All automated tests run through the fallback path headlessly. |
| External dataset URLs | Low | The Hugging Face and Stanford download URLs are external. If they change, `fetch_real_data.py` will fail. The test suite skips gracefully when files are absent, so this only affects tests that require those specific files. |
