# Milestone 1 — Full Codebase Analysis

## Table of Contents
1. [Project Goal & Deliverables](#1-project-goal--deliverables)
2. [Repository Layout](#2-repository-layout)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [Module Deep-Dives](#4-module-deep-dives)
   - [loader.py](#41-loaderpy)
   - [alignment.py](#42-alignmentpy)
   - [renderer.py](#43-rendererpy)
   - [app.py](#44-apppy)
   - [Shaders](#45-shaders)
   - [tests/](#46-tests)
5. [How the Modules Connect](#5-how-the-modules-connect)
6. [What Has Been Correctly Done](#6-what-has-been-correctly-done)
7. [Bugs & Problems](#7-bugs--problems)
8. [Missing Milestone 1 Deliverables](#8-missing-milestone-1-deliverables)
9. [M2 Hook Inventory](#9-m2-hook-inventory)

---

## 1. Project Goal & Deliverables

Milestone 1 is a **working baseline app** that:

| Deliverable | Where implemented |
|---|---|
| Preprocessing / downsampling | `loader.py` — `downsample_adaptive()` |
| Scene vs object classification | `alignment.py` — `ContentTypeClassifier` |
| Geometry-based alignment | `alignment.py` — `GeometryAxisEstimator` |
| Preview / apply / reset / manual tweak | `app.py` — all toolbar buttons |
| Axis gizmo in renderer | `app.py` — `Viewport._paint_axis_gizmo()` |

Success criteria from the spec:
- Indoor scenes align upright reliably
- Obvious objects get roughly aligned
- No meaningful FPS drop after load

---

## 2. Repository Layout

```
milestone1/
├── loader.py            # Layer 1 — PLY parse + voxel downsample + stats
├── alignment.py         # Layer 2 — outlier clean + classify + axis estimate
├── renderer.py          # Layer 3 — Vulkan offscreen renderer + software fallback
├── app.py               # Layer 4 — PyQt5 window, toolbar actions, gizmo
├── shaders/
│   ├── points.vert      # GLSL vertex shader (MVP + point size via push constant)
│   ├── points.frag      # GLSL fragment shader (round sprite discard)
│   ├── points.vert.spv  # compiled SPIR-V
│   └── points.frag.spv
├── tests/
│   ├── gen_synth_ply.py # generates indoor.ply + object.ply for testing
│   └── test_pipeline.py # end-to-end loader → alignment → renderer checks
├── data/
│   ├── indoor.ply       # synthetic tilted room (generated or real capture)
│   ├── object.ply       # synthetic tilted vase (generated or real capture)
│   └── big.ply          # larger real capture (not covered by tests)
├── Architecture Elements.md
├── Project Milestones.md
└── README.md
```

---

## 3. Data Flow — End to End

```
.ply file on disk
      │
      ▼
loader.load_point_cloud()
  ├── _read_ply_vertex()          parse XYZ + colours (RGB uint8 or 3DGS SH-DC)
  ├── downsample_adaptive()       voxel grid → binary-search refinement → random fallback
  └── compute_stats()             bbox, centroid, extent, scale
      │
      └─► PointCloud(positions, colors, stats, ...)
                │
                ▼
AlignmentEngine.align(positions)
  ├── OutlierCleaner()            k-NN mean-distance filter, drops statistical stragglers
  ├── ContentTypeClassifier()     RANSAC plane coverage → "indoor" | "object" + confidence
  └── GeometryAxisEstimator()
        ├── indoor path           reuses RANSAC plane normal from classifier, rotates to +Y
        └── object path           PCA smallest-variance axis, rotates to +Y
      │
      └─► AlignmentResult(transform 4×4, confidence, mode, ambiguity_flags, diagnostics)
                │
                ▼
VulkanRenderer / software fallback
  ├── set_points(positions, colors)      upload interleaved vertex buffer once
  ├── set_scene_root_transform(T)        stores 4×4 matrix, no re-upload needed
  └── render()                           MVP = P·V·T applied as push constant each frame
      │
      └─► (H,W,4) uint8 RGBA buffer
                │
                ▼
app.py  Viewport.paintEvent()
  ├── draws RGBA buffer via QImage
  ├── overlays axis gizmo (QPainter)     projected through view·sceneRoot, no perspective
  └── overlays HUD (device name, points, FPS)
```

---

## 4. Module Deep-Dives

### 4.1 `loader.py`

**Purpose:** CPU-only parsing and downsampling. Does not import torch so import time is minimal.

#### Key types

```
CloudStats  — bbox_min/max, centroid, extent, scale, point_count
PointCloud  — positions (N,3) float32, colors (N,3) float32, stats, extras
```

#### `_read_ply_vertex(path)` — PLY parsing

Handles three colour cases:
1. `red, green, blue` present → uint8 (0-255) or float (0-1) detected by `rgb.max() > 1.5`
2. `f_dc_0..2` present (3DGS) → `RGB = clip(0.5 + SH_C0 * f_dc, 0, 1)` using `SH_C0 = 0.28209...`
3. Neither → neutral grey `0.75`

Opacity (3DGS logit space) is sigmoid-transformed and stored in `extras["opacity"]`.

#### `voxel_downsample(pts, colors, voxel_size)`

1. Maps points to integer voxel indices `floor(pts / voxel_size)`
2. Encodes (i,j,k) into a single int64 using a shifted-span scheme (avoids negative-index issues)
3. `argsort` + `np.add.reduceat` to compute per-voxel mean position and colour — O(N log N)

#### `downsample_adaptive(pts, colors, cap=50_000)`

Two-pass binary search:
- **Grow pass**: doubles voxel size until output fits under `cap` (max 16 iterations)
- **Refine pass**: binary-searches downward to recover as much resolution as possible
- **Fallback**: `random_downsample` if voxel approach never converges

Initial voxel size is `diag / cap^(1/3)` — an analytic estimate for uniform density.

#### `compute_stats(pts)`

Trivial bbox, centroid (arithmetic mean), scale (longest extent). Called on the **decimated** cloud so downstream modules reason about exactly what they'll see.

---

### 4.2 `alignment.py`

**Purpose:** Geometry-only alignment pipeline. All PyTorch ops explicitly on `device='cpu'`.

#### `OutlierCleaner(k=16, std_ratio=2.0)`

- Fits `NearestNeighbors` (sklearn, `n_jobs=1`) on all points
- Computes mean k-NN distance per point
- Drops points where `mean_d > μ + 2σ`
- Falls back to identity mask if cleaner is too aggressive (< 16 pts survive)

#### `ContentTypeClassifier`

Decision is based on two signals combined into a soft score `[0,1]`:

**Primary — dominant plane coverage:**
- RANSAC on up to 8 000 subsampled points (speed cap), with 180 iterations
- Counts inliers on the **full** (un-subsampled) cloud for an honest ratio
- If `plane_ratio ≥ 0.22` → score ramps up toward 1.0 (saturates at 2× threshold)

**Backup — PCA flatness:**
- `flat_ratio = λ3 / λ2` (smallest / second eigenvalue of covariance)
- If `flat_ratio < 0.20` and `plane_ratio > 0.10` → score = 0.55 (nudge toward indoor for very flat clouds)

`mode = "indoor"` if `score ≥ 0.5`, else `"object"`.

Diagnostics dict passes the RANSAC plane normal and point through to `GeometryAxisEstimator` to avoid running RANSAC twice.

#### `_ransac_plane(points, iterations, inlier_thresh, rng_seed)`

Pure NumPy RANSAC:
1. Samples 3 random points, computes cross-product normal
2. Counts inliers within `inlier_thresh * extent` of the plane
3. After all iterations, **refits** best plane via eigendecomposition on all inliers (smallest eigenvector of covariance = refined normal)
4. Orientation disambiguated: normal points toward cloud centroid

#### `_rotation_align_vectors(src, dst)` — Rodrigues rotation

Builds a 3×3 rotation aligning unit vector `src` → `dst`:
- Near-parallel (d > 1-ε): returns identity
- Antiparallel (d < -1+ε): picks an orthogonal axis and uses `R = 2·a·aᵀ - I` (180° rotation via `I + 2·K·K` where K is the skew matrix of `a`)
- General case: standard Rodrigues formula `R = I + K + K²·(1-d)/(|v|²)`

#### `GeometryAxisEstimator`

**Indoor path (`_indoor_transform`):**
- Reads RANSAC plane normal directly from classifier diagnostics (no duplicate RANSAC)
- Falls back to fresh RANSAC if diagnostics are absent
- Orientation: flips normal so centroid sits on the positive side (above the floor)
- `R = _rotation_align_vectors(normal, world_up)` — maps floor normal to +Y
- `T[:3,3] = -R @ centroid` — re-centres cloud at world origin
- Confidence: `clip(0.4 + 1.5 * inlier_ratio, 0.25, 0.95)`
- Flag `"weak_ground_plane"` when inlier_ratio < 0.20

**Object path (`_object_transform`):**
- PCA via `torch.linalg.eigh` on CPU covariance matrix
- Smallest eigenvalue direction = `up_candidate`
- Up/down disambiguation: flip candidate so `median(projection) < 0` (more mass "below" the up direction)
- Anisotropy = `(λ3 - λ2) / λ3` between the two largest eigenvalues; if < 0.05 → `"yaw_ambiguous"` flag
- Confidence: `clip(0.3 + 0.6 * anisotropy, 0.2, 0.9)`

#### `AlignmentEngine` — glue

```
cleaned, _mask = cleaner(points)
mode, cls_conf, diag = classifier(cleaned)
T, est_conf, flags = estimator(cleaned, mode, diag)
confidence = sqrt(cls_conf * est_conf)   # geometric mean
```

#### UX static methods on `AlignmentEngine`

| Method | Operation |
|---|---|
| `rotate_about_up(T, deg)` | Premultiplies a +Y rotation: `R_y · T` |
| `flip_front_back(T)` | `rotate_about_up(T, 180)` |
| `keep_up_only(T)` | Strips yaw: finds cloud's current up via `R.T @ world_up`, maps it back to world up without introducing heading |

---

### 4.3 `renderer.py`

**Purpose:** Offscreen Vulkan point renderer with full NumPy software fallback.

#### Device selection priority

Score-based physical device picker:
```
integrated GPU  → 100  (preferred — iGPU-first constraint)
CPU (llvmpipe)  → 80
virtual GPU     → 60
other           → 40
discrete GPU    → 30  (last resort)
```

`prefer_integrated=False` promotes discrete to 150 (useful for testing).

#### Vulkan pipeline

Initialisation sequence (in `_init_vulkan`):
1. `VkInstance` (no extensions, no layers — headless offscreen only)
2. Physical device pick + queue family scan (graphics bit required)
3. Logical device with `largePoints = VK_TRUE` feature request
4. Shader modules from SPIR-V files in `shaders/`
5. Offscreen render targets: `VK_FORMAT_R8G8B8A8_UNORM` colour + `VK_FORMAT_D32_SFLOAT` depth
6. Host-visible readback buffer (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`)
7. Render pass → framebuffer
8. Pipeline: `VK_PRIMITIVE_TOPOLOGY_POINT_LIST`, no blending, depth test ≤, no multisampling
9. Push constant range: `mat4 mvp + float pointSize` = 68 bytes, vertex stage only
10. Command pool + single primary command buffer
11. Fence for CPU/GPU sync

#### Frame cycle (`_render_vulkan`)

```
if dirty: _upload_vertices()     # interleave pos/col into VBO, map/memcopy/unmap
_record_command_buffer()         # reset + begin + renderpass + bind + pushconstants + draw + blit to readback buf + end
vkQueueSubmit → vkWaitForFences  # 5-second timeout
vkMapMemory → np.frombuffer → .copy() → vkUnmapMemory
return (H,W,4) uint8
```

Vertex buffer grows with 2× slack to amortise reallocation.

#### Push constant layout

```python
push_bytes = mvp.T.tobytes() + struct.pack("f", self.point_size)
#            ↑ row-major → column-major for GLSL mat4
```

#### Software fallback (`_render_software`)

1. Transform all points to clip space via `MVP @ [pos | 1]ᵀ`
2. Cull behind-camera (`w ≤ 1e-4`) and out-of-depth (`z < 0` or `z > 1`)
3. Project NDC to pixel coords
4. Sort back-to-front (`argsort(-z)`) for painter's algorithm
5. Scatter RGB into image array — always 1 pixel per point regardless of `point_size`

#### `resize(w, h)`

Calls `_recreate_targets()` which rebuilds colour + depth images, readback buffer, framebuffer, and pipeline (pipeline has baked-in static viewport).

#### Teardown

`_teardown_vulkan_safe()` destroys all handles in reverse-creation order with individual try/except so partial-init failures don't cascade. No `__del__` — callers must call `close()` explicitly (documented tradeoff to avoid cffi/libvulkan race at interpreter shutdown on Linux).

---

### 4.4 `app.py`

**Purpose:** PyQt5 single-window GUI tying all three layers together.

#### `Viewport(QWidget)`

- Owns a `VulkanRenderer(900, 700)`
- `QTimer(33 ms)` drives ~30 Hz repaints via `_tick → update()`
- `paintEvent`: calls `renderer.render()` → `QImage` → `QPainter.drawImage` → overlay gizmo → overlay HUD
- Left-drag: orbit (`yaw += dx*0.4`, `pitch -= dy*0.3`, clamped ±85°)
- Wheel: zoom (`distance *= 0.9^steps`, floor 0.05)
- `resizeEvent`: forwards to `renderer.resize()`

#### Axis gizmo (`_paint_axis_gizmo`)

- Drawn in screen-space at bottom-right corner (90 px footprint, dark disc background)
- Projects world axes `[e₁, e₂, e₃] * 37.8px` through `view · sceneRoot` rotation only (no perspective) — keeps gizmo constant on-screen size
- Y-axis painter coordinates are flipped: `ey = cy - vec[1]`
- Colors: X=red, Y=green, Z=blue

#### HUD (`_paint_hud`)

Top-left black panel shows device name, device type, point count, and live FPS (rolling 1-second window).

#### `MainWindow`

| Action | Implementation |
|---|---|
| Open .ply | `load_point_cloud()` → `viewport.set_point_cloud()` → reset transform to identity |
| Auto Align | `engine.align(cloud.positions)` → `viewport.set_scene_transform(result.transform)` |
| Align Up Only | if no prior alignment → run Auto Align first; then `AlignmentEngine.keep_up_only(current_T)` |
| Rotate 90° | `AlignmentEngine.rotate_about_up(current_T, 90)` |
| Flip Front/Back | `AlignmentEngine.flip_front_back(current_T)` |
| Reset | `viewport.set_scene_transform(np.eye(4))` |
| Save Alignment | JSON sidecar next to .ply with `transform_4x4`, `auto_align` block, `source_ply`, counts |

`_last_alignment` holds the most recent `AlignmentResult` from Auto Align. Manual nudges do NOT update it — only the raw auto-align result is tracked.

`closeEvent` calls `viewport.close_renderer()` → `renderer.close()` → `_teardown_vulkan_safe()`.

Entry point calls `os._exit(rc)` instead of `sys.exit(rc)` to avoid cffi/libvulkan interpreter-shutdown race.

---

### 4.5 Shaders

#### `points.vert`

```glsl
layout(push_constant) uniform PushConstants {
    mat4 mvp;        // projection * view * sceneRoot
    float pointSize;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    gl_PointSize = pc.pointSize;
    vColor = inColor;
}
```

The scene-root transform is baked into `mvp` on the CPU (`P · V · T`), so the vertex shader needs no separate model matrix uniform. Alignment changes cost exactly one push-constant write.

#### `points.frag`

```glsl
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;   // unit-circle clip → round sprites
    outColor = vec4(vColor, 1.0);
}
```

`gl_PointCoord` ranges `[0,1]²` over the point quad. The discard condition `dot(d,d) > 0.25` = `|d| > 0.5` clips to a circle. Output is fully opaque.

---

### 4.6 Tests

#### `gen_synth_ply.py`

Generates two deterministic synthetic clouds:

**`indoor.ply`** (~10 900 pts):
- Floor: 5 000 pts, XZ ∈ [-2,2], Y ≈ 0 (σ = 0.005)
- 4 walls: 800 pts each at X = ±2, Z = ±2
- Ceiling: 1 500 pts at Y = 2.5
- 2 boxy fixtures: 600 pts each
- Applied rotation: 25° about X, then 18° about Z — the alignment engine must recover this

**`object.ply`** (~9 500 pts):
- 40 layers of 200 pts forming a sinusoidal-radius vase (height 1.4, max radius ~0.55)
- Base disk: 1 500 pts at Y ≈ 0, radius ≤ 0.30
- Applied rotation: 20° about X, 35° about Z

#### `test_pipeline.py`

Five checks per cloud:
1. **Loader**: `n > 0`, `n ≤ 50 000`, finite positions, `scale > 0`
2. **Alignment**: `T.shape == (4,4)`, finite, `R·Rᵀ ≈ I`, `det(R) ≈ 1`, correct mode, `confidence ∈ [0,1]`
3. **Up-axis accuracy**: applies T to cloud, measures angle between smallest PCA axis and +Y; tolerance 8° (indoor) / 10° (object)
4. **Renderer shape**: `render()` returns `(192,256,4)` uint8 with non-background pixels
5. **Scene-root responsiveness**: identity vs aligned transform produce different pixel buffers (`diff > 0`)

---

## 5. How the Modules Connect

```
app.py
  │  imports
  ├── loader.py           (PointCloud, load_point_cloud, DEFAULT_POINT_CAP)
  ├── alignment.py        (AlignmentEngine, AlignmentResult, UP_AXIS_WORLD)
  └── renderer.py         (VulkanRenderer)
           │
           └── shaders/points.{vert,frag}.spv  (loaded from disk at VkDevice creation)

tests/test_pipeline.py
  │  imports
  ├── loader.py
  ├── alignment.py
  ├── renderer.py
  └── tests/gen_synth_ply.py
```

**Transform ownership chain:**

```
AlignmentEngine.align()
    └─► AlignmentResult.transform   (4×4 float32)
            └─► Viewport.set_scene_transform()
                    └─► VulkanRenderer.set_scene_root_transform()
                            └─► renderer.scene_root   (stored on host)
                                    └─► mvp() = P · V · scene_root
                                            └─► pushed as VkPushConstant each frame
                                                    └─► points.vert gl_Position = mvp * pos
```

No vertex data is modified when the transform changes. The scene_root is just a matrix that gets included in MVP every frame — a single host-side matrix multiply + push constant write.

---

## 6. What Has Been Correctly Done

### Loader
- PLY parsing handles all three colour cases (uint8 RGB, 3DGS SH-DC, no colour) correctly
- SH_C0 constant is exactly right (`0.28209479177387814 = 1/(2√π)`)
- Opacity is sigmoid-decoded correctly (`1 / (1 + exp(-logit))`)
- Voxel downsampling avoids negative-index issues via the shift-then-span encoding
- `np.add.reduceat` pattern is correct and efficient for per-voxel aggregation
- Two-pass binary search (grow + refine) is well-designed; avoids the common problem of over-decimating
- Stats computed on the decimated cloud (consistent with what alignment and renderer see)

### Alignment
- `OutlierCleaner`: correct k-NN statistical filter; `+1` on k accounts for self-distance
- `_ransac_plane`: correct cross-product normal, correct relative→absolute threshold conversion, correct least-squares refit on inliers
- `_rotation_align_vectors`: correct Rodrigues formula including the tricky antiparallel case (`R = 2·a·aᵀ - I`)
- Re-use of RANSAC result from classifier in the estimator (avoids running RANSAC twice)
- Indoor path orients floor normal toward centroid — correctly identifies which side is "up"
- Re-centring via `T[:3,3] = -R @ centroid` is correct (translates after rotating)
- Geometric mean for confidence blending is appropriate for two probability-like numbers
- `keep_up_only` correctly strips yaw by extracting the cloud's current up vector from `R.T @ world_up`
- All UX static methods (`rotate_about_up`, `flip_front_back`, `keep_up_only`) are O(1) matrix ops

### Renderer
- iGPU preference is clearly implemented and documented
- Vulkan handles init failure gracefully; `is_vulkan` flag used everywhere
- Push constant layout is correct: `mvp.T.tobytes()` properly converts row-major numpy to column-major GLSL
- Vertex buffer growth with 2× slack correctly amortises reallocation
- Fragment shader round-sprite discard (`dot(d,d) > 0.25`) is correct
- Software fallback back-to-front sort is correct painter's algorithm
- `_teardown_vulkan_safe` destroys handles in correct reverse order with per-handle error handling
- `resize` correctly rebuilds pipeline (which has a baked-in static viewport)
- `os._exit` workaround for cffi shutdown race is the correct mitigation

### App
- Scene-root transform architecture (not baking into vertex data) is correctly implemented — transform changes are O(1)
- Gizmo projects through `view · sceneRoot` only (no perspective) — gives constant on-screen size
- Gizmo Y-axis flip (`ey = cy - vec[1]`) correctly handles screen vs world coordinate difference
- `_last_alignment` correctly tracks auto-align vs manual distinction in saved JSON
- Save alignment stores both `transform_4x4` (current, possibly manually adjusted) and `auto_align.transform` (original) — useful provenance
- FPS measurement is a rolling 1-second window, not frame-by-frame

### Tests
- Synthetic clouds are intentionally tilted so the alignment has real work to do
- Test correctly verifies orthonormality of the rotation block (`R·Rᵀ ≈ I`, `det(R) ≈ 1`)
- `check_renderer` verifies that a non-identity transform actually changes pixels — not just that `render()` runs
- `_ensure_synth()` generates test data on demand so tests are self-contained

---

## 7. Bugs & Problems

### Bug 1 — Object up-axis heuristic wrong for tall thin objects (alignment.py:401)

**Severity: High**

```python
# _object_transform, line ~401
up_candidate = eigvecs_np[:, 0]   # smallest-variance PCA direction
```

The assumption "smallest-variance axis = up" holds only when the object is **wider than it is tall** (a flat disc, a table, a bowl). For any elongated object (a bottle, a person, a pillar, a statue), the principal axis of greatest variance is the **height axis** — so the smallest-variance direction is a **horizontal axis**, and this code would map a horizontal direction to world up, aligning the object on its side.

The test passes because the synthetic vase has a large flat base disk (1 500 pts) that adds disproportionate XZ variance and partially compensates. Real elongated captures will fail.

**Correct fix:** For "object" mode, the largest-variance axis (eigvecs_np[:, 2]) should be the candidate up axis when the object is taller than it is wide.

---

### Bug 2 — Opacity always dropped after real downsampling (loader.py:341)

**Severity: Low (feature gap)**

```python
# load_point_cloud, line ~341
if "opacity" in extras and extras["opacity"].shape[0] == pos.shape[0]:
    kept_extras["opacity"] = extras["opacity"]
```

After voxel downsampling, `pos.shape[0]` is the number of occupied voxels — never equal to `extras["opacity"].shape[0]` which is the raw point count. So opacity is silently discarded for every cloud that gets voxel-decimated (i.e., every cloud > 50 000 raw points). The code comment acknowledges this (`"voxel-bucketing scrambled per-point identity"`), but it means the renderer never receives opacity data for real 3DGS captures.

---

### Bug 3 — ContentTypeClassifier confidence formula is misleading (alignment.py:185)

**Severity: Low (code quality)**

```python
confidence = float(np.clip(0.5 + (score - 0.5), 0.10, 0.95)) \
    if mode == "indoor" else \
    float(np.clip(0.5 + (0.5 - score), 0.10, 0.95))
```

For `mode == "indoor"`: `0.5 + (score - 0.5) = score`. The formula is just `clip(score, 0.10, 0.95)` — no margin, no boundary calculation.

For `mode == "object"`: `0.5 + (0.5 - score) = 1.0 - score`. Confidence is `1 - score`. Since `score` is high for indoor and low for object, this gives high confidence for strongly-classified objects. The formulas are accidentally correct but the intermediate expression `0.5 + (...)` obscures this completely.

---

### Bug 4 — Software renderer ignores point size (renderer.py:288)

**Severity: Low**

```python
# _render_software, lines ~288-306
img[y, x, 0] = (rgb[:, 0] * 255).astype(np.uint8)  # always 1 pixel per point
```

The Vulkan path respects `self.point_size` via `gl_PointSize`. The software fallback always writes exactly 1 pixel per point. At default `point_size=2.0` this produces a visually different result — software renders look sparser than Vulkan renders of the same cloud.

---

### Bug 5 — `_angle_to_up` test metric is wrong for indoor scenes (tests/test_pipeline.py:54)

**Severity: Medium (test validity)**

```python
def _angle_to_up(points: np.ndarray) -> float:
    """Angle (degrees) between the cloud's smallest-PCA axis and +Y."""
    _, V = np.linalg.eigh(cov)
    smallest = V[:, 0]
    return float(np.degrees(np.arccos(min(1.0, abs(smallest[1])))))
```

For indoor scenes, the alignment target is that the **RANSAC floor normal maps to +Y**. The smallest PCA axis of an indoor scene is NOT necessarily the floor normal — in a room with equal-density floor, ceiling, and walls, the PCA structure is complex. The test is asserting the wrong thing for the indoor case. It passes because the synthetic room has a very dense floor relative to walls, making the smallest PCA axis coincidentally close to the floor normal.

---

### Bug 6 — `action_align_up_only` may act on stale transform (app.py:406)

**Severity: Low**

```python
def action_align_up_only(self) -> None:
    if self._last_alignment is None:
        self.action_auto_align()
    current = self.viewport.scene_transform()    # reads AFTER auto_align
    new = AlignmentEngine.keep_up_only(current)
```

If `_last_alignment` is None, `action_auto_align()` runs and sets a new transform. Then `current` correctly picks up that transform. However `_last_alignment` is set inside `action_auto_align()` but `action_align_up_only` doesn't update it after calling `keep_up_only`. If the user clicks "Align Up Only" again, `_last_alignment` is not None (from the prior auto-align), so `action_auto_align()` is skipped, and `keep_up_only` is applied to the already-up-only transform — which is a no-op. This is technically correct but the flow is fragile and undocumented.

---

### Bug 7 — `keep_up_only` translation component is inherited unchanged (alignment.py:518)

**Severity: Low**

```python
def keep_up_only(transform: np.ndarray) -> np.ndarray:
    out[:3, :3] = R_up
    out[:3, 3] = transform[:3, 3]    # copied verbatim
```

The translation `t = -R @ centroid` was computed for the original `R`. After replacing `R` with `R_up` (yaw-stripped rotation), the translation should be `-R_up @ centroid`. Since `centroid` is not available here, using the original translation is an approximation — it will be wrong whenever the yaw-stripped `R_up` differs significantly from the original `R`, placing the cloud off-centre.

---

### Bug 8 — `_recreate_targets` does not null readback buffer handles before recreate (renderer.py:803)

**Severity: Very Low (code fragility)**

```python
def _recreate_targets(self) -> None:
    # destroys old readback buf/mem but does not set to None
    vk.vkDestroyBuffer(self._device, self._readback_buf, None)
    vk.vkFreeMemory(self._device, self._readback_mem, None)
    self._create_targets()   # overwrites self._readback_buf / _readback_mem
```

If `_create_targets()` throws partway through, `self._readback_buf` still holds the now-destroyed handle. A subsequent `_teardown_vulkan_safe` would attempt to destroy it again. Not an immediate crash (Vulkan spec allows double-destroy to be a validation error, not UB), but fragile.

---

### Issue 9 — `big.ply` not covered by tests

The `data/big.ply` file exists in the repo but the test suite only exercises `indoor.ply` and `object.ply`. Large real captures may expose downsampling edge cases or RANSAC instability at scale.

---

### Issue 10 — No guard against `torch` import failure in alignment.py

`alignment.py` imports `torch` unconditionally at the top. If PyTorch is not installed, `loader.py` still works fine (it has no torch dependency) but importing `alignment` fails. Since `app.py` imports all three at startup, a missing PyTorch install produces a confusing error at launch rather than a clear diagnostic.

---

## 8. Missing Milestone 1 Deliverables

| Deliverable | Status | Notes |
|---|---|---|
| Preprocessing / downsampling | ✅ Done | voxel + random fallback, 50 k cap |
| Scene vs object classification | ✅ Done | RANSAC plane coverage |
| Geometry-based alignment | ✅ Done | indoor RANSAC / object PCA |
| Preview / apply / reset / manual tweak | ✅ Done | all 5 buttons |
| Axis gizmo in renderer | ✅ Done | overlaid in paintEvent |
| Indoor scenes align upright reliably | ✅ Done for typical scenes | fragile for edge cases |
| Obvious objects roughly aligned | ⚠️ Partially | fails for tall thin objects (Bug 1) |
| No meaningful FPS drop after load | ✅ Done | 50 k cap + push-constant transform |
| 3DGS splat rendering (ellipsoids) | ❌ Not done | scale_*, rot_* parsed but ignored; only point cloud rendered |
| Opacity used in rendering | ❌ Not done | decoded but silently dropped (Bug 2) |

---

## 9. M2 Hook Inventory

Three `[M2 HOOK]` markers are placed at the exact integration points for the Milestone 2 ML module:

| File | Location | What goes there |
|---|---|---|
| `alignment.py:189` | `ContentTypeClassifier.__call__` | Blend a learnt classifier logit with `score` — weighted average of geometry + ML class confidence |
| `alignment.py:383` | `GeometryAxisEstimator._indoor_transform` | Weighted Slerp between RANSAC floor normal and ML-predicted up vector, weights ∝ each method's confidence |
| `alignment.py:425` | `GeometryAxisEstimator._object_transform` | Replace PCA `up_candidate` with ML-predicted up vector when ML confidence exceeds PCA anisotropy margin |

The `AlignmentFusion` and `ConfidenceScorer` submodules from the Architecture Elements doc are not yet implemented — they are the planned M2 additions.
