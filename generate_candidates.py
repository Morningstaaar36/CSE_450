"""
CanoVerse Stage-I candidate generator + render exporter.

Faithful to the paper (arXiv:2603.07144v1, Section 3.1):
  - ONE canonical template per category (not a pool).
  - 5 candidates in the paper's exact composition/order:
       R_S        : horizontal semantic          (Eq. 4)
       R_G        : horizontal geometric          (Eq. 3)
       R_invG     : R_G flipped 180 deg           (Eq. 3)
       R_S . R_Sup: support-surface vertical, then horizontal semantic
       R_S . R_Pca: PCA vertical, then horizontal semantic
  - Quality gate mirroring the paper's 51%/6% discards (thin-shell /
    degenerate meshes) so you don't waste annotation clicks on junk.

Approximations still required until the supplementary material is out
(clearly flagged in code):
  * R_S semantic term (Eq. 4) uses a height-band centroid signature as a
    stand-in for Find3D part segmentation, BUT now correctly modulates it
    with the geometric-extrema Gaussian, matching Eq. 4's structure.
  * R_Sup uses largest-convex-hull-face-down instead of the full
    center-of-mass-in-support-polygon enumeration from Fu et al. [9].
  * Template is one CANONICAL mesh you designate per category (via its
    ground-truth rotation), standing in for the paper's hand-built
    human-convention template.

Output: renders/<category>/<uid>/cand_{0..4}.png  + template.png
        renders/<category>/manifest.json  (consumed by annotate.html)

Usage:
    pip install objaverse trimesh numpy scipy pyrender pillow
    python generate_candidates.py
"""

import glob
import json
import os
import random

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from PIL import Image  # noqa: E402

# Defensively clear any stale headless-GL setting inherited from the shell,
# so importing pyrender never tries EGL/OSMesa on a desktop machine.
os.environ.pop("PYOPENGL_PLATFORM", None)

# Rendering backend. Default is 'matplotlib': pure CPU, no OpenGL / pyglet /
# EGL, so it always works on Windows. It flat-shades projected triangles —
# not pretty, but perfectly clear enough to judge orientation.
#   RENDER_BACKEND=pyrender  -> nicer shading, needs working GL + pyglet<2
#   RENDER_BACKEND=trimesh   -> trimesh's pyglet viewer (window may flash)
RENDER_BACKEND = os.environ.get("RENDER_BACKEND", "matplotlib")
USE_PYRENDER = RENDER_BACKEND == "pyrender"
if USE_PYRENDER:
    import pyrender  # noqa: E402

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

ANNO_GLOB = "canoverse_anno/canoverse_poses_part00*.json"
CATEGORY = "table"

N_OBJECTS = 20          # how many objects to prepare for annotation
N_POINTS = 2048
THETA_STEPS = 180       # 2-degree horizontal resolution
RENDER_SIZE = 320
RANDOM_SEED = 0

OUT_ROOT = "renders"

# Quality gate thresholds (approximate the paper's discard criteria)
MIN_THICKNESS_RATIO = 0.02   # min(extent)/max(extent): below = thin-shell/flat, discard
MIN_FACES = 50               # degenerate / meaningless structure guard


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def normalize_to_unit_sphere(mesh):
    """Paper: normalize translation & scale to a UNIT SPHERE (not bbox)."""
    mesh.apply_translation(-mesh.centroid)
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 0:
        mesh.apply_scale(1.0 / radius)
    return mesh


def sample_points(mesh, n=N_POINTS, seed=0):
    st = np.random.get_state()
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    np.random.set_state(st)
    return np.asarray(pts)


def chamfer_distance(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, _ = tb.query(a)
    d_ba, _ = ta.query(b)
    return d_ab.mean() + d_ba.mean()


def height_band_signature(points, n_bands=6):
    """Proxy for semantic part sets S_k: horizontal centroid per height band."""
    order = np.argsort(points[:, 1])
    bands = np.array_split(points[order], n_bands)
    return [b[:, [0, 2]].mean(axis=0) for b in bands if len(b) > 0]


def rotation_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


# ----------------------------------------------------------------------
# Candidate criteria (paper equations)
# ----------------------------------------------------------------------

def geometric_horizontal(obj_pts, tpl_pts, steps=THETA_STEPS):
    """Eq. 3: theta* = argmin CD(P^t, R_z(theta) P^o). Returns theta*, Eg curve."""
    thetas = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    Eg = np.array([chamfer_distance(tpl_pts, obj_pts @ rot_y(t).T) for t in thetas])
    return thetas[int(np.argmin(Eg))], thetas, Eg


def semantic_horizontal(obj_pts, tpl_pts, steps=THETA_STEPS):
    """Eq. 4: favour angles that are semantically aligned AND near geometric
    extrema of Eg. theta* = argmin -(e^{-Es} * sum_k N(theta | w_k, I))."""
    theta_g, thetas, Eg = geometric_horizontal(obj_pts, tpl_pts, steps)

    # geometric extrema omega_k = local minima of Eg (candidate "good" angles)
    omega = []
    for i in range(len(Eg)):
        if Eg[i] <= Eg[(i - 1) % len(Eg)] and Eg[i] <= Eg[(i + 1) % len(Eg)]:
            omega.append(thetas[i])
    if not omega:
        omega = [theta_g]

    tpl_sig = height_band_signature(tpl_pts)
    best_theta, best_obj = theta_g, -np.inf
    for t in thetas:
        obj_sig = height_band_signature(obj_pts @ rot_y(t).T)
        m = min(len(obj_sig), len(tpl_sig))
        Es = np.mean([np.linalg.norm(obj_sig[k] - tpl_sig[k]) for k in range(m)]) if m else 1.0
        # Gaussian mixture over geometric extrema (I = identity cov, sigma=1)
        gauss = sum(np.exp(-0.5 * (min(abs(t - w), 2 * np.pi - abs(t - w))) ** 2) for w in omega)
        objective = np.exp(-Es) * gauss   # we maximise this (== minimise its negative)
        if objective > best_obj:
            best_obj, best_theta = objective, t
    return best_theta


def pca_axes(points):
    c = points - points.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    v = v[:, np.argsort(w)[::-1]]
    return v[:, 0], v[:, 1]


def pca_vertical(obj_pts, tpl_pts):
    """Eq. 1-2: 4 polarity candidates, pick the one minimising semantic CD."""
    v1o, v2o = pca_axes(obj_pts)
    v1t, v2t = pca_axes(tpl_pts)
    Mt = np.stack([v1t, v2t, np.cross(v1t, v2t)], axis=1)
    tpl_sig = height_band_signature(tpl_pts)
    best_R, best_cost = np.eye(3), np.inf
    for s1 in (1, -1):
        for s2 in (1, -1):
            u1, u2 = s1 * v1o, s2 * v2o
            Mo = np.stack([u1, u2, np.cross(u1, u2)], axis=1)
            R = Mt @ Mo.T
            sig = height_band_signature(obj_pts @ R.T)
            m = min(len(sig), len(tpl_sig))
            cost = np.mean([np.linalg.norm(sig[k] - tpl_sig[k]) for k in range(m)]) if m else np.inf
            if cost < best_cost:
                best_cost, best_R = cost, R
    return best_R


def support_vertical(mesh):
    """Approx R_Sup: largest convex-hull face becomes the ground contact."""
    hull = mesh.convex_hull
    n = hull.face_normals[int(np.argmax(hull.area_faces))]
    return rotation_from_vectors(n, np.array([0.0, -1.0, 0.0]))


def generate_five_candidates(mesh, tpl_pts):
    obj_pts = sample_points(mesh)

    theta_g, _, _ = geometric_horizontal(obj_pts, tpl_pts)
    R_G = rot_y(theta_g)
    R_invG = rot_y(theta_g + np.pi)

    theta_s = semantic_horizontal(obj_pts, tpl_pts)
    R_S = rot_y(theta_s)

    R_sup = support_vertical(mesh)
    theta_s_sup = semantic_horizontal(obj_pts @ R_sup.T, tpl_pts)
    R_S_Sup = rot_y(theta_s_sup) @ R_sup

    R_pca = pca_vertical(obj_pts, tpl_pts)
    theta_s_pca = semantic_horizontal(obj_pts @ R_pca.T, tpl_pts)
    R_S_Pca = rot_y(theta_s_pca) @ R_pca

    # order matches the paper's candidate list
    return [
        ("R_S", R_S),
        ("R_G", R_G),
        ("R_invG", R_invG),
        ("R_S.R_Sup", R_S_Sup),
        ("R_S.R_Pca", R_S_Pca),
    ]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _render_matplotlib(m, out_path):
    """Pure-CPU flat-shaded render. No OpenGL/pyglet/EGL. Always works.

    Our meshes use the paper's canonical frame: y = up, z = front.
    matplotlib's 3D axes are z-up, so we remap (x, y, z)_mesh -> (x, z, y)_plot
    to display an upright object; otherwise canonical poses look tipped over.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, no window
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = m.vertices.copy()
    # axis remap: mesh y (up) -> plot z (up); mesh z -> plot y
    verts = verts[:, [0, 2, 1]]
    faces = m.faces
    tris = verts[faces]  # (F, 3, 3)

    # simple lambert shading from a fixed light direction
    light = np.array([0.3, 0.75, 0.6])
    light = light / np.linalg.norm(light)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals = normals / nlen
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0) * 0.7 + 0.3
    base = np.array([0.55, 0.60, 0.68])
    facecolors = np.clip(shade[:, None] * base[None, :], 0, 1)

    fig = plt.figure(figsize=(RENDER_SIZE / 100, RENDER_SIZE / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
    ax.add_collection3d(coll)

    lim = np.abs(verts).max() * 0.7
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    # 3/4 view looking slightly down; plot-z is now mesh-up
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, transparent=False, facecolor="white")
    plt.close(fig)


_renderer = None

def get_renderer():
    global _renderer
    if _renderer is None:
        _renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)
    return _renderer


def _render_pyrender(m, out_path):
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0.35],
                         [0, 0, 1, 2.6], [0, 0, 0, 1]])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=cam_pose)
    color, _ = get_renderer().render(scene)
    Image.fromarray(color).save(out_path)


def _render_trimesh(m, out_path):
    """trimesh pyglet backend (a window may flash open on Windows)."""
    import trimesh.transformations as tf
    scene = m.scene()
    cam = scene.camera_transform
    rot = tf.rotation_matrix(np.radians(30), [0, 1, 0]) @ tf.rotation_matrix(np.radians(20), [1, 0, 0])
    scene.camera_transform = cam @ rot
    png = scene.save_image(resolution=(RENDER_SIZE, RENDER_SIZE), visible=True)
    with open(out_path, "wb") as f:
        f.write(png)


def render_mesh_png(mesh, R, out_path):
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    if RENDER_BACKEND == "pyrender":
        _render_pyrender(m, out_path)
    elif RENDER_BACKEND == "trimesh":
        _render_trimesh(m, out_path)
    else:
        _render_matplotlib(m, out_path)


# ----------------------------------------------------------------------
# Quality gate (approximate paper's 51%/6% discard)
# ----------------------------------------------------------------------

def passes_quality(mesh):
    if len(mesh.faces) < MIN_FACES:
        return False, "too few faces (meaningless structure)"
    ext = mesh.bounding_box.extents
    if ext.max() <= 0 or (ext.min() / ext.max()) < MIN_THICKNESS_RATIO:
        return False, "thin-shell / near-planar mesh"
    return True, "ok"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_category(category):
    anno = {}
    for p in glob.glob(ANNO_GLOB):
        with open(p) as f:
            anno.update(json.load(f))
    return {u: v for u, v in anno.items() if v["category"] == category}


def download(uids):
    import objaverse
    objs = objaverse.load_objects(uids=uids, download_processes=4)
    return objs


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    items = load_category(CATEGORY)
    uids = list(items.keys())
    random.shuffle(uids)
    print(f"{len(uids)} '{CATEGORY}' objects available.")

    out_dir = os.path.join(OUT_ROOT, CATEGORY)
    os.makedirs(out_dir, exist_ok=True)

    # --- template: one canonical reference for the category ---
    template_uid = uids[0]
    objs = download([template_uid])
    tpl_mesh = normalize_to_unit_sphere(trimesh.load(objs[template_uid], force="mesh"))
    gt_R = np.array(items[template_uid]["rotation_matrix"])
    T = np.eye(4)
    T[:3, :3] = gt_R  # canonicalize template (consistent with pts @ R.T search convention)
    tpl_mesh.apply_transform(T)
    tpl_pts = sample_points(tpl_mesh)
    render_mesh_png(tpl_mesh, np.eye(3), os.path.join(out_dir, "template.png"))
    print(f"Template = {template_uid}")

    # --- prepare N objects for annotation ---
    manifest = {"category": CATEGORY, "template": "template.png", "objects": []}
    prepared, discarded = 0, 0

    for uid in uids[1:]:
        if prepared >= N_OBJECTS:
            break
        path = objs.get(uid) or download([uid]).get(uid)
        if path is None:
            continue
        try:
            mesh = normalize_to_unit_sphere(trimesh.load(path, force="mesh"))
        except Exception:
            continue

        ok, reason = passes_quality(mesh)
        if not ok:
            discarded += 1
            print(f"  discard {uid[:12]}: {reason}")
            continue

        obj_dir = os.path.join(out_dir, uid)
        os.makedirs(obj_dir, exist_ok=True)

        candidates = generate_five_candidates(mesh, tpl_pts)
        cand_records = []
        for i, (name, R) in enumerate(candidates):
            fn = f"{uid}/cand_{i}.png"
            render_mesh_png(mesh, R, os.path.join(out_dir, fn))
            cand_records.append({
                "index": i,
                "name": name,
                "image": fn,
                "rotation": R.tolist(),
            })

        manifest["objects"].append({
            "uid": uid,
            "candidates": cand_records,
            "ground_truth": items[uid]["rotation_matrix"],  # for later scoring only
        })
        prepared += 1
        print(f"  prepared {uid[:12]} ({prepared}/{N_OBJECTS})")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Prepared {prepared}, discarded {discarded}.")
    print(f"Manifest: {os.path.join(out_dir, 'manifest.json')}")
    print("Next: open annotate.html and load that manifest.json + renders folder.")


if __name__ == "__main__":
    main()