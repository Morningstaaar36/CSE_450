"""
CanoVerse pipeline: pick a small subset of UIDs, download meshes from
Objaverse, normalize + apply the annotated canonical rotation, export
as .obj, and render a preview grid.

Edit the CONFIG section below before running.
"""

import os
import glob
import json
import random

import numpy as np
import trimesh

# ----------------------------------------------------------------------
# CONFIG - edit these paths/values for your machine
# ----------------------------------------------------------------------

ANNO_GLOB = "canoverse_anno/canoverse_poses_part00*.json"  # annotation files
N_SUBSET = 30                                               # how many objects to pull
RANDOM_SEED = 0

# Redirect big downloads / outputs to a drive with more space if needed, e.g.:
# OBJAVERSE_CACHE = "E:/objaverse_cache"
# OUTPUT_DIR = "E:/canoverse_output"
OBJAVERSE_CACHE = None          # None = use objaverse's default cache location
OUTPUT_DIR = "canonicalized"    # where .obj files + preview image go

DOWNLOAD_PROCESSES = 4
RENDER_SIZE = 256               # pixel size of each preview thumbnail
GRID_COLS = 6                   # columns in the preview grid


# ----------------------------------------------------------------------
# STEP 1: Load annotations, pick a subset of UIDs
# ----------------------------------------------------------------------

def load_annotations():
    anno = {}
    paths = glob.glob(ANNO_GLOB)
    if not paths:
        raise FileNotFoundError(
            f"No annotation files matched '{ANNO_GLOB}'. "
            "Run this script from inside the cloned Canoverse folder, "
            "or fix ANNO_GLOB above."
        )
    for path in paths:
        with open(path, "r") as f:
            anno.update(json.load(f))
    print(f"Loaded {len(anno)} annotated objects from {len(paths)} file(s).")
    return anno


def pick_subset(anno, n, seed):
    random.seed(seed)
    uids = list(anno.keys())
    n = min(n, len(uids))
    subset = random.sample(uids, n)
    print(f"Selected {len(subset)} UIDs for this run.")
    return subset


# ----------------------------------------------------------------------
# STEP 2: Download meshes for just that subset
# ----------------------------------------------------------------------

def download_subset(uids):
    import objaverse  # imported here so CONFIG can set BASE_PATH first

    if OBJAVERSE_CACHE:
        os.makedirs(OBJAVERSE_CACHE, exist_ok=True)
        objaverse.BASE_PATH = OBJAVERSE_CACHE

    print(f"Downloading {len(uids)} objects from Objaverse ...")
    objects = objaverse.load_objects(uids=uids, download_processes=DOWNLOAD_PROCESSES)

    missing = [uid for uid in uids if uid not in objects or objects[uid] is None]
    if missing:
        print(f"Warning: {len(missing)} UID(s) failed to download and will be skipped.")

    return objects


# ----------------------------------------------------------------------
# STEPS 4-5: Normalize (center + unit scale) and apply canonical rotation
# ----------------------------------------------------------------------

def normalize_mesh(mesh):
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    extents = mesh.bounding_box.extents
    max_extent = np.max(extents)
    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)
    return mesh


def apply_rotation(mesh, rotation_matrix):
    R = np.array(rotation_matrix)
    T = np.eye(4)
    T[:3, :3] = R
    mesh.apply_transform(T)
    return mesh


def canonicalize_subset(objects, anno, uids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    canon_meshes = {}

    for uid in uids:
        src_path = objects.get(uid)
        if src_path is None:
            continue
        try:
            mesh = trimesh.load(src_path, force="mesh")
        except Exception as e:
            print(f"  Failed to load {uid}: {e}")
            continue

        mesh = normalize_mesh(mesh)
        mesh = apply_rotation(mesh, anno[uid]["rotation_matrix"])

        out_path = os.path.join(OUTPUT_DIR, f"{uid}.obj")
        mesh.export(out_path)
        canon_meshes[uid] = mesh
        print(f"  {uid}  [{anno[uid].get('category', '?')}]  -> {out_path}")

    print(f"Canonicalized {len(canon_meshes)}/{len(uids)} meshes.")
    return canon_meshes


# ----------------------------------------------------------------------
# STEP 6: Visualize a preview grid
# ----------------------------------------------------------------------

def render_mesh_offscreen(mesh, size=256):
    import pyrender

    scene = pyrender.Scene(bg_color=[255, 255, 255, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    cam_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.3],
        [0, 0, 1, 2.5],
        [0, 0, 0, 1],
    ])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    color, _ = renderer.render(scene)
    renderer.delete()
    return color


def save_preview_grid(canon_meshes, anno):
    import matplotlib.pyplot as plt

    if not canon_meshes:
        print("Nothing to visualize.")
        return

    n = len(canon_meshes)
    cols = min(GRID_COLS, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)  # flatten regardless of shape

    for ax, (uid, mesh) in zip(axes, canon_meshes.items()):
        try:
            img = render_mesh_offscreen(mesh, size=RENDER_SIZE)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "render failed", ha="center", va="center")
            print(f"  Render failed for {uid}: {e}")
        ax.set_title(anno[uid].get("category", uid[:8]), fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "preview_grid.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved preview grid -> {out_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    anno = load_annotations()
    uids = pick_subset(anno, N_SUBSET, RANDOM_SEED)
    objects = download_subset(uids)
    canon_meshes = canonicalize_subset(objects, anno, uids)
    save_preview_grid(canon_meshes, anno)


if __name__ == "__main__":
    main()