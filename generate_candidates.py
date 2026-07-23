"""
=============================================================================
  CanoVerse VNN -- Single-File Kaggle Bundle
=============================================================================

Complete self-contained pipeline for training and using a Vector Neuron
Network (VNN) for automatic canonical orientation selection of 3D objects.

This file merges:
  - vnn_model.py       (VNN architecture)
  - generate_candidates.py  (candidate rotation generation + rendering)
  - train_vnn.py        (self-supervised training pipeline)

FULLY AUTOMATED -- NO human labels needed!
Training uses Chamfer distance to the category template as self-supervised
ground truth.

Usage in Kaggle notebook:
  1. Paste this entire file into a cell
  2. Adjust the CONFIG section below (start with N_SAMPLES=5 to test)
  3. Call run_all() to train + generate for all categories

Or run individual steps:
  - run_pipeline("train")      # Train VNNs for all categories
  - run_pipeline("generate")   # Generate candidates using trained VNNs
  - run_pipeline("train", categories=["table"])  # Train one category
"""

# =====================================================================
# EASY-TUNE CONFIG -- Change these to control the pipeline!
# =====================================================================

# Categories to process
CATEGORIES = ["table", "chair", "car", "airplane"]

# Objects PER CATEGORY (set to 5 for quick smoke test, 30 for medium, 40 for full)
N_SAMPLES = 30

# Objects to generate candidates for (after training, per category)
N_GENERATE = 20

# Point cloud settings
N_POINTS = 2048          # Points sampled per mesh surface

# Training hyperparameters
BATCH_SIZE = 16          # Training batch size
NUM_EPOCHS = 50          # Max training epochs (early stopping will kick in)
LEARNING_RATE = 1e-3     # Adam learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
PATIENCE = 10            # Early stopping patience (epochs without improvement)

# VNN architecture
FEAT_DIMS = (32, 64, 128)  # VN encoder channel dimensions
HIDDEN_DIM = 256            # MLP head hidden size

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Candidate generation
THETA_STEPS = 180        # Horizontal rotation search resolution
RENDER_SIZE = 320        # Render image dimensions (pixels, square)

# Quality gate
MIN_THICKNESS_RATIO = 0.02
MIN_FACES = 50

# Paths
OUT_ROOT = "renders"
CACHE_DIR = "data_cache"
CHECKPOINT_DIR = "checkpoints"

# Seeds
RANDOM_SEED = 42

# =====================================================================
# END CONFIG
# =====================================================================

import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
!pip -q install trimesh objaverse
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.pop("PYOPENGL_PLATFORM", None)

# Per-category strict Objaverse keyword filtering
CATEGORY_FILTERS = {
    "table": {
        "include": ["table", "desk"],
        "exclude": [
            "periodic", "pool", "billiard", "turntable", "timetable",
            "printable", "stable", "notable", "portable", "vegetable",
            "comfortable", "charitable", "suitable", "tablet",
        ],
    },
    "chair": {
        "include": ["chair"],
        "exclude": [
            "wheelchair", "chairlift", "chairman", "stairchair",
        ],
    },
    "car": {
        "include": ["car"],
        "exclude": [
            "card", "cart", "carpet", "cargo", "carrot", "care",
            "cartoon", "cardinal", "carbon", "carnival", "oscar",
            "scar", "boxcar", "carriage", "carousel", "carrier",
            "card game", "playing card", "postcard", "carton",
        ],
    },
    "airplane": {
        "include": ["airplane", "aeroplane", "aircraft", "jet", "plane"],
        "exclude": [
            "jetski", "planet", "planer", "plank", "plant",
            "plantation", "planetarium",
        ],
    },
}


def get_cache_file(category):
    return os.path.join(CACHE_DIR, f"{category}_vnn_data.pt")


def get_best_model_path(category):
    return os.path.join(CHECKPOINT_DIR, f"vnn_{category}_best.pt")


# =====================================================================
# VNN MODEL -- SO(3)-Equivariant Architecture
# =====================================================================

class VNLinear(nn.Module):
    """SO(3)-equivariant linear layer for vector features.
    Maps (B, C_in, 3) -> (B, C_out, 3).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels) * (2.0 / in_channels) ** 0.5
        )

    def forward(self, x):
        return torch.einsum("oi,bid->bod", self.weight, x)


class VNLeakyReLU(nn.Module):
    """SO(3)-equivariant LeakyReLU for vector features."""
    def __init__(self, in_channels, negative_slope=0.2, share_nonlinearity=False):
        super().__init__()
        self.negative_slope = negative_slope
        if share_nonlinearity:
            self.direction = VNLinear(in_channels, 1)
        else:
            self.direction = VNLinear(in_channels, in_channels)
        self.share = share_nonlinearity

    def forward(self, x):
        q = self.direction(x)
        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        if self.share:
            q_norm = q_norm.expand_as(x)
        proj = (x * q_norm).sum(dim=-1, keepdim=True)
        mask = (proj >= 0).float()
        x_out = mask * x + (1 - mask) * (
            x - (1.0 - self.negative_slope) * proj * q_norm
        )
        return x_out


class VNMaxPool(nn.Module):
    """SO(3)-equivariant max pooling over the point dimension."""
    def forward(self, x):
        norms = torch.norm(x, dim=-1)
        max_indices = norms.argmax(dim=1)
        B, N, C, _ = x.shape
        idx = max_indices.unsqueeze(1).unsqueeze(-1).expand(B, 1, C, 3)
        pooled = x.gather(dim=1, index=idx).squeeze(1)
        return pooled


class VNStdFeature(nn.Module):
    """Extract rotation-INVARIANT scalar features from equivariant vectors."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = in_channels * (in_channels + 1) // 2
        idx = torch.triu_indices(in_channels, in_channels)
        self.register_buffer("tri_row", idx[0])
        self.register_buffer("tri_col", idx[1])

    def forward(self, x):
        gram = torch.bmm(x, x.transpose(1, 2))
        invariant = gram[:, self.tri_row, self.tri_col]
        return invariant


class VNPointNetEncoder(nn.Module):
    """VN-PointNet encoder: lifts raw 3D points to equivariant vector features."""
    def __init__(self, n_points=2048, feat_dims=(64, 128, 256)):
        super().__init__()
        self.n_points = n_points
        layers = []
        in_ch = 1
        for out_ch in feat_dims:
            layers.append(VNLinear(in_ch, out_ch))
            layers.append(VNLeakyReLU(out_ch, negative_slope=0.2))
            in_ch = out_ch
        self.vn_layers = nn.ModuleList(layers)
        self.pool = VNMaxPool()
        self.invariant = VNStdFeature(feat_dims[-1])

    def forward(self, points):
        B, N, _ = points.shape
        x = points.unsqueeze(2)
        for layer in self.vn_layers:
            if isinstance(layer, VNLinear):
                B_N = B * N
                x_flat = x.reshape(B_N, -1, 3)
                x_flat = layer(x_flat)
                C_out = x_flat.shape[1]
                x = x_flat.reshape(B, N, C_out, 3)
            elif isinstance(layer, VNLeakyReLU):
                B_N = B * N
                x_flat = x.reshape(B_N, -1, 3)
                x_flat = layer(x_flat)
                x = x_flat.reshape(B, N, -1, 3)
        x_global = self.pool(x)
        # Return EQUIVARIANT features (B, C, 3) instead of invariant ones,
        # so the selector can compute relative orientation!
        return x_global


class VNCandidateSelector(nn.Module):
    """VNN-based canonical orientation candidate selector."""
    def __init__(self, n_points=2048, feat_dims=(32, 64, 128), hidden_dim=256):
        super().__init__()
        self.n_points = n_points
        self.encoder = VNPointNetEncoder(n_points=n_points, feat_dims=feat_dims)
        
        last_ch = feat_dims[-1]
        
        # We concatenate the template and candidate EQUIVARIANT features
        # along the channel dimension, so in_channels = last_ch * 2
        self.invariant = VNStdFeature(last_ch * 2)
        inv_dim = (last_ch * 2) * (last_ch * 2 + 1) // 2
        
        self.head = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, template_pts, candidate_pts):
        # Extract equivariant features: (B, C, 3)
        feat_t = self.encoder(template_pts)
        feat_c = self.encoder(candidate_pts)
        
        # Concatenate along channel dimension: (B, 2C, 3)
        combined_eq = torch.cat([feat_t, feat_c], dim=1)
        
        # Apply invariant layer to the combined equivariant features.
        # This computes the joint Gram matrix, which includes the cross-terms
        # (feat_t @ feat_c^T) that capture the RELATIVE rotation between them!
        inv_feat = self.invariant(combined_eq)
        
        score = self.head(inv_feat)
        return score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    lines = [f"{'='*60}", f"  VNCandidateSelector -- Model Summary", f"{'='*60}"]
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            lines.append(f"  {name:45s}  {str(list(param.shape)):>20s}  ({n:,})")
    lines.append(f"{'-'*60}")
    lines.append(f"  Total trainable parameters: {total:,}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


# =====================================================================
# GEOMETRY HELPERS
# =====================================================================

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def normalize_to_unit_sphere(mesh):
    mesh.apply_translation(-mesh.centroid)
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 0:
        mesh.apply_scale(1.0 / radius)
    return mesh


def sample_points(mesh, n=None, seed=0):
    if n is None:
        n = N_POINTS
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


def geodesic_angle_deg(R1, R2):
    R_rel = R1.T @ R2
    trace_val = np.trace(R_rel)
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# =====================================================================
# CANDIDATE ROTATION CRITERIA (Paper Equations 1-4)
# =====================================================================

def geometric_horizontal(obj_pts, tpl_pts, steps=None):
    if steps is None:
        steps = THETA_STEPS
    thetas = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    Eg = np.array([chamfer_distance(tpl_pts, obj_pts @ rot_y(t).T) for t in thetas])
    return thetas[int(np.argmin(Eg))], thetas, Eg


def semantic_horizontal(obj_pts, tpl_pts, steps=None):
    if steps is None:
        steps = THETA_STEPS
    theta_g, thetas, Eg = geometric_horizontal(obj_pts, tpl_pts, steps)
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
        gauss = sum(
            np.exp(-0.5 * (min(abs(t - w), 2 * np.pi - abs(t - w))) ** 2)
            for w in omega
        )
        objective = np.exp(-Es) * gauss
        if objective > best_obj:
            best_obj, best_theta = objective, t
    return best_theta


def pca_axes(points):
    c = points - points.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    v = v[:, np.argsort(w)[::-1]]
    return v[:, 0], v[:, 1]


def pca_vertical(obj_pts, tpl_pts):
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
    hull = mesh.convex_hull
    largest_face_idx = int(np.argmax(hull.area_faces))
    n = hull.face_normals[largest_face_idx]
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
    return [
        ("R_S", R_S), ("R_G", R_G), ("R_invG", R_invG),
        ("R_S.R_Sup", R_S_Sup), ("R_S.R_Pca", R_S_Pca),
    ]


# =====================================================================
# QUALITY GATE
# =====================================================================

def passes_quality(mesh):
    if len(mesh.faces) < MIN_FACES:
        return False, "too few faces"
    ext = mesh.bounding_box.extents
    if ext.max() <= 0 or (ext.min() / ext.max()) < MIN_THICKNESS_RATIO:
        return False, "thin-shell"
    return True, "ok"


# =====================================================================
# RENDERING
# =====================================================================

def render_mesh_png(mesh, R, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)

    verts = m.vertices.copy()
    verts = verts[:, [0, 2, 1]]
    faces = m.faces
    tris = verts[faces]

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
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, transparent=False, facecolor="white")
    plt.close(fig)


# =====================================================================
# OBJAVERSE DATA LOADING
# =====================================================================

def load_category_uids(category, n_needed):
    """Fetch object UIDs from Objaverse with strict per-category filtering."""
    import objaverse

    filters = CATEGORY_FILTERS.get(category, {"include": [category], "exclude": []})
    include_words = filters["include"]
    exclude_substrings = filters["exclude"]

    print(f"  Querying Objaverse for '{category}' objects...")
    uids = objaverse.load_uids()
    annotations = objaverse.load_annotations(uids[:50000])

    selected_uids = []
    for uid, anno in annotations.items():
        tags = [t['name'].lower() for t in anno.get('tags', [])]
        name_lower = anno.get('name', '').lower()
        name_words = name_lower.replace('-', ' ').replace('_', ' ').split()

        has_include = any(word in tags or word in name_words for word in include_words)
        if not has_include:
            continue

        has_exclude = any(exc in name_lower for exc in exclude_substrings)
        if has_exclude:
            continue

        selected_uids.append(uid)
        if len(selected_uids) >= n_needed:
            break

    print(f"  Found {len(selected_uids)} '{category}' objects (needed {n_needed}).")
    return selected_uids


def download_objects(uids):
    """Download 3D objects from Objaverse by UID safely."""
    import objaverse
    try:
        objs = objaverse.load_objects(uids=uids, download_processes=1)
        return objs
    except Exception as e:
        print(f"  [!] Batch download error: {e}. Trying item-by-item...")
        objs = {}
        for uid in uids:
            try:
                res = objaverse.load_objects(uids=[uid], download_processes=1)
                objs.update(res)
            except Exception as item_err:
                print(f"  [!] Skipping {uid}: {item_err}")
        return objs


# =====================================================================
# SELF-SUPERVISED LABELING
# =====================================================================

def compute_auto_label(candidate_pts_list, template_pts):
    """Pick candidate with lowest Chamfer distance to template (self-supervised)."""
    distances = [chamfer_distance(cpts, template_pts) for cpts in candidate_pts_list]
    return int(np.argmin(distances)), distances


# =====================================================================
# PYTORCH DATASET
# =====================================================================

class CandidateDataset(Dataset):
    def __init__(self, template_pts, objects):
        self.template_pts = torch.tensor(template_pts, dtype=torch.float32)
        self.samples = []
        for obj_idx, obj in enumerate(objects):
            for cand_idx in range(5):
                self.samples.append({
                    "candidate_pts": torch.tensor(
                        obj["candidate_pts"][cand_idx], dtype=torch.float32
                    ),
                    "label": torch.tensor(
                        [1.0 if cand_idx == obj["best_idx"] else 0.0],
                        dtype=torch.float32
                    ),
                    "object_idx": obj_idx,
                    "candidate_idx": cand_idx,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "template_pts": self.template_pts,
            "candidate_pts": s["candidate_pts"],
            "label": s["label"],
            "object_idx": s["object_idx"],
            "candidate_idx": s["candidate_idx"],
        }


# =====================================================================
# TRAINING DATA PREPARATION (SELF-SUPERVISED)
# =====================================================================

def prepare_training_data(category, n_objects=None, force_rebuild=False):
    """Prepare training data with self-supervised labels for a category."""
    if n_objects is None:
        n_objects = N_SAMPLES
    cache_file = get_cache_file(category)

    if not force_rebuild and os.path.exists(cache_file):
        print(f"\n  Loading cached data from {cache_file}...")
        return torch.load(cache_file, weights_only=False)

    print("\n" + "=" * 60)
    print(f"  PREPARING TRAINING DATA [{category.upper()}]")
    print("=" * 60)

    needed = n_objects + 1
    uids = load_category_uids(category, needed + 50)
    if len(uids) < 10:
        raise RuntimeError(f"Only found {len(uids)} '{category}' objects. Need >= 10.")

    random.shuffle(uids)

    download_uids = uids[:min(len(uids), needed + 50)]
    print(f"\n  Downloading {len(download_uids)} objects...")
    objs = download_objects(download_uids)

    print("\n  Setting up canonical template...")
    template_uid = uids[0]
    tpl_path = objs.get(template_uid)
    if tpl_path is None:
        raise RuntimeError(f"Template {template_uid} failed to download!")
    tpl_mesh = normalize_to_unit_sphere(trimesh.load(tpl_path, force="mesh"))
    template_pts = sample_points(tpl_mesh)
    print(f"  Template UID: {template_uid[:16]}...")

    print(f"\n  Processing objects (target: {n_objects})...")
    data_objects = []
    discarded = 0

    pbar = tqdm(uids[1:], desc="  Processing", total=min(len(uids)-1, n_objects+50), unit="obj")
    for uid in pbar:
        if len(data_objects) >= n_objects:
            break
        path = objs.get(uid)
        if path is None:
            try:
                path = download_objects([uid]).get(uid)
            except Exception:
                continue
        if path is None:
            continue
        try:
            mesh = normalize_to_unit_sphere(trimesh.load(path, force="mesh"))
        except Exception:
            continue
        ok, _ = passes_quality(mesh)
        if not ok:
            discarded += 1
            pbar.set_postfix(good=len(data_objects), bad=discarded)
            continue

        candidates = generate_five_candidates(mesh, template_pts)
        obj_pts = sample_points(mesh)
        cand_pts_list, cand_rot_list, cand_name_list = [], [], []
        for name, R in candidates:
            cand_pts_list.append(obj_pts @ R.T)
            cand_rot_list.append(R)
            cand_name_list.append(name)

        best_idx, cd_dists = compute_auto_label(cand_pts_list, template_pts)
        data_objects.append({
            "uid": uid,
            "candidate_pts": cand_pts_list,
            "candidate_rotations": cand_rot_list,
            "candidate_names": cand_name_list,
            "best_idx": best_idx,
            "chamfer_distances": cd_dists,
        })
        pbar.set_postfix(good=len(data_objects), bad=discarded)
    pbar.close()

    print(f"\n  Prepared {len(data_objects)} objects, discarded {discarded}.")
    from collections import Counter
    counts = Counter(obj["best_idx"] for obj in data_objects)
    print(f"  Label distribution: {', '.join(f'cand_{i}: {counts.get(i,0)}' for i in range(5))}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {"category": category, "template_pts": template_pts,
            "template_uid": template_uid, "objects": data_objects}
    torch.save(data, cache_file)
    print(f"  Cached to {cache_file}")
    return data


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch:3d} [TRAIN]", leave=False, unit="batch")
    for batch in pbar:
        tpl = batch["template_pts"].to(device)
        cand = batch["candidate_pts"].to(device)
        labels = batch["label"].to(device)
        scores = model(tpl, cand)
        loss = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * tpl.size(0)
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")
    pbar.close()
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, phase="VAL"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    object_scores = {}
    pbar = tqdm(dataloader, desc=f"  [{phase:>5s}]", leave=False, unit="batch")
    for batch in pbar:
        tpl = batch["template_pts"].to(device)
        cand = batch["candidate_pts"].to(device)
        labels = batch["label"].to(device)
        scores = model(tpl, cand)
        loss = criterion(scores, labels)
        total_loss += loss.item() * tpl.size(0)
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for i in range(tpl.size(0)):
            oi = batch["object_idx"][i].item()
            ci = batch["candidate_idx"][i].item()
            if oi not in object_scores:
                object_scores[oi] = []
            object_scores[oi].append((ci, scores[i].item(), labels[i].item()))
    pbar.close()

    sel_correct, sel_total = 0, 0
    for entries in object_scores.values():
        if len(entries) != 5:
            continue
        best = max(entries, key=lambda x: x[1])
        if best[2] > 0.5:
            sel_correct += 1
        sel_total += 1

    return {
        "loss": total_loss / total if total else 0,
        "accuracy": correct / total if total else 0,
        "selection_accuracy": sel_correct / sel_total if sel_total else 0,
        "selection_correct": sel_correct,
        "selection_total": sel_total,
    }


def split_data(objects, seed=None):
    if seed is None:
        seed = RANDOM_SEED
    rng = random.Random(seed)
    indices = list(range(len(objects)))
    rng.shuffle(indices)
    n = len(objects)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_objs = [objects[i] for i in indices[:n_train]]
    val_objs = [objects[i] for i in indices[n_train:n_train+n_val]]
    test_objs = [objects[i] for i in indices[n_train+n_val:]]
    print(f"\n  Split: {len(train_objs)} train / {len(val_objs)} val / {len(test_objs)} test")
    return train_objs, val_objs, test_objs


# =====================================================================
# TRAIN ONE CATEGORY
# =====================================================================

def train_category(category):
    """Train VNN for a single category with self-supervised labels."""
    print("\n" + "=" * 60)
    print(f"  VNN TRAINING -- {category.upper()}")
    print("=" * 60)

    best_model_path = get_best_model_path(category)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    total_start = time.time()
    data_start = time.time()
    data = prepare_training_data(category)
    template_pts = data["template_pts"]
    objects = data["objects"]
    data_time = time.time() - data_start
    print(f"\n  [T] Data preparation: {data_time:.1f}s ({data_time/60:.1f}min)")

    if len(objects) < 10:
        print(f"\n  ERROR: Not enough objects ({len(objects)}). Need >= 10.")
        return

    train_objs, val_objs, test_objs = split_data(objects)

    train_loader = DataLoader(CandidateDataset(template_pts, train_objs),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(CandidateDataset(template_pts, val_objs),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(CandidateDataset(template_pts, test_objs),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                             pin_memory=(device.type == "cuda"))

    model = VNCandidateSelector(n_points=N_POINTS, feat_dims=FEAT_DIMS,
                                hidden_dim=HIDDEN_DIM).to(device)
    print(f"\n{model_summary(model)}")
    # 1 positive and 4 negative candidates per object, so we weight the positive class by 4.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    print("\n" + "-" * 60)
    print(f"  TRAINING [{category.upper()}]")
    print("-" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_sel_acc = -1.0
    patience_counter = 0
    train_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vm = evaluate(model, val_loader, criterion, device, "VAL")
        scheduler.step(vm["selection_accuracy"])
        et = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"TrLoss: {tm['loss']:.4f} Acc: {tm['accuracy']:.3f} | "
            f"VaLoss: {vm['loss']:.4f} Acc: {vm['accuracy']:.3f} "
            f"SelAcc: {vm['selection_accuracy']:.3f} "
            f"({vm['selection_correct']}/{vm['selection_total']}) | {et:.1f}s"
        )

        if vm["selection_accuracy"] > best_val_sel_acc:
            best_val_sel_acc = vm["selection_accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_selection_accuracy": best_val_sel_acc,
                "feat_dims": FEAT_DIMS, "hidden_dim": HIDDEN_DIM,
                "n_points": N_POINTS, "category": category,
            }, best_model_path)
            print(f"  * New best! (SelAcc: {best_val_sel_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - train_start
    print(f"\n  [T] Training: {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  Best val SelAcc: {best_val_sel_acc:.3f}")

    # Test evaluation
    print("\n" + "-" * 60)
    print(f"  TEST EVALUATION [{category.upper()}]")
    print("-" * 60)

    ckpt = torch.load(best_model_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_start = time.time()
    test_m = evaluate(model, test_loader, criterion, device, "TEST")
    test_time = time.time() - test_start

    print(f"\n  Test Results:")
    print(f"    Binary Accuracy:    {test_m['accuracy']:.3f}")
    print(f"    Selection Accuracy: {test_m['selection_accuracy']:.3f} "
          f"({test_m['selection_correct']}/{test_m['selection_total']})")
    print(f"    Loss:               {test_m['loss']:.4f}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  TIMING [{category.upper()}]")
    print(f"{'='*60}")
    print(f"  Data:     {data_time:8.1f}s ({data_time/60:.1f}min)")
    print(f"  Train:    {train_time:8.1f}s ({train_time/60:.1f}min)")
    print(f"  Test:     {test_time:8.1f}s")
    print(f"  TOTAL:    {total_time:8.1f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")
    print(f"  Model: {best_model_path}")


# =====================================================================
# GENERATE CANDIDATES FOR ONE CATEGORY
# =====================================================================

def generate_category(category):
    """Generate candidates + VNN selection for a category."""
    pipeline_start = time.time()

    print(f"\n{'='*60}")
    print(f"  GENERATING CANDIDATES -- {category.upper()}")
    print(f"{'='*60}")

    uids = load_category_uids(category, N_GENERATE + 10)
    if not uids:
        print(f"  [!] No '{category}' objects found. Skipping.")
        return 0, 0

    random.shuffle(uids)
    out_dir = os.path.join(OUT_ROOT, category)
    os.makedirs(out_dir, exist_ok=True)

    # Load VNN
    best_model_path = get_best_model_path(category)
    vnn_model_obj = None
    vnn_available = False
    if os.path.exists(best_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(best_model_path, weights_only=False, map_location=device)
        vnn_model_obj = VNCandidateSelector(
            n_points=ckpt.get("n_points", N_POINTS),
            feat_dims=ckpt.get("feat_dims", FEAT_DIMS),
            hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
        ).to(device)
        vnn_model_obj.load_state_dict(ckpt["model_state_dict"])
        vnn_model_obj.eval()
        vnn_available = True
        print(f"  [OK] Loaded VNN from {best_model_path}")
    else:
        print(f"  [!] No VNN checkpoint. Generating without auto-selection.")

    # Template
    template_uid = uids[0]
    objs = download_objects([template_uid])
    tpl_path = objs.get(template_uid)
    if tpl_path is None:
        print(f"  [!] Template download failed. Skipping.")
        return 0, 0
    tpl_mesh = normalize_to_unit_sphere(trimesh.load(tpl_path, force="mesh"))
    tpl_pts = sample_points(tpl_mesh)
    render_mesh_png(tpl_mesh, np.eye(3), os.path.join(out_dir, "template.png"))
    print(f"  Template = {template_uid}")

    manifest = {"category": category, "template": "template.png",
                "vnn_enabled": vnn_available, "objects": []}
    prepared, discarded = 0, 0

    pbar = tqdm(uids[1:], desc=f"  [{category}]", total=min(len(uids)-1, N_GENERATE), unit="obj")
    for uid in pbar:
        if prepared >= N_GENERATE:
            break
        path = objs.get(uid) or download_objects([uid]).get(uid)
        if path is None:
            continue
        try:
            mesh = normalize_to_unit_sphere(trimesh.load(path, force="mesh"))
        except Exception:
            continue
        ok, _ = passes_quality(mesh)
        if not ok:
            discarded += 1
            pbar.set_postfix(good=prepared, bad=discarded, status="skip")
            continue

        obj_dir = os.path.join(out_dir, uid)
        os.makedirs(obj_dir, exist_ok=True)
        candidates = generate_five_candidates(mesh, tpl_pts)

        cand_records, cand_rotations = [], []
        for i, (name, R) in enumerate(candidates):
            fn = f"{uid}/cand_{i}.png"
            render_mesh_png(mesh, R, os.path.join(out_dir, fn))
            cand_records.append({"index": i, "name": name, "image": fn, "rotation": R.tolist()})
            cand_rotations.append(R)

        vnn_selection = None
        if vnn_available:
            obj_pts = sample_points(mesh)
            tpl_tensor = torch.tensor(tpl_pts, dtype=torch.float32).unsqueeze(0).to(device)
            scores = []
            with torch.no_grad():
                for R in cand_rotations:
                    rotated = obj_pts @ R.T
                    ct = torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).to(device)
                    s = vnn_model_obj(tpl_tensor, ct)
                    scores.append(s.item())
            best_idx = int(np.argmax(scores))

            correct_dir = os.path.join(obj_dir, "correct")
            rejected_dir = os.path.join(obj_dir, "rejected")
            os.makedirs(correct_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)
            shutil.copy2(os.path.join(out_dir, f"{uid}/cand_{best_idx}.png"),
                         os.path.join(correct_dir, f"cand_{best_idx}.png"))
            for i in range(5):
                if i != best_idx:
                    shutil.copy2(os.path.join(out_dir, f"{uid}/cand_{i}.png"),
                                 os.path.join(rejected_dir, f"cand_{i}.png"))

            vnn_selection = {"selected_index": best_idx, "selected_name": candidates[best_idx][0],
                             "scores": scores, "selected_score": scores[best_idx]}
            pbar.set_postfix(good=prepared+1, bad=discarded, vnn=f"c{best_idx}({scores[best_idx]:.2f})")

        entry = {"uid": uid, "candidates": cand_records}
        if vnn_selection:
            entry["vnn_selected"] = vnn_selection
        manifest["objects"].append(entry)
        prepared += 1
    pbar.close()

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - pipeline_start
    print(f"\n  [{category}] Prepared: {prepared}, Discarded: {discarded}, Time: {elapsed:.1f}s")
    return prepared, discarded


# =====================================================================
# TOP-LEVEL ENTRY POINTS
# =====================================================================

def run_pipeline(mode, categories=None):
    """Run the pipeline in 'train' or 'generate' mode.

    Args:
        mode: 'train' or 'generate'
        categories: List of categories to process (default: all)
    """
    if categories is None:
        categories = CATEGORIES

    if mode == "train":
        print("\n" + "#" * 60)
        print("  TRAINING VNN FOR ALL CATEGORIES")
        print("#" * 60)
        t0 = time.time()
        for cat in categories:
            train_category(cat)
        print(f"\n  Total training time: {time.time()-t0:.1f}s")

    elif mode == "generate":
        print("\n" + "#" * 60)
        print("  GENERATING CANDIDATES FOR ALL CATEGORIES")
        print("#" * 60)
        t0 = time.time()
        total_p, total_d = 0, 0
        for cat in categories:
            p, d = generate_category(cat)
            total_p += p
            total_d += d
        print(f"\n  Total: {total_p} prepared, {total_d} discarded, {time.time()-t0:.1f}s")

    else:
        print(f"  Unknown mode: {mode}. Use 'train' or 'generate'.")


def run_all(categories=None):
    """Run the full pipeline: train all VNNs, then generate candidates."""
    if categories is None:
        categories = CATEGORIES

    grand_start = time.time()

    print("\n" + "#" * 60)
    print("  FULL PIPELINE: TRAIN + GENERATE")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Samples per category: {N_SAMPLES}")
    print("#" * 60)

    # Step 1: Train
    run_pipeline("train", categories)

    # Step 2: Generate
    run_pipeline("generate", categories)

    grand_time = time.time() - grand_start
    print(f"\n{'#'*60}")
    print(f"  COMPLETE! Total time: {grand_time:.1f}s ({grand_time/60:.1f}min)")
    print(f"{'#'*60}\n")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    run_all()