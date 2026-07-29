"""
=============================================================================
  CanoVerse VNN -- Single-File Kaggle Bundle  (v2 — Full CanoVerse Pipeline)
=============================================================================

Complete self-contained pipeline for training and using a Vector Neuron
Network (VNN) for automatic canonical orientation selection of 3D objects.

Implements the full CanoVerse paper method:
  - Semantic part segmentation (K-means spatial clustering as Find3D proxy)
  - Proper support-surface vertical alignment (Rodrigues rotation + stability)
  - PCA vertical with semantic part disambiguation (Eq. 1-2)
  - Joint semantic-geometric horizontal alignment (Eq. 4)
  - Symmetry-aware 5-candidate generation
  - SO(3) rotation augmentation (40 random rotations per sample)
  - CLIP-based zero-shot category discovery (Uni3D proxy)
  - Category-agnostic cross-category training with ranking loss
  - Geodesic evaluation metrics (Acc@10, Acc@30)

Two training modes:
  - Per-category:  One VNN per category (backward-compatible)
  - Agnostic:      Single VNN across all categories (paper-faithful)

FULLY AUTOMATED -- NO human labels or annotators needed!

Usage in Kaggle notebook:
  1. Paste this entire file into a cell
  2. Adjust the CONFIG section below
  3. Call run_all() for per-category, or run_agnostic() for cross-category

Or run individual steps:
  - run_pipeline("train")
  - run_pipeline("generate")
  - train_agnostic()
  - align_single_object(mesh)       # fully zero-shot on any mesh
"""

# =====================================================================
# EASY-TUNE CONFIG -- Change these to control the pipeline!
# =====================================================================

# Categories to process
CATEGORIES = ["table", "chair", "car", "airplane"]

# Use category-agnostic implementation or not
CATEGORY_AGNOSTIC = True

# Limit training samples per category to N_SAMPLES (True) or sample ALL existing point clouds (False)
USE_ONLY_N_SAMPLES = False

# Objects PER CATEGORY (used only if USE_ONLY_N_SAMPLES = True)
N_SAMPLES = 30

# Objects to generate candidates for (used only if USE_ONLY_N_SAMPLES = True)
N_GENERATE = 5

# Point cloud settings
N_POINTS = 1024          # Points sampled per mesh surface (consistent everywhere)

# Download from Objaverse (True) or use Kaggle ModelNet40 Point Clouds (False)
DOWNLOAD_OBJAVERSE = False

# Rotation augmentation (paper: 40 random rotations per sample)
N_ROTATIONS = 30

# Training hyperparameters
BATCH_SIZE = 4           # Micro-batch size to fit in memory
GRAD_ACCUM_STEPS = 8     # Effective batch size = 4 * 8 = 32
NUM_EPOCHS = 150          # Max training epochs (early stopping will kick in)
LEARNING_RATE = 1e-3     # Adam learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
PATIENCE = 10            # Early stopping patience

# VNN architecture (per-category mode)
FEAT_DIMS = (32, 64, 128)  # VN encoder channel dimensions
HIDDEN_DIM = 256            # MLP head hidden size

# VNN architecture (agnostic mode -- larger for cross-category capacity)
AGNOSTIC_FEAT_DIMS = (32, 64, 128)
AGNOSTIC_HIDDEN_DIM = 256
N_CATEGORY_EMBED = 32       # Category embedding dimension

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Candidate generation
THETA_STEPS = 72         # Horizontal rotation search resolution (was 180; 5-deg steps still precise)
RENDER_SIZE = 320        # Render image dimensions (pixels, square)

# Semantic part segmentation
N_SEMANTIC_PARTS = 6     # Number of spatial-semantic clusters

# Quality gate
MIN_THICKNESS_RATIO = 0.02
MIN_FACES = 50
MIN_CONNECTED_RATIO = 0.7   # Min fraction of faces in largest component
MAX_FACE_AREA_VAR = 50.0     # Max ratio of largest to mean face area

# Support surface
MAX_SUPPORT_FACES = 20   # Max convex hull faces to evaluate

# CLIP settings (for zero-shot category discovery)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

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
import tempfile
import glob
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
!pip -q install trimesh objaverse
import trimesh
from PIL import Image
from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.pop("PYOPENGL_PLATFORM", None)

def load_mesh_or_pc(path):
    """Load a Trimesh mesh, or convert a .txt point cloud into a faceless Trimesh."""
    if path.endswith('.txt'):
        try:
            data = np.loadtxt(path, delimiter=',')
        except ValueError:
            data = np.loadtxt(path)
        m = trimesh.Trimesh(vertices=data[:, :3])
        if data.shape[1] >= 6:
            m.vertex_normals = data[:, 3:6]
        return m
    else:
        return trimesh.load(path, force="mesh")


# =====================================================================
# CATEGORY CONFIGURATION
# =====================================================================

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

# Symmetry definitions per category (paper §3.1)
# angle=0 means no symmetry, angle=180 means bilateral, angle=360 means full rotational
SYMMETRY_INFO = {
    "table":    {"axis": "Y", "angle": 90},    # 4-fold symmetry
    "chair":    {"axis": "Y", "angle": 0},      # no symmetry (has a front)
    "car":      {"axis": "Y", "angle": 180},    # bilateral (left-right mirror)
    "airplane": {"axis": "Y", "angle": 180},    # bilateral
    "cup":      {"axis": "Y", "angle": 360},    # full rotational
    "bottle":   {"axis": "Y", "angle": 360},
    "bowl":     {"axis": "Y", "angle": 360},
    "vase":     {"axis": "Y", "angle": 360},
    "bed":      {"axis": "Y", "angle": 180},
    "lamp":     {"axis": "Y", "angle": 0},
    "sofa":     {"axis": "Y", "angle": 0},
    "bench":    {"axis": "Y", "angle": 180},
}

# Template axis conventions (paper Supp §1.1)
# Forward = +Z, Up = +Y  (all categories share these axes)
TEMPLATE_FORWARD = np.array([0.0, 0.0, 1.0])
TEMPLATE_UP = np.array([0.0, 1.0, 0.0])


# =====================================================================
# PATH HELPERS
# =====================================================================

def get_cache_file(category):
    return os.path.join(CACHE_DIR, f"{category}_vnn_data.pt")


def get_best_model_path(category):
    return os.path.join(CHECKPOINT_DIR, f"vnn_{category}_best.pt")


def get_latest_model_path(category):
    return os.path.join(CHECKPOINT_DIR, f"vnn_{category}_latest.pt")


def get_agnostic_model_path():
    return os.path.join(CHECKPOINT_DIR, "vnn_agnostic_best.pt")


def get_agnostic_latest_model_path():
    return os.path.join(CHECKPOINT_DIR, "vnn_agnostic_latest.pt")


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


class VNMeanPool(nn.Module):
    """SO(3)-equivariant mean pooling over the point dimension."""
    def forward(self, x):
        return x.mean(dim=1)


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


def knn_distance_invariant(x, k):
    """SO(3)-invariant KNN based on inner products of vector features.
    x: (B, C, 3, N) -> returns indices (B, N, k)
    """
    inner = torch.einsum('bcdn,bcdm->bnm', x, x) # (B, N, N)
    x2 = torch.diagonal(inner, dim1=1, dim2=2).unsqueeze(2) # (B, N, 1)
    dist = x2 + x2.transpose(1, 2) - 2 * inner
    idx = dist.topk(k=k, dim=-1, largest=False)[1] # (B, N, k)
    return idx

def get_vn_graph_feature(x, k=20, idx=None):
    """Extract VN EdgeConv features. x: (B, C, 3, N)"""
    B, C, _, N = x.size()
    if idx is None:
        idx = knn_distance_invariant(x, k=k) # (B, N, k)
        
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    
    x_perm = x.permute(0, 3, 1, 2) # (B, N, C, 3)
    x_flat = x_perm.reshape(B*N, C, 3) # (B*N, C, 3)
    x_neighbors = x_flat[idx_flat].view(B, N, k, C, 3).permute(0, 3, 4, 1, 2) # (B, C, 3, N, k)
    
    x_central = x.unsqueeze(-1).expand(B, C, 3, N, k)
    edge_feature = torch.cat([x_central, x_neighbors - x_central], dim=1) # (B, 2C, 3, N, k)
    return edge_feature, idx

class VNMaxPoolLocal(nn.Module):
    """SO(3)-equivariant max pooling over local K-NN neighborhoods."""
    def forward(self, x):
        norms = torch.norm(x, dim=2) # (B, C, N, k)
        max_idx = norms.argmax(dim=-1, keepdim=True) # (B, C, N, 1)
        max_idx_expanded = max_idx.unsqueeze(2).expand(-1, -1, 3, -1, -1) # (B, C, 3, N, 1)
        pooled = torch.gather(x, 4, max_idx_expanded).squeeze(-1) # (B, C, 3, N)
        return pooled

class VNDGCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = VNLinear(in_channels * 2, out_channels)
        self.act = VNLeakyReLU(out_channels)
        self.pool = VNMaxPoolLocal()
        
    def forward(self, x, idx=None):
        edge_feature, idx = get_vn_graph_feature(x, k=self.k, idx=idx) # (B, 2C, 3, N, k)
        B, C2, _, N, k = edge_feature.size()
        edge_flat = edge_feature.permute(0, 3, 4, 1, 2).reshape(B*N*k, C2, 3)
        out_flat = self.act(self.conv(edge_flat))
        out = out_flat.view(B, N, k, -1, 3).permute(0, 3, 4, 1, 2) # (B, C_out, 3, N, k)
        return self.pool(out), idx

class VNDGCNNEncoder(nn.Module):
    """VN-DGCNN encoder: preserves local geometry via EdgeConv with
    multi-scale feature aggregation.
    
    Input:  (B, N, 3)
    Output: (B, C_total, 3) global equivariant features.
    
    Instead of only pooling the final layer (which loses fine-grained detail),
    we concatenate features from ALL layers before global pooling, capturing
    both local (early layers) and global (later layers) geometry.
    """
    def __init__(self, n_points=1024, feat_dims=(32, 64, 128), k=20):
        super().__init__()
        self.n_points = n_points
        self.k = k
        
        layers = []
        in_ch = 1
        for out_ch in feat_dims:
            layers.append(VNDGCNNLayer(in_ch, out_ch, k=self.k))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        
        # Multi-scale: concatenate all layer outputs then project to final dim
        total_ch = sum(feat_dims)  # e.g. 32+64+128 = 224
        self.fuse = VNLinear(total_ch, feat_dims[-1])  # project back to last_ch
        self.fuse_act = VNLeakyReLU(feat_dims[-1])
        
        self.global_pool_max = VNMaxPool()
        self.global_pool_mean = VNMeanPool()

    def forward(self, points):
        x = points.unsqueeze(1).permute(0, 1, 3, 2) # (B, 1, 3, N)
        layer_outputs = []
        for layer in self.layers:
            x, _ = layer(x, idx=None)  # Recompute graph dynamically
            layer_outputs.append(x)    # Each is (B, C_i, 3, N)
            
        # Concatenate multi-scale features: (B, sum(C_i), 3, N)
        x_cat = torch.cat(layer_outputs, dim=1)
        
        # Fuse per-point: reshape to (B*N, total_ch, 3), apply VNLinear, reshape back
        B, C_total, _, N = x_cat.size()
        x_flat = x_cat.permute(0, 3, 1, 2).reshape(B * N, C_total, 3)
        x_fused = self.fuse_act(self.fuse(x_flat))  # (B*N, last_ch, 3)
        x_fused = x_fused.view(B, N, -1, 3)  # (B, N, last_ch, 3)
            
        x_max = self.global_pool_max(x_fused) # (B, C, 3)
        x_mean = self.global_pool_mean(x_fused) # (B, C, 3)
        x_global = torch.cat([x_max, x_mean], dim=1) # (B, 2C, 3)
        return x_global


class VNCandidateSelector(nn.Module):
    """VNN-based canonical orientation candidate selector.

    Scores a (template, candidate) pair.  In per-category mode the score
    is compared across the 5 candidates of one object.  In agnostic mode
    an optional learned category embedding is concatenated to the
    invariant features.

    Args:
        n_points:           expected point count (stored, not enforced)
        feat_dims:          VN encoder channel dimensions
        hidden_dim:         MLP head hidden width
        n_categories:       if >0, add a learned category embedding
        category_embed_dim: dimension of the category embedding
    """
    def __init__(self, n_points=1024, feat_dims=(32, 64, 128), hidden_dim=256,
                 n_categories=0, category_embed_dim=32, reduce_dim=64, k=12):
        super().__init__()
        self.n_points = n_points
        self.encoder = VNDGCNNEncoder(n_points=n_points, feat_dims=feat_dims, k=k)

        last_ch = feat_dims[-1] * 2  # Account for mean+max concat from encoder

        # Concatenate template and candidate equivariant features: 2*C channels.
        # Apply a VNLinear bottleneck to reduce channels before the 
        # O(C^2) Gram matrix computation.  reduce_dim=64 keeps rich geometric
        # detail (the old 32 was too lossy for distinguishing subtle rotations).
        self.reducer = VNLinear(last_ch * 2, reduce_dim)
        self.invariant = VNStdFeature(reduce_dim)
        inv_dim = reduce_dim * (reduce_dim + 1) // 2

        # Optional category embedding (agnostic mode)
        self.use_category = n_categories > 0
        head_input_dim = inv_dim
        if self.use_category:
            self.category_embed = nn.Embedding(n_categories, category_embed_dim)
            head_input_dim += category_embed_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, template_pts, candidate_pts, category_idx=None):
        """
        Args:
            template_pts:  (B, N, 3)
            candidate_pts: (B, N, 3)
            category_idx:  (B,) LongTensor, optional
        Returns:
            score: (B, 1) raw logit
        """
        feat_t = self.encoder(template_pts)          # (B, C, 3)
        feat_c = self.encoder(candidate_pts)          # (B, C, 3)
        combined_eq = torch.cat([feat_t, feat_c], dim=1)  # (B, 2C, 3)
        reduced_eq = self.reducer(combined_eq)        # (B, reduce_dim, 3)
        inv_feat = self.invariant(reduced_eq)        # (B, inv_dim)

        if self.use_category and category_idx is not None:
            cat_emb = self.category_embed(category_idx)  # (B, embed_dim)
            inv_feat = torch.cat([inv_feat, cat_emb], dim=1)

        score = self.head(inv_feat)
        return score

    def forward_with_cached_template(self, feat_t, candidate_pts, category_idx=None):
        """Score a candidate using pre-encoded template features.
        This avoids re-encoding the template for each of the 5 candidates.

        Args:
            feat_t:         (B, C, 3) pre-encoded equivariant template features
            candidate_pts:  (B, N, 3)
            category_idx:   (B,) LongTensor, optional
        Returns:
            score: (B, 1) raw logit
        """
        feat_c = self.encoder(candidate_pts)          # (B, C, 3)
        combined_eq = torch.cat([feat_t, feat_c], dim=1)  # (B, 2C, 3)
        reduced_eq = self.reducer(combined_eq)        # (B, reduce_dim, 3)
        inv_feat = self.invariant(reduced_eq)        # (B, inv_dim)

        if self.use_category and category_idx is not None:
            cat_emb = self.category_embed(category_idx)  # (B, embed_dim)
            inv_feat = torch.cat([inv_feat, cat_emb], dim=1)

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
    """Rotation matrix around the Y (up) axis by angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def chordal_l2_mean_np(rotations):
    """Compute the chordal L2 mean of a list of numpy rotation matrices."""
    R_sum = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    S = U @ Vt
    if np.linalg.det(S) < 0:
        U = U.copy()
        U[:, -1] *= -1
        S = U @ Vt
    return S

def rotation_angle_np(R1, R2):
    """Geodesic angle (radians) between two numpy rotation matrices.
    Uses ||R1-R2||_F = 2*sqrt(2)*sin(theta/2) which is exact near 0."""
    d = np.sqrt(np.sum((R1 - R2) ** 2, axis=(-2, -1)) + 1e-24)
    return 2.0 * np.arcsin(np.clip(d / (2.0 * np.sqrt(2.0)), a_min=None, a_max=1.0 - 1e-7))


def rodrigues_rotation(n_from, n_to):
    """Rotation matrix that aligns unit vector n_from to n_to via Rodrigues.

    Implements Supp §1.2 Equations 1-3:
      u = n_from × n_to          (rotation axis)
      θ = arccos(n_from · n_to)  (rotation angle)
      R = I + sinθ K + (1-cosθ) K²   (Rodrigues formula)
    where K is the skew-symmetric matrix of u.
    """
    a = np.asarray(n_from, dtype=np.float64)
    b = np.asarray(n_to, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        # Vectors are (anti-)parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180° rotation: find any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, a) * a
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            return 2.0 * np.outer(perp, perp) - np.eye(3)

    v_unit = v / s
    K = np.array([
        [0, -v_unit[2], v_unit[1]],
        [v_unit[2], 0, -v_unit[0]],
        [-v_unit[1], v_unit[0], 0],
    ])
    theta = np.arccos(np.clip(c, -1.0, 1.0))
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def random_rotation_matrix():
    """Sample a uniform random rotation from SO(3).

    Uses the subgroup algorithm (Arvo 1992) via uniform random quaternion.
    Applied as data augmentation: 40 rotations per sample (paper Supp §2.1).
    """
    u1, u2, u3 = np.random.random(3)
    q = np.array([
        np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
        np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
        np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
        np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def normalize_to_unit_sphere(mesh):
    """Centre mesh at origin and scale to fit inside a unit sphere."""
    center = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-center)
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 0:
        mesh.apply_scale(1.0 / radius)
    return mesh


def sample_points(mesh, n=None, seed=None, with_normals=False):
    """Sample n points uniformly from the mesh surface or point cloud.
    
    Args:
        seed: If None, uses fresh random state (avoids train/test leakage).
              If an int, uses deterministic seeding (for template reproducibility).
    """
    if n is None:
        n = N_POINTS
    st = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    
    has_normals = hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > 0
    
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        pts, faces_idx = trimesh.sample.sample_surface(mesh, n)
        if with_normals:
            if hasattr(mesh, 'face_normals') and len(mesh.face_normals) > 0:
                normals = mesh.face_normals[faces_idx]
            else:
                normals = np.zeros_like(pts)
    else:
        pts = mesh.vertices
        idx = None
        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            pts = pts[idx]
        elif len(pts) > 0 and len(pts) < n:
            idx = np.random.choice(len(pts), n, replace=True)
            pts = pts[idx]
        
        if with_normals:
            if has_normals:
                all_normals = mesh.vertex_normals
                normals = all_normals[idx] if idx is not None else all_normals
            else:
                normals = np.zeros_like(pts)
            
    if seed is not None:
        np.random.set_state(st)
    if with_normals:
        return np.asarray(pts, dtype=np.float32), np.asarray(normals, dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def chamfer_distance(a, b, na=None, nb=None, alpha=0.5):
    """Chamfer distance between two point clouds (mean of nearest-neighbour dists).
    Optional normal penalty: penalizes point matches with opposing normals.
    """
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, idx_ab = tb.query(a)
    d_ba, idx_ba = ta.query(b)
    
    cd_val = float(d_ab.mean() + d_ba.mean())
    if na is not None and nb is not None:
        dot_ab = np.sum(na * nb[idx_ab], axis=1)
        dot_ba = np.sum(nb * na[idx_ba], axis=1)
        normal_loss = float((1 - dot_ab).mean() + (1 - dot_ba).mean())
        return cd_val + alpha * normal_loss
    return cd_val


def chamfer_distance_fast(a, b, tree_b=None, max_pts=256):
    """Fast approximate Chamfer using subsampled point clouds + optional pre-built KDTree.
    Used during alignment search (geometric/semantic_horizontal) where many calls are made.
    """
    if len(a) > max_pts:
        a = a[np.random.choice(len(a), max_pts, replace=False)]
    if tree_b is None:
        if len(b) > max_pts:
            b = b[np.random.choice(len(b), max_pts, replace=False)]
        tree_b = cKDTree(b)
        tree_a = cKDTree(a)
        d_ba, _ = tree_a.query(b)
    else:
        tree_a = cKDTree(a)
        d_ba, _ = tree_a.query(tree_b.data if hasattr(tree_b, 'data') else b[:max_pts])
    d_ab, _ = tree_b.query(a)
    return float(d_ab.mean() + d_ba.mean())


def rotation_from_vectors(a, b):
    """Rotation matrix that rotates unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def geodesic_angle_deg(R1, R2):
    """Geodesic angle between two rotation matrices, in degrees."""
    R_rel = R1.T @ R2
    trace_val = np.trace(R_rel)
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pca_axes(points):
    """First two principal axes of a point cloud."""
    c = points - points.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    v = v[:, np.argsort(w)[::-1]]
    return v[:, 0], v[:, 1]


def point_in_polygon_2d(point, polygon):
    """Ray-casting algorithm for 2D point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    px, py = float(point[0]), float(point[1])
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


# =====================================================================
# SEMANTIC PART SEGMENTATION  (Find3D proxy — paper §3.1)
# =====================================================================

def segment_semantic_parts(points, n_parts=None):
    """Segment a point cloud into spatial-semantic parts via MiniBatch K-means clustering.

    MiniBatchKMeans is 3-10x faster than kmeans2 on 1024 points while producing
    essentially the same cluster quality for spatial segmentation.

    Returns a list of point-cloud arrays, one per cluster.
    """
    if n_parts is None:
        n_parts = N_SEMANTIC_PARTS
    n_parts = min(n_parts, max(1, len(points) // 4))
    if n_parts <= 1:
        return [points]
    try:
        km = MiniBatchKMeans(
            n_clusters=n_parts,
            n_init=1,
            max_iter=50,
            batch_size=min(256, len(points)),
            random_state=RANDOM_SEED,
        )
        labels = km.fit_predict(points.astype(np.float32))
    except Exception:
        # Fallback: uniform spatial binning along the longest axis
        axis = int(np.argmax(np.ptp(points, axis=0)))
        edges = np.linspace(points[:, axis].min(), points[:, axis].max(), n_parts + 1)
        labels = np.clip(np.digitize(points[:, axis], edges[1:-1]), 0, n_parts - 1)

    parts = []
    for k in range(n_parts):
        mask = labels == k
        if mask.sum() > 0:
            parts.append(points[mask])
    return parts if parts else [points]


def match_parts_chamfer(parts_a, parts_b, max_pts=256):
    """Match two sets of semantic parts by greedy nearest-centroid and return
    the average per-part fast Chamfer distance.
    """
    if not parts_a or not parts_b:
        return float("inf")
    centroids_a = [p.mean(axis=0) for p in parts_a]
    centroids_b = [p.mean(axis=0) for p in parts_b]
    used_b = set()
    total_cd = 0.0
    m = 0
    for i, ca in enumerate(centroids_a):
        best_j, best_dist = -1, float("inf")
        for j, cb in enumerate(centroids_b):
            if j in used_b:
                continue
            d = float(np.linalg.norm(ca - cb))
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j >= 0:
            used_b.add(best_j)
            total_cd += chamfer_distance_fast(parts_a[i], parts_b[best_j], max_pts=max_pts)
            m += 1
    return total_cd / m if m > 0 else float("inf")


# =====================================================================
# VERTICAL ALIGNMENT
# =====================================================================

def support_vertical(mesh):
    """Full support-surface vertical alignment (Supp §1.2).

    Algorithm:
      1. Compute convex hull, extract face normals.
      2. For each face k:
         - u_k = n_s^k × n_g            (rotation axis, Eq. 1)
         - θ_k = arccos(n_s^k · n_g)    (rotation angle, Eq. 2)
         - R_k via Rodrigues formula     (Eq. 3)
      3. Rotate point cloud, check centroid stability.
      4. Among valid stable poses, pick the most "upright" one
         (highest Y-extent and lowest centroid — replaces VLM selection).
    """
    try:
        hull = mesh.convex_hull
    except Exception:
        # Fallback for degenerate meshes
        return np.eye(3)

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    n_g = np.array([0.0, -1.0, 0.0])  # ground normal (pointing down for "floor")

    face_areas = hull.area_faces
    face_order = np.argsort(face_areas)[::-1][:MAX_SUPPORT_FACES]

    valid_poses = []

    for face_idx in face_order:
        n_s = hull.face_normals[face_idx].astype(np.float64)
        n_s_norm = np.linalg.norm(n_s)
        if n_s_norm < 1e-12:
            continue
        n_s = n_s / n_s_norm

        # Rodrigues rotation aligning face normal to ground
        R_k = rodrigues_rotation(n_s, n_g)

        # Rotate point cloud
        rotated = (R_k @ pts.T).T
        centroid = rotated.mean(axis=0)

        # Rotate face vertices for stability check
        face_vert_indices = hull.faces[face_idx]
        face_verts = hull.vertices[face_vert_indices].astype(np.float64)
        rotated_face = (R_k @ face_verts.T).T

        # Project centroid onto the ground plane (XZ plane at face level)
        proj_2d = np.array([centroid[0], centroid[2]])
        face_2d = rotated_face[:, [0, 2]]

        stable = point_in_polygon_2d(proj_2d, face_2d)

        # Score: even unstable poses get a low score (they may be the only option)
        y_extent = rotated[:, 1].max() - rotated[:, 1].min()
        y_min = rotated[:, 1].min()
        # Prefer: high vertical extent, centroid well above base, wide base
        base_width = np.ptp(face_2d, axis=0).mean() if len(face_2d) > 1 else 0.01
        score = y_extent * (base_width + 0.01)
        if stable:
            score *= 3.0  # strong bonus for centroid-stable poses

        valid_poses.append((R_k, score))

    if not valid_poses:
        # Absolute fallback: align largest face to ground
        largest_idx = int(np.argmax(face_areas))
        n = hull.face_normals[largest_idx].astype(np.float64)
        return rodrigues_rotation(n / (np.linalg.norm(n) + 1e-12), n_g)

    valid_poses.sort(key=lambda x: x[1], reverse=True)
    return valid_poses[0][0]


def pca_vertical(obj_pts, tpl_pts):
    """PCA-based vertical alignment with semantic part disambiguation.

    Implements Eq. 1 (rotation candidates) and Eq. 2 (semantic selection):
      R_i = [v1t v2t (v1t×v2t)] · [σ_i1·v1o  σ_i2·v2o  (σ_i1·v1o × σ_i2·v2o)]^T
      R_Pca = argmin_{R_i} (1/m) Σ_k CD(S_k^t, R_i S_k^o)
    """
    v1o, v2o = pca_axes(obj_pts)
    v1t, v2t = pca_axes(tpl_pts)
    Mt = np.stack([v1t, v2t, np.cross(v1t, v2t)], axis=1)

    # Segment template into semantic parts (once)
    tpl_parts = segment_semantic_parts(tpl_pts)

    best_R, best_cost = np.eye(3), np.inf
    for s1 in (1, -1):
        for s2 in (1, -1):
            u1, u2 = s1 * v1o, s2 * v2o
            Mo = np.stack([u1, u2, np.cross(u1, u2)], axis=1)
            R = Mt @ Mo.T

            # Rotate object and segment
            rotated_obj = (R @ obj_pts.T).T
            obj_parts = segment_semantic_parts(rotated_obj)

            # Eq. 2: average per-part Chamfer distance
            cost = match_parts_chamfer(tpl_parts, obj_parts)
            if cost < best_cost:
                best_cost, best_R = cost, R.copy()
    return best_R


# =====================================================================
# HORIZONTAL ALIGNMENT  (Paper §3.1, Equations 3-4)
# =====================================================================

def geometric_horizontal(obj_pts, tpl_pts, steps=None, tpl_tree=None):
    """Geometric horizontal alignment (Eq. 3) — fast version.

    Precomputes template KDTree once and reuses it across all theta steps.
    Uses downsampled Chamfer (256 pts) for the search, then returns best theta.
    """
    if steps is None:
        steps = THETA_STEPS
    thetas = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    if len(tpl_pts) > 256:
        tpl_sub = tpl_pts[np.random.choice(len(tpl_pts), 256, replace=False)]
    else:
        tpl_sub = tpl_pts
    if tpl_tree is None:
        tpl_tree = cKDTree(tpl_sub)
    Eg = np.array([chamfer_distance_fast((rot_y(t) @ obj_pts.T).T, tpl_sub, tree_b=tpl_tree) for t in thetas])
    return thetas[int(np.argmin(Eg))], thetas, Eg


def semantic_horizontal(obj_pts, tpl_pts, steps=None):
    """Joint semantic-geometric horizontal alignment (Eq. 4) — fast version.

    Key optimisations vs. the original:
    - Template KDTree and template segments computed ONCE, reused for all angles.
    - Uses fast downsampled Chamfer (256 pts) for the inner loop.
    - Object parts are pre-computed once and only rotated per angle (no re-clustering).
    """
    if steps is None:
        steps = THETA_STEPS

    # Precompute template tree & segments (done ONCE)
    tpl_sub = tpl_pts if len(tpl_pts) <= 256 else tpl_pts[np.random.choice(len(tpl_pts), 256, replace=False)]
    tpl_tree = cKDTree(tpl_sub)
    theta_g, thetas, Eg = geometric_horizontal(obj_pts, tpl_pts, steps, tpl_tree=tpl_tree)

    # Find geometric extrema Ω (local minima of E_g)
    omega = []
    for i in range(len(Eg)):
        if Eg[i] <= Eg[(i - 1) % len(Eg)] and Eg[i] <= Eg[(i + 1) % len(Eg)]:
            omega.append(thetas[i])
    if not omega:
        omega = [theta_g]

    # Segment template and object ONCE (rotate object parts per angle)
    tpl_parts = segment_semantic_parts(tpl_pts)
    obj_parts_base = segment_semantic_parts(obj_pts)

    best_theta, best_obj = theta_g, -np.inf
    for t in thetas:
        R = rot_y(t)
        # Rotate each object part (preserves cluster assignments)
        obj_parts_rotated = [(R @ part.T).T for part in obj_parts_base]

        # Semantic energy: average per-part fast Chamfer distance
        Es = match_parts_chamfer(tpl_parts, obj_parts_rotated)

        # Gaussian mixture centred at geometric extrema
        gauss = sum(
            np.exp(-0.5 * (min(abs(t - w), 2 * np.pi - abs(t - w))) ** 2)
            for w in omega
        )
        objective = np.exp(-Es) * gauss
        if objective > best_obj:
            best_obj, best_theta = objective, t
    return best_theta


# =====================================================================
# CANDIDATE ROTATION GENERATION  (Paper §3.1 — 5 poses)
# =====================================================================

def generate_five_candidates(mesh, tpl_pts, category=None):
    """Generate exactly 5 candidate rotations per object (paper §3.1).

    The set {R_S, R_G, R_invG, R_S·R_sup, R_S·R_Pca} covers:
      1. Horizontal-only (semantic)
      2. Horizontal-only (geometric)
      3. Horizontal-only (geometric + symmetry flip)
      4. Vertical (support) + horizontal (semantic)
      5. Vertical (PCA) + horizontal (semantic)

    Symmetry-aware: R_invG respects the category's symmetry angle instead
    of always flipping by 180°.
    """
    obj_pts = sample_points(mesh)

    # ---- Horizontal components ----
    theta_g, _, _ = geometric_horizontal(obj_pts, tpl_pts)
    R_G = rot_y(theta_g)

    # Symmetry-aware inverted geometric rotation
    sym = SYMMETRY_INFO.get(category, {}) if category else {}
    sym_angle = sym.get("angle", 0)
    if sym_angle >= 360:
        # Full rotational symmetry — 180° flip is redundant; use 90° offset
        R_invG = rot_y(theta_g + np.pi / 2)
    elif sym_angle > 0:
        R_invG = rot_y(theta_g + np.radians(sym_angle))
    else:
        R_invG = rot_y(theta_g + np.pi)

    theta_s = semantic_horizontal(obj_pts, tpl_pts)
    R_S = rot_y(theta_s)

    # ---- Vertical components ----
    R_sup = support_vertical(mesh)
    pts_after_sup = (R_sup @ obj_pts.T).T
    theta_s_sup = semantic_horizontal(pts_after_sup, tpl_pts)
    R_S_Sup = rot_y(theta_s_sup) @ R_sup

    R_pca = pca_vertical(obj_pts, tpl_pts)
    pts_after_pca = (R_pca @ obj_pts.T).T
    theta_s_pca = semantic_horizontal(pts_after_pca, tpl_pts)
    R_S_Pca = rot_y(theta_s_pca) @ R_pca

    return [
        ("R_S", R_S), ("R_G", R_G), ("R_invG", R_invG),
        ("R_S.R_Sup", R_S_Sup), ("R_S.R_Pca", R_S_Pca),
    ]


# =====================================================================
# QUALITY GATE  (Paper §3.2 — targets ~51% discard rate)
# =====================================================================

def passes_quality(mesh):
    """Enhanced quality screening matching the paper's filtering criteria.

    Checks:
      - Minimum face count
      - Thin-shell (degenerate bounding box)
      - Fragmented mesh (too many disconnected components)
      - Degenerate faces (extreme area variance)
    """
    # Basic geometry checks
    if not DOWNLOAD_OBJAVERSE or len(mesh.faces) == 0:
        return True, "ok"
    if len(mesh.faces) < MIN_FACES:
        return False, "too few faces"
    ext = mesh.bounding_box.extents
    if ext.max() <= 0 or (ext.min() / ext.max()) < MIN_THICKNESS_RATIO:
        return False, "thin-shell"

    # Connected component check (catches fragmented / broken meshes)
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda m: len(m.faces))
            if len(largest.faces) / len(mesh.faces) < MIN_CONNECTED_RATIO:
                return False, "fragmented"
    except Exception:
        pass  # some meshes resist splitting; allow them through

    # Degenerate face area check
    try:
        areas = mesh.area_faces
        if len(areas) > 10 and areas.mean() > 0:
            ratio = areas.max() / (areas.mean() + 1e-15)
            if ratio > MAX_FACE_AREA_VAR:
                return False, "degenerate faces"
    except Exception:
        pass

    return True, "ok"


# =====================================================================
# RENDERING
# =====================================================================

def render_mesh_png(mesh, R, out_path):
    """Render a mesh under rotation R and save to out_path as PNG."""
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
    
    fig = plt.figure(figsize=(RENDER_SIZE / 100, RENDER_SIZE / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    
    if len(faces) > 0:
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
        coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
        ax.add_collection3d(coll)
    else:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1, c='gray', depthshade=True)
    lim = np.abs(verts).max() * 0.7
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, transparent=False, facecolor="white")
    plt.close(fig)


def render_to_pil(mesh, R):
    """Render a mesh under rotation R and return as PIL Image (for CLIP)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        render_mesh_png(mesh, R, tmp_path)
        img = Image.open(tmp_path).convert("RGB")
        return img
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# =====================================================================
# CLIP-BASED CATEGORY DISCOVERY  (Uni3D proxy — paper §3.1)
# =====================================================================

class CategoryDiscovery:
    """Zero-shot category identification using multi-view CLIP.

    The paper uses Uni3D for category identification.  We approximate this
    with a multi-view CLIP approach (render from 4 viewpoints, average CLIP
    scores against "a 3D model of a [category]" text prompts).

    Uses HuggingFace `transformers` CLIPModel (pre-installed on Kaggle).
    """

    def __init__(self, categories=None, model_name=None):
        if categories is None:
            categories = CATEGORIES
        if model_name is None:
            model_name = CLIP_MODEL_NAME
        
        print("Initializing CLIP Model...")
        print("Categories: ", categories)
        self.categories = list(categories)
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._available = True

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
            print(f"  [CLIP] Loaded {self.model_name}")
        except Exception as e:
            print(f"  [!] CLIP unavailable ({e}). Category discovery disabled.")
            self._available = False

    def identify(self, mesh, n_views=4):
        """Identify the category of a 3D mesh via multi-view CLIP.

        Returns:
            (category_name, confidence, all_scores_dict)
        """
        self._load()
        if not self._available:
            return self.categories[0], 0.0, {}

        # Render from multiple viewpoints
        images = []
        for az_deg in np.linspace(0, 360, n_views, endpoint=False):
            R = rot_y(np.radians(az_deg))
            img = render_to_pil(mesh, R)
            images.append(img)

        # CLIP inference
        text_prompts = [f"a 3D model of a {cat}" for cat in self.categories]
        inputs = self._processor(
            text=text_prompts, images=images, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_image: (n_views, n_categories)
            logits = outputs.logits_per_image
            avg_logits = logits.mean(dim=0)  # (n_categories,)
            probs = torch.softmax(avg_logits, dim=0)

        best_idx = int(probs.argmax().item())
        scores_dict = {cat: float(p) for cat, p in zip(self.categories, probs.tolist())}
        return self.categories[best_idx], float(probs[best_idx].item()), scores_dict


# =====================================================================
# OBJAVERSE DATA LOADING
# =====================================================================

def load_category_uids(category, n_needed=None):
    """Fetch object UIDs from Objaverse with strict per-category filtering, or from local ModelNet40 if disabled."""
    if not DOWNLOAD_OBJAVERSE:
        pattern = f"/kaggle/input/datasets/tasnimulislammahi/modelnet40/modelnet40_normal_resampled/{category}/*.txt"
        files = glob.glob(pattern)
        if not files:
            # Fallback path if Kaggle structures it slightly differently
            pattern2 = f"/kaggle/input/*/modelnet40_normal_resampled/{category}/*.txt"
            files = glob.glob(pattern2)
        print(f"  Found {len(files)} local '{category}' files.")
        if n_needed is not None:
            return files[:n_needed]
        return files

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
        if n_needed is not None and len(selected_uids) >= n_needed:
            break

    print(f"  Found {len(selected_uids)} '{category}' objects.")
    return selected_uids


def download_objects(uids):
    """Download 3D objects from Objaverse by UID safely, or return local paths if disabled."""
    if not DOWNLOAD_OBJAVERSE:
        # uids are already local file paths when DOWNLOAD_OBJAVERSE is False
        return {uid: uid for uid in uids}

    import objaverse
    try:
        objs = objaverse.load_objects(uids=uids, download_processes=1)
        return objs
    except Exception as e:
        print(f"  [!] Batch download error: {e}. Trying item-by-item...")
        objs = {}
        for uid in tqdm(uids, desc="  Item-by-item download", leave=False):
            try:
                res = objaverse.load_objects(uids=[uid], download_processes=1)
                objs.update(res)
            except Exception as item_err:
                print(f"  [!] Skipping {uid}: {item_err}")
        return objs


# =====================================================================
# SELF-SUPERVISED LABELING
# =====================================================================

def compute_auto_label(candidate_pts_list, template_pts, candidate_normals_list=None, template_normals=None):
    """Pick candidate with lowest Chamfer distance to template (self-supervised)."""
    distances = []
    for i, cpts in enumerate(candidate_pts_list):
        cn = candidate_normals_list[i] if candidate_normals_list else None
        d = chamfer_distance(cpts, template_pts, na=cn, nb=template_normals)
        distances.append(d)
    return int(np.argmin(distances)), distances


# =====================================================================
# PYTORCH DATASETS
# =====================================================================

class CandidateDataset(Dataset):
    """Per-category pairwise dataset: one sample per (object, candidate) pair.

    Used for per-category BCE-based training.  Supports SO(3) rotation
    augmentation (paper Supp §2.1: 40 random rotations per sample).
    """
    def __init__(self, template_pts, objects, augment=False, n_rotations=None):
        if n_rotations is None:
            n_rotations = N_ROTATIONS
        self.template_pts = torch.tensor(template_pts, dtype=torch.float32)
        self.augment = augment
        self.n_rotations = n_rotations if augment else 1
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
        return len(self.samples) * self.n_rotations

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        s = self.samples[sample_idx]

        template = self.template_pts.clone()
        candidate = s["candidate_pts"].clone()

        if self.augment and self.n_rotations > 1:
            # Apply the SAME random rotation to both template and candidate.
            # Since VNN is SO(3)-equivariant, this preserves relative orientation.
            R = torch.tensor(random_rotation_matrix(), dtype=torch.float32)
            template = template @ R.T
            candidate = candidate @ R.T

        return {
            "template_pts": template,
            "candidate_pts": candidate,
            "label": s["label"],
            "object_idx": s["object_idx"],
            "candidate_idx": s["candidate_idx"],
        }


class AgnosticRankingDataset(Dataset):
    """Cross-category ranking dataset for category-agnostic training.

    Each sample contains ALL 5 candidates for one object plus the object's
    per-category template.  The model scores each candidate and the loss is
    5-way cross-entropy (ranking loss), which directly optimises selection
    accuracy.

    Supports SO(3) rotation augmentation and per-object category index.
    """
    def __init__(self, all_category_data, augment=False, n_rotations=None):
        """
        Args:
            all_category_data: list of dicts, each with keys
                "category", "template_pts", "objects"
            augment: whether to apply random SO(3) rotations
            n_rotations: augmentation factor
        """
        if n_rotations is None:
            n_rotations = N_ROTATIONS
        self.augment = augment
        self.n_rotations = n_rotations if augment else 1
        self.category_to_idx = {}
        self.items = []

        for cat_data in all_category_data:
            category = cat_data["category"]
            if category not in self.category_to_idx:
                self.category_to_idx[category] = len(self.category_to_idx)
            cat_idx = self.category_to_idx[category]
            tpl = torch.tensor(cat_data["template_pts"], dtype=torch.float32)

            for obj in cat_data["objects"]:
                cand_pts = torch.stack([
                    torch.tensor(obj["candidate_pts"][i], dtype=torch.float32)
                    for i in range(5)
                ])  # (5, N_POINTS, 3)
                self.items.append({
                    "template_pts": tpl,
                    "candidate_pts": cand_pts,
                    "best_idx": obj["best_idx"],
                    "category_idx": cat_idx,
                    "category": category,
                })

    def __len__(self):
        return len(self.items) * self.n_rotations

    def __getitem__(self, idx):
        item_idx = idx % len(self.items)
        item = self.items[item_idx]

        template = item["template_pts"].clone()        # (N, 3)
        candidates = item["candidate_pts"].clone()      # (5, N, 3)

        if self.augment and self.n_rotations > 1:
            R = torch.tensor(random_rotation_matrix(), dtype=torch.float32)
            template = template @ R.T
            candidates = candidates @ R.T  # (5, N, 3) @ (3, 3) broadcasts correctly

        return {
            "template_pts": template,
            "candidate_pts": candidates,
            "best_idx": torch.tensor(item["best_idx"], dtype=torch.long),
            "category_idx": torch.tensor(item["category_idx"], dtype=torch.long),
        }


# =====================================================================
# TRAINING DATA PREPARATION (SELF-SUPERVISED)
# =====================================================================

def prepare_training_data(category, n_objects=None, force_rebuild=False):
    """Prepare training data with self-supervised labels for a category."""
    if n_objects is None and USE_ONLY_N_SAMPLES:
        n_objects = N_SAMPLES
    cache_file = get_cache_file(category)

    if not force_rebuild and os.path.exists(cache_file):
        print(f"\n  Loading cached data from {cache_file}...")
        try:
            return torch.load(cache_file, weights_only=False)
        except (RuntimeError, EOFError, Exception) as e:
            print(f"  [!] Cache file is corrupt ({type(e).__name__}: {e}). Deleting and rebuilding...")
            os.remove(cache_file)
            # Fall through to rebuild below

    print("\n" + "=" * 60)
    print(f"  PREPARING TRAINING DATA [{category.upper()}]")
    print("=" * 60)

    needed = (n_objects + 1) if (USE_ONLY_N_SAMPLES and n_objects is not None) else None
    uids = load_category_uids(category, (needed + 50) if needed else None)
    if len(uids) < 10:
        raise RuntimeError(f"Only found {len(uids)} '{category}' objects. Need >= 10.")

    random.shuffle(uids)

    download_uids = uids[:min(len(uids), needed + 50)] if needed else uids
    if DOWNLOAD_OBJAVERSE:
        print(f"\n  Downloading {len(download_uids)} objects...")
    else:
        print(f"\n  Resolving {len(download_uids)} local ModelNet40 objects...")
    objs = download_objects(download_uids)

    print("\n  Setting up canonical template (Medoid Selection)...")
    # Evaluate first 15 objects to find the "most average" (medoid) template
    num_medoid_cands = min(15, len(uids))
    medoid_pts_list = []
    medoid_uids = uids[:num_medoid_cands]
    
    for uid in medoid_uids:
        tpl_path = objs.get(uid)
        if tpl_path is None: continue
        m = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
        pts = sample_points(m, seed=42, with_normals=False)
        medoid_pts_list.append((uid, pts))
        
    best_medoid_uid = medoid_uids[0]
    best_medoid_score = float('inf')
    for i, (uid_i, pts_i) in enumerate(medoid_pts_list):
        score = 0
        for j, (uid_j, pts_j) in enumerate(medoid_pts_list):
            if i != j:
                score += chamfer_distance(pts_i, pts_j)
        if score < best_medoid_score:
            best_medoid_score = score
            best_medoid_uid = uid_i

    template_uid = best_medoid_uid
    tpl_path = objs.get(template_uid)
    if tpl_path is None:
        raise RuntimeError(f"Template {template_uid} failed to download!")
    tpl_mesh = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
    template_pts, template_normals = sample_points(tpl_mesh, seed=42, with_normals=True)
    print(f"  Template UID selected: {template_uid[:16]}...")

    target_str = f"{n_objects}" if (USE_ONLY_N_SAMPLES and n_objects is not None) else f"ALL ({len(uids)-1})"
    print(f"\n  Processing objects (target: {target_str})...")
    data_objects = []
    all_best_dists = []
    discarded = 0

    total_items = (min(len(uids)-1, n_objects+50)) if (USE_ONLY_N_SAMPLES and n_objects is not None) else (len(uids)-1)
    pbar = tqdm(uids[1:], desc="  Processing", total=total_items, unit="obj")
    for uid in pbar:
        if USE_ONLY_N_SAMPLES and n_objects is not None and len(data_objects) >= n_objects:
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
            mesh = normalize_to_unit_sphere(load_mesh_or_pc(path))
        except Exception:
            continue
        ok, reason = passes_quality(mesh)
        if not ok:
            discarded += 1
            pbar.set_postfix(good=len(data_objects), bad=discarded)
            continue

        candidates = generate_five_candidates(mesh, template_pts, category=category)
        obj_pts, obj_normals = sample_points(mesh, with_normals=True)
        cand_pts_list, cand_normals_list, cand_rot_list, cand_name_list = [], [], [], []
        for name, R in candidates:
            cand_pts_list.append((R @ obj_pts.T).T)
            cand_normals_list.append((R @ obj_normals.T).T)
            cand_rot_list.append(R)
            cand_name_list.append(name)

        best_idx, cd_dists = compute_auto_label(
            cand_pts_list, template_pts,
            candidate_normals_list=cand_normals_list,
            template_normals=template_normals
        )
        
        all_best_dists.append(cd_dists[best_idx])
            
        safe_uid = os.path.basename(uid)
        data_objects.append({
            "uid": safe_uid,
            "candidate_pts": cand_pts_list,
            "candidate_rotations": cand_rot_list,
            "candidate_names": cand_name_list,
            "best_idx": best_idx,
            "chamfer_distances": cd_dists,
        })
        pbar.set_postfix(good=len(data_objects), bad=discarded)
    pbar.close()

    # ---- Adaptive distance threshold filtering (lenient: median + 3σ) ----
    if all_best_dists:
        dist_arr = np.array(all_best_dists)
        threshold = np.median(dist_arr) + 3.0 * np.std(dist_arr)
        print(f"  Adaptive threshold: {threshold:.4f} (median={np.median(dist_arr):.4f}, std={np.std(dist_arr):.4f})")
        before_count = len(data_objects)
        filtered_objects = []
        for obj, d in zip(data_objects, all_best_dists):
            if d <= threshold:
                filtered_objects.append(obj)
            else:
                discarded += 1
        data_objects = filtered_objects
        print(f"  Adaptive filter removed {before_count - len(data_objects)} outliers.")

    print(f"\n  Final: {len(data_objects)} objects, discarded {discarded} total.")
    if discarded + len(data_objects) > 0:
        print(f"  Discard rate: {discarded / (discarded + len(data_objects)) * 100:.1f}%")
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
# TRAINING FUNCTIONS  (Per-Category BCE Mode)
# =====================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train one epoch with pairwise BCE loss."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
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
    return {"loss": total_loss / total if total else 0, "accuracy": correct / total if total else 0}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, phase="VAL"):
    """Evaluate with pairwise BCE loss + compute object-level selection accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    object_scores = {}
    for batch in dataloader:
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


# =====================================================================
# TRAINING FUNCTIONS  (Agnostic Ranking Mode — 5-way Cross-Entropy)
# =====================================================================

def train_one_epoch_ranking(model, dataloader, optimizer, device, epoch,
                            use_category=False):
    """Train one epoch with ranking loss (5-way cross-entropy over candidates)."""
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        template = batch["template_pts"].to(device)      # (B, N, 3)
        candidates = batch["candidate_pts"].to(device)    # (B, 5, N, 3)
        best_idx = batch["best_idx"].to(device)           # (B,)
        cat_idx = batch["category_idx"].to(device) if use_category else None

        B = template.shape[0]

        # Encode template ONCE, reuse for all 5 candidates (saves 40% encoder compute)
        feat_t = model.encoder(template)  # (B, C, 3)

        logits_list = []
        for c in range(5):
            s = model.forward_with_cached_template(feat_t, candidates[:, c], category_idx=cat_idx)
            logits_list.append(s)
        logits = torch.cat(logits_list, dim=1)  # (B, 5)

        loss = F.cross_entropy(logits, best_idx, label_smoothing=0.05)

        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=1)
        correct += (preds == best_idx).sum().item()
        total += B
        total_loss += loss.item() * B

    if len(dataloader) % GRAD_ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {"loss": total_loss / total if total else 0, "selection_accuracy": correct / total if total else 0}


@torch.no_grad()
def evaluate_ranking(model, dataloader, device, phase="VAL", use_category=False):
    """Evaluate with ranking loss — reports selection accuracy."""
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        template = batch["template_pts"].to(device)
        candidates = batch["candidate_pts"].to(device)
        best_idx = batch["best_idx"].to(device)
        cat_idx = batch["category_idx"].to(device) if use_category else None

        B = template.shape[0]

        # Encode template ONCE, reuse for all 5 candidates
        feat_t = model.encoder(template)

        logits_list = []
        for c in range(5):
            s = model.forward_with_cached_template(feat_t, candidates[:, c], category_idx=cat_idx)
            logits_list.append(s)
        logits = torch.cat(logits_list, dim=1)

        loss = F.cross_entropy(logits, best_idx)
        preds = logits.argmax(dim=1)

        correct += (preds == best_idx).sum().item()
        total += B
        total_loss += loss.item() * B

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(best_idx.cpu().tolist())

    return {
        "loss": total_loss / total if total else 0,
        "selection_accuracy": correct / total if total else 0,
        "selection_correct": correct,
        "selection_total": total,
        "predictions": all_preds,
        "labels": all_labels,
    }


# =====================================================================
# GEODESIC EVALUATION METRICS  (Paper §4 — Acc@10, Acc@30)
# =====================================================================

def compute_geodesic_metrics(predicted_rotations, gt_rotations):
    """Compute geodesic error metrics between predicted and ground-truth rotations.

    Returns:
        dict with acc_at_10, acc_at_30, median_error_deg, mean_error_deg
    """
    errors = []
    for R_pred, R_gt in zip(predicted_rotations, gt_rotations):
        errors.append(geodesic_angle_deg(
            np.asarray(R_pred, dtype=np.float64),
            np.asarray(R_gt, dtype=np.float64),
        ))
    errors = np.array(errors)
    return {
        "acc_at_10": float((errors <= 10).mean()) if len(errors) else 0.0,
        "acc_at_30": float((errors <= 30).mean()) if len(errors) else 0.0,
        "median_error_deg": float(np.median(errors)) if len(errors) else 0.0,
        "mean_error_deg": float(np.mean(errors)) if len(errors) else 0.0,
        "n_samples": len(errors),
    }


# =====================================================================
# DATA SPLITTING
# =====================================================================

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
# PER-CATEGORY TRAINING
# =====================================================================

def train_category(category):
    """Train VNN for a single category with self-supervised labels.

    Uses pairwise BCE loss with SO(3) rotation augmentation.
    """
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

    # Training set WITH augmentation, val/test WITHOUT
    train_loader = DataLoader(
        CandidateDataset(template_pts, train_objs, augment=True, n_rotations=N_ROTATIONS),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        CandidateDataset(template_pts, val_objs, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        CandidateDataset(template_pts, test_objs, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = VNCandidateSelector(n_points=N_POINTS, feat_dims=FEAT_DIMS,
                                hidden_dim=HIDDEN_DIM).to(device)
    print(f"\n{model_summary(model)}")
    # 1 positive and 4 negative candidates per object → weight positive by 4.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    print("\n" + "-" * 60)
    print(f"  TRAINING [{category.upper()}]  (augment={N_ROTATIONS}x rotations)")
    print("-" * 60)

    best_model_path = get_best_model_path(category)
    latest_model_path = get_latest_model_path(category)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_sel_acc = -1.0
    patience_counter = 0
    start_epoch = 1

    if os.path.exists(latest_model_path):
        try:
            print(f"\n  [+] Found checkpoint at {latest_model_path}. Resuming training...")
            ckpt = torch.load(latest_model_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_sel_acc = ckpt.get("best_val_sel_acc", ckpt.get("val_selection_accuracy", -1.0))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"  [+] Resumed from epoch {start_epoch}/{NUM_EPOCHS} (Best Val SelAcc: {best_val_sel_acc:.3f}, Patience: {patience_counter}/{PATIENCE})")
        except Exception as e:
            print(f"  [!] Failed to load resume checkpoint ({e}). Starting fresh...")

    print(f"\n  [INFO] Checkpoint Path: /kaggle/working/{latest_model_path}")
    print("  [INFO] Download Location: Kaggle Right Panel -> Output -> checkpoints/")
    print("  [INFO] Advantage: Auto-saves every epoch so you can resume training seamlessly after any crash without losing progress.")

    train_start = time.time()

    if start_epoch <= NUM_EPOCHS:
        epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS + 1), desc=f"  [{category.upper()}] Training", unit="epoch")
        for epoch in epoch_pbar:
            t0 = time.time()
            tm = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            vm = evaluate(model, val_loader, criterion, device, "VAL")
            scheduler.step(vm["selection_accuracy"])
            et = time.time() - t0

            epoch_pbar.set_postfix(
                tr_loss=f"{tm['loss']:.3f}",
                tr_acc=f"{tm['accuracy']:.3f}",
                val_sel=f"{vm['selection_accuracy']:.3f}"
            )

            is_best = vm["selection_accuracy"] > best_val_sel_acc
            if is_best:
                best_val_sel_acc = vm["selection_accuracy"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_selection_accuracy": best_val_sel_acc,
                    "feat_dims": FEAT_DIMS, "hidden_dim": HIDDEN_DIM,
                    "n_points": N_POINTS, "category": category,
                    "n_categories": 0,
                }, best_model_path)
                epoch_pbar.write(
                    f"  * Epoch {epoch:3d}/{NUM_EPOCHS} [NEW BEST] | "
                    f"TrLoss: {tm['loss']:.4f} Acc: {tm['accuracy']:.3f} | "
                    f"VaLoss: {vm['loss']:.4f} SelAcc: {best_val_sel_acc:.3f} "
                    f"({vm['selection_correct']}/{vm['selection_total']}) | {et:.1f}s"
                )
            else:
                patience_counter += 1

            # Save latest checkpoint after EVERY epoch for instant resumption if crashed
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_sel_acc": best_val_sel_acc,
                "patience_counter": patience_counter,
                "feat_dims": FEAT_DIMS, "hidden_dim": HIDDEN_DIM,
                "n_points": N_POINTS, "category": category,
                "n_categories": 0,
            }, latest_model_path)

            if patience_counter >= PATIENCE:
                epoch_pbar.write(f"  [!] Early stopping at epoch {epoch}")
                break
        epoch_pbar.close()
    else:
        print(f"\n  [+] Training already completed up to epoch {NUM_EPOCHS}.")

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
# CATEGORY-AGNOSTIC TRAINING  (Cross-Category Ranking)
# =====================================================================

def train_agnostic(categories=None):
    """Train a single category-agnostic VNN across all categories.

    Uses ranking loss (5-way cross-entropy) with category embeddings and
    SO(3) rotation augmentation.  Produces a single checkpoint that works
    for any category (including unseen ones via zero-shot).
    """
    if categories is None:
        categories = CATEGORIES

    print("\n" + "=" * 60)
    print("  CATEGORY-AGNOSTIC VNN TRAINING")
    print(f"  Categories: {', '.join(categories)}")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    total_start = time.time()

    # ---- Prepare data for all categories ----
    print("\n  Preparing data for all categories...")
    all_data = []
    for cat in categories:
        data = prepare_training_data(cat)
        all_data.append(data)

    # ---- Split per-category, then pool ----
    train_cat_data, val_cat_data, test_cat_data = [], [], []
    for data in all_data:
        train_objs, val_objs, test_objs = split_data(data["objects"])
        for split_list, split_objs in [
            (train_cat_data, train_objs),
            (val_cat_data, val_objs),
            (test_cat_data, test_objs),
        ]:
            split_list.append({
                "category": data["category"],
                "template_pts": data["template_pts"],
                "objects": split_objs,
            })

    # ---- Build datasets ----
    train_ds = AgnosticRankingDataset(train_cat_data, augment=True, n_rotations=N_ROTATIONS)
    val_ds = AgnosticRankingDataset(val_cat_data, augment=False)
    test_ds = AgnosticRankingDataset(test_cat_data, augment=False)

    n_cats = len(train_ds.category_to_idx)
    print(f"\n  Total train samples: {len(train_ds)} ({len(train_ds)//N_ROTATIONS} base × {N_ROTATIONS} aug)")
    print(f"  Total val samples:   {len(val_ds)}")
    print(f"  Total test samples:  {len(test_ds)}")
    print(f"  Category count:      {n_cats}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    # ---- Build model ----
    model = VNCandidateSelector(
        n_points=N_POINTS,
        feat_dims=AGNOSTIC_FEAT_DIMS,
        hidden_dim=AGNOSTIC_HIDDEN_DIM,
        n_categories=n_cats,
        category_embed_dim=N_CATEGORY_EMBED,
    ).to(device)
    print(f"\n{model_summary(model)}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    agnostic_path = get_agnostic_model_path()
    latest_agnostic_path = get_agnostic_latest_model_path()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_sel_acc = -1.0
    patience_counter = 0
    start_epoch = 1

    if os.path.exists(latest_agnostic_path):
        try:
            print(f"\n  [+] Found checkpoint at {latest_agnostic_path}. Resuming training...")
            ckpt = torch.load(latest_agnostic_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_sel_acc = ckpt.get("best_val_sel_acc", ckpt.get("val_selection_accuracy", -1.0))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"  [+] Resumed from epoch {start_epoch}/{NUM_EPOCHS} (Best Val SelAcc: {best_val_sel_acc:.3f}, Patience: {patience_counter}/{PATIENCE})")
        except Exception as e:
            print(f"  [!] Failed to load resume checkpoint ({e}). Starting fresh...")

    print(f"\n  [INFO] Checkpoint Path: /kaggle/working/{latest_agnostic_path}")
    print("  [INFO] Download Location: Kaggle Right Panel -> Output -> checkpoints/")
    print("  [INFO] Advantage: Auto-saves every epoch so you can resume training seamlessly after any crash without losing progress.")

    train_start = time.time()

    if start_epoch <= NUM_EPOCHS:
        epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS + 1), desc="  [AGNOSTIC] Training", unit="epoch")
        for epoch in epoch_pbar:
            t0 = time.time()
            tm = train_one_epoch_ranking(model, train_loader, optimizer, device, epoch,
                                         use_category=True)
            vm = evaluate_ranking(model, val_loader, device, "VAL", use_category=True)
            scheduler.step(vm["selection_accuracy"])
            et = time.time() - t0

            epoch_pbar.set_postfix(
                tr_loss=f"{tm['loss']:.3f}",
                tr_sel=f"{tm['selection_accuracy']:.3f}",
                val_sel=f"{vm['selection_accuracy']:.3f}"
            )

            is_best = vm["selection_accuracy"] > best_val_sel_acc
            if is_best:
                best_val_sel_acc = vm["selection_accuracy"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_selection_accuracy": best_val_sel_acc,
                    "feat_dims": AGNOSTIC_FEAT_DIMS,
                    "hidden_dim": AGNOSTIC_HIDDEN_DIM,
                    "n_points": N_POINTS,
                    "n_categories": n_cats,
                    "category_embed_dim": N_CATEGORY_EMBED,
                    "category_to_idx": train_ds.category_to_idx,
                    "categories": categories,
                }, agnostic_path)
                epoch_pbar.write(
                    f"  * Epoch {epoch:3d}/{NUM_EPOCHS} [NEW BEST] | "
                    f"TrLoss: {tm['loss']:.4f} TrSel: {tm['selection_accuracy']:.3f} | "
                    f"VaLoss: {vm['loss']:.4f} ValSel: {best_val_sel_acc:.3f} "
                    f"({vm['selection_correct']}/{vm['selection_total']}) | {et:.1f}s"
                )
            else:
                patience_counter += 1

            # Save latest checkpoint after EVERY epoch for instant resumption if crashed
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_sel_acc": best_val_sel_acc,
                "patience_counter": patience_counter,
                "feat_dims": AGNOSTIC_FEAT_DIMS,
                "hidden_dim": AGNOSTIC_HIDDEN_DIM,
                "n_points": N_POINTS,
                "n_categories": n_cats,
                "category_embed_dim": N_CATEGORY_EMBED,
                "category_to_idx": train_ds.category_to_idx,
                "categories": categories,
            }, latest_agnostic_path)

            if patience_counter >= PATIENCE:
                epoch_pbar.write(f"  [!] Early stopping at epoch {epoch}")
                break
        epoch_pbar.close()
    else:
        print(f"\n  [+] Training already completed up to epoch {NUM_EPOCHS}.")

    train_time = time.time() - train_start
    print(f"\n  [T] Training: {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  Best val SelAcc: {best_val_sel_acc:.3f}")

    # ---- Test evaluation ----
    print("\n" + "-" * 60)
    print("  TEST EVALUATION [AGNOSTIC]")
    print("-" * 60)

    ckpt = torch.load(agnostic_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_m = evaluate_ranking(model, test_loader, device, "TEST", use_category=True)
    print(f"\n  Test Results:")
    print(f"    Selection Accuracy: {test_m['selection_accuracy']:.3f} "
          f"({test_m['selection_correct']}/{test_m['selection_total']})")
    print(f"    Loss:               {test_m['loss']:.4f}")

    # ---- Per-category breakdown ----
    print("\n  Per-Category Breakdown:")
    for cat_data_entry in test_cat_data:
        cat = cat_data_entry["category"]
        cat_ds = AgnosticRankingDataset([cat_data_entry], augment=False)
        if len(cat_ds) == 0:
            continue
        cat_loader = DataLoader(cat_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        cat_m = evaluate_ranking(model, cat_loader, device, cat.upper(), use_category=True)
        print(f"    {cat:12s}: SelAcc {cat_m['selection_accuracy']:.3f} "
              f"({cat_m['selection_correct']}/{cat_m['selection_total']})")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  AGNOSTIC TRAINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Model: {agnostic_path}")
    print(f"{'='*60}")


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

    # ---- Load VNN (try agnostic first, then per-category) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vnn_model_obj = None
    vnn_available = False
    vnn_cat_idx = None

    # Try agnostic model
    agnostic_path = get_agnostic_model_path()
    if os.path.exists(agnostic_path):
        ckpt = torch.load(agnostic_path, weights_only=False, map_location=device)
        cat_to_idx = ckpt.get("category_to_idx", {})
        vnn_model_obj = VNCandidateSelector(
            n_points=ckpt.get("n_points", N_POINTS),
            feat_dims=ckpt.get("feat_dims", AGNOSTIC_FEAT_DIMS),
            hidden_dim=ckpt.get("hidden_dim", AGNOSTIC_HIDDEN_DIM),
            n_categories=ckpt.get("n_categories", 0),
            category_embed_dim=ckpt.get("category_embed_dim", N_CATEGORY_EMBED),
        ).to(device)
        vnn_model_obj.load_state_dict(ckpt["model_state_dict"])
        vnn_model_obj.eval()
        vnn_available = True
        if category in cat_to_idx:
            vnn_cat_idx = cat_to_idx[category]
        print(f"  [OK] Loaded AGNOSTIC VNN from {agnostic_path}")
    else:
        # Fall back to per-category model
        best_model_path = get_best_model_path(category)
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, weights_only=False, map_location=device)
            vnn_model_obj = VNCandidateSelector(
                n_points=ckpt.get("n_points", N_POINTS),
                feat_dims=ckpt.get("feat_dims", FEAT_DIMS),
                hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                n_categories=ckpt.get("n_categories", 0),
            ).to(device)
            vnn_model_obj.load_state_dict(ckpt["model_state_dict"])
            vnn_model_obj.eval()
            vnn_available = True
            print(f"  [OK] Loaded per-category VNN from {best_model_path}")

    if not vnn_available:
        print(f"  [!] No VNN checkpoint. Generating without auto-selection.")

    # Template
    template_uid = uids[0]
    objs = download_objects([template_uid])
    tpl_path = objs.get(template_uid)
    if tpl_path is None:
        print(f"  [!] Template download failed. Skipping.")
        return 0, 0
    tpl_mesh = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
    tpl_pts = sample_points(tpl_mesh)
    render_mesh_png(tpl_mesh, np.eye(3), os.path.join(out_dir, "template.png"))
    print(f"  Template = {template_uid}")

    manifest = {"category": category, "template": "template.png",
                "vnn_enabled": vnn_available, "objects": []}
    prepared, discarded = 0, 0

    total_items = min(len(uids)-1, N_GENERATE)
    pbar = tqdm(uids[1:], desc=f"  [{category}]", total=total_items, unit="obj")
    for uid in pbar:
        if prepared >= N_GENERATE:
            break
        path = objs.get(uid) or download_objects([uid]).get(uid)
        if path is None:
            continue
        try:
            mesh = normalize_to_unit_sphere(load_mesh_or_pc(path))
        except Exception:
            continue
        ok, _ = passes_quality(mesh)
        if not ok:
            discarded += 1
            pbar.set_postfix(good=prepared, bad=discarded, status="skip")
            continue

        # Sanitize UID for file paths (handles absolute local paths)
        safe_uid = os.path.basename(uid).replace('.txt', '')
        
        obj_dir = os.path.join(out_dir, safe_uid)
        os.makedirs(obj_dir, exist_ok=True)
        candidates = generate_five_candidates(mesh, tpl_pts, category=category)

        cand_records, cand_rotations = [], []
        for i, (name, R) in enumerate(candidates):
            fn = f"{safe_uid}/cand_{i}.png"
            render_mesh_png(mesh, R, os.path.join(out_dir, fn))
            cand_records.append({"index": i, "name": name, "image": fn, "rotation": R.tolist()})
            cand_rotations.append(R)

        vnn_selection = None
        if vnn_available:
            obj_pts = sample_points(mesh)
            tpl_tensor = torch.tensor(tpl_pts, dtype=torch.float32).unsqueeze(0).to(device)
            cat_tensor = (torch.tensor([vnn_cat_idx], dtype=torch.long).to(device)
                          if vnn_cat_idx is not None else None)
            scores = []
            with torch.no_grad():
                for R in cand_rotations:
                    rotated = (R @ obj_pts.T).T
                    ct = torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).to(device)
                    s = vnn_model_obj(tpl_tensor, ct, category_idx=cat_tensor)
                    scores.append(s.item())
            best_idx = int(np.argmax(scores))

            correct_dir = os.path.join(obj_dir, "correct")
            rejected_dir = os.path.join(obj_dir, "rejected")
            os.makedirs(correct_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)
            shutil.copy2(os.path.join(out_dir, f"{safe_uid}/cand_{best_idx}.png"),
                         os.path.join(correct_dir, f"cand_{best_idx}.png"))
            for i in range(5):
                if i != best_idx:
                    shutil.copy2(os.path.join(out_dir, f"{safe_uid}/cand_{i}.png"),
                                 os.path.join(rejected_dir, f"cand_{i}.png"))

            vnn_selection = {"selected_index": best_idx, "selected_name": candidates[best_idx][0],
                             "scores": scores, "selected_score": scores[best_idx]}
            pbar.set_postfix(good=prepared+1, bad=discarded, vnn=f"c{best_idx}({scores[best_idx]:.2f})")

        entry = {"uid": safe_uid, "candidates": cand_records}
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
# FULL AUTOMATION -- align_single_object()
# =====================================================================

def align_single_object(mesh_or_path, template_mesh_or_pts=None, category=None):
    """Fully automatic canonical alignment of a single 3D object.

    Pipeline:
      1. Load mesh if path is given
      2. Auto-detect category via CLIP (if no category specified)
      3. Generate 5 candidate rotations
      4. Score with best available VNN (agnostic → per-category → Chamfer fallback)
      5. Return (best_rotation, candidate_name, candidate_index)
    """
    # ---- Load mesh ----
    if isinstance(mesh_or_path, str):
        mesh = normalize_to_unit_sphere(load_mesh_or_pc(mesh_or_path))
    else:
        mesh = mesh_or_path

    # ---- Auto-detect category ----
    if category is None:
        discovery = CategoryDiscovery()
        category, confidence, all_scores = discovery.identify(mesh)
        print(f"  Auto-detected category: {category} (confidence: {confidence:.2f})")
        if all_scores:
            top3 = sorted(all_scores.items(), key=lambda x: -x[1])[:3]
            print(f"  Top-3: {', '.join(f'{c}: {s:.2f}' for c, s in top3)}")

    # ---- Get template ----
    if template_mesh_or_pts is not None:
        if isinstance(template_mesh_or_pts, np.ndarray):
            template_pts = template_mesh_or_pts
        else:
            template_pts = sample_points(normalize_to_unit_sphere(template_mesh_or_pts))
    else:
        cache_file = get_cache_file(category)
        if os.path.exists(cache_file):
            data = torch.load(cache_file, weights_only=False)
            template_pts = data["template_pts"]
        else:
            raise ValueError(
                f"No template for '{category}'. Run prepare_training_data('{category}') first, "
                f"or pass template_mesh_or_pts explicitly."
            )

    # ---- Generate candidates ----
    candidates = generate_five_candidates(mesh, template_pts, category=category)
    obj_pts = sample_points(mesh)

    # ---- Score with VNN ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try agnostic model first
    model_loaded = False
    agnostic_path = get_agnostic_model_path()
    if os.path.exists(agnostic_path):
        ckpt = torch.load(agnostic_path, weights_only=False, map_location=device)
        cat_to_idx = ckpt.get("category_to_idx", {})
        model = VNCandidateSelector(
            n_points=ckpt.get("n_points", N_POINTS),
            feat_dims=ckpt.get("feat_dims", AGNOSTIC_FEAT_DIMS),
            hidden_dim=ckpt.get("hidden_dim", AGNOSTIC_HIDDEN_DIM),
            n_categories=ckpt.get("n_categories", 0),
            category_embed_dim=ckpt.get("category_embed_dim", N_CATEGORY_EMBED),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        cat_idx_tensor = None
        if category in cat_to_idx:
            cat_idx_tensor = torch.tensor([cat_to_idx[category]], dtype=torch.long).to(device)
        model_loaded = True
        print(f"  Using agnostic VNN")

    if not model_loaded:
        per_cat_path = get_best_model_path(category)
        if os.path.exists(per_cat_path):
            ckpt = torch.load(per_cat_path, weights_only=False, map_location=device)
            model = VNCandidateSelector(
                n_points=ckpt.get("n_points", N_POINTS),
                feat_dims=ckpt.get("feat_dims", FEAT_DIMS),
                hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                n_categories=ckpt.get("n_categories", 0),
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            cat_idx_tensor = None
            model_loaded = True
            print(f"  Using per-category VNN for '{category}'")

    if model_loaded:
        tpl_tensor = torch.tensor(template_pts, dtype=torch.float32).unsqueeze(0).to(device)
        scores = []
        with torch.no_grad():
            for name, R in candidates:
                rotated = (R @ obj_pts.T).T
                ct = torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).to(device)
                s = model(tpl_tensor, ct, category_idx=cat_idx_tensor)
                scores.append(s.item())
        best_idx = int(np.argmax(scores))
        print(f"  VNN scores: {[f'{s:.3f}' for s in scores]}")
    else:
        # Fallback: pure Chamfer distance
        print(f"  [!] No VNN available — using Chamfer distance fallback")
        cand_pts = [(R @ obj_pts.T).T for _, R in candidates]
        best_idx, scores = compute_auto_label(cand_pts, template_pts)
        scores = [-d for d in scores]  # negate so higher = better

    best_name, best_R = candidates[best_idx]
    print(f"  Selected: {best_name} (index {best_idx})")
    return best_R, best_name, best_idx


# =====================================================================
# METRICS EVALUATION (Stability & Consistency)
# =====================================================================

def evaluate_stability_consistency(category, num_objects=10, k_rotations=5):
    """Evaluate per-category Stability and Consistency metrics."""
    print(f"\n" + "=" * 60)
    print(f"  EVALUATING STABILITY & CONSISTENCY [{category.upper()}]")
    print(f"=" * 60)

    uids = load_category_uids(category, num_objects + 10)
    if not uids or len(uids) < 2:
        print(f"  [!] Not enough '{category}' objects found.")
        return 0, 0
        
    random.shuffle(uids)
    download_uids = uids[:min(len(uids), num_objects + 10)]
    objs = download_objects(download_uids)

    cache_file = get_cache_file(category)
    if os.path.exists(cache_file):
        data = torch.load(cache_file, weights_only=False)
        template_pts = data["template_pts"]
    else:
        print(f"  [!] No template for '{category}'. Skipping evaluation.")
        return 0, 0

    valid_objects = []
    for uid in download_uids:
        path = objs.get(uid)
        if not path: continue
        try:
            mesh = normalize_to_unit_sphere(load_mesh_or_pc(path))
            ok, _ = passes_quality(mesh)
            if ok:
                valid_objects.append(mesh)
            if len(valid_objects) >= num_objects:
                break
        except Exception:
            continue
            
    if len(valid_objects) < 2:
        print(f"  [!] Need at least 2 valid objects. Found {len(valid_objects)}.")
        return 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    vnn_cat_idx = None

    agnostic_path = get_agnostic_model_path()
    if os.path.exists(agnostic_path):
        ckpt = torch.load(agnostic_path, weights_only=False, map_location=device)
        cat_to_idx = ckpt.get("category_to_idx", {})
        model = VNCandidateSelector(
            n_points=ckpt.get("n_points", N_POINTS),
            feat_dims=ckpt.get("feat_dims", AGNOSTIC_FEAT_DIMS),
            hidden_dim=ckpt.get("hidden_dim", AGNOSTIC_HIDDEN_DIM),
            n_categories=ckpt.get("n_categories", 0),
            category_embed_dim=ckpt.get("category_embed_dim", N_CATEGORY_EMBED),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if category in cat_to_idx:
            vnn_cat_idx = cat_to_idx[category]
    else:
        per_cat_path = get_best_model_path(category)
        if os.path.exists(per_cat_path):
            ckpt = torch.load(per_cat_path, weights_only=False, map_location=device)
            model = VNCandidateSelector(
                n_points=ckpt.get("n_points", N_POINTS),
                feat_dims=ckpt.get("feat_dims", FEAT_DIMS),
                hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                n_categories=ckpt.get("n_categories", 0),
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

    if model is None:
        print(f"  [!] No VNN model found for '{category}'. Skipping.")
        return 0, 0

    print(f"  Evaluating {len(valid_objects)} objects with {k_rotations} rotations each...")
    tpl_tensor = torch.tensor(template_pts, dtype=torch.float32).unsqueeze(0).to(device)
    cat_tensor = torch.tensor([vnn_cat_idx], dtype=torch.long).to(device) if vnn_cat_idx is not None else None

    def get_predicted_rotation(mesh_input, R_rand):
        # We manually rotate points and normals to be safe instead of using mesh.apply_transform
        # because Trimesh apply_transform might drop or incorrectly transform custom vertex_normals.
        pts, norms = sample_points(mesh_input, with_normals=True)
        pts_rot = (R_rand @ pts.T).T
        norms_rot = (R_rand @ norms.T).T
        
        rot_mesh = trimesh.Trimesh(vertices=pts_rot)
        rot_mesh.vertex_normals = norms_rot
        
        candidates = generate_five_candidates(rot_mesh, template_pts, category=category)
        
        scores = []
        cand_R = []
        with torch.no_grad():
            for name, R_cand in candidates:
                cand_R.append(R_cand)
                rotated = (R_cand @ pts_rot.T).T
                ct = torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).to(device)
                s = model(tpl_tensor, ct, category_idx=cat_tensor)
                scores.append(s.item())
        best_idx = int(np.argmax(scores))
        
        # The true canonicalizing rotation from the ORIGINAL shape is R_cand * R_rand
        return cand_R[best_idx] @ R_rand

    stability_scores = []
    canonical_rotations_per_object = []

    for mesh in valid_objects:
        obj_canon_rotations = []
        for _ in range(k_rotations):
            R_rand = random_rotation_matrix()
            R_pred = get_predicted_rotation(mesh, R_rand)
            obj_canon_rotations.append(R_pred)
            
        # Stability: RMS spread from the chordal L2 mean
        R_mean = chordal_l2_mean_np(obj_canon_rotations)
        angs = [rotation_angle_np(R, R_mean) for R in obj_canon_rotations]
        rms_ang_deg = np.sqrt(np.mean(np.array(angs)**2)) * 180.0 / math.pi
        
        stability_scores.append(rms_ang_deg)
        canonical_rotations_per_object.append(obj_canon_rotations[0])

    avg_stability = np.mean(stability_scores)
    
    # Consistency: RMS spread across different objects (vs their mean)
    R_mean_cons = chordal_l2_mean_np(canonical_rotations_per_object)
    angs_cons = [rotation_angle_np(R, R_mean_cons) for R in canonical_rotations_per_object]
    rms_cons_deg = np.sqrt(np.mean(np.array(angs_cons)**2)) * 180.0 / math.pi
                
    avg_consistency = rms_cons_deg

    print(f"\n  [RESULTS] Category: {category.upper()}")
    print(f"    Stability (k={k_rotations}):   {avg_stability:.2f}° (Ideal: 0°)")
    print(f"    Consistency (n={n}): {avg_consistency:.2f}° (Ideal: 0°)\n")
    
    return avg_stability, avg_consistency

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
        print("  TRAINING VNN FOR ALL CATEGORIES (per-category)")
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

    elif mode == "evaluate_metrics":
        print("\n" + "#" * 60)
        print("  EVALUATING STABILITY & CONSISTENCY FOR ALL CATEGORIES")
        print("#" * 60)
        t0 = time.time()
        for cat in categories:
            evaluate_stability_consistency(cat)
        print(f"\n  Total evaluation time: {time.time()-t0:.1f}s")

    else:
        print(f"  Unknown mode: {mode}. Use 'train', 'generate', or 'evaluate_metrics'.")


def run_all(categories=None):
    """Run the full per-category pipeline: train all VNNs, then generate candidates."""
    if categories is None:
        categories = CATEGORIES

    grand_start = time.time()

    print("\n" + "#" * 60)
    print("  FULL PIPELINE: PER-CATEGORY TRAIN + GENERATE")
    print(f"  Categories: {', '.join(categories)}")
    if USE_ONLY_N_SAMPLES: print(f"  Samples per category: {N_SAMPLES}")
    print(f"  Rotation augmentation: {N_ROTATIONS}x")
    print("#" * 60)

    # Step 1: Train
    run_pipeline("train", categories)

    # Step 2: Generate
    run_pipeline("generate", categories)

    # Step 3: Calculate stablility and consistency
    run_pipeline("evaluate_metrics")

    grand_time = time.time() - grand_start
    print(f"\n{'#'*60}")
    print(f"  COMPLETE! Total time: {grand_time:.1f}s ({grand_time/60:.1f}min)")
    print(f"{'#'*60}\n")


def run_agnostic(categories=None):
    """Run the full category-agnostic pipeline: train unified VNN, then generate.

    This is the paper-faithful approach: a single model is trained across all
    categories using ranking loss, then used to select canonical orientations.
    """
    if categories is None:
        categories = CATEGORIES

    grand_start = time.time()

    print("\n" + "#" * 60)
    print("  FULL PIPELINE: AGNOSTIC TRAIN + GENERATE")
    print(f"  Categories: {', '.join(categories)}")
    if USE_ONLY_N_SAMPLES: print(f"  Samples per category: {N_SAMPLES}")
    print(f"  Rotation augmentation: {N_ROTATIONS}x")
    print("#" * 60)

    # Step 1: Train agnostic model
    train_agnostic(categories)

    # Step 2: Generate using agnostic model
    run_pipeline("generate", categories)

    # Step 3: Calculate stablility and consistency
    run_pipeline("evaluate_metrics")

    grand_time = time.time() - grand_start
    print(f"\n{'#'*60}")
    print(f"  AGNOSTIC PIPELINE COMPLETE! Total time: {grand_time:.1f}s ({grand_time/60:.1f}min)")
    print(f"{'#'*60}\n")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    if CATEGORY_AGNOSTIC: run_agnostic()
    else: run_all()