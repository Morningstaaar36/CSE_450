"""
=============================================================================
  CanoVerse VNN -- Qwen VLM Distillation Bundle (Category-Agnostic Mode)
=============================================================================

Complete self-contained Kaggle pipeline that replaces noisy Chamfer-distance
pseudo-labels with semantically-aware Qwen2.5-VL-7B vision-language model
labels for training a Vector Neuron Network (VNN) canonical orientation
selector.

KEY CHANGE FROM ORIGINAL:
  During training data preparation, instead of using Chamfer Distance to
  pick the "best" candidate rotation (which injects ~30% label noise on
  symmetric objects), we render a composite panel showing the template and
  5 candidates, then ask Qwen VLM to select the best match.  The VN-DGCNN
  then learns from these clean, semantically-grounded labels.

  At inference time, ONLY VN-DGCNN runs (no Qwen needed).

Usage:
  1. Upload Qwen2.5-VL-7B GGUF + mmproj to Kaggle as a dataset
  2. Paste this entire file into a single Kaggle cell
  3. Adjust CONFIG section below
  4. Run the cell — it will build llama.cpp, prepare data with Qwen labels,
     train VN-DGCNN, and evaluate
"""

# =====================================================================
# EASY-TUNE CONFIG
# =====================================================================

# Categories to process (leave empty to dynamically discover from dataset)
CATEGORIES = []

# Maximum categories to process (None = all, integer N = first N only)
MAX_CATEGORIES = 4

# Limit training samples per category
USE_ONLY_N_SAMPLES = False
N_SAMPLES = 20

# Objects to generate candidates for per rendered category
N_GENERATE = 5

# Candidate rendering & generation settings
CATEGORIES_TO_RENDER = 12
MIN_GLBS_TO_RENDER = 100
MIN_GLBS_TO_CALC_METRICS = 50

# Point cloud settings
N_POINTS = 1024

# Rotation augmentation
N_ROTATIONS = 1

# Training hyperparameters
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 50

# VNN architecture
FEAT_DIMS = (32, 64, 128)
HIDDEN_DIM = 256
N_CATEGORY_EMBED = 32
AGNOSTIC_FEAT_DIMS = FEAT_DIMS
AGNOSTIC_HIDDEN_DIM = HIDDEN_DIM

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Candidate generation
THETA_STEPS = 36
RENDER_SIZE = 320

# Semantic part segmentation
CENTROID_MATCH_WEIGHT = 0.2

# Quality gate
MIN_THICKNESS_RATIO = 0.02
MIN_FACES = 50
MIN_CONNECTED_RATIO = 0.7
MAX_FACE_AREA_VAR = 50.0

# Support surface
MAX_SUPPORT_FACES = 50

# CLIP settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
USE_PYRENDER = True

# Dataset Paths
DATASET_DIR = "/kaggle/input/datasets/tasnimulislammahi/canoverse-2"
OUT_ROOT = "renders"
CACHE_DIR = "data_cache"
CHECKPOINT_DIR = "checkpoints"

# Seeds
RANDOM_SEED = 42

# ---- QWEN VLM DISTILLATION CONFIG ----
QWEN_MODEL_DIR = "/kaggle/input/qwen-model"       # Directory containing GGUF files
QWEN_DUAL_GPU = True                                # Use -dev CUDA0,CUDA1 for dual T4
QWEN_RENDER_SIZE = 320                              # Per-panel render size for Qwen composite
QWEN_TIMEOUT = 120                                  # Seconds before Qwen subprocess times out
QWEN_PANEL_DIR = "qwen_panels"                      # Save rendered panels here for debugging
SOFT_LABEL_SIGMA_DEG = 30.0                         # Gaussian sigma for geodesic soft labels
MAX_CANDIDATE_GT_ERROR = 45.0                       # Discard object if ALL candidates have GT error > this
# =====================================================================
# END CONFIG
# =====================================================================

import base64
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import math
from collections import defaultdict
from pathlib import Path

for pkg in ["trimesh", "transformers", "pyrender", "fast-simplification"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        except Exception:
            print("Could NOT install", pkg)
            pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.pop("PYOPENGL_PLATFORM", None)


# =====================================================================
# LLAMA.CPP BUILD & QWEN MODEL DISCOVERY
# =====================================================================

LLAMA_MTMD_CLI = None
QWEN_MODEL_PATH = None
QWEN_MMPROJ_PATH = None
_QWEN_AVAILABLE = False

def setup_llama_cpp():
    """Build llama.cpp from source if not already built. Locate the multimodal CLI binary."""
    global LLAMA_MTMD_CLI

    candidate_paths = []
    # Prefer mtmd-cli over deprecated qwen2vl-cli
    for b in ["llama-mtmd-cli", "llama-qwen2vl-cli", "llama-llava-cli"]:
        candidate_paths.extend([
            f"/kaggle/working/llama.cpp/build/bin/{b}",
            f"/kaggle/working/llama-cpp-t4/bin/{b}",
            f"/kaggle/working/llama-cpp/build/bin/{b}",
        ])
    for p in candidate_paths:
        if os.path.exists(p):
            LLAMA_MTMD_CLI = p
            print(f"  [llama.cpp] Found existing binary: {p}")
            return

    print("  [llama.cpp] Building from source (this takes ~5 minutes)...")
    build_dir = "/kaggle/working/llama.cpp"
    try:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "cmake", "build-essential", "git"],
                       check=True)

        if not os.path.exists(build_dir):
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/ggml-org/llama.cpp.git", build_dir],
                           check=True)

        subprocess.run([
            "cmake", "-B", os.path.join(build_dir, "build"),
            "-S", build_dir,
            "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            "-DCMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64",
            "-DCMAKE_PREFIX_PATH=/usr/local/cuda",
            "-DLLAMA_CURL=OFF",
        ], check=True)

        subprocess.run([
            "cmake", "--build", os.path.join(build_dir, "build"),
            "--config", "Release", "-j4"
        ], check=True)

        for b in ["llama-mtmd-cli", "llama-qwen2vl-cli", "llama-llava-cli"]:
            bin_path = os.path.join(build_dir, "build", "bin", b)
            if os.path.exists(bin_path):
                LLAMA_MTMD_CLI = bin_path
                print(f"  [llama.cpp] Build successful: {bin_path}")
                break
        else:
            print(f"  [!] multimodal CLI not found after build.")
    except Exception as e:
        print(f"  [!] llama.cpp build failed: {e}")


def find_qwen_models():
    """Locate Qwen2.5-VL GGUF model and mmproj files."""
    global QWEN_MODEL_PATH, QWEN_MMPROJ_PATH, _QWEN_AVAILABLE

    search_dirs = [QWEN_MODEL_DIR, "/kaggle/input"]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        gguf_files = list(Path(search_dir).rglob("*.gguf"))
        if not gguf_files:
            continue

        model = next((f for f in gguf_files
                       if "mmproj" not in f.name.lower()), None)
        mmproj = next((f for f in gguf_files
                        if "mmproj" in f.name.lower()), None)

        if model and mmproj:
            QWEN_MODEL_PATH = str(model)
            QWEN_MMPROJ_PATH = str(mmproj)
            print(f"  [Qwen] Model:  {QWEN_MODEL_PATH} ({model.stat().st_size / 1e9:.1f} GB)")
            print(f"  [Qwen] MMProj: {QWEN_MMPROJ_PATH} ({mmproj.stat().st_size / 1e6:.0f} MB)")
            break

    if QWEN_MODEL_PATH and QWEN_MMPROJ_PATH and LLAMA_MTMD_CLI:
        _QWEN_AVAILABLE = True
        print("  [Qwen] VLM distillation: ENABLED")
    else:
        _QWEN_AVAILABLE = False
        print("  [!] Qwen VLM distillation: DISABLED (falling back to Chamfer labels)")
        exit(0)


def create_qwen_grammar_file():
    """Create a GBNF grammar file that constrains Qwen output to a single digit 0-4."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    grammar_path = os.path.join(CACHE_DIR, "_qwen_grammar.gbnf")
    with open(grammar_path, "w") as f:
        f.write('root ::= [01234]\n')
    return grammar_path


# Run setup at import time
print("\n" + "=" * 60)
print("  SETTING UP QWEN VLM DISTILLATION")
print("=" * 60)
setup_llama_cpp()
find_qwen_models()
GRAMMAR_FILE = create_qwen_grammar_file() if _QWEN_AVAILABLE else None
print("=" * 60 + "\n")


# =====================================================================
# DATASET DISCOVERY
# =====================================================================

_DATASET_INDEX = None
_DISCOVERED_ALL_CATEGORIES = None


def find_dataset_root():
    """Locate the exact CanoVerse dataset root containing category folders."""
    candidates = [
        DATASET_DIR,
        "/kaggle/input/datasets/tasnimulislammahi/canoverse-2",
        "/kaggle/input/canoverse-2",
        "/kaggle/input/canoverse",
        "../input/datasets/tasnimulislammahi/canoverse-2",
        "../input/canoverse-2",
        "./datasets/canoverse-2",
        "./data_cache/canoverse-2",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d)) and not d.startswith(".")]
            if len(subdirs) >= 2:
                return os.path.realpath(c)

    for search_root in ["/kaggle/input", "../input", "./datasets", "./data_cache", "."]:
        if os.path.isdir(search_root):
            for root, dirs, _ in os.walk(search_root, followlinks=True):
                if "canoverse" in os.path.basename(root).lower():
                    subdirs = [d for d in dirs if not d.startswith(".") and d not in ("renders", "checkpoints")]
                    if len(subdirs) >= 2:
                        return os.path.realpath(root)
    return None


def get_dataset_index():
    """Build or retrieve a fast in-memory dictionary mapping category names to .glb files."""
    global _DATASET_INDEX
    if _DATASET_INDEX is not None:
        return _DATASET_INDEX

    root = find_dataset_root()
    index = defaultdict(list)

    if root and os.path.isdir(root):
        print(f"  [Dataset] Found dataset root: {root}")
        for entry in sorted(os.listdir(root)):
            cat_dir = os.path.join(root, entry)
            if os.path.isdir(cat_dir) and not entry.startswith("."):
                cat_name = entry.lower().strip()
                for dirpath, dirnames, filenames in os.walk(cat_dir):
                    glbs = [
                        os.path.join(dirpath, f)
                        for f in sorted(filenames) if f.lower().endswith(".glb")
                    ]
                    if glbs:
                        index[cat_name].extend(glbs)

    _DATASET_INDEX = dict(index)
    return _DATASET_INDEX


def get_categories():
    """Dynamically discover available categories from local CanoVerse GLB dataset."""
    global CATEGORIES, _DISCOVERED_ALL_CATEGORIES
    if CATEGORIES:
        cats = list(CATEGORIES)
    elif _DISCOVERED_ALL_CATEGORIES is not None:
        cats = list(_DISCOVERED_ALL_CATEGORIES)
    else:
        index = get_dataset_index()
        cats = sorted(list(index.keys()))

        if not cats:
            print("  [WARNING] No CanoVerse categories found at dataset path!")
            raise ValueError("No categories found in dataset directory!")
        _DISCOVERED_ALL_CATEGORIES = list(cats)
        print(f"\n  [INFO] Dynamically discovered {len(cats)} CanoVerse categories: {cats}\n")

    if MAX_CATEGORIES is not None and isinstance(MAX_CATEGORIES, int) and MAX_CATEGORIES > 0:
        cats = cats[:MAX_CATEGORIES]
        print(f"  [INFO] MAX_CATEGORIES={MAX_CATEGORIES}: Running on first {len(cats)} categories: {cats}\n")

    return cats


def sample_surface_points(geometry, n=5000, seed=42):
    """Uniformly sample n points from a Trimesh (area-weighted surface sampling) or PointCloud."""
    if seed is not None:
        np.random.seed(seed)
        
    if hasattr(geometry, 'faces') and geometry.faces is not None and len(geometry.faces) > 0:
        try:
            pts, _ = trimesh.sample.sample_surface(geometry, n)
            return np.asarray(pts, dtype=np.float32)
        except Exception:
            pass
    if hasattr(geometry, 'vertices') and len(geometry.vertices) > 0:
        verts = np.asarray(geometry.vertices, dtype=np.float32)
        if len(verts) <= n:
            return verts
        idx = np.random.choice(len(verts), n, replace=False)
        return verts[idx]
    return np.zeros((n, 3), dtype=np.float32)


def load_mesh_or_pc(path):
    """Load a CanoVerse .glb mesh with safe multi-geometry concatenation."""
    loaded = trimesh.load(path, skip_materials=True, process=True)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, 'vertices') and len(g.vertices) > 0]
        if not geoms:
            return trimesh.Trimesh()
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded

    MAX_GLB_VERTS = 50_000
    if hasattr(mesh, 'vertices') and len(mesh.vertices) > MAX_GLB_VERTS:
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            try:
                target_faces = min(len(mesh.faces), MAX_GLB_VERTS)
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            except Exception as e:
                print(f"  [decimate] failed ({type(e).__name__}: {e}) — falling back to point sampling")
                pts, face_idx = trimesh.sample.sample_surface(mesh, MAX_GLB_VERTS)
                normals = mesh.face_normals[face_idx] if hasattr(mesh, 'face_normals') and len(mesh.face_normals) > 0 else np.zeros_like(pts)
                mesh = trimesh.Trimesh(vertices=pts, vertex_normals=normals)
        else:
            idx = np.random.choice(len(mesh.vertices), MAX_GLB_VERTS, replace=False)
            mesh = trimesh.Trimesh(vertices=mesh.vertices[idx])

    return mesh


TEMPLATE_FORWARD = np.array([0.0, 0.0, 1.0])
TEMPLATE_UP = np.array([0.0, 1.0, 0.0])


# =====================================================================
# PATH HELPERS
# =====================================================================

def get_cache_file(category):
    return os.path.join(CACHE_DIR, f"{category}_vnn_data.pt")

def get_agnostic_model_path():
    return os.path.join(CHECKPOINT_DIR, "vnn_agnostic_best.pt")

def get_agnostic_latest_model_path():
    return os.path.join(CHECKPOINT_DIR, "vnn_agnostic_latest.pt")


# =====================================================================
# VNN MODEL -- SO(3)-Equivariant Architecture
# =====================================================================

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels) * (2.0 / in_channels) ** 0.5
        )

    def forward(self, x):
        return torch.einsum("oi,bid->bod", self.weight, x)


class VNLeakyReLU(nn.Module):
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
        proj_neg = torch.clamp(proj, max=0.0)
        x_out = x - ((1.0 - self.negative_slope) * proj_neg) * q_norm
        return x_out


class VNMaxPool(nn.Module):
    def forward(self, x):
        norms = torch.norm(x, dim=-1)
        max_indices = norms.argmax(dim=1)
        B, N, C, _ = x.shape
        idx = max_indices.unsqueeze(1).unsqueeze(-1).expand(B, 1, C, 3)
        pooled = x.gather(dim=1, index=idx).squeeze(1)
        return pooled


class VNMeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class VNChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        norms = torch.norm(x, dim=-1)
        attn = F.relu(self.fc1(norms))
        attn = torch.sigmoid(self.fc2(attn))
        return x * attn.unsqueeze(-1)


class VNStdFeature(nn.Module):
    """Extract rotation-INVARIANT scalar features from equivariant vectors via normalized Gram matrix."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = in_channels * (in_channels + 1) // 2
        idx = torch.triu_indices(in_channels, in_channels)
        self.register_buffer("tri_row", idx[0])
        self.register_buffer("tri_col", idx[1])

    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        x_norm = x / norms
        gram = torch.bmm(x_norm, x_norm.transpose(1, 2))
        invariant = gram[:, self.tri_row, self.tri_col]
        return invariant


def knn_distance_invariant(x, k):
    inner = torch.einsum('bcdn,bcdm->bnm', x, x)
    x2 = torch.diagonal(inner, dim1=1, dim2=2).unsqueeze(2)
    dist = x2 + x2.transpose(1, 2) - 2 * inner
    idx = dist.topk(k=k + 1, dim=-1, largest=False)[1][..., 1:]
    return idx

def get_vn_graph_feature(x, k=20, idx=None):
    B, C, _, N = x.size()
    if idx is None:
        idx = knn_distance_invariant(x, k=k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    x_perm = x.permute(0, 3, 1, 2)
    x_flat = x_perm.reshape(B*N, C, 3)
    x_neighbors = x_flat[idx_flat].view(B, N, k, C, 3).permute(0, 3, 4, 1, 2)

    x_central = x.unsqueeze(-1).expand(B, C, 3, N, k)
    edge_feature = torch.cat([x_central, x_neighbors - x_central], dim=1)
    return edge_feature, idx

class VNMaxPoolLocal(nn.Module):
    def forward(self, x):
        norms = torch.norm(x, dim=2)
        max_idx = norms.argmax(dim=-1, keepdim=True)
        max_idx_expanded = max_idx.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        pooled = torch.gather(x, 4, max_idx_expanded).squeeze(-1)
        return pooled

class VNDGCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = VNLinear(in_channels * 2, out_channels)
        self.act = VNLeakyReLU(out_channels)
        self.pool = VNMaxPoolLocal()

    def forward(self, x, idx=None):
        edge_feature, idx = get_vn_graph_feature(x, k=self.k, idx=idx)
        B, C2, _, N, k = edge_feature.size()
        edge_flat = edge_feature.permute(0, 3, 4, 1, 2).reshape(B*N*k, C2, 3)
        out_flat = self.act(self.conv(edge_flat))
        out = out_flat.view(B, N, k, -1, 3).permute(0, 3, 4, 1, 2)
        return self.pool(out), idx

class VNDGCNNEncoder(nn.Module):
    """VN-DGCNN encoder with multi-scale feature aggregation and gradient checkpointing."""
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

        total_ch = sum(feat_dims)
        self.fuse = VNLinear(total_ch, feat_dims[-1])
        self.fuse_act = VNLeakyReLU(feat_dims[-1])

        self.global_pool_max = VNMaxPool()
        self.global_pool_mean = VNMeanPool()

        self.channel_attention = VNChannelAttention(feat_dims[-1] * 2)

    def forward(self, points):
        x = points.unsqueeze(1).permute(0, 1, 3, 2)
        layer_outputs = []
        for layer in self.layers:
            if self.training:
                # Force requires_grad so PyTorch checkpointing actually activates
                x = x.requires_grad_()
                x, _ = torch.utils.checkpoint.checkpoint(layer, x, None, use_reentrant=False)
            else:
                x, _ = layer(x, idx=None)
            layer_outputs.append(x)

        x_cat = torch.cat(layer_outputs, dim=1)

        B, C_total, _, N = x_cat.size()
        x_flat = x_cat.permute(0, 3, 1, 2).reshape(B * N, C_total, 3)
        x_fused = self.fuse_act(self.fuse(x_flat))
        x_fused = x_fused.view(B, N, -1, 3)

        x_max = self.global_pool_max(x_fused)
        x_mean = self.global_pool_mean(x_fused)
        x_global = torch.cat([x_max, x_mean], dim=1)

        x_global = self.channel_attention(x_global)
        return x_global


class VNCandidateSelector(nn.Module):
    """VNN-based canonical orientation candidate selector."""
    def __init__(self, n_points=1024, feat_dims=(32, 64, 128), hidden_dim=256,
                 n_categories=0, category_embed_dim=32, reduce_dim=64, k=12):
        super().__init__()
        self.n_points = n_points
        self.encoder = VNDGCNNEncoder(n_points=n_points, feat_dims=feat_dims, k=k)

        last_ch = feat_dims[-1] * 2

        self.reducer = VNLinear(last_ch * 2, reduce_dim)
        self.invariant = VNStdFeature(reduce_dim)
        inv_dim = reduce_dim * (reduce_dim + 1) // 2

        self.use_category = n_categories > 0
        head_input_dim = inv_dim
        if self.use_category:
            self.category_embed = nn.Embedding(n_categories, category_embed_dim)
            head_input_dim += category_embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
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
        feat_t = self.encoder(template_pts)
        feat_c = self.encoder(candidate_pts)
        combined_eq = torch.cat([feat_t, feat_c], dim=1)
        reduced_eq = self.reducer(combined_eq)
        inv_feat = self.invariant(reduced_eq)

        if self.use_category and category_idx is not None:
            cat_emb = self.category_embed(category_idx)
            inv_feat = torch.cat([inv_feat, cat_emb], dim=1)

        score = self.head(inv_feat)
        return score

    def forward_with_cached_template(self, feat_t, candidate_pts, category_idx=None):
        feat_c = self.encoder(candidate_pts)
        combined_eq = torch.cat([feat_t, feat_c], dim=1)
        reduced_eq = self.reducer(combined_eq)
        inv_feat = self.invariant(reduced_eq)

        if self.use_category and category_idx is not None:
            cat_emb = self.category_embed(category_idx)
            inv_feat = torch.cat([inv_feat, cat_emb], dim=1)

        score = self.head(inv_feat)
        return score

    def forward_all_candidates(self, template_pts, candidates_pts, category_idx=None, chunk_size=10):
        B, K, N, _ = candidates_pts.shape

        feat_t = self.encoder(template_pts)

        cands_flat = candidates_pts.reshape(B * K, N, 3)
        if B * K <= chunk_size:
            feat_c_flat = self.encoder(cands_flat)
        else:
            feat_c_chunks = []
            for start in range(0, B * K, chunk_size):
                end = min(start + chunk_size, B * K)
                feat_c_chunks.append(self.encoder(cands_flat[start:end]))
            feat_c_flat = torch.cat(feat_c_chunks, dim=0)

        feat_c = feat_c_flat.reshape(B, K, -1, 3)

        feat_t_exp = feat_t.unsqueeze(1).expand(B, K, -1, 3)
        combined_eq = torch.cat([feat_t_exp, feat_c], dim=2)

        C2 = combined_eq.shape[2]
        combined_flat = combined_eq.reshape(B * K, C2, 3)
        reduced_flat = self.reducer(combined_flat)
        inv_flat = self.invariant(reduced_flat)

        if self.use_category and category_idx is not None:
            cat_emb = self.category_embed(category_idx)
            cat_emb_exp = cat_emb.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            inv_flat = torch.cat([inv_flat, cat_emb_exp], dim=1)

        scores_flat = self.head(inv_flat)
        logits = scores_flat.reshape(B, K)
        return logits

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

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def chordal_l2_mean_np(rotations):
    R_sum = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    S = U @ Vt
    if np.linalg.det(S) < 0:
        U = U.copy()
        U[:, -1] *= -1
        S = U @ Vt
    return S

def rotation_angle_np(R1, R2):
    R_rel = np.matmul(np.swapaxes(R1, -1, -2), R2)
    trace = np.trace(R_rel, axis1=-2, axis2=-1)
    trace = np.clip(trace, -1.0, 3.0)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def rodrigues_rotation(n_from, n_to):
    a = np.asarray(n_from, dtype=np.float64)
    b = np.asarray(n_to, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
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


def random_rotation_matrix_torch(device="cpu"):
    u = torch.rand(3, device=device)
    u1, u2, u3 = u[0], u[1], u[2]
    pi2 = 2.0 * torch.pi

    r1 = torch.sqrt(1.0 - u1)
    r2 = torch.sqrt(u1)

    w = r1 * torch.sin(pi2 * u2)
    x = r1 * torch.cos(pi2 * u2)
    y = r2 * torch.sin(pi2 * u3)
    z = r2 * torch.cos(pi2 * u3)

    R = torch.stack([
        torch.stack([1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)]),
        torch.stack([2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)]),
        torch.stack([2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]),
    ])
    return R


def random_realistic_input_rotation(p_azimuth=0.60, p_tilted=0.20):
    u = float(np.random.random())
    if u < p_azimuth:
        theta = float(np.random.uniform(0, 2 * np.pi))
        return rot_y(theta).astype(np.float32)
    elif u < p_azimuth + p_tilted:
        theta = float(np.random.uniform(0, 2 * np.pi))
        R_yaw = rot_y(theta)
        tilt = float(np.random.uniform(np.radians(-25), np.radians(25)))
        phi = float(np.random.uniform(0, 2 * np.pi))
        axis = np.array([np.cos(phi), 0.0, np.sin(phi)], dtype=np.float32)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        R_tilt = np.eye(3, dtype=np.float32) + np.sin(tilt) * K + (1.0 - np.cos(tilt)) * (K @ K)
        return (R_tilt @ R_yaw).astype(np.float32)
    else:
        return random_rotation_matrix().astype(np.float32)


def normalize_to_unit_sphere(mesh):
    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2.0
    mesh.apply_translation(-center)
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 0:
        mesh.apply_scale(1.0 / radius)
    return mesh


def sample_points(mesh, n=None, seed=None, with_normals=False):
    if n is None:
        n = N_POINTS
    if seed is not None:
        st = np.random.get_state()
        np.random.seed(seed)

    pts, normals = None, None
    has_normals = hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0
    has_faces = hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0

    if has_faces:
        try:
            if with_normals:
                pts, face_indices = trimesh.sample.sample_surface(mesh, n)
                if hasattr(mesh, "face_normals") and len(mesh.face_normals) > 0:
                    normals = mesh.face_normals[face_indices]
                else:
                    normals = np.zeros_like(pts)
            else:
                pts, _ = trimesh.sample.sample_surface(mesh, n)
        except Exception:
            pts = None

    if pts is None:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        idx = None
        if len(pts) >= n:
            idx = np.random.choice(len(pts), n, replace=False)
            pts = pts[idx]
        else:
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


@torch.no_grad()
def chamfer_distance(a, b, na=None, nb=None, alpha=0.5):
    """Fast Chamfer distance — kept for geometric alignment internals."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
        b_t = torch.as_tensor(b, dtype=torch.float32, device=device)

        dists = torch.cdist(a_t.unsqueeze(0), b_t.unsqueeze(0)).squeeze(0)
        min_ab, idx_ab = dists.min(dim=1)
        min_ba, idx_ba = dists.min(dim=0)
        cd_val = float((min_ab.mean() + min_ba.mean()).item())

        if na is not None and nb is not None:
            na_t = torch.as_tensor(na, dtype=torch.float32, device=device)
            nb_t = torch.as_tensor(nb, dtype=torch.float32, device=device)
            dot_ab = (na_t * nb_t[idx_ab]).sum(dim=1)
            dot_ba = (nb_t * na_t[idx_ba]).sum(dim=1)
            normal_loss = float(((1.0 - dot_ab).mean() + (1.0 - dot_ba).mean()).item())
            return cd_val + alpha * normal_loss
        return cd_val
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        ta, tb = cKDTree(a), cKDTree(b)
        d_ab, idx_ab = tb.query(a)
        d_ba, idx_ba = ta.query(b)
        cd_val = float(d_ab.mean() + d_ba.mean())
        if na is not None and nb is not None:
            dot_ab = np.sum(na * nb[idx_ab], axis=1)
            dot_ba = np.sum(nb * na[idx_ba], axis=1)
            normal_loss = float((1.0 - dot_ab).mean() + (1.0 - dot_ba).mean())
            return cd_val + alpha * normal_loss
        return cd_val

def chamfer_distance_fast(a, b, tree_b=None, max_pts=256):
    """Fast approximate Chamfer — kept for geometric/semantic horizontal alignment."""
    if tree_b is None and torch.cuda.is_available():
        try:
            b_sub = b[np.random.choice(len(b), max_pts, replace=False)] if len(b) > max_pts else b
            a_t = torch.as_tensor(a, dtype=torch.float32, device="cuda")
            b_t = torch.as_tensor(b_sub, dtype=torch.float32, device="cuda")
            dists = torch.cdist(a_t.unsqueeze(0), b_t.unsqueeze(0)).squeeze(0)
            d_ab = dists.min(dim=1).values.mean().item()
            d_ba = dists.min(dim=0).values.mean().item()
            return float(d_ab + d_ba)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            pass

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
    R_rel = R1.T @ R2
    trace_val = np.trace(R_rel)
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_geodesic_soft_labels(best_idx, cand_rot_list, sigma_deg=None):
    """Convert a hard 'best_idx' into a soft probability distribution based on relative rotation."""
    if sigma_deg is None:
        sigma_deg = SOFT_LABEL_SIGMA_DEG
    R_best = cand_rot_list[best_idx]
    scores = []
    sigma_rad = np.deg2rad(sigma_deg)
    
    for R in cand_rot_list:
        R_rel = R.T @ R_best
        trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
        angle_rad = np.arccos((trace_val - 1.0) / 2.0)
        
        # Gaussian smoothing
        score = np.exp(- (angle_rad**2) / (2 * sigma_rad**2))
        scores.append(score)
        
    scores = np.array(scores, dtype=np.float32)
    return scores / scores.sum()


def pca_axes(points):
    c = points - points.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    v = v[:, np.argsort(w)[::-1]]
    return v[:, 0], v[:, 1]


def point_in_polygon_2d(point, polygon):
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
# SPECTRAL GEOMETRIC PART SEGMENTATION
# =====================================================================

class SpectralPartSegmenter:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SpectralPartSegmenter()
        return cls._instance

    def compute_features(self, points, k_neighbors=16, n_components=None):
        N = len(points)
        if N < 8:
            return points, 1

        tree = cKDTree(points)
        k_eff = min(k_neighbors, N - 1)
        dists, indices = tree.query(points, k=k_eff + 1)

        sigma = np.median(dists[:, 1:]) + 1e-6
        rows = np.repeat(np.arange(N), k_eff)
        cols = indices[:, 1:].flatten()
        weights = np.exp(-(dists[:, 1:].flatten() ** 2) / (2 * sigma ** 2))

        W = csr_matrix((weights, (rows, cols)), shape=(N, N), dtype=np.float32)
        W = 0.5 * (W + W.T)

        deg = np.array(W.sum(axis=1)).flatten() + 1e-7
        inv_sqrt_deg = 1.0 / np.sqrt(deg)
        D_inv_sqrt = csr_matrix((inv_sqrt_deg, (np.arange(N), np.arange(N))), shape=(N, N), dtype=np.float32)

        L_sym = speye(N, dtype=np.float32, format='csr') - D_inv_sqrt @ W @ D_inv_sqrt

        k_max_eig = min(9, max(4, N - 2))
        try:
            evals, evecs = eigsh(L_sym, k=k_max_eig + 1, which='SM', tol=1e-3, maxiter=200)
            idx_sort = np.argsort(evals)
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]

            if n_components is None:
                k_min_cand = 3
                k_max_cand = min(8, k_max_eig - 1)
                if k_max_cand >= k_min_cand:
                    gaps = np.diff(evals[1:k_max_cand + 2])
                    best_offset = int(np.argmax(gaps[k_min_cand - 1:k_max_cand]))
                    n_comp = k_min_cand + best_offset
                else:
                    n_comp = k_min_cand
            else:
                n_comp = min(n_components, k_max_eig)

            spectral_emb = evecs[:, 1:n_comp + 1]
            
            # Canonicalize eigenvector signs for stability across runs/rotations
            for j in range(spectral_emb.shape[1]):
                if spectral_emb[np.argmax(np.abs(spectral_emb[:, j])), j] < 0:
                    spectral_emb[:, j] *= -1
                    
        except Exception:
            n_comp = 6 if n_components is None else min(n_components, 6)
            spectral_emb = np.zeros((N, n_comp), dtype=np.float32)

        center = points.mean(axis=0)
        norm_coords = (points - center) / (np.linalg.norm(points - center, axis=1).max() + 1e-8)
        radius = np.linalg.norm(norm_coords, axis=1, keepdims=True)

        # Drop norm_coords (raw xyz) to maintain strict rotation invariance!
        features = np.concatenate([radius, spectral_emb], axis=-1).astype(np.float32)
        return features, n_comp

    def segment(self, points, n_parts=None):
        if len(points) < 4:
            return [points]

        target_n_parts = n_parts if (n_parts is not None and isinstance(n_parts, int)) else None
        combined_rep, auto_k = self.compute_features(points, n_components=target_n_parts)
        effective_n_parts = target_n_parts if target_n_parts is not None else auto_k
        effective_n_parts = max(2, min(effective_n_parts, len(points) // 4))

        try:
            km = MiniBatchKMeans(
                n_clusters=effective_n_parts, n_init=3, max_iter=100,
                batch_size=min(256, len(points)), random_state=RANDOM_SEED,
            )
            labels = km.fit_predict(combined_rep)
        except Exception:
            split_dim = int(np.argmax(np.ptp(points, axis=0)))
            edges = np.linspace(points[:, split_dim].min(), points[:, split_dim].max(), effective_n_parts + 1)
            labels = np.clip(np.digitize(points[:, split_dim], edges[1:-1]), 0, effective_n_parts - 1)

        parts = []
        for k in range(effective_n_parts):
            mask = labels == k
            if mask.sum() > 0:
                parts.append(points[mask])

        return parts if parts else [points]


def segment_semantic_parts(points, n_parts=None):
    return SpectralPartSegmenter.get_instance().segment(points, n_parts=n_parts)


def match_parts_chamfer(parts_a, parts_b, max_pts=256, centroid_weight=CENTROID_MATCH_WEIGHT, trees_a=None):
    """Match semantic parts via Hungarian algorithm — kept for candidate generation alignment."""
    if not parts_a or not parts_b:
        return float("inf")

    na, nb = len(parts_a), len(parts_b)
    cost_matrix = np.zeros((na, nb), dtype=np.float32)

    centroids_a = [p.mean(axis=0) for p in parts_a]
    centroids_b = [p.mean(axis=0) for p in parts_b]

    for i in range(na):
        tree_ai = trees_a[i] if trees_a is not None and i < len(trees_a) else None
        for j in range(nb):
            cd = chamfer_distance_fast(parts_b[j], parts_a[i], tree_b=tree_ai, max_pts=max_pts)
            cent_dist = np.linalg.norm(centroids_a[i] - centroids_b[j])
            cost_matrix[i, j] = cd + centroid_weight * cent_dist

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        m = len(row_ind)
        return float(total_cost / m) if m > 0 else float("inf")
    except Exception:
        return float("inf")


# =====================================================================
# VERTICAL ALIGNMENT
# =====================================================================

def support_vertical(mesh):
    """Full support-surface vertical alignment."""
    pts_sampled = None
    try:
        pts_sampled = sample_surface_points(mesh, n=5000).astype(np.float64)
        hull = trimesh.points.PointCloud(pts_sampled).convex_hull
    except Exception:
        print("Degenerate mesh found!")
        return np.eye(3)

    pts = pts_sampled
    n_g = np.array([0.0, -1.0, 0.0])

    from collections import defaultdict as _dd
    face_normals = hull.face_normals.astype(np.float64)
    face_areas   = hull.area_faces

    normal_groups = _dd(list)
    for fi in range(len(face_normals)):
        n = face_normals[fi]
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n_unit = (n / nn).round(2)
        key = tuple(n_unit)
        normal_groups[key].append(fi)

    sorted_groups = sorted(
        normal_groups.values(),
        key=lambda idxs: sum(face_areas[i] for i in idxs),
        reverse=True,
    )[:MAX_SUPPORT_FACES]

    stable_poses = []
    unstable_poses = []

    for face_idxs in sorted_groups:
        total_area = sum(face_areas[fi] for fi in face_idxs)
        n_s = sum(face_normals[fi] * face_areas[fi] for fi in face_idxs) / max(total_area, 1e-12)
        n_s_norm = np.linalg.norm(n_s)
        if n_s_norm < 1e-12:
            continue
        n_s = n_s / n_s_norm

        R_k = rodrigues_rotation(n_s, n_g)
        rotated = (R_k @ pts.T).T
        centroid = rotated.mean(axis=0)

        all_vert_indices = np.unique(np.concatenate([hull.faces[fi] for fi in face_idxs]))
        facet_verts = hull.vertices[all_vert_indices].astype(np.float64)
        rotated_facet = (R_k @ facet_verts.T).T
        facet_2d = rotated_facet[:, [0, 2]]

        proj_2d = np.array([centroid[0], centroid[2]])
        try:
            from scipy.spatial import ConvexHull as _CH
            ch2d = _CH(facet_2d)
            support_polygon = facet_2d[ch2d.vertices]
            stable = point_in_polygon_2d(proj_2d, support_polygon)
        except Exception:
            stable = False

        y_extent = rotated[:, 1].max() - rotated[:, 1].min()
        base_width = np.ptp(facet_2d, axis=0).mean() if len(facet_2d) > 1 else 0.01
        score = y_extent * (base_width + 0.01)

        if stable:
            stable_poses.append((R_k, score))
        else:
            unstable_poses.append((R_k, score))

    candidate_poses = stable_poses if stable_poses else unstable_poses

    if not candidate_poses:
        largest_group = sorted_groups[0] if sorted_groups else [int(np.argmax(face_areas))]
        n_fb = sum(face_normals[fi] * face_areas[fi] for fi in largest_group)
        n_fb_norm = np.linalg.norm(n_fb)
        n_fb = n_fb / max(n_fb_norm, 1e-12)
        return rodrigues_rotation(n_fb, n_g)

    candidate_poses.sort(key=lambda x: x[1], reverse=True)
    return candidate_poses[0][0]


def pca_vertical(obj_pts, tpl_pts):
    v1o, v2o = pca_axes(obj_pts)
    v1t, v2t = pca_axes(tpl_pts)
    Mt = np.stack([v1t, v2t, np.cross(v1t, v2t)], axis=1)

    tpl_parts = segment_semantic_parts(tpl_pts)
    obj_parts_base = segment_semantic_parts(obj_pts)

    np.random.seed(42)
    tpl_part_trees = []
    for p in tpl_parts:
        p_sub = p if len(p) <= 256 else p[np.random.choice(len(p), 256, replace=False)]
        tpl_part_trees.append(cKDTree(p_sub))

    best_R, best_cost = np.eye(3), np.inf
    for swap in (False, True):
        a1, a2 = (v2o, v1o) if swap else (v1o, v2o)
        for s1 in (1, -1):
            for s2 in (1, -1):
                u1, u2 = s1 * a1, s2 * a2
                Mo = np.stack([u1, u2, np.cross(u1, u2)], axis=1)
                R = Mt @ Mo.T

                rot_obj_parts = [(R @ part.T).T for part in obj_parts_base]

                cost = match_parts_chamfer(tpl_parts, rot_obj_parts, trees_a=tpl_part_trees)
                if cost < best_cost:
                    best_cost, best_R = cost, R.copy()
    return best_R


# =====================================================================
# HORIZONTAL ALIGNMENT
# =====================================================================

def geometric_horizontal(obj_pts, tpl_pts, steps=None, tpl_tree=None, tpl_sub=None):
    if steps is None:
        steps = THETA_STEPS
    thetas = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    np.random.seed(42)
    if tpl_sub is None:
        tpl_sub = tpl_pts[np.random.choice(len(tpl_pts), min(512, len(tpl_pts)), replace=False)]
    if tpl_tree is None:
        tpl_tree = cKDTree(tpl_sub)

    obj_sub = obj_pts[np.random.choice(len(obj_pts), min(512, len(obj_pts)), replace=False)]

    Eg = np.array([chamfer_distance_fast(obj_sub @ rot_y(t).T, tpl_sub, tree_b=tpl_tree) for t in thetas])
    return thetas[int(np.argmin(Eg))], thetas, Eg


def semantic_horizontal(obj_pts, tpl_pts, steps=None):
    if steps is None:
        steps = THETA_STEPS

    tpl_sub = tpl_pts if len(tpl_pts) <= 256 else tpl_pts[np.random.choice(len(tpl_pts), 256, replace=False)]
    tpl_tree = cKDTree(tpl_sub)
    theta_g, thetas, Eg = geometric_horizontal(obj_pts, tpl_pts, steps, tpl_tree=tpl_tree, tpl_sub=tpl_sub)

    omega = []
    for i in range(len(Eg)):
        if Eg[i] < Eg[(i - 1) % len(Eg)] and Eg[i] <= Eg[(i + 1) % len(Eg)]:
            omega.append((thetas[i], Eg[i]))
    omega.sort(key=lambda x: x[1])
    omega = [w for w, _ in omega[:4]]
    if not omega:
        omega = [theta_g]

    tpl_parts = segment_semantic_parts(tpl_pts)
    obj_parts_base = segment_semantic_parts(obj_pts)

    tpl_part_trees = []
    for p in tpl_parts:
        p_sub = p if len(p) <= 256 else p[np.random.choice(len(p), 256, replace=False)]
        tpl_part_trees.append(cKDTree(p_sub))

    sigma_g = np.radians(15.0)
    step_size = thetas[1] - thetas[0] if len(thetas) > 1 else np.radians(10.0)
    NEIGHBORHOOD = 2
    eval_indices = set()
    for w in omega:
        closest_idx = int(np.argmin(np.abs(thetas - w)))
        for offset in range(-NEIGHBORHOOD, NEIGHBORHOOD + 1):
            eval_indices.add(closest_idx + offset)
    eval_indices = sorted(set(i % len(thetas) for i in eval_indices))

    Es_local = np.array([
        match_parts_chamfer(tpl_parts, [(rot_y(thetas[i]) @ part.T).T for part in obj_parts_base],
                            trees_a=tpl_part_trees)
        for i in eval_indices
    ])
    Es_mean = Es_local.mean()
    Es_std  = Es_local.std() + 1e-8
    Es_z_local = (Es_local - Es_mean) / Es_std

    best_theta, best_obj = theta_g, -np.inf
    for local_i, global_i in enumerate(eval_indices):
        t = thetas[global_i]
        ang_diffs = np.array([min(abs(t - w), 2 * np.pi - abs(t - w)) for w in omega])
        log_gauss = np.log(np.sum(np.exp(-0.5 * (ang_diffs / sigma_g) ** 2)) + 1e-8)

        LAMBDA = 1.0
        objective = -Es_z_local[local_i] + LAMBDA * log_gauss
        if objective > best_obj:
            best_obj, best_theta = objective, t
    return best_theta


# =====================================================================
# CANDIDATE ROTATION GENERATION
# =====================================================================

def generate_five_candidates(mesh, tpl_pts, category=None):
    """Generate exactly 5 candidate rotations per object.

    All 5 candidates incorporate a vertical alignment step first, so Qwen (and later
    the VNN) discriminates between genuinely different full-pose hypotheses rather than
    rejecting 3 useless yaw-only candidates every time.

    Candidate layout:
      0 — support_vertical  + geometric_horizontal
      1 — support_vertical  + semantic_horizontal      (different yaw estimator)
      2 — pca_vertical (no swap) + semantic_horizontal
      3 — pca_vertical (axis-swapped) + semantic_horizontal (handles degenerate PCA)
      4 — support_vertical 180-flip + semantic_horizontal  (upside-down hypothesis)
    """
    obj_pts = sample_points(mesh, seed=42)

    # ── Vertical estimator 1: physics-based (convex hull support surface) ──
    R_sup = support_vertical(mesh)
    pts_sup = (R_sup @ obj_pts.T).T

    # ── Vertical estimator 2a: PCA, no axis swap ──
    R_pca = pca_vertical(obj_pts, tpl_pts)
    pts_pca = (R_pca @ obj_pts.T).T

    # ── Vertical estimator 2b: PCA axis-swapped (v1↔v2 for near-degenerate eigenvalues) ──
    v1o, v2o = pca_axes(obj_pts)
    v1t, v2t = pca_axes(tpl_pts)
    Mt = np.stack([v1t, v2t, np.cross(v1t, v2t)], axis=1)
    tpl_parts_c = segment_semantic_parts(tpl_pts)
    obj_parts_c = segment_semantic_parts(obj_pts)
    np.random.seed(42)
    tpl_part_trees_c = [
        cKDTree(p if len(p) <= 256 else p[np.random.choice(len(p), 256, replace=False)])
        for p in tpl_parts_c
    ]
    best_swap_R, best_swap_cost = np.eye(3), np.inf
    # Only evaluate the swapped variant (non-swap is already R_pca above)
    for s1 in (1, -1):
        for s2 in (1, -1):
            u1, u2 = s1 * v2o, s2 * v1o   # swapped!
            Mo = np.stack([u1, u2, np.cross(u1, u2)], axis=1)
            R = Mt @ Mo.T
            rot_parts = [(R @ part.T).T for part in obj_parts_c]
            cost = match_parts_chamfer(tpl_parts_c, rot_parts, trees_a=tpl_part_trees_c)
            if cost < best_swap_cost:
                best_swap_cost, best_swap_R = cost, R.copy()
    R_pca_swap = best_swap_R
    pts_pca_swap = (R_pca_swap @ obj_pts.T).T

    # ── Vertical estimator 3: 180° flip of support_vertical (upside-down hypothesis) ──
    R_flip = rot_z(np.pi) @ R_sup
    pts_flip = (R_flip @ obj_pts.T).T

    # ── Horizontal refinement for each vertical hypothesis ──
    np.random.seed(42)
    theta0, _, _ = geometric_horizontal(pts_sup,     tpl_pts)
    theta1        = semantic_horizontal(pts_sup,     tpl_pts)
    theta2        = semantic_horizontal(pts_pca,     tpl_pts)
    theta3        = semantic_horizontal(pts_pca_swap, tpl_pts)
    theta4        = semantic_horizontal(pts_flip,    tpl_pts)

    R0 = rot_y(theta0) @ R_sup       # cand 0: support + geometric yaw
    R1 = rot_y(theta1) @ R_sup       # cand 1: support + semantic  yaw
    R2 = rot_y(theta2) @ R_pca       # cand 2: PCA (no swap) + semantic yaw
    R3 = rot_y(theta3) @ R_pca_swap  # cand 3: PCA (axis-swapped) + semantic yaw
    R4 = rot_y(theta4) @ R_flip      # cand 4: upside-down + semantic yaw

    return [
        ("Sup+Geo",     R0),
        ("Sup+Sem",     R1),
        ("PCA+Sem",     R2),
        ("PCAswap+Sem", R3),
        ("Flip+Sem",    R4),
    ]



# =====================================================================
# QUALITY GATE
# =====================================================================

def passes_quality(mesh):
    if not hasattr(mesh, "vertices") or len(mesh.vertices) < min(MIN_FACES, 50):
        return False, "too few vertices"

    ext = np.ptp(mesh.vertices, axis=0)
    if ext.max() <= 0 or (ext.min() / ext.max()) < MIN_THICKNESS_RATIO:
        return False, "thin-shell"

    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        if len(mesh.faces) < MIN_FACES:
            return False, "too few faces"

    return True, "ok"


# =====================================================================
# RENDERING
# =====================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MeshRenderer:
    """Configurable 3D Mesh Renderer with pyrender (GPU) and matplotlib (CPU) backends."""

    def __init__(self, render_size=RENDER_SIZE, use_pyrender=USE_PYRENDER):
        self.render_size = render_size
        self.use_pyrender = use_pyrender
        self.backend = None
        self._pyrenderer = None
        self._pyrender = None
        self._init_backend()

    def __del__(self):
        if self._pyrenderer is not None:
            try:
                self._pyrenderer.delete()
            except Exception:
                pass

    def _init_backend(self):
        if not self.use_pyrender:
            self.backend = "matplotlib (CPU)"
            print(f"  [Renderer] Using Matplotlib Renderer (USE_PYRENDER=False)")
            return

        try:
            for platform in ["egl", "osmesa"]:
                cmd = (
                    f"import os; os.environ['PYOPENGL_PLATFORM'] = '{platform}'; "
                    f"import pyrender; r = pyrender.OffscreenRenderer(64, 64); r.delete()"
                )
                res = subprocess.run(
                    [sys.executable, "-c", cmd],
                    capture_output=True, timeout=60
                )
                if res.returncode == 0:
                    os.environ["PYOPENGL_PLATFORM"] = platform
                    import pyrender
                    self._pyrender = pyrender
                    self._pyrenderer = pyrender.OffscreenRenderer(
                        viewport_width=self.render_size,
                        viewport_height=self.render_size
                    )
                    self.backend = f"pyrender ({platform.upper()})"
                    print(f"  [Renderer] Initialized Hardware Renderer: {self.backend}")
                    return
        except Exception:
            pass

        self.backend = "matplotlib (CPU fallback)"
        print("  [!] Pyrender unavailable. Falling back to Matplotlib.")

    def render(self, mesh, R):
        if self._pyrenderer is not None:
            try:
                return self._render_pyrender(mesh, R)
            except Exception:
                pass
        return self._render_matplotlib(mesh, R)

    def _render_pyrender(self, mesh, R):
        verts = (R @ np.asarray(mesh.vertices).T).T
        verts = verts - verts.mean(axis=0)
        faces = getattr(mesh, 'faces', None)
        if faces is None or len(faces) == 0:
            faces = np.zeros((0, 3), dtype=np.int64)

        if len(faces) == 0:
            return self._render_matplotlib(mesh, R)

        mesh_oriented = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        material = self._pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.55, 0.60, 0.68, 1.0],
            roughnessFactor=0.6, metallicFactor=0.1
        )
        py_mesh = self._pyrender.Mesh.from_trimesh(mesh_oriented, material=material, smooth=True)

        scene = self._pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.45, 0.45, 0.45]
        )
        scene.add(py_mesh)

        max_extent = float(np.abs(verts).max() * 1.1 + 1e-4)
        fov_y = np.pi / 3.0
        dist = max_extent / np.tan(fov_y / 2.0)

        camera = self._pyrender.PerspectiveCamera(yfov=fov_y)
        cam_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dist],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=cam_pose)

        key_light = self._pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.array([
            [1.0, 0.0, 0.0, dist * 0.4],
            [0.0, 1.0, 0.0, dist * 0.6],
            [0.0, 0.0, 1.0, dist * 0.8],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(key_light, pose=light_pose)

        color, _ = self._pyrenderer.render(scene)
        return Image.fromarray(color)

    def _render_matplotlib(self, mesh, R):
        verts = (R @ np.asarray(mesh.vertices).T).T
        verts = verts - verts.mean(axis=0)
        faces = getattr(mesh, 'faces', None)
        if faces is None or len(faces) == 0:
            faces = np.zeros((0, 3), dtype=np.int64)

        fig = plt.figure(figsize=(self.render_size / 100, self.render_size / 100), dpi=100)
        try:
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

            lim = float(np.abs(verts).max() * 1.05 + 1e-6)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=0, azim=0)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", transparent=False, facecolor="white")
            buf.seek(0)
            img = Image.open(buf).convert("RGB").copy()
            buf.close()
            return img
        finally:
            plt.close(fig)


_RENDERER_LOCK = threading.Lock()
_GLOBAL_RENDERER = None

def get_mesh_renderer():
    global _GLOBAL_RENDERER
    if _GLOBAL_RENDERER is None:
        with _RENDERER_LOCK:
            if _GLOBAL_RENDERER is None:
                _GLOBAL_RENDERER = MeshRenderer(render_size=RENDER_SIZE)
    return _GLOBAL_RENDERER

def render_to_pil(mesh, R):
    renderer = get_mesh_renderer()
    return renderer.render(mesh, R)

def render_mesh_png(mesh, R, out_path):
    img = render_to_pil(mesh, R)
    img.save(out_path)


# =====================================================================
# QWEN VLM COMPOSITE PANEL RENDERER  (NEW)
# =====================================================================

def render_candidate_panel(tpl_mesh, obj_mesh, candidate_rotations, render_size=None):
    """Render a composite image: [TEMPLATE] | [0] | [1] | [2] | [3] | [4]

    The template mesh is rendered at identity (canonical pose).
    Each candidate shows the object mesh rotated by the candidate's rotation matrix.
    Clear text labels are drawn above each panel for Qwen to read.

    Args:
        tpl_mesh:  Trimesh — the category template mesh (rendered at Identity)
        obj_mesh:  Trimesh — the object being evaluated
        candidate_rotations:  list of 5 numpy (3,3) rotation matrices
        render_size: pixel size per sub-panel (default: QWEN_RENDER_SIZE)

    Returns:
        PIL.Image — composite panel image
    """
    if render_size is None:
        render_size = QWEN_RENDER_SIZE

    HEADER_H = 48
    N_PANELS = 6  # 1 template + 5 candidates
    panel_w = render_size
    total_w = N_PANELS * panel_w
    total_h = render_size + HEADER_H

    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)

    # Try to get a readable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

    def _center_text(text, x_start, width):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        return x_start + (width - tw) // 2

    # Panel 0: Template at identity rotation
    tpl_img = render_to_pil(tpl_mesh, np.eye(3))
    tpl_img = tpl_img.resize((render_size, render_size), Image.LANCZOS)
    composite.paste(tpl_img, (0, HEADER_H))
    tx = _center_text("TEMPLATE", 0, panel_w)
    draw.text((tx, 8), "TEMPLATE", fill=(200, 0, 0), font=font)

    # Panels 1-5: Candidates
    for i, R_cand in enumerate(candidate_rotations):
        cand_img = render_to_pil(obj_mesh, R_cand)
        cand_img = cand_img.resize((render_size, render_size), Image.LANCZOS)
        x_off = (i + 1) * panel_w
        composite.paste(cand_img, (x_off, HEADER_H))
        label = str(i)
        lx = _center_text(label, x_off, panel_w)
        draw.text((lx, 8), label, fill=(0, 0, 180), font=font)

    # Draw vertical separator lines between panels
    for i in range(1, N_PANELS):
        x = i * panel_w
        draw.line([(x, 0), (x, total_h)], fill=(180, 180, 180), width=2)

    return composite


# =====================================================================
# QWEN VLM ORACLE  (NEW)
# =====================================================================

QWEN_PROMPT = (
    "The leftmost image labeled TEMPLATE shows a 3D object in the correct canonical orientation. "
    "The 5 images to the right labeled 0 through 4 show a similar object in different rotations. "
    "Which numbered image (0, 1, 2, 3, or 4) has the orientation most similar to the TEMPLATE? "
    "Consider which surfaces face forward and which direction is up. "
    "Reply with ONLY the single digit number."
)


def qwen_select_best_candidate(panel_image, debug_save_path=None):
    """Query Qwen2.5-VL via llama-mtmd-cli to select the best candidate from a composite panel.

    Args:
        panel_image:      PIL.Image of the composite panel [TEMPLATE | 0 | 1 | 2 | 3 | 4]
        debug_save_path:  If provided, save the panel image here for debugging

    Returns:
        int (0-4) on success, None if Qwen is unavailable or parsing fails
    """
    if not _QWEN_AVAILABLE:
        return None

    # Save panel to temp file
    os.makedirs(CACHE_DIR, exist_ok=True)
    panel_path = os.path.join(CACHE_DIR, f"_qwen_panel_{os.getpid()}_{threading.get_ident()}.png")
    panel_image.save(panel_path, format="PNG")

    if debug_save_path:
        panel_image.save(debug_save_path, format="PNG")

    # Free PyTorch GPU cache before launching Qwen subprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cmd = [
        LLAMA_MTMD_CLI,
        "-m", str(QWEN_MODEL_PATH),
        "--mmproj", str(QWEN_MMPROJ_PATH),
        "-ngl", "999",
        "-c", "4096",
        "-n", "16",                   # Enough tokens for any short answer
        "--temp", "0.0",              # Greedy decoding for determinism
        "--image", panel_path,
        "-p", QWEN_PROMPT,
    ]

    if QWEN_DUAL_GPU:
        cmd.extend(["-dev", "CUDA0,CUDA1", "-sm", "layer", "-ts", "1,1"])

    if GRAMMAR_FILE and os.path.exists(GRAMMAR_FILE):
        cmd.extend(["--grammar-file", GRAMMAR_FILE])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=QWEN_TIMEOUT,
        )

        output = result.stdout.strip()
        
        import re
        # Try a direct standalone digit first (most reliable)
        for line in reversed(output.splitlines()):
            line = line.strip()
            if re.fullmatch(r"[01234]", line):
                return int(line)

        # Try: last occurrence of a standalone digit word in the full output
        tokens = re.findall(r'\b([01234])\b', output)
        if tokens:
            selected = int(tokens[-1])  # take the LAST match, not the first
            return selected

        # Fallback: scan stderr for a standalone digit line
        for line in result.stderr.splitlines():
            line = line.strip()
            if re.fullmatch(r"[01234]", line):
                return int(line)

        print(f"  [Qwen] Could not parse output: stdout='{output[:80]}' stderr='{result.stderr[-200:]}'")
        return None

    except subprocess.TimeoutExpired:
        print(f"  [Qwen] Timeout after {QWEN_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"  [Qwen] Subprocess error: {type(e).__name__}: {e}")
        return None
    finally:
        # Clean up temp panel file
        try:
            if os.path.exists(panel_path):
                os.remove(panel_path)
        except Exception:
            pass


# =====================================================================
# CHAMFER FALLBACK LABELING (kept as safety net)
# =====================================================================

@torch.no_grad()
def compute_auto_label(candidate_pts_list, template_pts, candidate_normals_list=None, template_normals=None):
    """Chamfer-based fallback label — used ONLY when Qwen is unavailable."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cands_t = torch.tensor(np.stack(candidate_pts_list), dtype=torch.float32, device=device)
        tpl_t = torch.tensor(template_pts, dtype=torch.float32, device=device).unsqueeze(0)

        dists = torch.cdist(cands_t, tpl_t)
        min_c2t, idx_c2t = dists.min(dim=2)
        min_t2c, idx_t2c = dists.min(dim=1)

        cd_vals = (min_c2t.mean(dim=1) + min_t2c.mean(dim=1))

        if candidate_normals_list and template_normals is not None:
            cn_t = torch.tensor(np.stack(candidate_normals_list), dtype=torch.float32, device=device)
            tn_t = torch.tensor(template_normals, dtype=torch.float32, device=device)

            tn_matched = tn_t[idx_c2t]
            dot_c2t = (cn_t * tn_matched).sum(dim=-1)

            cn_matched = torch.gather(cn_t, 1, idx_t2c.unsqueeze(-1).expand(-1, -1, 3))
            dot_t2c = (tn_t.unsqueeze(0) * cn_matched).sum(dim=-1)

            normal_loss = (1.0 - dot_c2t).mean(dim=1) + (1.0 - dot_t2c).mean(dim=1)
            cd_vals = cd_vals + 0.5 * normal_loss

        distances = cd_vals.tolist()
        best_idx = int(cd_vals.argmin().item())
        return best_idx, distances
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        distances = []
        for i, cpts in enumerate(candidate_pts_list):
            cn = candidate_normals_list[i] if candidate_normals_list else None
            d = chamfer_distance(cpts, template_pts, na=cn, nb=template_normals)
            distances.append(d)
        return int(np.argmin(distances)), distances


# =====================================================================
# CLIP-BASED CATEGORY DISCOVERY
# =====================================================================

def _extract_tensor(feats):
    if isinstance(feats, torch.Tensor):
        return feats
    if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
        return feats.text_embeds
    if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
        return feats.image_embeds
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
        return feats.last_hidden_state[:, 0]
    return feats[0]


class CategoryDiscovery:
    def __init__(self, categories=None, model_name=None):
        if categories is None:
            categories = get_categories()
        if model_name is None:
            model_name = CLIP_MODEL_NAME

        self.categories = list(categories)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._text_features = None
        self._available = True

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

            prompts = [f"a 3D model of a {cat.replace('_', ' ')}" for cat in self.categories]
            text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.no_grad():
                text_feats = self._model.get_text_features(**text_inputs)
                text_feats = _extract_tensor(text_feats)
                self._text_features = F.normalize(text_feats, dim=-1)

            print(f"  [CLIP] Loaded {self.model_name} on {self.device}")
        except Exception as e:
            print(f"  [!] CLIP unavailable ({e}).")
            self._available = False

    def identify(self, mesh, n_views=8):
        self._load()
        if not self._available or self._model is None or self._text_features is None:
            return None, 0.0, {}

        VIEW_ANGLES = [
            (0, 0), (90, 0), (180, 0), (270, 0),
            (45, 30), (135, 30), (225, 30), (315, 30),
        ]
        images = []
        for az, el in VIEW_ANGLES[:n_views]:
            R = rot_y(np.radians(az)) @ rot_x(np.radians(el))
            img = render_to_pil(mesh, R)
            images.append(img)

        image_inputs = self._processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_feats = self._model.get_image_features(**image_inputs)
            image_feats = _extract_tensor(image_feats)
            image_feats = F.normalize(image_feats, dim=-1)
            avg_feat = image_feats.mean(dim=0, keepdim=True)

            logit_scale = self._model.logit_scale.exp()
            scores = (logit_scale * (avg_feat @ self._text_features.T)).squeeze(0)
            probs = torch.softmax(scores, dim=0)

        best_idx = int(probs.argmax().item())
        scores_dict = {cat: float(p) for cat, p in zip(self.categories, probs.tolist())}
        return self.categories[best_idx], float(probs[best_idx].item()), scores_dict


# =====================================================================
# DATASET LOADING
# =====================================================================

def load_category_uids(category, n_needed=None, verbose=True):
    index = get_dataset_index()
    cat_key = category.lower().strip()
    files = index.get(cat_key, [])

    if n_needed is not None:
        selected_files = files[:n_needed]
    else:
        selected_files = list(files)

    if verbose:
        print(f"  Found {len(selected_files)} '{category}' objects in CanoVerse.")
    return selected_files


def get_render_categories(categories=None, max_categories=CATEGORIES_TO_RENDER, min_glbs=MIN_GLBS_TO_RENDER):
    if categories is None:
        categories = get_categories()

    qualifying = []
    for cat in categories:
        uids = load_category_uids(cat, n_needed=min_glbs, verbose=False)
        if len(uids) >= min_glbs:
            qualifying.append(cat)

    if qualifying:
        if max_categories is not None and len(qualifying) > max_categories:
            selected = random.sample(qualifying, max_categories)
        else:
            selected = list(qualifying)
            random.shuffle(selected)
    else:
        print(f"  [!] No category found with >= {min_glbs} .glb files.")
        available = [cat for cat in categories if len(load_category_uids(cat, n_needed=1, verbose=False)) > 0]
        if max_categories is not None and len(available) > max_categories:
            selected = random.sample(available, max_categories)
        else:
            selected = list(available)
            random.shuffle(selected)

    print(f"\n  [INFO] Selected {len(selected)} categories for rendering: {selected}\n")
    return selected


def download_objects(uids):
    return {uid: uid for uid in uids}


# =====================================================================
# PYTORCH DATASETS
# =====================================================================

class AgnosticRankingDataset(Dataset):
    """Cross-category ranking dataset. Identical to original — label source is transparent."""
    def __init__(self, all_category_data, augment=False, n_rotations=None, category_to_idx=None):
        if n_rotations is None:
            n_rotations = N_ROTATIONS
        self.augment = augment
        self.n_rotations = n_rotations if augment else 1
        self.category_to_idx = dict(category_to_idx) if category_to_idx is not None else {}

        templates = []
        candidates = []
        best_idxs = []
        soft_scores = []
        category_idxs = []
        candidate_rotations = []
        r_rands = []

        for cat_data in all_category_data:
            category = cat_data["category"]
            if category not in self.category_to_idx:
                self.category_to_idx[category] = len(self.category_to_idx)
            cat_idx = self.category_to_idx[category]
            tpl = torch.tensor(cat_data["template_pts"], dtype=torch.float32)

            for obj in cat_data["objects"]:
                cands = obj.get("candidate_pts", [])
                if len(cands) != 5:
                    continue
                cand_pts = torch.stack([
                    torch.tensor(cands[i], dtype=torch.float32)
                    for i in range(5)
                ])

                templates.append(tpl)
                candidates.append(cand_pts)
                best_idxs.append(obj["best_idx"])
                category_idxs.append(cat_idx)
                
                # Default to hard 1-hot if soft_scores are missing in older cached datasets
                s_scores = obj.get("soft_scores", None)
                if s_scores is None:
                    s_scores = [1.0 if i == obj["best_idx"] else 0.0 for i in range(5)]
                soft_scores.append(s_scores)

                c_rots = obj.get("candidate_rotations", [np.eye(3)] * 5)
                candidate_rotations.append(torch.tensor(np.stack(c_rots), dtype=torch.float32))
                r_rands.append(torch.tensor(obj.get("R_rand", np.eye(3)), dtype=torch.float32))

        if not templates:
            self.templates = torch.zeros((0, N_POINTS, 3), dtype=torch.float32)
            self.candidates = torch.zeros((0, 5, N_POINTS, 3), dtype=torch.float32)
            self.best_idxs = torch.zeros((0,), dtype=torch.long)
            self.soft_scores = torch.zeros((0, 5), dtype=torch.float32)
            self.category_idxs = torch.zeros((0,), dtype=torch.long)
            self.candidate_rotations = torch.zeros((0, 5, 3, 3), dtype=torch.float32)
            self.r_rands = torch.zeros((0, 3, 3), dtype=torch.float32)
        else:
            self.templates = torch.stack(templates)
            self.candidates = torch.stack(candidates)
            self.best_idxs = torch.tensor(best_idxs, dtype=torch.long)
            self.soft_scores = torch.tensor(soft_scores, dtype=torch.float32)
            self.category_idxs = torch.tensor(category_idxs, dtype=torch.long)
            self.candidate_rotations = torch.stack(candidate_rotations)
            self.r_rands = torch.stack(r_rands)

    def __len__(self):
        return len(self.best_idxs) * self.n_rotations

    def __getitem__(self, idx):
        item_idx = idx % len(self.best_idxs)

        template = self.templates[item_idx].clone()
        candidates = self.candidates[item_idx].clone()
        best_idx = self.best_idxs[item_idx]
        category_idx = self.category_idxs[item_idx]
        cand_rots = self.candidate_rotations[item_idx].clone()
        r_rand = self.r_rands[item_idx].clone()

        if self.augment and self.n_rotations > 1:
            R = random_rotation_matrix_torch()
            template = template @ R.T
            candidates = candidates @ R.T
            r_rand = R @ r_rand

        if self.augment:
            N = template.shape[0]

            sigma = 0.007
            template  = template  + sigma * torch.randn_like(template)
            candidates = candidates + sigma * torch.randn(candidates.shape[0], N, 3)

            perm_t = torch.randperm(N)
            template = template[perm_t]
            perm_c = torch.randperm(N)
            candidates = candidates[:, perm_c, :]

            scale = 0.95 + 0.10 * torch.rand(1).item()
            template  = template  * scale
            candidates = candidates * scale

        return {
            "template_pts": template,
            "candidate_pts": candidates,
            "best_idx": best_idx,
            "soft_scores": self.soft_scores[item_idx],
            "category_idx": category_idx,
            "candidate_rotations": cand_rots,
            "R_rand": r_rand,
        }


# =====================================================================
# TRAINING DATA PREPARATION  (MODIFIED: QWEN VLM LABELS)
# =====================================================================

def prepare_training_data(category, n_objects=None, force_rebuild=False):
    """Prepare training data with Qwen VLM labels (Chamfer fallback if Qwen unavailable)."""
    if n_objects is None and USE_ONLY_N_SAMPLES:
        n_objects = N_SAMPLES
    cache_file = get_cache_file(category)

    if not force_rebuild and os.path.exists(cache_file):
        print(f"  Loading cached data from {cache_file}...")
        try:
            return torch.load(cache_file, weights_only=False)
        except (RuntimeError, EOFError, Exception) as e:
            print(f"  [!] Cache file is corrupt ({type(e).__name__}: {e}). Rebuilding...")
            os.remove(cache_file)

    print("\n" + "=" * 60)
    print(f"  PREPARING TRAINING DATA [{category.upper()}]")
    if _QWEN_AVAILABLE:
        print(f"  Label Source: QWEN VLM DISTILLATION")
    else:
        print(f"  Label Source: CHAMFER DISTANCE (fallback)")
    print("=" * 60)

    needed = (n_objects + 1) if (USE_ONLY_N_SAMPLES and n_objects is not None) else None
    uids = load_category_uids(category, (needed + 50) if needed else None)
    if len(uids) == 0:
        print(f"  [!] No objects found for category '{category}'. Skipping.")
        return None
    elif len(uids) == 1:
        uids = [uids[0], uids[0]]

    random.shuffle(uids)

    download_uids = uids[:min(len(uids), needed + 50)] if needed else uids
    objs = download_objects(download_uids)

    # ---- Medoid template selection (unchanged) ----
    num_medoid_cands = min(15, len(uids))
    medoid_pts_list = []
    medoid_uids = uids[:num_medoid_cands]

    for uid in medoid_uids:
        tpl_path = objs.get(uid)
        if tpl_path is None: continue
        try:
            m = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
            pts = sample_points(m, seed=42, with_normals=False)
            medoid_pts_list.append((uid, pts))
        except Exception as e:
            print(f"  [!] Could not load template candidate ({type(e).__name__}: {e}).")
            continue

    if len(medoid_pts_list) > 1:
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            pts_tensor = torch.tensor(
                np.stack([item[1] for item in medoid_pts_list]),
                dtype=torch.float32, device=dev
            )
            K = len(medoid_pts_list)
            scores = torch.zeros(K, device=dev)
            with torch.no_grad():
                for i in range(K):
                    for j in range(i + 1, K):
                        dists = torch.cdist(pts_tensor[i:i+1], pts_tensor[j:j+1]).squeeze(0)
                        cd = (dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()).item()
                        scores[i] += cd
                        scores[j] += cd
            best_medoid_uid = medoid_pts_list[scores.argmin().item()][0]
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            best_medoid_score = float('inf')
            best_medoid_uid = medoid_pts_list[0][0]
            for i, (uid_i, pts_i) in enumerate(medoid_pts_list):
                score = 0
                for j, (uid_j, pts_j) in enumerate(medoid_pts_list):
                    if i != j:
                        score += chamfer_distance(pts_i, pts_j)
                if score < best_medoid_score:
                    best_medoid_score = score
                    best_medoid_uid = uid_i
    elif len(medoid_pts_list) == 1:
        best_medoid_uid = medoid_pts_list[0][0]
    else:
        best_medoid_uid = medoid_uids[0]

    template_uid = best_medoid_uid
    tpl_path = objs.get(template_uid)
    if tpl_path is None:
        raise RuntimeError(f"Template {template_uid} failed to resolve!")
    tpl_mesh = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
    template_pts, template_normals = sample_points(tpl_mesh, seed=42, with_normals=True)

    # ---- Process objects ----
    data_objects = []
    discarded = 0
    qwen_labels = 0
    chamfer_labels = 0

    # Create debug panel directory
    panel_debug_dir = os.path.join(QWEN_PANEL_DIR, category)
    if _QWEN_AVAILABLE:
        os.makedirs(panel_debug_dir, exist_ok=True)

    total_items = (min(len(uids)-1, n_objects+50)) if (USE_ONLY_N_SAMPLES and n_objects is not None) else (len(uids)-1)
    for idx_obj, uid in enumerate(uids[1:]):
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
        except Exception as e:
            print(f"  [!] Failed to load mesh ({type(e).__name__}: {e}). Skipping.")
            continue
        ok, reason = passes_quality(mesh)
        if not ok:
            discarded += 1
            continue

        # Apply random rotation to simulate unaligned input
        has_custom_normals = hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0
        if has_custom_normals:
            saved_normals = mesh.vertex_normals.copy()

        T_rand = np.eye(4)
        R_rand = random_realistic_input_rotation()
        T_rand[:3, :3] = R_rand
        mesh.apply_transform(T_rand)

        if has_custom_normals:
            mesh.vertex_normals = (R_rand @ saved_normals.T).T

        # Generate 5 candidate rotations
        candidates = generate_five_candidates(mesh, template_pts, category=category)
        obj_pts, obj_normals = sample_points(mesh, with_normals=True)
        cand_pts_list, cand_normals_list, cand_rot_list, cand_name_list = [], [], [], []
        for name, R in candidates:
            cand_pts_list.append((R @ obj_pts.T).T)
            cand_normals_list.append((R @ obj_normals.T).T)
            cand_rot_list.append(R)
            cand_name_list.append(name)

        # ---- LABEL ASSIGNMENT: Qwen VLM (primary) or Chamfer (fallback) ----
        best_idx = None
        label_source = "chamfer"
        safe_uid = os.path.basename(uid)

        # Check if ALL candidates are terrible (far from ground truth)
        min_gt_err = min(geodesic_angle_deg(R @ R_rand, np.eye(3)) for R in cand_rot_list)
        if min_gt_err > MAX_CANDIDATE_GT_ERROR:
            print(f"\r  [!] Discarding {safe_uid}: all candidates are bad (min GT error {min_gt_err:.1f}° > {MAX_CANDIDATE_GT_ERROR}°)                 \n", end="")
            discarded += 1
            continue

        if _QWEN_AVAILABLE:
            # Render composite panel for Qwen
            debug_path = os.path.join(panel_debug_dir, f"panel_{idx_obj:04d}.png")
            panel = render_candidate_panel(tpl_mesh, mesh, cand_rot_list)
            qwen_idx = qwen_select_best_candidate(panel, debug_save_path=debug_path)

            if qwen_idx is not None:
                best_idx = qwen_idx
                label_source = "qwen"
                qwen_labels += 1

        if best_idx is None:
            # Chamfer fallback
            best_idx, _ = compute_auto_label(
                cand_pts_list, template_pts,
                candidate_normals_list=cand_normals_list,
                template_normals=template_normals
            )
            label_source = "chamfer"
            chamfer_labels += 1
            
        soft_scores = compute_geodesic_soft_labels(best_idx, cand_rot_list)

        data_objects.append({
            "uid": safe_uid,
            "candidate_pts": cand_pts_list,
            "candidate_rotations": cand_rot_list,
            "candidate_names": cand_name_list,
            "best_idx": best_idx,
            "soft_scores": soft_scores.tolist(),
            "label_source": label_source,
            "R_rand": R_rand,
        })

        if (idx_obj + 1) % 10 == 0:
            print(f"\r  Processing [{category}]: {idx_obj+1}/{total_items} | "
                  f"Kept: {len(data_objects)} | Qwen: {qwen_labels} | Chamfer: {chamfer_labels} | "
                  f"Discarded: {discarded}", end="", flush=True)

    print(f"\r  Processing [{category}]: Completed {len(data_objects)} objects "
          f"(Qwen: {qwen_labels}, Chamfer: {chamfer_labels}, Discarded: {discarded}).        ")

    print(f"\n  Final: {len(data_objects)} objects, discarded {discarded} total.")
    from collections import Counter
    counts = Counter(obj["best_idx"] for obj in data_objects)
    print(f"  Label distribution: {', '.join(f'cand_{i}: {counts.get(i,0)}' for i in range(5))}")
    src_counts = Counter(obj.get("label_source", "chamfer") for obj in data_objects)
    print(f"  Label sources: {dict(src_counts)}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {"category": category, "template_pts": template_pts,
            "template_uid": template_uid, "objects": data_objects}
    torch.save(data, cache_file)
    return data


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_one_epoch_ranking(model, dataloader, optimizer, device, epoch,
                             use_category=False, scaler=None):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    use_amp = (device.type == "cuda")

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        template = batch["template_pts"].to(device)
        candidates = batch["candidate_pts"].to(device)
        best_idx = batch["best_idx"].to(device)
        soft_scores = batch["soft_scores"].to(device)
        cat_idx = batch["category_idx"].to(device) if use_category else None

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model.forward_all_candidates(template, candidates, category_idx=cat_idx)
            # Use geometric soft scores instead of hard labels (no need for uniform label_smoothing)
            loss = F.cross_entropy(logits, soft_scores)
            scaled_loss = loss / GRAD_ACCUM_STEPS

        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=1)
        correct += (preds == best_idx).sum().item()
        total += batch["template_pts"].shape[0]
        total_loss += loss.item() * batch["template_pts"].shape[0]

    if len(dataloader) % GRAD_ACCUM_STEPS != 0:
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return {"loss": total_loss / total if total else 0, "selection_accuracy": correct / total if total else 0}


@torch.no_grad()
def evaluate_ranking(model, dataloader, device, phase="VAL", use_category=False):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    geodesic_errors = []

    for batch in dataloader:
        template = batch["template_pts"].to(device)
        candidates = batch["candidate_pts"].to(device)
        best_idx = batch["best_idx"].to(device)
        cat_idx = batch["category_idx"].to(device) if use_category else None

        logits = model.forward_all_candidates(template, candidates, category_idx=cat_idx)

        loss = F.cross_entropy(logits, best_idx)
        preds = logits.argmax(dim=1)

        correct += (preds == best_idx).sum().item()
        total += template.shape[0]
        total_loss += loss.item() * template.shape[0]

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(best_idx.cpu().tolist())

        if "candidate_rotations" in batch and "R_rand" in batch:
            cand_rots = batch["candidate_rotations"].cpu().numpy()
            r_rands = batch["R_rand"].cpu().numpy()
            preds_np = preds.cpu().numpy()
            for b in range(len(preds_np)):
                pred_cand_R = cand_rots[b, preds_np[b]]
                R_rand_b = r_rands[b]
                R_pred_canonical = pred_cand_R @ R_rand_b
                err = geodesic_angle_deg(R_pred_canonical, np.eye(3))
                geodesic_errors.append(err)

    results = {
        "loss": total_loss / total if total else 0,
        "selection_accuracy": correct / total if total else 0,
        "selection_correct": correct,
        "selection_total": total,
        "predictions": all_preds,
        "labels": all_labels,
    }

    if geodesic_errors:
        errs = np.array(geodesic_errors)
        results["acc_at_10"] = float((errs <= 10.0).mean())
        results["acc_at_30"] = float((errs <= 30.0).mean())
        results["mean_geodesic_error"] = float(np.mean(errs))
        results["median_geodesic_error"] = float(np.median(errs))

    return results


def compute_geodesic_metrics(predicted_rotations, gt_rotations):
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


def split_data(objects, seed=None):
    if seed is None:
        seed = RANDOM_SEED
    rng = random.Random(seed)
    indices = list(range(len(objects)))
    rng.shuffle(indices)
    n = len(objects)

    if n == 0:
        return [], [], []

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    if n_train == 0: n_train = 1
    if n_val == 0: n_val = 1

    train_objs = [objects[i] for i in indices[:n_train]]

    if n_train < n:
        val_objs = [objects[i] for i in indices[n_train:min(n_train+n_val, n)]]
    else:
        val_objs = [objects[indices[-1]]]

    if n_train + n_val < n:
        test_objs = [objects[i] for i in indices[n_train+n_val:]]
    else:
        test_objs = [objects[indices[-1]]]

    if not val_objs: val_objs = list(train_objs)
    if not test_objs: test_objs = list(train_objs)

    return train_objs, val_objs, test_objs


# =====================================================================
# CATEGORY-AGNOSTIC TRAINING
# =====================================================================

def train_agnostic(categories=None):
    if categories is None:
        categories = get_categories()

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

    print("\n  Preparing data for all categories...")
    all_data = []
    for cat in categories:
        data = prepare_training_data(cat)
        if data is not None:
            all_data.append(data)

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

    train_ds = AgnosticRankingDataset(train_cat_data, augment=True, n_rotations=N_ROTATIONS)
    val_ds = AgnosticRankingDataset(val_cat_data, augment=False, category_to_idx=train_ds.category_to_idx)
    test_ds = AgnosticRankingDataset(test_cat_data, augment=False, category_to_idx=train_ds.category_to_idx)

    n_cats = len(train_ds.category_to_idx)
    print(f"\n  Total train samples: {len(train_ds)} ({len(train_ds)//N_ROTATIONS} base × {N_ROTATIONS} aug)")
    print(f"  Total val samples:   {len(val_ds)}")
    print(f"  Total test samples:  {len(test_ds)}")
    print(f"  Category count:      {n_cats}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    model = VNCandidateSelector(
        n_points=N_POINTS,
        feat_dims=AGNOSTIC_FEAT_DIMS,
        hidden_dim=AGNOSTIC_HIDDEN_DIM,
        n_categories=n_cats,
        category_embed_dim=N_CATEGORY_EMBED,
    ).to(device)
    print(f"\n{model_summary(model)}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    agnostic_path = get_agnostic_model_path()
    latest_agnostic_path = get_agnostic_latest_model_path()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_sel_acc = -1.0
    patience_counter = 0
    start_epoch = 1

    if os.path.exists(latest_agnostic_path):
        try:
            print(f"\n  [+] Found checkpoint at {latest_agnostic_path}. Resuming...")
            ckpt = torch.load(latest_agnostic_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_sel_acc = ckpt.get("best_val_sel_acc", ckpt.get("val_selection_accuracy", -1.0))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"  [+] Resumed from epoch {start_epoch}/{NUM_EPOCHS}")
        except Exception as e:
            print(f"  [!] Failed to load checkpoint ({e}). Starting fresh...")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_start = time.time()

    if start_epoch <= NUM_EPOCHS:
        epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS + 1), desc="  [AGNOSTIC] Training", unit="epoch")
        for epoch in epoch_pbar:
            t0 = time.time()
            tm = train_one_epoch_ranking(model, train_loader, optimizer, device, epoch,
                                         use_category=True, scaler=scaler)
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
    if "acc_at_10" in test_m:
        print(f"    Acc@10° (Geodesic): {test_m['acc_at_10']*100:.1f}%")
        print(f"    Acc@30° (Geodesic): {test_m['acc_at_30']*100:.1f}%")
        print(f"    Mean Geodesic Err:  {test_m['mean_geodesic_error']:.2f}°")
        print(f"    Median Geodesic Err:{test_m['median_geodesic_error']:.2f}°")
    print(f"    Loss:               {test_m['loss']:.4f}")

    print("\n  Per-Category Breakdown:")
    for cat_data_entry in test_cat_data:
        cat = cat_data_entry["category"]
        cat_ds = AgnosticRankingDataset([cat_data_entry], augment=False, category_to_idx=train_ds.category_to_idx)
        if len(cat_ds) == 0:
            continue
        cat_loader = DataLoader(cat_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        cat_m = evaluate_ranking(model, cat_loader, device, cat.upper(), use_category=True)
        acc10_str = f"| Acc@10: {cat_m['acc_at_10']*100:.1f}%" if "acc_at_10" in cat_m else ""
        print(f"    {cat:12s}: SelAcc {cat_m['selection_accuracy']:.3f} "
              f"({cat_m['selection_correct']}/{cat_m['selection_total']}) {acc10_str}")

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
    """Generate candidates + VNN selection for a category (inference — no Qwen)."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vnn_model_obj = None
    vnn_available = False
    vnn_cat_idx = None

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

    objs = download_objects(uids)
    cache_file = get_cache_file(category)
    if os.path.exists(cache_file):
        data = torch.load(cache_file, weights_only=False)
        tpl_pts = data["template_pts"]
        template_uid = data.get("template_uid", uids[0])
        tpl_path = objs.get(template_uid) or download_objects([template_uid]).get(template_uid)
        if tpl_path:
            try:
                tpl_mesh = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
                render_mesh_png(tpl_mesh, np.eye(3), os.path.join(out_dir, "template.png"))
            except Exception:
                pass
    else:
        template_uid = uids[0]
        tpl_path = objs.get(template_uid)
        if tpl_path is None:
            return 0, 0
        tpl_mesh = normalize_to_unit_sphere(load_mesh_or_pc(tpl_path))
        tpl_pts = sample_points(tpl_mesh)
        render_mesh_png(tpl_mesh, np.eye(3), os.path.join(out_dir, "template.png"))

    manifest = {"category": category, "template": "template.png",
                "vnn_enabled": vnn_available, "objects": []}
    prepared, discarded = 0, 0

    for uid in uids[1:]:
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
            continue

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
            cand_pts_list = [(R @ obj_pts.T).T for R in cand_rotations]
            cand_tensor = torch.tensor(np.stack(cand_pts_list), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = vnn_model_obj.forward_all_candidates(tpl_tensor, cand_tensor, category_idx=cat_tensor)
                scores = logits.squeeze(0).cpu().tolist()
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

        entry = {"uid": safe_uid, "candidates": cand_records}
        if vnn_selection:
            entry["vnn_selected"] = vnn_selection
        manifest["objects"].append(entry)
        prepared += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - pipeline_start
    print(f"  [{category}] Prepared: {prepared}, Discarded: {discarded}, Time: {elapsed:.1f}s")
    return prepared, discarded


# =====================================================================
# FULL AUTOMATION -- align_single_object()
# =====================================================================

def align_single_object(mesh_or_path, template_mesh_or_pts=None, category=None):
    """Fully automatic canonical alignment (inference — VN-DGCNN only, no Qwen)."""
    if isinstance(mesh_or_path, str):
        mesh = normalize_to_unit_sphere(load_mesh_or_pc(mesh_or_path))
    else:
        mesh = mesh_or_path

    if category is None:
        discovery = CategoryDiscovery()
        category, confidence, all_scores = discovery.identify(mesh)
        if category is None:
            raise ValueError("Zero-shot CLIP category discovery failed.")
        print(f"  Auto-detected category: {category} (confidence: {confidence:.2f})")

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
            raise ValueError(f"No template for '{category}'. Run prepare_training_data first.")

    candidates = generate_five_candidates(mesh, template_pts, category=category)
    obj_pts = sample_points(mesh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if model_loaded:
        tpl_tensor = torch.tensor(template_pts, dtype=torch.float32).unsqueeze(0).to(device)
        cand_pts_list = [(R @ obj_pts.T).T for _, R in candidates]
        cand_tensor = torch.tensor(np.stack(cand_pts_list), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward_all_candidates(tpl_tensor, cand_tensor, category_idx=cat_idx_tensor)
            scores = logits.squeeze(0).cpu().tolist()
        best_idx = int(np.argmax(scores))
    else:
        print(f"  [!] No VNN available — using Chamfer distance fallback")
        cand_pts = [(R @ obj_pts.T).T for _, R in candidates]
        best_idx, scores = compute_auto_label(cand_pts, template_pts)
        scores = [-d for d in scores]

    best_name, best_R = candidates[best_idx]
    print(f"  Selected: {best_name} (index {best_idx})")
    return best_R, best_name, best_idx


# =====================================================================
# METRICS EVALUATION (Stability & Consistency)
# =====================================================================

def evaluate_stability_consistency(category, num_objects=10, k_rotations=5):
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
        print(f"  [!] No template for '{category}'. Skipping.")
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
        print(f"  [!] No VNN model found. Skipping.")
        return 0, 0

    print(f"  Evaluating {len(valid_objects)} objects with {k_rotations} rotations each...")
    tpl_tensor = torch.tensor(template_pts, dtype=torch.float32).unsqueeze(0).to(device)
    cat_tensor = torch.tensor([vnn_cat_idx], dtype=torch.long).to(device) if vnn_cat_idx is not None else None

    def get_predicted_rotation(mesh_input, R_rand):
        rot_mesh = mesh_input.copy()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_rand
        rot_mesh.apply_transform(T)
        if hasattr(mesh_input, 'vertex_normals') and len(mesh_input.vertex_normals) > 0:
            rot_mesh.vertex_normals = (R_rand @ mesh_input.vertex_normals.T).T

        candidates = generate_five_candidates(rot_mesh, template_pts, category=category)
        pts_rot = sample_points(rot_mesh, seed=42)

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

        return cand_R[best_idx] @ R_rand

    stability_scores = []
    canonical_rotations_per_object = []

    for mesh in valid_objects:
        obj_canon_rotations = []
        for _ in range(k_rotations):
            R_rand = random_rotation_matrix()
            R_pred = get_predicted_rotation(mesh, R_rand)
            obj_canon_rotations.append(R_pred)

        R_mean = chordal_l2_mean_np(obj_canon_rotations)
        angs = [rotation_angle_np(R, R_mean) for R in obj_canon_rotations]
        rms_ang_deg = np.sqrt(np.mean(np.array(angs)**2)) * 180.0 / math.pi

        stability_scores.append(rms_ang_deg)
        canonical_rotations_per_object.append(obj_canon_rotations[0])

    avg_stability = np.mean(stability_scores)

    R_mean_cons = chordal_l2_mean_np(canonical_rotations_per_object)
    angs_cons = [rotation_angle_np(R, R_mean_cons) for R in canonical_rotations_per_object]
    rms_cons_deg = np.sqrt(np.mean(np.array(angs_cons)**2)) * 180.0 / math.pi

    avg_consistency = rms_cons_deg

    abs_errors = [geodesic_angle_deg(R, np.eye(3)) for R in canonical_rotations_per_object]
    acc_10 = float(np.mean([e <= 10.0 for e in abs_errors]) * 100.0) if abs_errors else 0.0
    acc_30 = float(np.mean([e <= 30.0 for e in abs_errors]) * 100.0) if abs_errors else 0.0
    mean_abs_err = float(np.mean(abs_errors)) if abs_errors else 0.0

    print(f"\n  [RESULTS] Category: {category.upper()}")
    print(f"    Stability (k={k_rotations}):          {avg_stability:.2f}° (Ideal: 0°)")
    print(f"    Consistency (n={len(valid_objects)}):        {avg_consistency:.2f}° (Ideal: 0°)")
    print(f"    Ground-Truth Acc@10°:       {acc_10:.1f}%")
    print(f"    Ground-Truth Acc@30°:       {acc_30:.1f}%")
    print(f"    Mean Geodesic Error vs GT:  {mean_abs_err:.2f}°\n")

    return avg_stability, avg_consistency


# =====================================================================
# TOP-LEVEL ENTRY POINTS
# =====================================================================

def run_pipeline(mode, categories=None):
    if categories is None:
        categories = get_categories()

    if mode == "train":
        train_agnostic(categories)

    elif mode == "generate":
        print("\n" + "#" * 60)
        print(f"  GENERATING CANDIDATES (Max {CATEGORIES_TO_RENDER} categories with >= {MIN_GLBS_TO_RENDER} GLBs)")
        print("#" * 60)
        render_cats = get_render_categories(categories, max_categories=CATEGORIES_TO_RENDER, min_glbs=MIN_GLBS_TO_RENDER)
        t0 = time.time()
        total_p, total_d = 0, 0
        for cat in render_cats:
            p, d = generate_category(cat)
            total_p += p
            total_d += d
        print(f"\n  Total: {total_p} prepared, {total_d} discarded, {time.time()-t0:.1f}s")

    elif mode == "evaluate_metrics":
        print("\n" + "#" * 60)
        print(f"  EVALUATING STABILITY & CONSISTENCY (Categories with >= {MIN_GLBS_TO_CALC_METRICS} GLBs)")
        print("#" * 60)

        metrics_cats = []
        for cat in categories:
            uids = load_category_uids(cat, n_needed=MIN_GLBS_TO_CALC_METRICS, verbose=False)
            if len(uids) >= MIN_GLBS_TO_CALC_METRICS:
                metrics_cats.append(cat)

        if not metrics_cats:
            print(f"  [!] No categories found with >= {MIN_GLBS_TO_CALC_METRICS} .glb files.")
        else:
            print(f"  [INFO] Evaluating {len(metrics_cats)} qualifying categories.")

        t0 = time.time()
        for cat in metrics_cats:
            evaluate_stability_consistency(cat)
        print(f"\n  Total evaluation time: {time.time()-t0:.1f}s")

    else:
        print(f"  Unknown mode: {mode}.")


def run_agnostic(categories=None):
    """Run the full pipeline: train with Qwen labels → generate → evaluate."""
    if categories is None:
        categories = get_categories()

    grand_start = time.time()

    print("\n" + "#" * 60)
    print("  FULL PIPELINE: QWEN VLM DISTILLATION + TRAIN + GENERATE + EVALUATE")
    if MAX_CATEGORIES is not None: print(f"  Max categories: {MAX_CATEGORIES}")
    if USE_ONLY_N_SAMPLES: print(f"  Samples per category: {N_SAMPLES}")
    print(f"  Qwen VLM: {'ENABLED' if _QWEN_AVAILABLE else 'DISABLED (Chamfer fallback)'}")
    print("#" * 60)

    # Step 1: Train agnostic model (data prep uses Qwen labels)
    train_agnostic(categories)

    # Step 2: Generate using agnostic model (inference — no Qwen)
    run_pipeline("generate", categories)

    # Step 3: Calculate stability and consistency
    run_pipeline("evaluate_metrics", categories)

    grand_time = time.time() - grand_start
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE! Total time: {grand_time:.1f}s ({grand_time/60:.1f}min)")
    print(f"{'#'*60}\n")

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    run_agnostic()
