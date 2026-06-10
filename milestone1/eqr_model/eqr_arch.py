import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        out = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        out = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid(
            [torch.arange(j) for j in x.size()[:-1]], indexing='ij'
        ) + (idx,)
        return x[index_tuple]


class VNEncoder(nn.Module):
    """SO(3)-equivariant encoder: R^{N×3} → feature tensor."""
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = VNLinearLeakyReLU(in_channels, hidden_channels, dim=4)
        self.conv2 = VNLinearLeakyReLU(hidden_channels, hidden_channels, dim=4)
        self.conv3 = VNLinearLeakyReLU(hidden_channels, out_channels, dim=4)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VNInvariantLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=4)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=4)
        self.vn_lin = VNLinear(in_channels // 4, 3)

    def forward(self, x):
        x = self.vn1(x)
        x = self.vn2(x)
        return self.vn_lin(x)


class RotationPredictor(nn.Module):
    """Reduces to 2 vectors, mean-pools, Gram-Schmidt → SO(3)."""
    def __init__(self, in_channels):
        super().__init__()
        self.vn_lin = VNLinear(in_channels, 2)

    def forward(self, x):
        basis = self.vn_lin(x).mean(dim=-1)
        v1, v2 = basis[:, 0, :], basis[:, 1, :]
        e1 = F.normalize(v1, dim=-1)
        e2 = F.normalize(v2 - (v2 * e1).sum(-1, keepdim=True) * e1, dim=-1, eps=1e-6)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)


class OrientationHypothesizer(nn.Module):
    """Stage h — SO(3)-equivariant."""
    def __init__(self, hidden_channels=64, out_channels=128):
        super().__init__()
        self.encoder = VNEncoder(1, hidden_channels, out_channels)
        self.rotation_predictor = RotationPredictor(out_channels)
        self.inv_layer = VNInvariantLayer(out_channels)

    def forward(self, x):
        equiv_feat = self.encoder(x)
        R_hyp = self.rotation_predictor(equiv_feat)
        V = self.inv_layer(equiv_feat)
        return R_hyp, equiv_feat, V


class InvariantFeatureExtractor(nn.Module):
    """SO(3)-invariant features via inner-product trick."""
    def __init__(self, in_channels, frame_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.frame_channels = frame_channels

    def forward(self, phi_feat, V_feat):
        inv_feat = torch.einsum('bcin,bdin->bcdn', phi_feat, V_feat)
        B, C, D, N = inv_feat.shape
        return inv_feat.permute(0, 3, 1, 2).reshape(B, N, C * D)


class PointTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, num_neighbors=16):
        super().__init__()
        out_dim = out_dim or in_dim
        self.num_neighbors = num_neighbors
        self.phi = nn.Linear(in_dim, out_dim)
        self.psi = nn.Linear(in_dim, out_dim)
        self.alpha = nn.Linear(in_dim, out_dim)
        self.pos_enc = nn.Sequential(
            nn.Linear(3, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.gamma = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x, pos):
        B, N, _ = x.shape
        k = min(self.num_neighbors, N)
        with torch.no_grad():
            _, knn_idx = torch.cdist(pos, pos).topk(k, dim=-1, largest=False)
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        x_neighbors = x[batch_idx, knn_idx]
        pos_neighbors = pos[batch_idx, knn_idx]
        delta = self.pos_enc(pos.unsqueeze(2) - pos_neighbors)
        attn = F.softmax(self.gamma(self.phi(x).unsqueeze(2) - self.psi(x_neighbors) + delta), dim=2)
        return (attn * (self.alpha(x_neighbors) + delta)).sum(dim=2)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, num_neighbors=16):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.pt_layer = PointTransformerLayer(dim, dim, num_neighbors)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x, pos):
        return self.linear_out(self.pt_layer(self.linear_in(x), pos)) + x


class ResidualPredictor(nn.Module):
    """Stage g — SO(3)-invariant residual: 6D output → Gram-Schmidt → SO(3)."""
    def __init__(self, in_channels=128, hidden_dim=256, inv_feat_channels=3, num_neighbors=16):
        super().__init__()
        self.inv_feat_extractor = InvariantFeatureExtractor(in_channels, inv_feat_channels)
        inv_feat_dim = in_channels * inv_feat_channels
        self.point_embed = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.feat_embed = nn.Sequential(
            nn.Linear(inv_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pt_block1 = PointTransformerBlock(hidden_dim, num_neighbors)
        self.pt_block2 = PointTransformerBlock(hidden_dim, num_neighbors)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6))

    def forward(self, canon_pc, phi_feat, V_feat):
        inv_feat = self.inv_feat_extractor(phi_feat, V_feat)
        combined = self.point_embed(canon_pc) + self.feat_embed(inv_feat)
        combined = self.pt_block1(combined, canon_pc)
        combined = self.pt_block2(combined, canon_pc)
        out = self.out_mlp(combined.mean(dim=1))
        v1, v2 = out[:, :3], out[:, 3:]
        e1 = F.normalize(v1, dim=-1)
        e2 = F.normalize(v2 - (v2 * e1).sum(-1, keepdim=True) * e1, dim=-1, eps=1e-6)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)


class EQR(nn.Module):
    """Equivariant Quotient-Residual model.

    Forward: R_final, R_hyp, R_res, equiv_feat = model(x)
    x shape: (B, N, 3)
    R_final shape: (B, 3, 3)  — predicted rotation matrix (Gram-Schmidt SO(3))
    Up vector in input space: R_final[b, 1, :]  (second row)
    """
    def __init__(self, hidden_channels=64, out_channels=128,
                 residual_hidden_dim=256, num_neighbors=16):
        super().__init__()
        self.hypothesizer = OrientationHypothesizer(hidden_channels, out_channels)
        self.residual_predictor = ResidualPredictor(
            out_channels, residual_hidden_dim, inv_feat_channels=3,
            num_neighbors=num_neighbors)

    def forward(self, x):
        R_hyp, equiv_feat, V_feat = self.hypothesizer(x)
        canon_pc = x @ R_hyp.transpose(-1, -2)
        R_res = self.residual_predictor(canon_pc, equiv_feat, V_feat)
        R_final = R_res @ R_hyp
        return R_final, R_hyp, R_res, equiv_feat
