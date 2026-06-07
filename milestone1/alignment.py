"""
alignment.py
============

Baseline geometric alignment for the 3DGS engine (Milestone 1).

Pipeline:
    raw analysis cloud
        -> OutlierCleaner        (statistical k-NN filter)
        -> ContentTypeClassifier (indoor scene vs isolated object)
        -> GeometryAxisEstimator (RANSAC ground for scenes / PCA for objects)
        -> 4x4 alignment transform + confidence + mode + ambiguity flags

All tensor / math work runs on CPU explicitly (`device='cpu'`) — the target
machine has no dedicated GPU.  The ML-assist hooks for Milestone 2 are marked
with `[M2 HOOK]` comments so the integration points are obvious.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

CPU = torch.device("cpu")
UP_AXIS_WORLD = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # +Y is "up"
FORWARD_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32) # -Z is "forward"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Everything the renderer/UI needs to consume an alignment decision."""
    transform: np.ndarray                  # (4,4) float32 — scene-root transform
    confidence: float                      # 0..1, higher == more trusted
    mode: str                              # 'indoor' | 'object' | 'fallback'
    ambiguity_flags: List[str] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "transform": self.transform.tolist(),
            "confidence": float(self.confidence),
            "mode": self.mode,
            "ambiguity_flags": list(self.ambiguity_flags),
            "diagnostics": {k: float(v) for k, v in self.diagnostics.items()},
        }


# ---------------------------------------------------------------------------
# OutlierCleaner
# ---------------------------------------------------------------------------

class OutlierCleaner:
    """
    Statistical outlier removal based on mean k-NN distance.

    For every point we compute the mean distance to its `k` nearest
    neighbours.  Points whose mean distance exceeds  μ + std_ratio · σ
    are considered stragglers and dropped.
    """

    def __init__(self, k: int = 16, std_ratio: float = 2.0):
        # k caps memory: 16 NN of 50k points is ~6.4 MB of float distances.
        self.k = int(k)
        self.std_ratio = float(std_ratio)

    def __call__(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (kept_points, mask) where mask is a boolean array over the
        input axis 0 — useful so callers can filter parallel colour arrays.
        """
        if points.shape[0] <= self.k + 1:
            return points, np.ones(points.shape[0], dtype=bool)

        k = min(self.k, points.shape[0] - 1) + 1   # +1: NN of a point is itself
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1)
        nbrs.fit(points)
        dists, _ = nbrs.kneighbors(points)
        # Drop self-distance (first column) before averaging.
        mean_d = dists[:, 1:].mean(axis=1)

        mu = float(mean_d.mean())
        sd = float(mean_d.std())
        thresh = mu + self.std_ratio * sd
        mask = mean_d <= thresh
        return points[mask], mask


# ---------------------------------------------------------------------------
# ContentTypeClassifier
# ---------------------------------------------------------------------------

class ContentTypeClassifier:
    """
    Heuristic classifier: indoor scene vs isolated object.

    The primary signal is **dominant-plane coverage** — what fraction of the
    cloud can be explained by a single best-fit plane.  This is rotation-
    invariant (works even when the cloud is tilted off the world axes):

        * Indoor scans: floors/walls/ceilings are big planar surfaces, so a
          good RANSAC plane typically covers ≥25 % of points.
        * Isolated objects: curved or sculpted surfaces, so the best plane
          rarely explains more than ~15 % of points (mostly the base disc
          or a single facet).

    Plane coverage **alone** is not enough: a flat-ish object (e.g. a 3DGS
    figurine captured front-on) has one big planar face and would look like a
    scene.  A real room is distinguished by a **second large plane roughly
    orthogonal to the first** — a wall meeting the floor.  So we require either
    a floor *plus* an orthogonal wall, or a single overwhelmingly dominant
    floor, before declaring "indoor".  Two near-parallel planes (the front and
    back faces of a flat object) do not qualify.
    """

    def __init__(
        self,
        indoor_inlier_threshold: float = 0.22,   # plane coverage ratio
        ransac_iters: int = 180,
        plane_thresh_rel: float = 0.015,         # relative to cloud extent
        wall_min_ratio: float = 0.12,            # 2nd plane coverage to count as a wall
        wall_angle_tol_deg: float = 35.0,        # wall must be 90°±tol to the floor
        huge_floor_ratio: float = 0.45,          # a floor this dominant => scene alone
    ):
        self.indoor_inlier_threshold = indoor_inlier_threshold
        self.ransac_iters = ransac_iters
        self.plane_thresh_rel = plane_thresh_rel
        self.wall_min_ratio = wall_min_ratio
        self.wall_angle_tol_deg = wall_angle_tol_deg
        self.huge_floor_ratio = huge_floor_ratio

    def __call__(self, points: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Returns (mode, confidence, diagnostics)."""
        n = points.shape[0]
        if n < 32:
            return "object", 0.25, {"n_points": float(n), "reason_small": 1.0}

        # ---- PCA on centred cloud (torch CPU, as required) ----------------
        pts_t = torch.from_numpy(points.astype(np.float32)).to(CPU)
        centred = pts_t - pts_t.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.pca_lowrank(centred, q=3, niter=2)
            lam = (S.cpu().numpy() ** 2) / max(n - 1, 1)
        except Exception:
            cov = (centred.T @ centred) / max(n - 1, 1)
            lam = np.linalg.eigvalsh(cov.numpy())[::-1]
        lam = np.sort(lam)[::-1].astype(np.float64)        # λ1 ≥ λ2 ≥ λ3
        l1, l2, l3 = float(lam[0]), float(lam[1]), float(lam[2])
        flat_ratio = l3 / max(l2, 1e-12)                   # λ3/λ2 — flatness

        # ---- Dominant-plane RANSAC (the floor candidate) -----------------
        extent = float((points.max(axis=0) - points.min(axis=0)).max())
        tol = max(self.plane_thresh_rel * extent, 1e-6)

        def _coverage(normal, point):
            """Fraction of the FULL cloud within `tol` of a plane."""
            return float(np.mean(np.abs((points - point) @ normal) < tol))

        # Sub-sample for speed — RANSAC quality saturates well below 50 k.
        rng = np.random.default_rng(0)
        sample = points[rng.choice(n, 8000, replace=False)] if n > 8000 else points
        plane_normal, plane_point, _ = _ransac_plane(
            sample, iterations=self.ransac_iters, inlier_thresh=self.plane_thresh_rel,
        )
        plane_ratio = _coverage(plane_normal, plane_point)

        # ---- Second plane on the residual (the wall candidate) -----------
        # Remove the floor's inliers and look for another large plane.  An
        # orthogonal one is a wall (→ scene); a parallel one is just the
        # opposite face of a flat object (→ not a scene).
        off = np.abs((points - plane_point) @ plane_normal) >= tol
        wall_ratio, wall_angle = 0.0, 0.0
        residual = points[off]
        if residual.shape[0] > 200:
            rs = (residual[np.random.default_rng(1).choice(residual.shape[0], 8000,
                  replace=False)] if residual.shape[0] > 8000 else residual)
            n2, p2, _ = _ransac_plane(
                rs, iterations=self.ransac_iters, inlier_thresh=self.plane_thresh_rel)
            wall_ratio = float(np.sum(np.abs((residual - p2) @ n2) < tol)) / n
            wall_angle = float(np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(plane_normal, n2)))))))
        orthogonal_wall = (wall_ratio >= self.wall_min_ratio and
                           abs(wall_angle - 90.0) <= self.wall_angle_tol_deg)

        # ---- Decision ----------------------------------------------------
        # Soft score in [0,1] favouring 'indoor'.  A dominant plane is
        # necessary but not sufficient — it must be a floor+wall pair, or an
        # overwhelmingly dominant floor, to count as a scene.
        score = 0.0
        if plane_ratio >= self.indoor_inlier_threshold:
            base = min(plane_ratio / (2 * self.indoor_inlier_threshold), 1.0)
            if orthogonal_wall:
                score = max(base, 0.70)              # floor + wall → scene
            elif plane_ratio >= self.huge_floor_ratio:
                score = base                          # huge floor alone → scene
            else:
                score = 0.35                          # one mid plane, no wall → flat object

        mode = "indoor" if score >= 0.5 else "object"
        # Confidence = distance of the score from the 0.5 boundary, rescaled.
        confidence = float(np.clip(0.10 + 1.7 * abs(score - 0.5), 0.10, 0.95))

        # [M2 HOOK] — A learnt classifier output (logit) would be blended
        # here with `score` once the Milestone-2 model is available.

        diagnostics = {
            "lambda1": l1, "lambda2": l2, "lambda3": l3,
            "flat_ratio": flat_ratio,
            "plane_inlier_ratio": float(plane_ratio),
            "wall_inlier_ratio": float(wall_ratio),
            "wall_angle_deg": float(wall_angle),
            "orthogonal_wall": float(orthogonal_wall),
            "n_points": float(n),
            "extent": float(extent),
            "score_indoor": score,
            "plane_normal_x": float(plane_normal[0]),
            "plane_normal_y": float(plane_normal[1]),
            "plane_normal_z": float(plane_normal[2]),
            "plane_point_x": float(plane_point[0]),
            "plane_point_y": float(plane_point[1]),
            "plane_point_z": float(plane_point[2]),
        }
        return mode, confidence, diagnostics


# ---------------------------------------------------------------------------
# GeometryAxisEstimator
# ---------------------------------------------------------------------------

def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 rotation that rotates unit-vector `src` onto unit-vector `dst`.
    Robust to the antiparallel case (180° turn).
    """
    a = _normalise(src.astype(np.float64))
    b = _normalise(dst.astype(np.float64))
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float32)
    if d < -1.0 + 1e-9:
        # Pick any axis orthogonal to `a` and rotate 180°.
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _normalise(np.cross(a, axis))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return (np.eye(3) + 2.0 * (K @ K)).astype(np.float32)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + K + K @ K * ((1 - d) / (s * s + 1e-12))
    return R.astype(np.float32)


def _ransac_plane(points: np.ndarray,
                  iterations: int = 200,
                  inlier_thresh: float = 0.02,
                  rng_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Tiny pure-NumPy RANSAC plane fit.  Returns (normal, point_on_plane, n_inliers).
    inlier_thresh is relative — it's scaled by the cloud's extent.

    We keep `iterations` modest so this stays well under 100 ms on iGPU
    machines even for 50k points.
    """
    n = points.shape[0]
    if n < 3:
        return np.array([0, 1, 0], dtype=np.float32), points.mean(axis=0), 0

    rng = np.random.default_rng(rng_seed)
    extent = float((points.max(axis=0) - points.min(axis=0)).max())
    tol = max(inlier_thresh * extent, 1e-6)

    best_n = np.array([0, 1, 0], dtype=np.float32)
    best_p = points.mean(axis=0)
    best_count = 0

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        nrm = np.cross(v1, v2)
        nlen = float(np.linalg.norm(nrm))
        if nlen < 1e-9:
            continue
        nrm = nrm / nlen
        d = np.abs((points - p0) @ nrm)
        inliers = int(np.sum(d < tol))
        if inliers > best_count:
            best_count = inliers
            best_n = nrm.astype(np.float32)
            best_p = p0.astype(np.float32)

    # Refit the best plane with least-squares on its inliers for stability.
    d = np.abs((points - best_p) @ best_n)
    inlier_mask = d < tol
    if int(inlier_mask.sum()) >= 16:
        pts_i = points[inlier_mask]
        c = pts_i.mean(axis=0)
        cov = (pts_i - c).T @ (pts_i - c)
        # Smallest-eigenvector of cov is the refined normal.
        w, V = np.linalg.eigh(cov)
        refined = V[:, 0].astype(np.float32)
        # Keep orientation consistent with the RANSAC normal.
        if float(np.dot(refined, best_n)) < 0:
            refined = -refined
        best_n = refined
        best_p = c.astype(np.float32)

    return best_n, best_p, best_count


class GeometryAxisEstimator:
    """
    Produce a 4x4 alignment transform from raw point geometry.

    Two execution paths:
        * indoor : RANSAC ground-plane fit on a lower-shell of the cloud,
                   then rotate plane normal -> world up.
        * object : PCA on the whole cloud, smallest-variance axis -> world up.

    Both paths re-centre the cloud at the origin so the renderer's camera
    framing logic does not have to care about input coordinate ranges.
    """

    def __init__(self,
                 lower_shell_pct: float = 0.30,
                 ransac_iters: int = 200,
                 ransac_thresh: float = 0.02):
        self.lower_shell_pct = lower_shell_pct
        self.ransac_iters = ransac_iters
        self.ransac_thresh = ransac_thresh

    # -- indoor ------------------------------------------------------------
    def _indoor_transform(self,
                          points: np.ndarray,
                          diagnostics: Dict[str, float]
                          ) -> Tuple[np.ndarray, float, List[str]]:
        """
        Use the dominant plane discovered by the classifier (if available)
        as the floor.  Falls back to a fresh RANSAC pass when the diagnostics
        dict is missing the plane params (e.g. user invoked the estimator
        directly).  The plane normal becomes the world up axis after rotation.
        """
        n_pts = points.shape[0]
        normal: Optional[np.ndarray] = None
        plane_point: Optional[np.ndarray] = None
        inlier_ratio: float = 0.0

        if {"plane_normal_x", "plane_normal_y", "plane_normal_z",
            "plane_point_x", "plane_point_y", "plane_point_z"}.issubset(
                diagnostics.keys()):
            normal = np.array([diagnostics["plane_normal_x"],
                               diagnostics["plane_normal_y"],
                               diagnostics["plane_normal_z"]], dtype=np.float32)
            plane_point = np.array([diagnostics["plane_point_x"],
                                    diagnostics["plane_point_y"],
                                    diagnostics["plane_point_z"]], dtype=np.float32)
            inlier_ratio = float(diagnostics.get("plane_inlier_ratio", 0.0))

        if normal is None:
            normal, plane_point, inlier_n = _ransac_plane(
                points,
                iterations=self.ransac_iters,
                inlier_thresh=self.ransac_thresh,
            )
            inlier_ratio = inlier_n / max(n_pts, 1)

        # Orient normal away from the floor — the world *centroid* should sit
        # on the positive side of the plane after orientation.
        centroid = points.mean(axis=0)
        if float(np.dot(normal, centroid - plane_point)) < 0:
            normal = -normal

        R = _rotation_align_vectors(normal, UP_AXIS_WORLD)

        flags: List[str] = []
        if inlier_ratio < 0.20:
            flags.append("weak_ground_plane")
        # Confidence rises with inlier coverage but is capped to leave room
        # for the classifier's confidence to influence the fused score.
        confidence = float(np.clip(0.4 + 1.5 * inlier_ratio, 0.25, 0.95))

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3, 3] = -R.astype(np.float32) @ centroid.astype(np.float32)

        # [M2 HOOK] — The ML up-axis predictor's prior would be averaged in
        # here (e.g. weighted Slerp between RANSAC normal and ML normal,
        # weights ∝ each method's confidence).

        return T, confidence, flags

    # -- object ------------------------------------------------------------
    def _object_transform(self,
                          points: np.ndarray
                          ) -> Tuple[np.ndarray, float, List[str]]:
        """
        Estimate the up axis of an isolated object from its PCA shape.

        Pure geometry can only orient an object when its shape *defines* an
        up axis.  Two shape classes carry that signal:

            * Prolate ("elongated"): one dominant long axis (a bottle, statue,
              pillar, person).  The long axis is up.
            * Oblate ("flat / disc"): one dominant thin axis (a plate, table,
              coin, a vase whose base disc flattens the cloud).  The thin axis
              is up.

        For a triaxial / near-isotropic object (e.g. the Stanford Bunny) the
        up direction is *semantic*, not geometric — no PCA axis recovers it.
        We pick the best available axis but flag `up_ambiguous` with low
        confidence rather than reporting a confident wrong answer.  Resolving
        these is the job of the Milestone-2 ML predictor.
        """
        pts_t = torch.from_numpy(points.astype(np.float32)).to(CPU)
        centred = pts_t - pts_t.mean(dim=0, keepdim=True)

        # PCA — covariance eigenvectors via torch.linalg on CPU.
        cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov.to(CPU))
        # eigvecs[:, i] is the eigenvector for eigvals[i]; ascending order,
        # so column 0 is the smallest-variance axis and column 2 the largest.
        eigvals_np = eigvals.cpu().numpy().astype(np.float64)
        eigvecs_np = eigvecs.cpu().numpy().astype(np.float64)
        l3, l2, l1 = eigvals_np[0], eigvals_np[1], eigvals_np[2]   # l1 ≥ l2 ≥ l3
        v_small, v_large = eigvecs_np[:, 0], eigvecs_np[:, 2]

        # Relative eigen-gaps tell us which axis is the "odd one out".
        gap_top = (l1 - l2) / max(l1, 1e-12)    # large gap → prolate (long axis up)
        gap_bot = (l2 - l3) / max(l2, 1e-12)    # large gap → oblate  (thin axis up)

        if gap_top >= gap_bot:
            up_candidate = v_large              # elongated → long axis is up
            strength = gap_top
        else:
            up_candidate = v_small              # flat/disc → thin axis is up
            strength = gap_bot

        # Ambiguity: a weak winning gap, or two gaps too close to call, means
        # the shape does not pin an up axis (triaxial / isotropic blob).
        margin = abs(gap_top - gap_bot)
        up_ambiguous = (strength < 0.25) or (margin < 0.15)

        # Disambiguate up/down: project points onto candidate, pick the sign
        # so the median of the projection is *negative* (more mass below up —
        # objects are usually bottom-heavy, resting on a base).
        centroid = points.mean(axis=0)
        proj = (points - centroid) @ up_candidate
        if float(np.median(proj)) > 0:
            up_candidate = -up_candidate

        R = _rotation_align_vectors(up_candidate.astype(np.float32), UP_AXIS_WORLD)

        flags: List[str] = []
        if up_ambiguous:
            flags.append("up_ambiguous")
        # Yaw ambiguity: once up is fixed, the heading is set by the remaining
        # two axes.  If their eigenvalues are nearly equal the object is
        # rotationally symmetric about up (a bottle, vase) → heading undefined.
        if gap_top >= gap_bot:
            yaw_pair = (l2, l3)                 # up was l1; remaining are l2,l3
        else:
            yaw_pair = (l1, l2)                 # up was l3; remaining are l1,l2
        yaw_anisotropy = (yaw_pair[0] - yaw_pair[1]) / max(yaw_pair[0], 1e-12)
        if yaw_anisotropy < 0.05:
            flags.append("yaw_ambiguous")

        # Confidence tracks how clearly the shape defines an up axis; ambiguous
        # objects are heavily discounted so the UI/fusion can distrust them.
        confidence = float(np.clip(
            0.25 + 0.7 * strength * (0.4 if up_ambiguous else 1.0), 0.15, 0.9))

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3, 3] = -R.astype(np.float32) @ centroid.astype(np.float32)

        # [M2 HOOK] — replace `up_candidate` with the ML-predicted up vector
        # when its confidence exceeds this geometric `strength` margin; this
        # is exactly where the `up_ambiguous` cases get rescued.

        return T, confidence, flags

    # -- dispatch ----------------------------------------------------------
    def __call__(self,
                 points: np.ndarray,
                 mode: str,
                 classifier_diag: Optional[Dict[str, float]] = None,
                 ) -> Tuple[np.ndarray, float, List[str]]:
        if mode == "indoor":
            return self._indoor_transform(points, classifier_diag or {})
        return self._object_transform(points)


# ---------------------------------------------------------------------------
# AlignmentEngine — wires the submodules together
# ---------------------------------------------------------------------------

class AlignmentEngine:
    """
    Glue the three baseline submodules into a single call:

        engine = AlignmentEngine()
        result = engine.align(points)            # AlignmentResult
        renderer.set_scene_root_transform(result.transform)
    """

    def __init__(self):
        self.cleaner = OutlierCleaner()
        self.classifier = ContentTypeClassifier()
        self.estimator = GeometryAxisEstimator()

    def align(self, points: np.ndarray) -> AlignmentResult:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"alignment expects (N,3) points; got {points.shape}")

        cleaned, _mask = self.cleaner(points)
        if cleaned.shape[0] < 16:
            cleaned = points  # cleaner was too aggressive — fall back

        mode, cls_conf, diag = self.classifier(cleaned)
        T, est_conf, flags = self.estimator(cleaned, mode, diag)

        # Geometric mean blends two probability-like numbers smoothly.
        confidence = float(math.sqrt(max(cls_conf, 1e-6) * max(est_conf, 1e-6)))

        return AlignmentResult(
            transform=T.astype(np.float32),
            confidence=confidence,
            mode=mode,
            ambiguity_flags=flags,
            diagnostics={**diag, "cls_confidence": cls_conf,
                         "est_confidence": est_conf,
                         "cleaned_points": float(cleaned.shape[0])},
        )

    # ---- discrete UX nudges (used by the toolbar buttons) ---------------

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def rotate_about_up(transform: np.ndarray, degrees: float) -> np.ndarray:
        """Premultiply a rotation about world +Y by `degrees`."""
        a = math.radians(degrees)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=np.float32)
        return (R @ transform).astype(np.float32)

    @staticmethod
    def flip_front_back(transform: np.ndarray) -> np.ndarray:
        """Premultiply a 180° turn about world +Y (front/back swap)."""
        return AlignmentEngine.rotate_about_up(transform, 180.0)

    @staticmethod
    def keep_up_only(transform: np.ndarray) -> np.ndarray:
        """
        Strip yaw: keep just the rotation that maps the cloud's current
        up direction onto world up, and zero out the heading.  Useful when
        the auto-yaw is wrong but the auto-up is right.
        """
        R = transform[:3, :3].astype(np.float64)
        t = transform[:3, 3].astype(np.float64)
        # Where does world up land if we applied R^{-1}? -> the cloud's up.
        # We want a rotation that maps that vector back to world up but
        # introduces no heading.
        cloud_up = R.T @ UP_AXIS_WORLD.astype(np.float64)
        R_up = _rotation_align_vectors(
            cloud_up.astype(np.float32), UP_AXIS_WORLD
        ).astype(np.float64)
        # The original translation re-centred the cloud: t = -R @ centroid, so
        # centroid = -Rᵀ t.  Recompute the translation for the new rotation so
        # the cloud stays centred at the origin instead of drifting off-screen.
        centroid = -R.T @ t
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R_up.astype(np.float32)
        out[:3, 3] = (-R_up @ centroid).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json
    from loader import load_point_cloud

    ap = argparse.ArgumentParser(description="Baseline geometric alignment.")
    ap.add_argument("ply")
    ap.add_argument("--cap", type=int, default=50_000)
    args = ap.parse_args()

    pc = load_point_cloud(args.ply, point_cap=args.cap)
    engine = AlignmentEngine()
    result = engine.align(pc.positions)
    print(json.dumps(result.as_dict(), indent=2))
