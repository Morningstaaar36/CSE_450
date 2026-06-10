import os
import numpy as np
import torch

from eqr_model.eqr_arch import EQR

_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'eqr_model', 'eqr_d_phase3_best.pt'
)
_NUM_SUBSAMPLES = 5
_SUBSAMPLE_SIZE = 1024
_CONFIDENCE_SPREAD_DEG = 45.0


class MLAxisEstimator:
    """
    Up-axis predictor backed by the EQR model.

    Confidence is self-consistency: run K random subsamples, measure angular
    spread of predicted up vectors. Tight spread → high confidence → trust ML
    over PCA. Wide spread → low confidence → fall back to geometry.

    Always returns (None, 0.0) silently if the checkpoint is missing or any
    forward pass fails — never raises.
    """

    def __init__(self, checkpoint_path: str = _DEFAULT_CHECKPOINT):
        self._available = False
        try:
            model = EQR(
                hidden_channels=64,
                out_channels=64,
                residual_hidden_dim=128,
                num_neighbors=16,
            )
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            self._model = model
            self._available = True
            print(f'[MLAxisEstimator] loaded checkpoint: {os.path.basename(checkpoint_path)}')
        except FileNotFoundError:
            print(f'[MLAxisEstimator] checkpoint not found, running geometry-only')
        except Exception as e:
            print(f'[MLAxisEstimator] failed to load model ({e}), running geometry-only')

    def predict(self, pts: np.ndarray) -> tuple:
        """
        Args:
            pts: (N, 3) float32, already outlier-cleaned, arbitrary scale/position.
        Returns:
            (up_vector (3,) float32, confidence float 0-1)
            or (None, 0.0) if unavailable or forward fails.
        """
        if not self._available:
            return None, 0.0
        try:
            return self._predict(pts)
        except Exception as e:
            print(f'[MLAxisEstimator] forward failed ({e}), returning (None, 0.0)')
            return None, 0.0

    def _predict(self, pts: np.ndarray) -> tuple:
        pts = pts.astype(np.float32)

        # Preprocessing must match training: center → unit sphere
        pts = pts - pts.mean(axis=0)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm < 1e-6:
            return None, 0.0
        pts = pts / max_norm

        N = len(pts)
        replace = N < _SUBSAMPLE_SIZE

        print(f"[ML] running {_NUM_SUBSAMPLES} subsamples on {N} pts "
              f"(unit-sphere normalised, subsample={_SUBSAMPLE_SIZE})")

        up_vectors = []
        with torch.no_grad():
            for k in range(_NUM_SUBSAMPLES):
                idx = np.random.choice(N, _SUBSAMPLE_SIZE, replace=replace)
                x = torch.from_numpy(pts[idx]).unsqueeze(0)  # (1, 1024, 3)
                R_final = self._model(x)[0]                  # (1, 3, 3)
                up = R_final[0, 1, :].numpy().copy()         # second row = predicted Y-axis
                norm = np.linalg.norm(up)
                if norm < 1e-6:
                    continue
                up_unit = up / norm
                up_vectors.append(up_unit)
                print(f"[ML]   sample {k+1}: up=[{up_unit[0]:+.4f}, {up_unit[1]:+.4f}, {up_unit[2]:+.4f}]")

        if len(up_vectors) < 2:
            print(f"[ML] too few valid samples, returning (None, 0.0)")
            return None, 0.0

        up_arr = np.stack(up_vectors)  # (K, 3)

        # Mean pairwise angle → confidence
        K = len(up_arr)
        dots = np.clip(up_arr @ up_arr.T, -1.0, 1.0)
        angles_deg = [
            np.degrees(np.arccos(dots[i, j]))
            for i in range(K) for j in range(i + 1, K)
        ]
        mean_angle = float(np.mean(angles_deg))
        confidence = max(0.0, 1.0 - mean_angle / _CONFIDENCE_SPREAD_DEG)

        mean_up = up_arr.mean(axis=0)
        norm = np.linalg.norm(mean_up)
        if norm < 1e-6:
            return None, 0.0
        mean_up = (mean_up / norm).astype(np.float32)

        print(f"[ML] spread: mean_pairwise_angle={mean_angle:.1f}°  "
              f"confidence={confidence:.3f}  "
              f"(0 when spread≥{_CONFIDENCE_SPREAD_DEG}°)")
        print(f"[ML] mean_up=[{mean_up[0]:+.4f}, {mean_up[1]:+.4f}, {mean_up[2]:+.4f}]")

        return mean_up, confidence
