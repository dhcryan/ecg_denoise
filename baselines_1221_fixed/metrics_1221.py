from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from utils.metrics import COS_SIM, PRD, SNR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-sample metrics.

    Inputs can be (N,L) or (N,L,1). Outputs are 1D arrays of length N.
    """
    if y_true.ndim == 3:
        y_true_2d = np.squeeze(y_true, axis=-1)
    else:
        y_true_2d = y_true

    if y_pred.ndim == 3:
        y_pred_2d = np.squeeze(y_pred, axis=-1)
    else:
        y_pred_2d = y_pred

    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_2d.shape}, y_pred={y_pred_2d.shape}")

    rmse = np.sqrt(np.mean((y_pred_2d - y_true_2d) ** 2, axis=1))
    prd = np.squeeze(PRD(y_true_2d, y_pred_2d))
    cos = np.squeeze(COS_SIM(np.expand_dims(y_true_2d, -1), np.expand_dims(y_pred_2d, -1)))
    snr = np.squeeze(SNR(y_true_2d, y_pred_2d))

    return {
        "RMSE": rmse,
        "PRD": prd,
        "COS_SIM": cos,
        "SNR": snr,
    }
