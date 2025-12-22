from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_2d(x: np.ndarray) -> np.ndarray:
    """(N, L, 1) -> (N, L) for metrics."""
    if x.ndim == 3 and x.shape[-1] == 1:
        return np.squeeze(x, axis=-1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected (N,L,1) or (N,L), got {x.shape}")


def to_ncl(x: np.ndarray) -> np.ndarray:
    """(N, L, 1) -> (N, 1, L) for PyTorch."""
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError(f"Expected (N,L,1), got {x.shape}")
    return np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False)


def to_nlc(x: np.ndarray) -> np.ndarray:
    """(N, 1, L) -> (N, L, 1)"""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected (N,1,L), got {x.shape}")
    return np.transpose(x, (0, 2, 1))


def save_test_results(out_dir: str, exp_name: str, x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> str:
    import _pickle as pickle

    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"test_results_{exp_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump([x_test, y_test, y_pred], f)
    return path
