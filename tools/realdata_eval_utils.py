"""Utilities for label-free evaluation on real ECG txt files (multi-column).

Designed for:
- wet electrode dataset: `ajou_real/<name>_<date>/*.txt` (phase-split)
- dry electrode dataset: `ajou_data/*.txt`, `ajou_data2/*.txt` (session-only)

Provides:
- multi-column txt loading + lead selection
- wet phase parsing
- preprocessing (baseline + resample + trim)
- unified denoising via framing + per-window z-normalization + overlap-add
- label-free metrics (HF gain, R-peak stability) + sliding-window worst-window
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import gcd
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from ajou_rpeak_utils import detect_rpeaks, evaluate_rpeak_performance


PHASE_ORDER = ["Stable", "Breathe", "Walk", "Recovery"]


def _looks_like_time_index(col: np.ndarray) -> bool:
    x = np.asarray(col)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return False
    dx = np.diff(x[: min(2000, x.size)])
    if not np.all(dx > 0):
        return False
    med = float(np.median(dx))
    return abs(med - 1.0) < 1e-2


def read_multicol_txt(path: str) -> np.ndarray:
    arr = np.genfromtxt(path, dtype=np.float64, invalid_raise=False)
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
    if arr.shape[0] < 2:
        raise RuntimeError(f"Too few samples in {path}")
    return arr


def select_lead_column(arr: np.ndarray, lead_index_1based: int = 2) -> np.ndarray:
    """Select one lead from multi-column array.

    - Drops columns with <50% finite values
    - Drops first col if it looks like time/index
    - lead_index_1based applies to remaining columns
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    # keep usable cols
    keep_cols = [j for j in range(arr.shape[1]) if float(np.isfinite(arr[:, j]).mean()) >= 0.5]
    if not keep_cols:
        raise RuntimeError("No usable columns (too many NaNs)")

    a = arr[:, keep_cols]
    if a.shape[1] >= 2 and _looks_like_time_index(a[:, 0]):
        a = a[:, 1:]

    lead_idx0 = int(lead_index_1based) - 1
    if lead_idx0 < 0 or lead_idx0 >= a.shape[1]:
        raise ValueError(f"lead_index={lead_index_1based} out of range; available 1..{a.shape[1]}")

    sig = np.asarray(a[:, lead_idx0], dtype=np.float32)
    if not np.all(np.isfinite(sig)):
        med = float(np.nanmedian(sig))
        sig = np.nan_to_num(sig, nan=med)
    return sig


def baseline_center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x - np.median(x)


def resample(x: np.ndarray, fs_src: int, fs_tgt: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if fs_src == fs_tgt:
        return x
    g = gcd(int(fs_src), int(fs_tgt))
    up = int(fs_tgt) // g
    down = int(fs_src) // g
    return resample_poly(x, up=up, down=down).astype(np.float32)


def frame_signal(x: np.ndarray, win: int, hop: int, pad_mode: str = "reflect") -> Tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < win:
        x = np.pad(x, (0, win - n), mode=pad_mode)
    n = len(x)
    n_frames = 1 + (n - win + hop - 1) // hop
    total_needed = (n_frames - 1) * hop + win
    if total_needed > n:
        x = np.pad(x, (0, total_needed - n), mode=pad_mode)
    frames = np.stack([x[i * hop : i * hop + win] for i in range(n_frames)], axis=0)
    return frames.astype(np.float32), len(x)


def overlap_add(frames: np.ndarray, win: int, hop: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, _L = frames.shape
    total_len = (n_frames - 1) * hop + win
    y = np.zeros(total_len, dtype=np.float32)
    w = np.hanning(win).astype(np.float32)
    w = w / (np.max(w) + 1e-12)
    acc = np.zeros(total_len, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        y[s : s + win] += frames[i] * w
        acc[s : s + win] += w
    acc[acc == 0] = 1.0
    return y / acc


def denoise_full_signal(model, x_360: np.ndarray, *, win: int, hop: int, batch_size: int = 128) -> np.ndarray:
    """Unified denoising: per-window z-norm -> model.predict -> rescale -> overlap-add."""
    frames, _padded_len = frame_signal(x_360, win=win, hop=hop)
    mu = np.mean(frames, axis=1, keepdims=True).astype(np.float32)
    sigma = (np.std(frames, axis=1, keepdims=True) + 1e-6).astype(np.float32)
    X = ((frames - mu) / sigma)[..., None]

    Y = model.predict(X, batch_size=int(batch_size), verbose=0)
    Y = np.squeeze(Y, axis=2).astype(np.float32)
    Y = Y * sigma + mu
    deno_full = overlap_add(Y, win=win, hop=hop)
    return deno_full[: len(x_360)].astype(np.float32)


def infer_wet_phase_from_filename(fname: str) -> Optional[str]:
    s = os.path.splitext(os.path.basename(fname))[0]

    if re.search(r"session|세션", s, flags=re.IGNORECASE):
        return "Session"

    m = re.search(r"_(\d{2})", s)
    if m:
        idx = int(m.group(1))
        if idx == 1:
            return "Stable"
        if idx == 2:
            return "Breathe"
        if idx == 3:
            return "Walk"
        if idx == 4:
            return "Recovery"

    if re.search(r"stable|안정", s, flags=re.IGNORECASE):
        return "Stable"
    if re.search(r"breathe|호흡", s, flags=re.IGNORECASE):
        return "Breathe"
    if re.search(r"walk|운동", s, flags=re.IGNORECASE):
        return "Walk"
    if re.search(r"recovery|회복", s, flags=re.IGNORECASE):
        return "Recovery"

    return None


@dataclass
class WetCase:
    subject: str
    date: str
    src_dir: str
    phase_paths: Dict[str, str]  # Stable/Breathe/Walk/Recovery (and maybe Session)


def parse_wet_cases(wet_root: str) -> List[WetCase]:
    cases: List[WetCase] = []
    for entry in sorted(os.listdir(wet_root)):
        full = os.path.join(wet_root, entry)
        if not os.path.isdir(full):
            continue
        m = re.match(r"^(?P<name>.+)_(?P<date>\d{6})$", entry)
        if not m:
            continue
        name = m.group("name")
        date = m.group("date")
        phase_paths: Dict[str, str] = {}
        for fn in sorted(os.listdir(full)):
            if not fn.lower().endswith(".txt"):
                continue
            ph = infer_wet_phase_from_filename(fn)
            if not ph:
                continue
            phase_paths[ph] = os.path.join(full, fn)
        if "Session" not in phase_paths and not all(p in phase_paths for p in PHASE_ORDER):
            continue
        cases.append(WetCase(subject=name, date=date, src_dir=full, phase_paths=phase_paths))
    return cases


def load_wet_case_signals_267(case: WetCase, *, lead_index_1based: int) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Return phase->signal_267. If Session exists, return only {'Session': ...}"""
    if "Session" in case.phase_paths:
        arr = read_multicol_txt(case.phase_paths["Session"])
        sig = select_lead_column(arr, lead_index_1based)
        return {"Session": sig}, ["Session"]

    out: Dict[str, np.ndarray] = {}
    for ph in PHASE_ORDER:
        p = case.phase_paths[ph]
        arr = read_multicol_txt(p)
        out[ph] = select_lead_column(arr, lead_index_1based)
    return out, list(PHASE_ORDER)


def build_concat_360_and_bounds(
    phase_signals_267: Dict[str, np.ndarray],
    phase_order: List[str],
    *,
    fs_src: int,
    fs_tgt: int,
    trim_s: float,
) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Resample phases independently (for exact bounds), concat, then trim only at start."""
    parts_360: List[np.ndarray] = []
    seg_lens: List[int] = []
    for ph in phase_order:
        x = baseline_center(phase_signals_267[ph])
        x = resample(x, fs_src=fs_src, fs_tgt=fs_tgt)
        parts_360.append(x)
        seg_lens.append(int(len(x)))

    x_360 = np.concatenate(parts_360, axis=0) if parts_360 else np.zeros((0,), dtype=np.float32)

    # Bounds before trim
    bounds: List[Tuple[str, int, int]] = []
    s = 0
    for ph, n in zip(phase_order, seg_lens):
        e = s + int(n)
        bounds.append((ph, int(s), int(e)))
        s = e

    # Apply trim at start and shift bounds
    trim_n = int(round(float(trim_s) * float(fs_tgt)))
    if trim_n > 0 and len(x_360) > trim_n:
        x_360 = x_360[trim_n:]
        new_bounds = []
        for ph, s, e in bounds:
            s2 = max(0, s - trim_n)
            e2 = max(0, e - trim_n)
            if e2 > 0 and s2 < e2:
                new_bounds.append((ph, s2, min(e2, len(x_360))))
        bounds = new_bounds

    return x_360.astype(np.float32), bounds


def compute_overall_metrics(
    *,
    raw_360: np.ndarray,
    deno_360: np.ndarray,
    fs: int,
    tol_ms: float,
    qrs_band: Tuple[float, float],
    hf_band: Tuple[float, float],
    snr_hf_fn,
    snr_residual_fn,
    sliding_eval_fn=None,
    window_sizes_s: Tuple[int, int] | Tuple[()] = (10, 30),
    step_s: int = 1,
    qrs_window_ms: float = 300.0,
) -> Dict[str, object]:
    """Compute overall label-free metrics (HF gain + rpeak stability + sliding worst windows)."""
    rp_raw = detect_rpeaks(
        raw_360,
        fs,
        trim_start_s=0.0,
        prefer_neurokit=True,
        prefer_positive=True,
        assume_stable_hr=True,
        min_rr_s=0.30,
    )
    rp_deno = detect_rpeaks(
        deno_360,
        fs,
        trim_start_s=0.0,
        prefer_neurokit=True,
        prefer_positive=True,
        assume_stable_hr=True,
        min_rr_s=0.30,
    )

    snr_hf_raw_db = snr_hf_fn(raw_360, rp_raw, fs, qrs_band=qrs_band, noise_band=hf_band)
    snr_hf_deno_db = snr_hf_fn(deno_360, rp_deno, fs, qrs_band=qrs_band, noise_band=hf_band)
    snr_hf_gain_db = (
        float(snr_hf_deno_db - snr_hf_raw_db)
        if (np.isfinite(snr_hf_raw_db) and np.isfinite(snr_hf_deno_db))
        else float("nan")
    )

    snr_residual_db = snr_residual_fn(raw_360, deno_360, rp_deno, fs, qrs_band=qrs_band)

    stab = evaluate_rpeak_performance(rp_raw, rp_deno, fs, tol_ms=tol_ms)

    summary_sw = None
    if sliding_eval_fn is not None and window_sizes_s:
        # Sliding-window eval uses the same cfg object layout as base runner; use a tiny shim.
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            fs_tgt=fs,
            window_sizes_s=window_sizes_s,
            step_s=step_s,
            tol_ms=tol_ms,
            qrs_window_ms=qrs_window_ms,
            qrs_band=qrs_band,
            hf_band=hf_band,
        )

        _df_sw, summary_sw = sliding_eval_fn(raw_360, deno_360, rp_raw, rp_deno, cfg)

    out: Dict[str, object] = {
        "n_rpeaks_raw": int(len(rp_raw)),
        "n_rpeaks_deno": int(len(rp_deno)),
        "snr_hf_raw_db": float(snr_hf_raw_db) if np.isfinite(snr_hf_raw_db) else float("nan"),
        "snr_hf_deno_db": float(snr_hf_deno_db) if np.isfinite(snr_hf_deno_db) else float("nan"),
        "snr_hf_gain_db": float(snr_hf_gain_db) if np.isfinite(snr_hf_gain_db) else float("nan"),
        "snr_residual_db": float(snr_residual_db) if np.isfinite(snr_residual_db) else float("nan"),
        "rpeak_f1": float(stab.get("f1")) if isinstance(stab, dict) else float("nan"),
        "rpeak_precision": float(stab.get("precision")) if isinstance(stab, dict) else float("nan"),
        "rpeak_recall": float(stab.get("recall")) if isinstance(stab, dict) else float("nan"),
    }

    if summary_sw is not None:
        for W_s in window_sizes_s:
            ww = summary_sw.get("worst_windows", {}).get(str(W_s), {}).get("match_rate_min")
            out[f"worst_match_rate_{W_s}s"] = ww.get("value") if isinstance(ww, dict) else float("nan")
            out[f"worst_match_center_s_{W_s}s"] = ww.get("center_s") if isinstance(ww, dict) else float("nan")

    return out
