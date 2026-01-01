"""Ajou ECG utilities (minimal, import-safe).

This module intentionally provides only the two functions used by the batch
pipeline:

- detect_rpeaks
- evaluate_rpeak_performance

The R-peak detector is designed to prefer *upward (positive)* R-peaks and to
optionally enforce a minimum RR interval to reject physiologically implausible
double-detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_1d_float(x: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")
    return arr


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


def _robust_scale(x: np.ndarray) -> float:
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad + 1e-12)


def _local_maxima_indices(x: np.ndarray, min_distance: int) -> np.ndarray:
    # Strict local maxima (no plateaus handling).
    if x.size < 3:
        return np.array([], dtype=int)
    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if candidates.size == 0:
        return candidates

    if min_distance <= 1:
        return candidates

    # Non-maximum suppression by amplitude.
    order = np.argsort(x[candidates])[::-1]
    keep: List[int] = []
    taken = np.zeros(x.size, dtype=bool)
    half = int(min_distance)
    for idx in candidates[order]:
        if taken[idx]:
            continue
        keep.append(int(idx))
        lo = max(0, idx - half)
        hi = min(x.size, idx + half + 1)
        taken[lo:hi] = True
    keep.sort()
    return np.asarray(keep, dtype=int)


def _refine_peaks_positive(signal: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    radius = int(max(1, radius))
    refined = []
    n = signal.size
    for p in peaks:
        lo = max(0, int(p) - radius)
        hi = min(n, int(p) + radius + 1)
        window = signal[lo:hi]
        if window.size == 0:
            continue
        refined.append(int(lo + int(np.argmax(window))))
    return np.asarray(refined, dtype=int)


def _apply_min_rr(peaks: np.ndarray, signal_for_amp: np.ndarray, min_rr: int) -> np.ndarray:
    if peaks.size <= 1:
        return peaks
    min_rr = int(min_rr)
    if min_rr <= 0:
        return peaks

    peaks = np.asarray(peaks, dtype=int)
    keep = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        if p - keep[-1] >= min_rr:
            keep.append(p)
            continue

        # Too close: keep the one with larger amplitude (prefer upward R).
        prev = keep[-1]
        if signal_for_amp[p] > signal_for_amp[prev]:
            keep[-1] = p

    return np.asarray(keep, dtype=int)


@dataclass(frozen=True)
class RPeakDebug:
    chosen_polarity: str
    threshold: float
    min_distance: int
    min_rr: int


def detect_rpeaks(
    ecg: np.ndarray | Sequence[float],
    fs: float,
    *,
    prefer_positive: bool = True,
    smooth_ms: float = 12.0,
    threshold_k: float = 3.5,
    min_distance_s: float = 0.20,
    refine_radius_ms: float = 40.0,
    min_rr_s: Optional[float] = 0.28,
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float | int | str]]:
    """Detect R-peaks with positive-peak preference.

    Parameters
    - ecg: 1D ECG signal.
    - fs: sampling rate (Hz).
    - prefer_positive: if True, evaluate both polarities but prefer upward peaks.
    - smooth_ms: smoothing window (ms) for a simple QRS-band proxy.
    - threshold_k: threshold in robust-sigma units.
    - min_distance_s: minimum distance between candidate peaks (seconds).
    - refine_radius_ms: refine each peak by taking the max in a local window.
    - min_rr_s: if set, apply min RR constraint after refinement.

    Returns
    - peaks (sample indices). If return_debug=True, also returns a debug dict.
    """
    x = _ensure_1d_float(ecg)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")

    # Simple high-pass-ish proxy: subtract moving average baseline.
    smooth_win = max(1, int(round((smooth_ms / 1000.0) * fs)))
    baseline = _moving_average(x, max(1, int(round(0.200 * fs))))
    hp = x - baseline
    qrs_proxy = _moving_average(hp, smooth_win)

    robust_sigma = _robust_scale(qrs_proxy)
    thr = float(threshold_k) * robust_sigma
    min_dist = max(1, int(round(float(min_distance_s) * fs)))
    refine_radius = max(1, int(round((float(refine_radius_ms) / 1000.0) * fs)))
    min_rr = 0
    if min_rr_s is not None:
        min_rr = max(0, int(round(float(min_rr_s) * fs)))

    def _detect_on(signal: np.ndarray) -> np.ndarray:
        candidates = _local_maxima_indices(signal, min_distance=min_dist)
        if candidates.size == 0:
            return candidates
        return candidates[signal[candidates] >= thr]

    pos_peaks = _detect_on(qrs_proxy)
    neg_peaks = _detect_on(-qrs_proxy)

    chosen = pos_peaks
    chosen_polarity = "+"
    if prefer_positive:
        # Prefer positive unless clearly worse.
        if pos_peaks.size == 0 and neg_peaks.size > 0:
            chosen = neg_peaks
            chosen_polarity = "-"
    else:
        # Pick polarity with more detections (tie -> positive).
        if neg_peaks.size > pos_peaks.size:
            chosen = neg_peaks
            chosen_polarity = "-"

    refined = _refine_peaks_positive(x, chosen, radius=refine_radius)
    refined = np.unique(refined)
    refined.sort()
    if min_rr > 0:
        refined = _apply_min_rr(refined, signal_for_amp=x, min_rr=min_rr)

    debug: Dict[str, float | int | str] = {
        "chosen_polarity": chosen_polarity,
        "threshold": thr,
        "min_distance": min_dist,
        "min_rr": min_rr,
        "n_pos": int(pos_peaks.size),
        "n_neg": int(neg_peaks.size),
    }
    return (refined, debug) if return_debug else refined


def evaluate_rpeak_performance(
    rp_a: np.ndarray | Sequence[int],
    rp_b: np.ndarray | Sequence[int],
    fs: float,
    *,
    tol_ms: float = 50.0,
) -> Dict[str, float]:
    """Compare two R-peak sequences with a matching tolerance.

    This is intentionally label-agnostic: it returns symmetric matching stats.
    """
    a = np.asarray(rp_a, dtype=int)
    b = np.asarray(rp_b, dtype=int)
    a = np.unique(a[a >= 0])
    b = np.unique(b[b >= 0])
    a.sort()
    b.sort()

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")
    tol = max(0, int(round((float(tol_ms) / 1000.0) * fs)))

    if a.size == 0 and b.size == 0:
        return {
            "n_a": 0.0,
            "n_b": 0.0,
            "matches": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_abs_dt_ms": 0.0,
        }
    if a.size == 0:
        return {
            "n_a": 0.0,
            "n_b": float(b.size),
            "matches": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_abs_dt_ms": float("nan"),
        }
    if b.size == 0:
        return {
            "n_a": float(a.size),
            "n_b": 0.0,
            "matches": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_abs_dt_ms": float("nan"),
        }

    i = 0
    j = 0
    matches = 0
    dts: List[int] = []
    while i < a.size and j < b.size:
        dt = int(b[j] - a[i])
        if abs(dt) <= tol:
            matches += 1
            dts.append(dt)
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    precision = matches / float(b.size) if b.size else 0.0
    recall = matches / float(a.size) if a.size else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_abs_dt_ms = float(np.mean(np.abs(dts)) * 1000.0 / fs) if dts else float("nan")

    return {
        "n_a": float(a.size),
        "n_b": float(b.size),
        "matches": float(matches),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_abs_dt_ms": mean_abs_dt_ms,
    }
"""
Ajou ECG fine‑tuning & denoising pipeline (enhanced version).

This script replicates the core functionality of the original Jupyter
notebook used for Ajou ECG data processing but refactors the code
into a single runnable module and incorporates several improvements
motivated by the literature review provided.  Key changes include:

* **Segmentation and SNR estimation:**  When no ground truth exists,
  assess denoising quality by computing the signal‑to‑noise ratio (SNR).
  Two approaches are supported: (1) a peak‑to‑peak SNR estimate using
  the ratio of R‑peak amplitude to residual noise, and (2) a surrogate
  reference approach in which a relatively clean segment (such as a
  resting period) is used as a proxy for the true signal and compared
  against a noisy segment【25†L89-L97】.  These functions allow
  objective quantification of noise reduction without labels【7†L200-L208】.

* **HRV‑based evaluation:**  In addition to classical precision/recall
  metrics derived from R‑peak matching, this script calculates heart
  rate variability (HRV) indices – mean RR, SDNN, RMSSD and pNN50 – from
  the detected R‑peaks【34†L85-L89】.  Stable HRV values after
  denoising provide evidence that the cardiac rhythm has been preserved.

* **Morphological quality metrics:**  The `ensemble_std` function
  computes the mean of the standard deviation of aligned heart beats,
  similar to the ensemble standard deviation used in Halvaei et al.
  (2021)【18†L754-L763】.  Lower values indicate reduced beat‑to‑beat
  variability due to noise.  Another optional time‑frequency recurrence
  metric could be added following the literature【18†L774-L782】, but
  is omitted here for brevity.

* **Fine‑tuning strategy:**  A simple example of partial fine‑tuning is
  provided.  Only the final decoder layers are trainable by default,
"""Lightweight helpers used by the phase batch runner.

Only two functions are imported by tools/run_phase_batch_transformer_dae.py:
- detect_rpeaks
- evaluate_rpeak_performance

This module intentionally stays dependency-light (NumPy + SciPy only).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(x: np.ndarray, fs: int, lo: float, hi: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    lo = max(1e-6, float(lo))
    hi = min(float(hi), nyq - 1e-6)
    if not (0 < lo < hi < nyq):
        return x
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.nan_to_num(x, nan=np.nanmean(x)))


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def _refine_to_signed_max(ecg: np.ndarray, peaks: np.ndarray, fs: int, search_ms: float = 80.0) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(ecg))]
    if peaks.size == 0:
        return peaks
    qrs = _bandpass(ecg, fs, 5.0, 20.0)
    half = int(round((search_ms / 1000.0) * fs))
    refined: List[int] = []
    for p in peaks:
        a = max(0, p - half)
        b = min(len(ecg), p + half + 1)
        if b - a <= 1:
            refined.append(int(p))
            continue
        refined.append(int(a + np.argmax(qrs[a:b])))
    out = np.unique(np.asarray(refined, dtype=int))
    out.sort()
    return out


def _regularize_rr(peaks: np.ndarray, ecg: np.ndarray, fs: int, min_rr_s: float = 0.30) -> np.ndarray:
    """Remove peaks that are too close (stable-HR assumption).

    For any RR < min_rr_s, drop the weaker of the two adjacent peaks
    (based on signed QRS-band amplitude).
    """
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 3:
        return peaks
    peaks = np.unique(peaks)
    peaks.sort()
    qrs = _bandpass(ecg, fs, 5.0, 20.0)
    amps = qrs[peaks]
    min_rr = int(round(min_rr_s * fs))

    changed = True
    guard = 0
    while changed and peaks.size >= 3 and guard < 5000:
        guard += 1
        changed = False
        rr = np.diff(peaks)
        bad = np.where(rr < min_rr)[0]
        if bad.size == 0:
            break
        i = int(bad[0])
        drop = i if amps[i] < amps[i + 1] else (i + 1)
        peaks = np.delete(peaks, drop)
        amps = np.delete(amps, drop)
        changed = True
    return peaks


def detect_rpeaks(
    ecg: np.ndarray,
    fs: int,
    trim_start_s: float = 0.0,
    *,
    prefer_positive: bool = True,
    assume_stable_hr: bool = True,
    min_rr_s: float = 0.30,
) -> np.ndarray:
    """Detect R-peaks in an ECG-like 1D signal.

    Key behaviors (to address common failure modes):
    - Polarity-aware candidate generation (sig vs -sig)
    - Refinement to signed QRS-band maxima (avoids snapping to negative peaks)
    - Optional RR regularization to remove implausibly-close peaks
    """
    x = np.asarray(ecg, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmean(x))
    trim = int(max(0.0, float(trim_start_s)) * fs)
    x0 = x[trim:] if trim < x.size else x.copy()
    if x0.size < int(fs):
        return np.asarray([], dtype=int)

    # Candidate detection signal
    det = _robust_z(_bandpass(x0, fs, 0.5, 40.0))

    def _pt_events(sig: np.ndarray) -> np.ndarray:
        d = np.concatenate(([0.0], np.diff(sig)))
        sq = d**2
        integ_len = max(int(0.12 * fs), 1)
        integ = np.convolve(sq, np.ones(integ_len) / integ_len, mode="same")
        i_med = np.median(integ)
        i_mad = np.median(np.abs(integ - i_med)) + 1e-12
        thr = i_med + 5.0 * (1.4826 * i_mad)
        dist = max(int(min_rr_s * fs), 1)
        peaks, _ = find_peaks(integ, height=thr, distance=dist)
        return np.asarray(peaks, dtype=int)

    cand1 = _refine_to_signed_max(x0, _pt_events(det), fs)
    cand2 = _refine_to_signed_max(x0, _pt_events(-det), fs)

    qrs = _bandpass(x0, fs, 5.0, 20.0)
    s1 = float(np.median(qrs[cand1])) if cand1.size else -1e18
    s2 = float(np.median(qrs[cand2])) if cand2.size else -1e18
    if not prefer_positive:
        s1 = abs(s1)
        s2 = abs(s2)
    peaks = cand1 if s1 >= s2 else cand2

    if assume_stable_hr and peaks.size >= 3:
        peaks = _regularize_rr(peaks, x0, fs, min_rr_s=min_rr_s)

    peaks = peaks + trim
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]
    return peaks


def evaluate_rpeak_performance(
    raw_peaks: np.ndarray,
    deno_peaks: np.ndarray,
    fs: int,
    tol_ms: float = 50.0,
) -> Dict[str, float]:
    """One-to-one matching metrics between raw and deno peaks."""
    tol = int(round((tol_ms / 1000.0) * fs))
    ref = np.asarray(raw_peaks, dtype=int)
    test = np.asarray(deno_peaks, dtype=int)
    ref = ref[np.isfinite(ref)]
    test = test[np.isfinite(test)]
    ref.sort()
    test.sort()

    used = np.zeros(len(test), dtype=bool)
    tp = 0
    for pr in ref:
        if test.size == 0:
            break
        d = np.abs(test - pr)
        if used.any():
            d = d.copy()
            d[used] = 1_000_000_000
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1

    fn = int(len(ref) - tp)
    fp = int(np.sum(~used))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_raw_peaks": int(len(ref)),
        "n_deno_peaks": int(len(test)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    2) Refine each event to the nearest local maximum on the *signed* QRS-band signal.
       (Avoids using abs(), which can snap to negative peaks.)
    3) Choose polarity that yields stronger positive QRS amplitude (if prefer_positive).
    4) Optionally remove implausibly-close peaks by dropping the weaker one.
    """
    x = np.asarray(ecg, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmean(x))
    if x.size < int(fs):
        return np.asarray([], dtype=int)

    nyquist = fs / 2.0
    lo = max(1e-6, float(lowcut))
    hi = min(float(highcut), nyquist - 1e-6)
    if not (0 < lo < hi < nyquist):
        return np.asarray([], dtype=int)

    b, a = butter(3, [lo / nyquist, hi / nyquist], btype="band")
    qrs = filtfilt(b, a, x)

    def _robust_z(sig: np.ndarray) -> np.ndarray:
        sig = np.asarray(sig, dtype=float)
        med = np.median(sig)
        mad = np.median(np.abs(sig - med)) + 1e-12
        return (sig - med) / (1.4826 * mad)

    def _pt_events(sig_z: np.ndarray) -> np.ndarray:
        d = np.concatenate(([0.0], np.diff(sig_z)))
        sq = d**2
        integ_len = max(int(0.12 * fs), 1)
        integ = np.convolve(sq, np.ones(integ_len) / integ_len, mode="same")
        i_med = np.median(integ)
        i_mad = np.median(np.abs(integ - i_med)) + 1e-12
        thr = i_med + 5.0 * (1.4826 * i_mad)
        dist = max(int(min_rr_s * fs), 1)
        peaks, _ = find_peaks(integ, height=thr, distance=dist)
        return np.asarray(peaks, dtype=int)

    def _refine_to_signed_max(sig_qrs: np.ndarray, peaks: np.ndarray, search_ms: float = 80.0) -> np.ndarray:
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < sig_qrs.size)]
        if peaks.size == 0:
            return peaks
        half = int(round((search_ms / 1000.0) * fs))
        out = []
        for p in peaks:
            a0 = max(0, p - half)
            b0 = min(sig_qrs.size, p + half + 1)
            if b0 - a0 <= 1:
                out.append(int(p))
                continue
            out.append(int(a0 + np.argmax(sig_qrs[a0:b0])))
        out = np.unique(np.asarray(out, dtype=int))
        out.sort()
        return out

    def _score(sig_qrs: np.ndarray, peaks: np.ndarray) -> float:
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return -1e18
        amp = float(np.median(sig_qrs[peaks]))  # signed
        if peaks.size >= 2:
            rr = np.diff(peaks) / float(fs)
            frac_short = float(np.mean(rr < min_rr_s))
            cv = float(np.std(rr) / (np.mean(rr) + 1e-12))
        else:
            frac_short, cv = 0.0, 1.0
        if prefer_positive:
            return amp - 10.0 * frac_short - 0.5 * cv + 0.001 * peaks.size
        return abs(amp) - 10.0 * frac_short - 0.5 * cv + 0.001 * peaks.size

    qrs_z = _robust_z(qrs)
    p_pos = _refine_to_signed_max(qrs, _pt_events(qrs_z))
    p_neg = _refine_to_signed_max(qrs, _pt_events(-qrs_z))
    peaks = p_pos if _score(qrs, p_pos) >= _score(qrs, p_neg) else p_neg

    if assume_stable_hr and peaks.size >= 3:
        min_rr = int(round(min_rr_s * fs))
        amps = qrs[peaks]
        changed = True
        guard = 0
        while changed and peaks.size >= 3 and guard < 5000:
            guard += 1
            changed = False
            rr = np.diff(peaks)
            bad = np.where(rr < min_rr)[0]
            if bad.size == 0:
                break
            i = int(bad[0])
            drop = i if amps[i] < amps[i + 1] else (i + 1)
            peaks = np.delete(peaks, drop)
            amps = np.delete(amps, drop)
            changed = True

    return peaks


def train_model(model: tf.keras.Model, X_train: np.ndarray, F_train: np.ndarray) -> tf.keras.callbacks.History:
    """Fine‑tune the model on the given data with early stopping.

    Uses a small learning rate and limited epochs to avoid overfitting.
    Only the trainable layers (decoder) will be updated if the model has
    been built with `trainable_decoder_only=True`.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x=[X_train, F_train],
        y=X_train,
        validation_split=0.2,
        epochs=10,  # fewer epochs mitigate overfitting
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def reconstruct_signal(frames: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Reconstruct a signal from overlapping frames using a Hann window."""
    n_frames, L = frames.shape
    total_len = (n_frames - 1) * hop + win
    y = np.zeros(total_len)
    w = np.hanning(win)
    w /= np.max(w)
    accw = np.zeros(total_len)
    for i in range(n_frames):
        start = i * hop
        end = start + win
        y[start:end] += frames[i] * w
        accw[start:end] += w
    accw[accw == 0] = 1.0
    return y / accw


def evaluate_rpeak_performance(raw_peaks: np.ndarray, deno_peaks: np.ndarray, fs: int, tol_ms: float = 50.0) -> Dict[str, float]:
    """Compute basic R‑peak detection metrics between raw and denoised peaks."""
    tol_samples = int(tol_ms * fs / 1000.0)
    TP = 0
    FP = 0
    FN = 0
    matched_deno = set()
    matched_raw = set()
    for idx_raw in raw_peaks:
        if len(deno_peaks) == 0:
            FN += 1
            continue
        distances = np.abs(deno_peaks - idx_raw)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tol_samples:
            TP += 1
            matched_deno.add(closest_idx)
            matched_raw.add(idx_raw)
        else:
            FN += 1
    FP = len(deno_peaks) - len(matched_deno)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_raw_peaks": int(len(raw_peaks)),
        "n_deno_peaks": int(len(deno_peaks)),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    # 1. Load data
    segments, meta = load_ecg_segments(DATA_DIR)
    print(f"Loaded {len(segments)} segments from {DATA_DIR}")
    # 2. Resample to 360 Hz
    resampled_segments, res_lengths = resample_segments(segments, FS_SRC, FS_TGT)
    lead_concat = np.concatenate(resampled_segments)
    # 3. Baseline centering
    lead_centered = baseline_centering(lead_concat, res_lengths)
    # 4. Frame and compute Fourier features
    X2d, padded_len = frame_signal(lead_centered, WIN, HOP)
    X_train = np.expand_dims(X2d, axis=2)
    F_train = np.expand_dims(make_fourier(X2d, n=WIN, fs=FS_TGT), axis=2)
    # 5. Build model and optionally fine‑tune
    model = build_model(WIN, PRETRAINED_PATH, trainable_decoder_only=True)
    # Fine‑tuning may be skipped by commenting out the next two lines
    print("Starting fine‑tuning...")
    _ = train_model(model, X_train, F_train)
    # 6. Denoise frames and reconstruct signal
    Y_hat_frames = model.predict([X_train, F_train], batch_size=32, verbose=0)
    Y_hat_frames = np.squeeze(Y_hat_frames, axis=2)
    denoised_full = reconstruct_signal(Y_hat_frames, WIN, HOP)
    # Trim to match length
    min_len = min(len(lead_centered), len(denoised_full))
    raw_sig = lead_centered[:min_len]
    deno_sig = denoised_full[:min_len]
    # 7. R‑peak detection
    raw_peaks = detect_rpeaks(raw_sig, FS_TGT)
    deno_peaks = detect_rpeaks(deno_sig, FS_TGT)
    # 8. Compute evaluation metrics
    rpeak_metrics = evaluate_rpeak_performance(raw_peaks, deno_peaks, FS_TGT)
    # SNR improvements
    snr_ptp_raw = compute_snr_ptp(raw_sig, raw_sig)
    snr_ptp_deno = compute_snr_ptp(raw_sig, deno_sig)
    snr_improvement = snr_ptp_deno - snr_ptp_raw
    # HRV metrics
    hrv_raw = compute_hrv_metrics(raw_peaks, FS_TGT)
    hrv_deno = compute_hrv_metrics(deno_peaks, FS_TGT)
    # Morphological consistency
    ens_std_raw = ensemble_std(raw_sig, raw_peaks, FS_TGT)
    ens_std_deno = ensemble_std(deno_sig, deno_peaks, FS_TGT)
    # Package metrics
    results = {
        **rpeak_metrics,
        "snr_ptp_raw_db": snr_ptp_raw,
        "snr_ptp_deno_db": snr_ptp_deno,
        "snr_improvement_db": snr_improvement,
        "hrv_raw": hrv_raw,
        "hrv_deno": hrv_deno,
        "ensemble_std_raw": ens_std_raw,
        "ensemble_std_deno": ens_std_deno,
        "ensemble_std_reduction": ens_std_raw - ens_std_deno if not np.isnan(ens_std_raw) and not np.isnan(ens_std_deno) else np.nan,
    }
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "ajou_denoising_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    # Print summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()