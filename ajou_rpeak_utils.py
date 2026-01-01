"""R-peak utilities used by Ajou generalization/batch runners.

This module is intentionally small and import-safe.

Key behaviors:
- Prefer upward (positive) R-peaks when polarity is ambiguous.
- Refine peaks by snapping to the local *positive* maximum (no abs()).
- Optionally enforce a minimum RR interval to remove implausibly-close peaks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    nyq = float(fs) / 2.0
    lo2 = max(1e-6, float(lo))
    hi2 = min(float(hi), nyq - 1e-6)
    if not (0 < lo2 < hi2 < nyq):
        return sig
    b, a = butter(order, [lo2 / nyq, hi2 / nyq], btype="band")
    return filtfilt(b, a, sig)


def _robust_z(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) + 1e-12
    return (sig - med) / (1.4826 * mad)


def infer_ecg_polarity(
    ecg: np.ndarray | Sequence[float],
    fs: float,
    *,
    min_rr_s: float = 0.30,
    search_ms: float = 80.0,
) -> Dict[str, float | int | str]:
    """Infer whether the ECG is inverted using an objective QRS-band criterion.

    Method (label-free):
    - build polarity-invariant QRS event candidates via a PT-like energy signal
      (derivative^2 + moving integration)
    - refine each event to the *max absolute* extremum in the 5–20Hz QRS band
    - use the median signed QRS value at these extrema:
        median(qrs[extrema]) < 0  => inverted (R deflection is downward)

    Returns a dict with:
    - polarity_sign: +1 (as-is) or -1 (should flip)
    - median_qrs_at_extrema, n_extrema
    """
    x = _ensure_1d_float(ecg)
    x = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")

    qrs = _bandpass(x, fs, 5.0, 20.0)
    det = _robust_z(_bandpass(x, fs, 0.5, 40.0))

    d = np.concatenate(([0.0], np.diff(det)))
    sq = d**2
    integ_len = max(int(0.12 * fs), 1)
    integ = np.convolve(sq, np.ones(integ_len) / float(integ_len), mode="same")

    i_med = float(np.median(integ))
    i_mad = float(np.median(np.abs(integ - i_med)) + 1e-12)
    thr = i_med + 5.0 * (1.4826 * i_mad)
    dist = max(int(float(min_rr_s) * fs), 1)
    peaks, _ = find_peaks(integ, height=thr, distance=dist)
    peaks = np.asarray(peaks, dtype=int)

    half = int((float(search_ms) / 1000.0) * fs)
    extrema: List[int] = []
    for p in peaks:
        a = max(0, int(p) - half)
        b = min(qrs.size, int(p) + half + 1)
        if b - a <= 1:
            continue
        j = int(a + int(np.argmax(np.abs(qrs[a:b]))))
        extrema.append(j)
    if len(extrema) == 0:
        return {
            "polarity_sign": 1,
            "median_qrs_at_extrema": 0.0,
            "n_extrema": 0,
        }

    extrema_arr = np.asarray(extrema, dtype=int)
    med_qrs = float(np.median(qrs[extrema_arr]))
    sign = 1 if med_qrs >= 0 else -1
    return {
        "polarity_sign": int(sign),
        "median_qrs_at_extrema": float(med_qrs),
        "n_extrema": int(extrema_arr.size),
    }


def normalize_ecg_polarity(
    ecg: np.ndarray | Sequence[float],
    fs: float,
    *,
    min_rr_s: float = 0.30,
    search_ms: float = 80.0,
) -> Tuple[np.ndarray, int, Dict[str, float | int | str]]:
    """Flip ECG (if needed) so that R deflections are upward (positive)."""
    x = _ensure_1d_float(ecg)
    info = infer_ecg_polarity(x, fs, min_rr_s=min_rr_s, search_ms=search_ms)
    sign = int(info["polarity_sign"])
    return (x * float(sign), sign, info)


def align_ecg_polarity_to_reference(
    reference: np.ndarray | Sequence[float],
    target: np.ndarray | Sequence[float],
    fs: float,
    *,
    band: Tuple[float, float] = (5.0, 20.0),
) -> Tuple[np.ndarray, int, Dict[str, float | int]]:
    """Align target polarity to reference using QRS-band correlation.

    Objective criterion:
      corr(qrs_ref, qrs_tgt) < 0  => flip target.
    """
    ref = _ensure_1d_float(reference)
    tgt = _ensure_1d_float(target)
    L = min(ref.size, tgt.size)
    if L <= 5:
        return tgt.copy(), 1, {"corr_qrs": 0.0}

    qrs_ref = _bandpass(ref[:L], fs, band[0], band[1])
    qrs_tgt = _bandpass(tgt[:L], fs, band[0], band[1])
    qrs_ref = qrs_ref - float(np.mean(qrs_ref))
    qrs_tgt = qrs_tgt - float(np.mean(qrs_tgt))
    denom = float(np.std(qrs_ref) * np.std(qrs_tgt) + 1e-12)
    corr = float(np.mean(qrs_ref * qrs_tgt) / denom)
    sign = 1 if corr >= 0 else -1
    return tgt[:L] * float(sign), int(sign), {"corr_qrs": corr}


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
    if x.size < 3:
        return np.array([], dtype=int)
    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if candidates.size == 0:
        return candidates

    if min_distance <= 1:
        return candidates

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


def _refine_to_local_positive_max(signal: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    radius = int(max(1, radius))
    refined: List[int] = []
    n = signal.size
    for p in peaks:
        lo = max(0, int(p) - radius)
        hi = min(n, int(p) + radius + 1)
        window = signal[lo:hi]
        if window.size == 0:
            continue
        refined.append(int(lo + int(np.argmax(window))))
    return np.asarray(refined, dtype=int)


def _apply_min_rr(peaks: np.ndarray, amp_signal: np.ndarray, min_rr: int) -> np.ndarray:
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

        prev = keep[-1]
        if amp_signal[p] > amp_signal[prev]:
            keep[-1] = p

    return np.asarray(keep, dtype=int)


def detect_rpeaks(
    ecg: np.ndarray | Sequence[float],
    fs: float,
    *,
    trim_start_s: float = 0.0,
    prefer_neurokit: bool = False,
    prefer_positive: bool = True,
    assume_stable_hr: bool = True,
    min_rr_s: float = 0.30,
    return_debug: bool = False,
    **_ignored_legacy_kwargs,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float | int | str]]:
    """Detect R-peaks (polarity-aware; positive-peaks preferred).

    This matches the notebook logic used in Ajou generalization experiments:
    - generate candidates from both polarities
    - refine on *signed* QRS-band maxima (no abs())
    - optionally remove implausibly-close peaks (stable-HR assumption)

    Notes
    - neurokit2 is optional. If prefer_neurokit=True but neurokit2 is not
      available, it falls back automatically.
    """

    x = _ensure_1d_float(ecg)
    x = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid fs={fs}")

    trim = int(max(0.0, float(trim_start_s)) * fs)
    x0 = x[trim:] if trim < x.size else x.copy()
    if x0.size < int(fs):
        debug = {
            "chosen_polarity": "+",
            "n_pos": 0,
            "n_neg": 0,
            "min_rr": int(round(min_rr_s * fs)),
            "used_neurokit": False,
        }
        empty = np.asarray([], dtype=int)
        return (empty, debug) if return_debug else empty

    def _bandpass(sig: np.ndarray, lo: float, hi: float, order: int = 3) -> np.ndarray:
        nyq = fs / 2.0
        lo2 = max(1e-6, float(lo))
        hi2 = min(float(hi), nyq - 1e-6)
        if not (0 < lo2 < hi2 < nyq):
            return sig
        b, a = butter(order, [lo2 / nyq, hi2 / nyq], btype="band")
        return filtfilt(b, a, sig)

    def _robust_z(sig: np.ndarray) -> np.ndarray:
        med = np.median(sig)
        mad = np.median(np.abs(sig - med)) + 1e-12
        return (sig - med) / (1.4826 * mad)

    def _refine_on_qrs_signed_max(peaks: np.ndarray, search_ms: float = 80.0) -> np.ndarray:
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < x0.size)]
        if peaks.size == 0:
            return peaks
        qrs = _bandpass(x0, 5.0, 20.0)
        half = int((search_ms / 1000.0) * fs)
        refined: List[int] = []
        for p in peaks:
            a = max(0, int(p) - half)
            b = min(x0.size, int(p) + half + 1)
            if b - a <= 1:
                refined.append(int(p))
                continue
            refined.append(int(a + int(np.argmax(qrs[a:b]))))
        out = np.unique(np.asarray(refined, dtype=int))
        out.sort()
        return out

    def _regularize_rr(peaks: np.ndarray) -> np.ndarray:
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < x0.size)]
        if peaks.size < 3:
            return peaks
        peaks = np.unique(peaks)
        peaks.sort()
        qrs = _bandpass(x0, 5.0, 20.0)
        amp = qrs[peaks]

        min_rr = int(round(float(min_rr_s) * fs))
        min_rr = max(1, min_rr)

        # 1) hard: drop one of too-close pair
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
            drop = i if amp[i] < amp[i + 1] else (i + 1)
            peaks = np.delete(peaks, drop)
            amp = np.delete(amp, drop)
            changed = True

        if peaks.size < 3:
            return peaks

        # 2) soft: drop extra peaks making RR unusually short vs median
        rr = np.diff(peaks).astype(float)
        med = float(np.median(rr))
        if not np.isfinite(med) or med <= 0:
            return peaks
        rel_min = int(round(max(0.5 * med, min_rr)))

        changed = True
        guard = 0
        while changed and peaks.size >= 3 and guard < 5000:
            guard += 1
            changed = False
            rr = np.diff(peaks)
            bad = np.where(rr < rel_min)[0]
            if bad.size == 0:
                break
            i = int(bad[0])
            drop = i if amp[i] < amp[i + 1] else (i + 1)
            peaks = np.delete(peaks, drop)
            amp = np.delete(amp, drop)
            changed = True

        return peaks

    def _score_candidate(peaks: np.ndarray) -> float:
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < x0.size)]
        if peaks.size == 0:
            return -1e18
        qrs = _bandpass(x0, 5.0, 20.0)
        amp = float(np.median(qrs[peaks]))
        if not prefer_positive:
            amp = abs(amp)
        if peaks.size >= 2:
            rr_s = np.diff(peaks) / fs
            frac_too_short = float(np.mean(rr_s < float(min_rr_s)))
            cv = float(np.std(rr_s) / (np.mean(rr_s) + 1e-12))
        else:
            frac_too_short, cv = 0.0, 1.0
        return amp - 10.0 * frac_too_short - 0.5 * cv + 0.001 * float(peaks.size)

    # Candidate detection signal
    det = _robust_z(_bandpass(x0, 0.5, 40.0))

    used_neurokit = False
    cand_pos: np.ndarray
    cand_neg: np.ndarray

    if prefer_neurokit:
        try:
            import neurokit2 as nk  # type: ignore

            used_neurokit = True
            sig = nk.signal_sanitize(det)
            _, info = nk.ecg_peaks(sig, sampling_rate=float(fs), method="neurokit")
            cand_pos = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)

            sig_inv = nk.signal_sanitize(-det)
            _, info2 = nk.ecg_peaks(sig_inv, sampling_rate=float(fs), method="neurokit")
            cand_neg = np.asarray(info2.get("ECG_R_Peaks", []), dtype=int)
        except Exception:
            used_neurokit = False
            cand_pos = np.asarray([], dtype=int)
            cand_neg = np.asarray([], dtype=int)
    else:
        cand_pos = np.asarray([], dtype=int)
        cand_neg = np.asarray([], dtype=int)

    if cand_pos.size == 0 and cand_neg.size == 0:
        # Fallback: Pan-Tompkins-like events on QRS-band proxy
        f = _robust_z(_bandpass(x0, 5.0, 20.0))

        def _pt_events(sig: np.ndarray) -> np.ndarray:
            d = np.concatenate(([0.0], np.diff(sig)))
            sq = d**2
            integ_len = max(int(0.12 * fs), 1)
            integ = np.convolve(sq, np.ones(integ_len) / float(integ_len), mode="same")
            i_med = np.median(integ)
            i_mad = np.median(np.abs(integ - i_med)) + 1e-12
            thr = i_med + 5.0 * (1.4826 * i_mad)
            dist = max(int(float(min_rr_s) * fs), 1)
            peaks, _ = find_peaks(integ, height=thr, distance=dist)
            return np.asarray(peaks, dtype=int)

        cand_pos = _pt_events(f)
        cand_neg = _pt_events(-f)

    refined_pos = _refine_on_qrs_signed_max(cand_pos)
    refined_neg = _refine_on_qrs_signed_max(cand_neg)

    s_pos = _score_candidate(refined_pos)
    s_neg = _score_candidate(refined_neg)

    chosen_polarity = "+" if s_pos >= s_neg else "-"
    peaks = refined_pos if s_pos >= s_neg else refined_neg
    if assume_stable_hr and peaks.size >= 3:
        peaks = _regularize_rr(peaks)

    peaks = peaks + trim
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]

    debug: Dict[str, float | int | str] = {
        "chosen_polarity": chosen_polarity,
        "n_pos": int(refined_pos.size),
        "n_neg": int(refined_neg.size),
        "min_rr": int(round(float(min_rr_s) * fs)),
        "used_neurokit": bool(used_neurokit),
    }
    return (peaks, debug) if return_debug else peaks


def evaluate_rpeak_performance(
    rp_a: np.ndarray | Sequence[int],
    rp_b: np.ndarray | Sequence[int],
    fs: float,
    *,
    tol_ms: float = 50.0,
) -> Dict[str, float]:
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
