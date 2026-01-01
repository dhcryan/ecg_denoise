#!/usr/bin/env python3
"""Generate a multi-user, multi-phase ECG dataset with controlled noise.

Creates per-user, per-phase text files named like:
  User01_Stable_251231.txt
  User01_Breathe_251231.txt
  User01_Walk_251231.txt
  User01_Recovery_251231.txt

Phases (default durations):
- Stable:   2 min (clean baseline)
- Breathe:  2 min (baseline wander)
- Walk:     4 min (motion + EMG-like artifacts)
- Recovery: 2 min (reduced motion + residual wander)

Input source:
- A 1D numpy array from .npy (recommended: denoised signal), OR
- A text file with a single numeric column.

Output format:
- One sample per line (single column). Also writes a metadata.json.

This is meant for *scenario simulation*, not for claiming physiological realism.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    label: str
    duration_s: int


DEFAULT_PHASES: list[PhaseSpec] = [
    PhaseSpec(name="Stable", label="Stable", duration_s=2 * 60),
    PhaseSpec(name="Breathe", label="Breathe", duration_s=2 * 60),
    PhaseSpec(name="Move", label="Walk", duration_s=4 * 60),
    PhaseSpec(name="Recovery", label="Recovery", duration_s=2 * 60),
]


def _today_yymmdd() -> str:
    return datetime.now().strftime("%y%m%d")


def _read_1d_signal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        x = np.load(path)
        x = np.asarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array in {path}, got shape={x.shape}")
        return x.astype(np.float64)

    # txt/csv: take first column of numeric values
    vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split by whitespace/comma/tab
            parts = line.replace(",", " ").split()
            try:
                vals.append(float(parts[0]))
            except Exception:
                # skip non-numeric lines
                continue
    if not vals:
        raise ValueError(f"No numeric samples found in {path}")
    return np.asarray(vals, dtype=np.float64)


def _tile_to_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n].copy()
    reps = int(math.ceil(n / len(x)))
    y = np.tile(x, reps)[:n].copy()
    return y


def _resample_linear(x: np.ndarray, fs_src: int, fs_tgt: int) -> np.ndarray:
    """Simple NumPy-only resampler using linear interpolation.

    This is intentionally dependency-free (no SciPy). For dataset synthesis/augmentation
    this is typically sufficient.
    """
    fs_src = int(fs_src)
    fs_tgt = int(fs_tgt)
    if fs_src == fs_tgt:
        return x.astype(np.float64, copy=True)
    if fs_src <= 0 or fs_tgt <= 0:
        raise ValueError("fs_src/fs_tgt must be positive")

    n_src = int(len(x))
    dur_s = n_src / fs_src
    n_tgt = int(round(dur_s * fs_tgt))
    if n_tgt < 2:
        return np.asarray([float(np.mean(x))], dtype=np.float64)

    t_src = np.arange(n_src, dtype=np.float64) / fs_src
    t_tgt = np.arange(n_tgt, dtype=np.float64) / fs_tgt
    y = np.interp(t_tgt, t_src, x.astype(np.float64))
    return y


def _colored_noise(rng: np.random.Generator, n: int, fs: int, lo: float | None, hi: float | None) -> np.ndarray:
    """Generate band-limited noise without SciPy (FFT spectral shaping).

    lo/hi are in Hz. Use lo=None for lowpass, hi=None for highpass.
    """
    if lo is None and hi is None:
        raise ValueError("At least one of lo/hi must be set")

    w = rng.standard_normal(n).astype(np.float64)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = np.ones_like(freqs, dtype=np.float64)
    if lo is not None:
        mask *= (freqs >= float(lo)).astype(np.float64)
    if hi is not None:
        mask *= (freqs <= float(hi)).astype(np.float64)

    # Light edge smoothing to reduce ringing: cosine taper over 0.5 Hz.
    taper_hz = 0.5
    if lo is not None:
        a = float(lo)
        idx = (freqs >= a) & (freqs <= a + taper_hz)
        if np.any(idx):
            x = (freqs[idx] - a) / taper_hz
            mask[idx] *= 0.5 - 0.5 * np.cos(np.pi * x)
    if hi is not None:
        b = float(hi)
        idx = (freqs >= b - taper_hz) & (freqs <= b)
        if np.any(idx):
            x = (b - freqs[idx]) / taper_hz
            mask[idx] *= 0.5 - 0.5 * np.cos(np.pi * x)

    y = np.fft.irfft(W * mask, n=n)
    y = y - float(np.mean(y))
    y = y / (float(np.std(y)) + 1e-12)
    return y


def _scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # snr_db = 20*log10(rms(clean)/rms(noise_scaled))
    eps = 1e-12
    rms_c = float(np.sqrt(np.mean(clean * clean) + eps))
    rms_n = float(np.sqrt(np.mean(noise * noise) + eps))
    target_rms_n = rms_c / (10 ** (snr_db / 20.0))
    return noise * (target_rms_n / max(rms_n, eps))


def _baseline_wander(rng: np.random.Generator, n: int, fs: int, amp: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    # breathing-like low freq mixture
    freqs = rng.uniform(0.08, 0.35, size=3)
    phases = rng.uniform(0, 2 * np.pi, size=3)
    mix = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)) / 3.0
    # add slow random walk drift
    rw = rng.standard_normal(n)
    rw = np.cumsum(rw)
    rw = (rw - rw.mean()) / (rw.std() + 1e-12)
    drift = 0.35 * rw
    bw = mix + drift
    bw = (bw - bw.mean()) / (bw.std() + 1e-12)
    return amp * bw


def _motion_artifacts(rng: np.random.Generator, clean: np.ndarray, fs: int, snr_db: float) -> np.ndarray:
    n = len(clean)
    # Low-freq motion + high-freq EMG-like
    low = _colored_noise(rng, n, fs, lo=0.5, hi=5.0)
    emg = _colored_noise(rng, n, fs, lo=20.0, hi=80.0)

    # Burst mask to simulate intermittent motion
    mask = np.zeros(n, dtype=np.float64)
    n_bursts = max(3, int((n / fs) / 10))
    for _ in range(n_bursts):
        start = int(rng.integers(0, max(1, n - fs)))
        dur = int(rng.integers(int(0.5 * fs), int(2.0 * fs)))
        end = min(n, start + dur)
        # smooth window
        w = np.hanning(max(4, end - start))
        mask[start:end] = np.maximum(mask[start:end], w)

    noise = 0.8 * low * (0.3 + 0.7 * mask) + 0.6 * emg * (0.1 + 0.9 * mask)

    # occasional spikes
    n_spikes = max(2, int((n / fs) / 15))
    for _ in range(n_spikes):
        idx = int(rng.integers(0, n))
        width = int(rng.integers(1, max(2, int(0.03 * fs))))
        amp = float(rng.uniform(2.0, 6.0))
        j0 = max(0, idx - width)
        j1 = min(n, idx + width)
        span = int(j1 - j0)
        if span <= 0:
            continue
        if span >= 4:
            spike = np.hanning(span)
        else:
            spike = np.ones(span, dtype=np.float64)
        noise[j0:j1] += amp * spike * float(np.sign(rng.standard_normal()))

    return _scale_to_snr(clean, noise, snr_db)


def _phase_noise(
    rng: np.random.Generator,
    phase: PhaseSpec,
    clean: np.ndarray,
    fs: int,
    snr_db_map: dict[str, float],
    baseline_amp_frac_map: dict[str, float],
) -> np.ndarray:
    snr_db = float(snr_db_map.get(phase.label, 25.0))
    # baseline wander amplitude as a fraction of clean std
    clean_std = float(np.std(clean) + 1e-12)
    bw_amp = float(baseline_amp_frac_map.get(phase.label, 0.0)) * clean_std

    if phase.label == "Stable":
        w = rng.standard_normal(len(clean))
        noise = _scale_to_snr(clean, w, snr_db)
        return noise

    if phase.label == "Breathe":
        bw = _baseline_wander(rng, len(clean), fs, amp=bw_amp)
        w = _colored_noise(rng, len(clean), fs, lo=0.5, hi=40.0)
        w = _scale_to_snr(clean, w, snr_db)
        return bw + w

    if phase.label == "Walk":
        bw = _baseline_wander(rng, len(clean), fs, amp=bw_amp)
        mot = _motion_artifacts(rng, clean, fs, snr_db=snr_db)
        return bw + mot

    if phase.label == "Recovery":
        # decaying baseline + mild EMG
        bw = _baseline_wander(rng, len(clean), fs, amp=bw_amp)
        decay = np.linspace(1.0, 0.3, len(clean), dtype=np.float64)
        bw = bw * decay
        emg = _colored_noise(rng, len(clean), fs, lo=15.0, hi=60.0)
        emg = _scale_to_snr(clean, emg, snr_db)
        return bw + 0.5 * emg

    w = rng.standard_normal(len(clean))
    return _scale_to_snr(clean, w, snr_db)


def _write_txt(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in x:
            f.write(f"{float(v):.8f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="Input base signal (.npy or .txt). Recommended: denoised .npy")
    ap.add_argument("--source_fs", type=int, default=None, help="Sampling rate of the source signal (if different from --fs)")
    ap.add_argument("--fs", type=int, default=360, help="Sampling rate of the source/output")
    ap.add_argument("--out_dir", type=str, default="ajou_phase_augmented", help="Output directory")
    ap.add_argument("--n_users", type=int, default=5, help="How many synthetic users to generate")
    ap.add_argument("--date", type=str, default=_today_yymmdd(), help="YYMMDD string used in filenames")
    ap.add_argument("--seed", type=int, default=123, help="Base seed")

    # Noise policy knobs (simple defaults)
    ap.add_argument("--snr_stable", type=float, default=30.0)
    ap.add_argument("--snr_breathe", type=float, default=25.0)
    ap.add_argument("--snr_walk", type=float, default=15.0)
    ap.add_argument("--snr_recovery", type=float, default=22.0)

    ap.add_argument("--bw_breathe_frac", type=float, default=0.35, help="Baseline wander amplitude as fraction of clean std")
    ap.add_argument("--bw_walk_frac", type=float, default=0.60)
    ap.add_argument("--bw_recovery_frac", type=float, default=0.25)

    ap.add_argument("--also_session", action="store_true", help="Also write a concatenated 10-min session file per user")

    args = ap.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    if not src.exists():
        raise FileNotFoundError(src)

    base = _read_1d_signal(src)
    fs = int(args.fs)
    fs_src = int(args.source_fs) if args.source_fs is not None else fs
    if fs_src != fs:
        base = _resample_linear(base, fs_src=fs_src, fs_tgt=fs)

    snr_db_map = {
        "Stable": float(args.snr_stable),
        "Breathe": float(args.snr_breathe),
        "Walk": float(args.snr_walk),
        "Recovery": float(args.snr_recovery),
    }
    bw_amp_frac_map = {
        "Breathe": float(args.bw_breathe_frac),
        "Walk": float(args.bw_walk_frac),
        "Recovery": float(args.bw_recovery_frac),
    }

    meta = {
        "fs": fs,
        "source_fs": fs_src,
        "source": str(src),
        "date": args.date,
        "phases": [{"label": p.label, "duration_s": p.duration_s} for p in DEFAULT_PHASES],
        "snr_db": snr_db_map,
        "baseline_wander_amp_frac_of_std": bw_amp_frac_map,
        "seed": int(args.seed),
        "format": "single-column txt (one sample per line)",
    }

    total_s = sum(p.duration_s for p in DEFAULT_PHASES)
    total_n = int(total_s * fs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for user_idx in range(1, int(args.n_users) + 1):
        user = f"User{user_idx:02d}"
        rng = np.random.default_rng(int(args.seed) + user_idx * 10007)

        # tile base to cover full session
        tiled = _tile_to_length(base, total_n)

        cursor = 0
        session_parts: list[np.ndarray] = []

        for phase in DEFAULT_PHASES:
            n = int(phase.duration_s * fs)
            clean = tiled[cursor : cursor + n]
            cursor += n

            noise = _phase_noise(
                rng=rng,
                phase=phase,
                clean=clean,
                fs=fs,
                snr_db_map=snr_db_map,
                baseline_amp_frac_map=bw_amp_frac_map,
            )

            # mild per-user gain/offset
            gain = float(rng.uniform(0.95, 1.05))
            offset = float(rng.uniform(-0.03, 0.03) * (np.std(clean) + 1e-12))
            y = gain * clean + offset + noise

            fname = f"{user}_{phase.label}_{args.date}.txt"
            _write_txt(out_dir / fname, y)

            session_parts.append(y)

        if args.also_session:
            sess = np.concatenate(session_parts)
            _write_txt(out_dir / f"{user}_Session_{args.date}.txt", sess)

    print(f"✓ Wrote dataset to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
