#!/usr/bin/env python3
"""
Standalone runner for Dual_FreqDAE denoising model.

Features
- Loads Dual_FreqDAE from deepFilter.dl_models
- Loads pretrained weights (defaults to 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5)
- Accepts .npy input array(s): shape (512,), (N,512), or (N,512,1)
- Automatically builds the frequency-branch input from time-domain using FFT magnitude (duplicated to 512)
- Saves denoised output as .npy (same batch shape)

Usage examples
  python tools/run_dual_freqdae.py --input path/to/signals.npy --output out/denoised.npy \
    --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5

Quick demo (synthetic input)
  python tools/run_dual_freqdae.py --demo
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TF INFO logs

# Ensure repository root is on PYTHONPATH for direct script invocation
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from deepFilter.dl_models import Dual_FreqDAE  # noqa: E402
except Exception as e:
    print("[ERROR] Failed to import Dual_FreqDAE from deepFilter.dl_models.\n"
          "Hint: run from repository root or check that deepFilter/__init__.py exists.\n"
          f"Exception: {e}")
    sys.exit(1)


SIG_LEN = 512


def make_fourier_magnitude(inputs: np.ndarray, n: int, fs: int) -> np.ndarray:
    """
    Build frequency-domain input as magnitude(FFT) duplicated to length n to match time-domain shape.

    inputs: 2D array (batch, n)
    returns: 2D array (batch, n)
    """
    from scipy.fft import fft

    assert inputs.ndim == 2 and inputs.shape[1] == n, f"Expected (batch,{n}) got {inputs.shape}"
    T = n / fs
    k = np.arange(n)
    freq = k / T
    _ = freq[: n // 2]  # not used, kept for parity with reference code

    out = []
    for i in range(inputs.shape[0]):
        y = inputs[i, :]
        Y = fft(y) / n
        mag = np.abs(Y[: n // 2])
        # duplicate to full length n
        Y_full = np.hstack([mag, mag])
        out.append(Y_full)
    return np.asarray(out)


def _segment_1d(arr_1d: np.ndarray, win: int, hop: int, pad: bool) -> np.ndarray:
    """Segment a 1D array into (N, win) windows with hop. If pad, zero-pad the tail."""
    L = arr_1d.shape[0]
    if pad and (L < win or (L - win) % hop != 0):
        # pad to cover last incomplete window
        rem = (L - win) % hop
        pad_len = 0 if L >= win and rem == 0 else (hop - rem) % hop
        pad_len = max(pad_len, max(0, win - L))
        arr_1d = np.pad(arr_1d, (0, pad_len), mode='constant')
        L = arr_1d.shape[0]
    if L < win:
        raise ValueError(f"Input length {L} is shorter than window {win}; use --pad to allow padding.")
    idxs = list(range(0, L - win + 1, hop))
    windows = np.stack([arr_1d[i:i+win] for i in idxs], axis=0)
    return windows.astype(np.float32)


def load_inputs(path: str | None, demo: bool, batch: int, *, segment: bool, hop: int, pad: bool) -> np.ndarray:
    """Load or generate inputs as float32 array shape (N, 512). Supports 1D segmentation if --segment."""
    if demo:
        # generate smooth synthetic ECG-like beats (sine+cubic noise)
        t = np.linspace(0, 1, SIG_LEN, endpoint=False)
        signals = []
        rng = np.random.default_rng(42)
        for _ in range(batch):
            base = 0.7 * np.sin(2 * np.pi * 5 * t) + 0.15 * np.sin(2 * np.pi * 50 * t)
            noise = 0.05 * rng.standard_normal(SIG_LEN)
            signals.append(base + noise)
        X = np.asarray(signals, dtype=np.float32)
        return X

    if not path:
        raise ValueError("--input path is required unless --demo is set")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    if p.is_file():
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix} (only .npy supported)")
    else:
        # If a directory is passed, stack all .npy files sorted
        files = sorted([f for f in p.glob("*.npy")])
        if not files:
            raise ValueError(f"No .npy files found in directory: {p}")
        arrays = [np.load(f) for f in files]
        arr = np.stack(arrays, axis=0) if arrays[0].ndim == 1 else np.concatenate(arrays, axis=0)

    # Normalize shapes to (N,512) or segment 1D arrays if requested
    if arr.ndim == 1:
        if segment:
            arr = _segment_1d(arr.astype(np.float32, copy=False), SIG_LEN, hop, pad)
        else:
            # treat as a single beat if length is exactly 512
            arr = arr[None, :]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected input shape: {arr.shape}, expected (N,512) or (N,512,1)")
    if arr.shape[1] != SIG_LEN:
        raise ValueError(f"Signals must have length {SIG_LEN}, got {arr.shape[1]}")
    return arr.astype(np.float32, copy=False)


def build_model(weights_path: str | None, *, fusion: str = "concat"):
    import tensorflow as tf
    from keras import losses

    model = Dual_FreqDAE(fusion=fusion)
    # compile with same metrics (loss not used for inference but keeps parity)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[losses.mean_squared_error, losses.mean_absolute_error],
    )

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model.load_weights(weights_path)
        print(f"✓ Loaded weights: {weights_path}")
    else:
        print("! No weights provided — using randomly initialized model (for debugging only)")
    return model


def main():
    parser = argparse.ArgumentParser(description="Run Dual_FreqDAE inference")
    parser.add_argument("--input", type=str, default=None, help=".npy file or directory of .npy files")
    parser.add_argument("--output", type=str, default="denoised.npy", help="Path to save denoised output .npy")
    parser.add_argument("--weights", type=str, default=str(Path("0221_FIXED")/"Dual_FreqDAE_weights.best.weights.h5"), help="Weights path (.h5)")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "concat_mlp", "cross_attn"], help="Fusion mode (must match weights)")
    parser.add_argument("--fs", type=int, default=360, help="Sampling rate used for Fourier features (default 360)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo inputs instead of loading files")
    parser.add_argument("--demo-batch", type=int, default=4, help="Demo: number of synthetic samples")
    parser.add_argument("--segment", action="store_true", help="If input is 1D, segment into 512-length windows")
    parser.add_argument("--hop", type=int, default=512, help="Hop size for segmentation when --segment is used")
    parser.add_argument("--pad", action="store_true", help="Zero-pad tail to complete the last segment when --segment is used")

    args = parser.parse_args()

    # 1) Load or generate inputs (N,512)
    X = load_inputs(args.input, args.demo, args.demo_batch, segment=args.segment, hop=args.hop, pad=args.pad)
    print(f"[INFO] Loaded inputs: {X.shape}")

    # 2) Build frequency-branch inputs (N,512)
    F = make_fourier_magnitude(X, n=SIG_LEN, fs=args.fs)
    print(f"[INFO] Built frequency inputs: {F.shape}")

    # 3) Expand dims to (N,512,1)
    X_in = X[..., None]
    F_in = F[..., None]

    # 4) Build model and load weights
    model = build_model(args.weights, fusion=args.fusion)

    # 5) Run inference
    y_hat = model.predict([X_in, F_in], batch_size=args.batch_size, verbose=1)
    # squeeze trailing channel if present
    if y_hat.ndim == 3 and y_hat.shape[-1] == 1:
        y_out = y_hat[..., 0]
    else:
        y_out = y_hat
    print(f"[INFO] Denoised output: {y_out.shape}")

    # 6) Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, y_out.astype(np.float32))
    print(f"✓ Saved denoised signals to {out_path}")


if __name__ == "__main__":
    main()
