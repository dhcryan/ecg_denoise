import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

import tensorflow as tf
from tensorflow import keras

# Ensure repo root is on sys.path when running as a script (sys.path[0] becomes tools/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepFilter.dl_models import (
    AttentionSkipDAE,
    CNN_DAE,
    DRRN_denoising,
    Dual_FreqDAE,
    FCN_DAE,
    Transformer_DAE,
    deep_filter_model_I_LANL_dilated,
)
from Data_Preparation.data_preparation_with_fourier import make_fourier


TARGET_FS = 360
SRC_FS = 500
SEQ_SEC = 10
SEQ_LEN = TARGET_FS * SEQ_SEC  # 3600
WIN = 512
HOP = 256
SAMPLES_PER_MV = 200.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float64)
    return float(np.sqrt(np.mean(d * d) + 1e-12))


def prd(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    d = (a - b).astype(np.float64)
    denom = np.sum(a64 * a64) + 1e-12
    return float(100.0 * np.sqrt(np.sum(d * d) / denom))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = (np.linalg.norm(a64) * np.linalg.norm(b64)) + 1e-12
    return float(np.dot(a64, b64) / denom)


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    clean64 = clean.astype(np.float64)
    err = (clean - est).astype(np.float64)
    p_clean = np.mean(clean64 * clean64) + 1e-12
    p_err = np.mean(err * err) + 1e-12
    return float(10.0 * np.log10(p_clean / p_err))


def _parse_simemg_name(path: Path) -> Dict[str, object]:
    # Examples: P10_1_Ag-AgCl.mat, P10_1_ORB.mat, P10_1_lead I.mat
    name = path.name
    out: Dict[str, object] = {"subject": None, "session": None, "modality": None, "path": str(path)}
    if not name.startswith("P") or not name.endswith(".mat"):
        return out
    try:
        stem = name[:-4]
        parts = stem.split("_")
        subj = int(parts[0][1:])
        sess = int(parts[1])
        modality = "_".join(parts[2:])
        out.update({"subject": subj, "session": sess, "modality": modality})
    except Exception:
        pass
    return out


def load_simemg_two_channel(path: Path) -> np.ndarray:
    d = loadmat(path)
    arrays: List[Tuple[str, np.ndarray]] = []
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            arrays.append((k, v))
    if not arrays:
        raise ValueError(f"No 2D array found in {path}")

    key, a = None, None
    for k, v in arrays:
        if v.shape[1] == 2 or v.shape[0] == 2:
            key, a = k, v
            break
    if a is None:
        key, a = max(arrays, key=lambda kv: kv[1].size)

    a = np.asarray(a)
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if a.shape[1] != 2:
        raise ValueError(f"Expected 2 channels; got {a.shape} from key={key}")
    return a.astype(np.float32)


def infer_clean_noisy(two_ch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x0 = two_ch[:, 0]
    x1 = two_ch[:, 1]
    hf0 = float(np.mean(np.square(np.diff(x0))))
    hf1 = float(np.mean(np.square(np.diff(x1))))
    if hf0 >= hf1:
        noisy, clean = x0, x1
        noisy_idx = 0
    else:
        noisy, clean = x1, x0
        noisy_idx = 1
    return clean.astype(np.float32), noisy.astype(np.float32), {"hf0": hf0, "hf1": hf1, "noisy_idx": float(noisy_idx)}


def resample_to_target(x: np.ndarray) -> np.ndarray:
    # 500 -> 360 exactly via 18/25
    return resample_poly(x, up=18, down=25).astype(np.float32)


def make_sequences(simemg_dir: Path, max_files: Optional[int] = None, max_seqs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    paths = sorted(simemg_dir.glob("*.mat"))
    if max_files is not None:
        paths = paths[: int(max_files)]

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for p in paths:
        base = _parse_simemg_name(p)
        try:
            two = load_simemg_two_channel(p)
            clean_1d, noisy_1d, info = infer_clean_noisy(two)

            # convert to mV (SimEMG: 200 samples per mV)
            clean_1d = (clean_1d / SAMPLES_PER_MV).astype(np.float32)
            noisy_1d = (noisy_1d / SAMPLES_PER_MV).astype(np.float32)

            # resample to 360 Hz
            clean_rs = resample_to_target(clean_1d)
            noisy_rs = resample_to_target(noisy_1d)
            L = min(clean_rs.shape[0], noisy_rs.shape[0])
            clean_rs = clean_rs[:L]
            noisy_rs = noisy_rs[:L]

            n_seq = L // SEQ_LEN
            for si in range(n_seq):
                s = si * SEQ_LEN
                e = s + SEQ_LEN
                X_list.append(noisy_rs[s:e].reshape(SEQ_LEN, 1))
                Y_list.append(clean_rs[s:e].reshape(SEQ_LEN, 1))
                meta_rows.append({
                    **base,
                    "seq_idx": si,
                    "seq_sec": SEQ_SEC,
                    "noisy_idx": int(info["noisy_idx"]),
                })
                if max_seqs is not None and len(X_list) >= int(max_seqs):
                    break
            if max_seqs is not None and len(X_list) >= int(max_seqs):
                break
        except Exception as e:
            meta_rows.append({**base, "error": repr(e)})

    if not X_list:
        raise RuntimeError("No sequences built (check SimEMG dir / parsing)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    # keep only rows that correspond to built sequences
    meta = meta.iloc[: X.shape[0]].reset_index(drop=True)
    return X, Y, meta


def _hann(win: int) -> np.ndarray:
    if win <= 1:
        return np.ones((win,), dtype=np.float32)
    return np.hanning(win).astype(np.float32)


def _baseline_offset_per_window(windows: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "mean":
        return windows.mean(axis=1, keepdims=True)
    if mode in ("endpoints", "endpoint", "ends"):
        return ((windows[:, :1] + windows[:, -1:]) / 2.0).astype(np.float32, copy=False)
    if mode in ("none", "no", "raw"):
        return np.zeros((windows.shape[0], 1), dtype=np.float32)
    raise ValueError("baseline_mode must be one of: 'mean', 'endpoints', 'none'")


def _frame_1d(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    if x.shape[0] < win:
        return np.zeros((0, win), dtype=np.float32)
    n = 1 + (x.shape[0] - win) // hop
    out = np.empty((n, win), dtype=np.float32)
    for i in range(n):
        s = i * hop
        out[i] = x[s : s + win]
    return out


def make_fourier_batch(x_batch: np.ndarray, fs: int = TARGET_FS, *, fourier_source: str = "centered") -> np.ndarray:
    x = np.asarray(x_batch)
    if x.ndim != 3 or x.shape[1] != WIN:
        raise ValueError(f"Expected (N,{WIN},1), got {x.shape}")
    out = np.zeros((x.shape[0], WIN), dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = make_fourier(x[i, :, 0].reshape(1, -1), WIN, fs).reshape(-1)
    return np.expand_dims(out, axis=2)


def denoise_one_sequence_sliding(
    model: keras.Model,
    needs_fourier: bool,
    x_seq: np.ndarray,
    *,
    win: int = WIN,
    hop: int = HOP,
    batch_size: int = 64,
    baseline_mode: str = "mean",
    fourier_source: str = "centered",
) -> np.ndarray:
    x = np.asarray(x_seq).reshape(-1).astype(np.float32, copy=False)
    L = len(x)
    if L < win:
        x = np.pad(x, (0, win - L), mode="constant")
        L = len(x)

    w = _hann(win)
    windows = _frame_1d(x, win=win, hop=hop)  # (Nw, win)
    if windows.shape[0] == 0:
        return np.zeros((L,), dtype=np.float32)

    offsets = _baseline_offset_per_window(windows, baseline_mode)
    x_in = (windows - offsets).astype(np.float32)
    x_in_3d = x_in[:, :, None]

    if needs_fourier:
        if fourier_source == "raw":
            f_in = make_fourier_batch(windows[:, :, None], fs=TARGET_FS)
        else:
            f_in = make_fourier_batch(x_in_3d, fs=TARGET_FS)
        y_hat = model.predict([x_in_3d, f_in], batch_size=batch_size, verbose=0)
    else:
        y_hat = model.predict(x_in_3d, batch_size=batch_size, verbose=0)

    y_hat = np.asarray(y_hat).squeeze(-1)  # (Nw, win)
    if y_hat.shape != windows.shape:
        raise RuntimeError(f"Prediction shape mismatch: {y_hat.shape} vs {windows.shape}")

    # add baseline back
    y_hat = (y_hat + offsets).astype(np.float32)

    # overlap-add
    y = np.zeros((L,), dtype=np.float32)
    ws = np.zeros((L,), dtype=np.float32)
    for i in range(y_hat.shape[0]):
        s = i * hop
        e = s + win
        y[s:e] += y_hat[i] * w
        ws[s:e] += w
    y = y / (ws + 1e-8)
    return y[: len(x_seq)].astype(np.float32)


def evaluate_model_on_sequences(
    model: keras.Model,
    needs_fourier: bool,
    X_seq: np.ndarray,
    Y_seq: np.ndarray,
    *,
    win: int = WIN,
    hop: int = HOP,
    batch_size: int = 64,
    baseline_mode: str = "mean",
    fourier_source: str = "centered",
) -> pd.DataFrame:
    rows = []

    for i in range(X_seq.shape[0]):
        noisy = X_seq[i, :, 0]
        clean = Y_seq[i, :, 0]
        den = denoise_one_sequence_sliding(
            model,
            needs_fourier,
            noisy,
            win=win,
            hop=hop,
            batch_size=batch_size,
            baseline_mode=baseline_mode,
            fourier_source=fourier_source,
        )
        rows.append({
            "rmse_noisy": rmse(clean, noisy),
            "rmse_denoised": rmse(clean, den),
            "prd_noisy": prd(clean, noisy),
            "prd_denoised": prd(clean, den),
            "cos_noisy": cos_sim(clean, noisy),
            "cos_denoised": cos_sim(clean, den),
            "snr_noisy_db": snr_db(clean, noisy),
            "snr_denoised_db": snr_db(clean, den),
        })

    return pd.DataFrame(rows)


def _looks_like_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "resource_exhausted" in msg
        or "cuda_error_out_of_memory" in msg
        or "out of memory" in msg
        or "oom" in msg
    )


def _try_enable_gpu_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Must be set before GPUs are initialized; ignore if too late.
                pass
    except Exception:
        pass


def build_model_and_load_weights(model_key: str, weights_path: Path) -> Tuple[keras.Model, bool]:
    model_key = model_key.strip()
    needs_fourier = (model_key == "Dual_FreqDAE")

    name = weights_path.name.lower()
    suffix = weights_path.suffix.lower()

    # Prefer explicit Keras v3 SavedModel
    if suffix == ".keras":
        model = keras.models.load_model(str(weights_path), compile=False)
        return model, needs_fourier

    # Some "keras3" h5 may still be weights-only; try load_model, fall back.
    if suffix in {".h5", ".hdf5"} and ("keras3" in name) and (not name.endswith(".weights.h5")):
        try:
            model = keras.models.load_model(str(weights_path), compile=False)
            return model, needs_fourier
        except ValueError as e:
            if "no model config found" not in str(e).lower():
                raise

    if model_key == "CNN_DAE":
        model = CNN_DAE(signal_size=WIN)
        needs_fourier = False
    elif model_key == "FCN_DAE":
        model = FCN_DAE()
        needs_fourier = False
    elif model_key == "DRNN":
        model = DRRN_denoising()
        needs_fourier = False
    elif model_key == "DeepFilter":
        model = deep_filter_model_I_LANL_dilated()
        needs_fourier = False
    elif model_key == "AttentionSkipDAE":
        model = AttentionSkipDAE(signal_size=WIN)
        needs_fourier = False
    elif model_key == "Transformer_DAE":
        model = Transformer_DAE(signal_size=WIN)
        needs_fourier = False
    elif model_key == "Dual_FreqDAE":
        model = Dual_FreqDAE(signal_size=WIN)
        needs_fourier = True
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # build variables
    if needs_fourier:
        dummy_x = np.zeros((1, WIN, 1), dtype=np.float32)
        dummy_f = np.zeros((1, WIN, 1), dtype=np.float32)
        _ = model([dummy_x, dummy_f], training=False)
    else:
        dummy_x = np.zeros((1, WIN, 1), dtype=np.float32)
        _ = model(dummy_x, training=False)

    model.load_weights(str(weights_path))
    return model, needs_fourier


def default_model_paths(weights_dir: Path) -> List[Tuple[str, Path]]:
    # Only include models that exist in most dirs
    candidates = [
        ("DeepFilter", weights_dir / "DeepFilter_weights.best.weights.h5"),
        ("Transformer_DAE", weights_dir / "Transformer_DAE_weights.best.weights.h5"),
        ("Dual_FreqDAE", weights_dir / "Dual_FreqDAE_weights.best.weights.h5"),
    ]
    return candidates


def run_one_setting(
    *,
    tag: str,
    weights_dir: Path,
    simemg_dir: Path,
    out_dir: Path,
    max_files: Optional[int],
    max_seqs: Optional[int],
    models: Optional[List[str]],
    batch_size: int,
) -> None:
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"weights_dir not found: {weights_dir} (did you mean something like '0221_FIXED'?)"
        )
    if not weights_dir.is_dir():
        raise NotADirectoryError(f"weights_dir is not a directory: {weights_dir}")

    X_seq, Y_seq, meta = make_sequences(simemg_dir, max_files=max_files, max_seqs=max_seqs)
    print(f"[INFO] Built sequences: X={X_seq.shape} Y={Y_seq.shape}")

    model_list = default_model_paths(weights_dir)
    if models is not None:
        models_set = set(models)
        model_list = [(n, p) for (n, p) in model_list if n in models_set]

    missing_weights = [(n, p) for (n, p) in model_list if not p.exists()]
    if len(missing_weights) == len(model_list):
        expected = "\n".join([f"- {n}: {p}" for (n, p) in model_list])
        raise FileNotFoundError(
            "No expected weight files found under weights_dir. Expected one or more of:\n"
            + expected
        )

    all_rows = []
    n_fail = 0
    n_skip = 0

    for model_name, wpath in model_list:
        if not wpath.exists():
            print(f"[SKIP] {tag}:{model_name} missing {wpath}")
            n_skip += 1
            continue

        print(f"[RUN] {tag}:{model_name} -> {wpath.name}")
        attempt_bs = int(batch_size)
        while True:
            try:
                model, needs_fourier = build_model_and_load_weights(model_name, wpath)
                mdf = evaluate_model_on_sequences(
                    model,
                    needs_fourier,
                    X_seq,
                    Y_seq,
                    win=WIN,
                    hop=HOP,
                    batch_size=attempt_bs,
                    baseline_mode="mean",
                    fourier_source="centered",
                )
                break
            except Exception as e:
                if _looks_like_oom(e) and attempt_bs > 1:
                    next_bs = max(1, attempt_bs // 2)
                    print(
                        f"[OOM] {tag}:{model_name} batch_size={attempt_bs} -> retry with batch_size={next_bs}"
                    )
                    attempt_bs = next_bs
                    keras.backend.clear_session()
                    gc.collect()
                    continue
                print(f"[FAIL] {tag}:{model_name}: {type(e).__name__}: {e}")
                n_fail += 1
                keras.backend.clear_session()
                gc.collect()
                mdf = None
                break

        if mdf is None:
            continue

        df = pd.concat([meta.reset_index(drop=True), mdf.reset_index(drop=True)], axis=1)
        df["model"] = model_name
        df["train_setting"] = tag
        all_rows.append(df)

        keras.backend.clear_session()
        gc.collect()

    if not all_rows:
        raise RuntimeError(
            f"No results produced for tag={tag} (skipped={n_skip}, failed={n_fail}). "
            "Check weights_dir path and that TensorFlow/Keras can load the weights."
        )

    results = pd.concat(all_rows, axis=0, ignore_index=True)

    metric_cols = [
        "rmse_noisy",
        "rmse_denoised",
        "prd_noisy",
        "prd_denoised",
        "cos_noisy",
        "cos_denoised",
        "snr_noisy_db",
        "snr_denoised_db",
    ]

    overall_mean = results.groupby("model")[metric_cols].mean().add_suffix("_mean")
    overall_std = results.groupby("model")[metric_cols].std().add_suffix("_std")
    overall_wide = pd.concat([overall_mean, overall_std], axis=1).reset_index()

    by_mod_mean = results.groupby(["model", "modality"])[metric_cols].mean().add_suffix("_mean")
    by_mod_std = results.groupby(["model", "modality"])[metric_cols].std().add_suffix("_std")
    by_mod_wide = pd.concat([by_mod_mean, by_mod_std], axis=1).reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_path = out_dir / f"simemg_{tag}_10s_sequence_metrics.csv"
    overall_path = out_dir / f"simemg_{tag}_10s_overall_summary_wide.csv"
    bymod_path = out_dir / f"simemg_{tag}_10s_by_modality_summary_wide.csv"

    results.to_csv(seq_path, index=False)
    overall_wide.to_csv(overall_path, index=False)
    by_mod_wide.to_csv(bymod_path, index=False)

    print("[WROTE]", seq_path)
    print("[WROTE]", overall_path)
    print("[WROTE]", bymod_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simemg-dir", type=Path, default=Path("data") / "SimEMG")
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation_results_simemg"))
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--models", type=str, default=None, help="Comma-separated subset: DeepFilter,Transformer_DAE,Dual_FreqDAE")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only execution (disables visible GPUs). Useful if CUDA OOM persists even at batch_size=1.",
    )
    ap.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        help="Disable TF GPU memory growth (by default, we try to enable it to reduce OOMs)",
    )
    args = ap.parse_args()

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("[INFO] Forced CPU-only execution (GPUs hidden)")
        except Exception as e:
            print(f"[WARN] Failed to force CPU-only execution: {type(e).__name__}: {e}")

    if not args.no_gpu_memory_growth:
        _try_enable_gpu_memory_growth()
    print("TF:", tf.__version__)
    run_one_setting(
        tag=args.tag,
        weights_dir=args.weights_dir,
        simemg_dir=args.simemg_dir,
        out_dir=args.out_dir,
        max_files=args.max_files,
        max_seqs=args.max_seqs,
        models=models,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
