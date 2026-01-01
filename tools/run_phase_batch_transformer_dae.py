#!/usr/bin/env python3
"""Batch evaluation for phase-augmented Ajou-style ECG txt files.

- Loads 267Hz single-column txt (Session if present, else concat phases)
- Resamples to 360Hz, trims initial seconds
- Runs pretrained Transformer_DAE (deepFilter.dl_models) on full signal via framing + overlap-add
- Computes label-free metrics + sliding-window distribution/worst-window

This intentionally imports existing project modules as requested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from math import gcd
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

# Ensure repo root is importable even when executed via wrappers.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Existing project modules (requested)
import h5py
import keras
import deepFilter.dl_models as dl_models
from deepFilter.dl_models import Transformer_DAE
from ajou_rpeak_utils import detect_rpeaks, evaluate_rpeak_performance

import tensorflow as tf


def _infer_transformer_dae_arch_from_weights(weights_path: str) -> Tuple[Optional[int], bool]:
    """Infer architecture knobs from a Keras 3 .weights.h5 file.

    Returns:
      (num_transformer_blocks or None, use_gated_noise)
    """
    if not weights_path.endswith(".h5"):
        return None, False
    try:
        with h5py.File(weights_path, "r") as f:
            if "layers" not in f:
                return None, False
            layer_keys = list(f["layers"].keys())
    except Exception:
        return None, False

    n_mha = sum(1 for k in layer_keys if k.startswith("multi_head_attention"))
    use_gated_noise = any(k.startswith("add_gated_noise") for k in layer_keys)
    return (n_mha if n_mha > 0 else None), use_gated_noise


def _manual_load_attention_qkvo_from_weights(model: keras.Model, weights_path: str, num_blocks: int) -> None:
    """Manually load q/k/v/out projections for attention blocks from a Keras3 `.weights.h5` file.

    This is a pragmatic fallback when `model.load_weights()` can't map nested attention sublayers.
    """

    def attn_name(i: int) -> str:
        return "multi_head_attention" if i == 0 else f"multi_head_attention_{i}"

    with h5py.File(weights_path, "r") as f:
        layers = f.get("layers")
        if layers is None:
            raise RuntimeError("weights file has no 'layers' group")

        for i in range(int(num_blocks)):
            name = attn_name(i)
            if name not in layers:
                raise RuntimeError(f"weights file missing attention group: {name}")
            g = layers[name]

            for sub, attr in [
                ("query_dense", "query_dense"),
                ("key_dense", "key_dense"),
                ("value_dense", "value_dense"),
                ("output_dense", "output_dense"),
            ]:
                if sub not in g or "vars" not in g[sub]:
                    raise RuntimeError(f"weights file missing {name}/{sub}/vars")

            qk = np.array(g["query_dense"]["vars"]["0"])
            qb = np.array(g["query_dense"]["vars"]["1"])
            kk = np.array(g["key_dense"]["vars"]["0"])
            kb = np.array(g["key_dense"]["vars"]["1"])
            vk = np.array(g["value_dense"]["vars"]["0"])
            vb = np.array(g["value_dense"]["vars"]["1"])
            ok = np.array(g["output_dense"]["vars"]["0"])
            ob = np.array(g["output_dense"]["vars"]["1"])

            layer = model.get_layer(name)
            layer.query_dense.set_weights([qk, qb])
            layer.key_dense.set_weights([kk, kb])
            layer.value_dense.set_weights([vk, vb])
            layer.output_dense.set_weights([ok, ob])


def _build_transformer_dae_compat(
    *,
    signal_size: int,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    num_transformer_blocks: int,
    dropout: float,
    use_gated_noise: bool,
):
    # Build with Keras 3 + a checkpoint-compatible custom attention block.
    input_shape = (int(signal_size), 1)
    inp = keras.layers.Input(shape=input_shape)

    ks = int(getattr(dl_models, "ks", 13))

    class AddGatedNoise(keras.layers.Layer):
        def call(self, x, training=None):
            if training:
                noise = tf.random.normal(tf.shape(x), stddev=0.01)
                return x + noise
            return x

    class TFPositionalEncoding1D(keras.layers.Layer):
        def call(self, inputs):
            # Stateless positional encoding (no variables) to match weight files.
            # Shape: (B, T, C)
            b = tf.shape(inputs)[0]
            t = tf.shape(inputs)[1]
            c = tf.shape(inputs)[2]
            pos = tf.cast(tf.range(t)[:, None], tf.float32)  # (T,1)
            i = tf.cast(tf.range(c)[None, :], tf.float32)  # (1,C)
            angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (i // 2.0)) / tf.cast(c, tf.float32))
            angles = pos * angle_rates
            sines = tf.sin(angles[:, 0::2])
            cosines = tf.cos(angles[:, 1::2])
            emb = tf.concat([sines, cosines], axis=-1)
            emb = emb[None, :, :]
            emb = emb[:, :, :c]
            return tf.repeat(emb, b, axis=0)

    class ExpandDimsLayer(keras.layers.Layer):
        def __init__(self, axis: int, **kwargs):
            super().__init__(**kwargs)
            self.axis = int(axis)

        def call(self, inputs):
            return tf.expand_dims(inputs, axis=self.axis)

    def conv1d_transpose(input_tensor, filters, kernel_size, strides=2, activation="relu", padding="same"):
        x = ExpandDimsLayer(axis=2)(input_tensor)
        x = keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding,
            activation=activation,
        )(x)
        x = keras.layers.Lambda(lambda t: tf.squeeze(t, axis=2))(x)
        return x

    class CompatMultiHeadAttention(keras.layers.Layer):
        def __init__(self, *, key_dim: int, num_heads: int, out_dim: int, dropout: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.key_dim = int(key_dim)
            self.num_heads = int(num_heads)
            self.out_dim = int(out_dim)
            self.dropout = float(dropout)
            self.query_dense = keras.layers.EinsumDense(
                equation="btd,dhk->bthk",
                output_shape=(None, self.num_heads, self.key_dim),
                bias_axes="hk",
                name="query_dense",
            )
            self.key_dense = keras.layers.EinsumDense(
                equation="btd,dhk->bthk",
                output_shape=(None, self.num_heads, self.key_dim),
                bias_axes="hk",
                name="key_dense",
            )
            self.value_dense = keras.layers.EinsumDense(
                equation="btd,dhk->bthk",
                output_shape=(None, self.num_heads, self.key_dim),
                bias_axes="hk",
                name="value_dense",
            )
            self.output_dense = keras.layers.EinsumDense(
                equation="bthk,hkd->btd",
                output_shape=(None, self.out_dim),
                bias_axes="d",
                name="output_dense",
            )
            self._softmax = keras.layers.Softmax(axis=-1, name="softmax")
            self._dropout_layer = keras.layers.Dropout(self.dropout, name="dropout_layer")

        def call(self, query, value, training=None):
            q = self.query_dense(query)
            k = self.key_dense(value)
            v = self.value_dense(value)
            scale = tf.math.rsqrt(tf.cast(self.key_dim, tf.float32))
            attn_scores = tf.einsum("bthk,bshk->bhts", q, k) * scale
            attn = self._softmax(attn_scores)
            attn = self._dropout_layer(attn, training=training)
            ctx = tf.einsum("bhts,bshk->bthk", attn, v)
            return self.output_dense(ctx)

        def build(self, input_shape):
            # input_shape may be (query_shape, value_shape)
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
                q_shape = input_shape[0]
            else:
                q_shape = input_shape
            if q_shape is None:
                super().build(input_shape)
                return
            self.query_dense.build(q_shape)
            self.key_dense.build(q_shape)
            self.value_dense.build(q_shape)
            # ctx shape: (B, T, H, K)
            self.output_dense.build((q_shape[0], q_shape[1], self.num_heads, self.key_dim))
            super().build(input_shape)

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0, block_idx: int = 0):
        x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        attn_name = "multi_head_attention" if block_idx == 0 else f"multi_head_attention_{block_idx}"
        x = CompatMultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            out_dim=int(inputs.shape[-1]),
            dropout=dropout,
            name=attn_name,
        )(x, x)
        x = keras.layers.Dropout(dropout)(x)
        res = x + inputs
        x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    x0 = keras.layers.Conv1D(
        filters=16,
        input_shape=(input_shape, 1),
        kernel_size=ks,
        activation="linear",
        strides=2,
        padding="same",
    )(inp)
    if use_gated_noise:
        x0 = AddGatedNoise()(x0)
    x0 = keras.layers.Activation("sigmoid")(x0)

    x0_ = keras.layers.Conv1D(
        filters=16,
        input_shape=(input_shape, 1),
        kernel_size=ks,
        activation=None,
        strides=2,
        padding="same",
    )(inp)
    xmul0 = keras.layers.Multiply()([x0, x0_])
    xmul0 = keras.layers.BatchNormalization()(xmul0)

    x1 = keras.layers.Conv1D(filters=32, kernel_size=ks, activation="linear", strides=2, padding="same")(xmul0)
    if use_gated_noise:
        x1 = AddGatedNoise()(x1)
    x1 = keras.layers.Activation("sigmoid")(x1)

    x1_ = keras.layers.Conv1D(filters=32, kernel_size=ks, activation=None, strides=2, padding="same")(xmul0)
    xmul1 = keras.layers.Multiply()([x1, x1_])
    xmul1 = keras.layers.BatchNormalization()(xmul1)

    x2 = keras.layers.Conv1D(filters=64, kernel_size=ks, activation="linear", strides=2, padding="same")(xmul1)
    if use_gated_noise:
        x2 = AddGatedNoise()(x2)
    x2 = keras.layers.Activation("sigmoid")(x2)

    x2_ = keras.layers.Conv1D(filters=64, kernel_size=ks, activation="elu", strides=2, padding="same")(xmul1)
    xmul2 = keras.layers.Multiply()([x2, x2_])
    xmul2 = keras.layers.BatchNormalization()(xmul2)

    position_embed = TFPositionalEncoding1D(name="tf_positional_encoding1d")
    x3 = xmul2 + position_embed(xmul2)
    for i in range(int(num_transformer_blocks)):
        x3 = transformer_encoder(x3, head_size, num_heads, ff_dim, dropout, block_idx=i)

    x4 = x3
    x5 = conv1d_transpose(x4, filters=64, kernel_size=ks, activation="elu", strides=1, padding="same")
    x5 = x5 + xmul2
    x5 = keras.layers.BatchNormalization()(x5)

    x6 = conv1d_transpose(x5, filters=32, kernel_size=ks, activation="elu", strides=2, padding="same")
    x6 = x6 + xmul1
    x6 = keras.layers.BatchNormalization()(x6)

    x7 = conv1d_transpose(x6, filters=16, kernel_size=ks, activation="elu", strides=2, padding="same")
    x7 = x7 + xmul0
    x8 = keras.layers.BatchNormalization()(x7)
    preds = conv1d_transpose(x8, filters=1, kernel_size=ks, activation="linear", strides=2, padding="same")

    return keras.Model(inputs=[inp], outputs=preds)


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    out_root: str
    date: str
    users: List[str]
    phases: List[str]
    fs_src: int = 267
    fs_tgt: int = 360
    trim_s: float = 1.0
    win: int = 512
    hop: int = 256
    batch_size: int = 128
    pretrained_weights: str = "0221_FIXED/Transformer_DAE_weights.best.weights.h5"
    window_sizes_s: Tuple[int, int] = (10, 30)
    step_s: int = 1
    tol_ms: float = 50.0
    qrs_window_ms: float = 300.0
    qrs_band: Tuple[float, float] = (5.0, 20.0)
    hf_band: Tuple[float, float] = (20.0, 90.0)


def _read_single_column_txt(path: str) -> np.ndarray:
    vals: List[float] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                # ignore malformed header lines if present
                continue
    x = np.asarray(vals, dtype=np.float32)
    if x.ndim != 1 or len(x) == 0:
        raise RuntimeError(f"Failed to read samples from: {path}")
    return x


def _load_user_session(cfg: RunConfig, user: str) -> Tuple[np.ndarray, Dict[str, str]]:
    # Prefer explicit session file
    sess = os.path.join(cfg.data_dir, f"{user}_Session_{cfg.date}.txt")
    meta: Dict[str, str] = {"user": user, "date": cfg.date}
    if os.path.exists(sess):
        meta["source"] = os.path.basename(sess)
        return _read_single_column_txt(sess), meta

    # Fallback: concat phases in order
    parts: List[np.ndarray] = []
    srcs: List[str] = []
    for p in cfg.phases:
        fp = os.path.join(cfg.data_dir, f"{user}_{p}_{cfg.date}.txt")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        parts.append(_read_single_column_txt(fp))
        srcs.append(os.path.basename(fp))

    meta["source"] = "+".join(srcs)
    return np.concatenate(parts, axis=0), meta


def _baseline_center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x - np.median(x)


def _resample(x: np.ndarray, fs_src: int, fs_tgt: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if fs_src == fs_tgt:
        return x
    g = gcd(fs_src, fs_tgt)
    up = fs_tgt // g
    down = fs_src // g
    return resample_poly(x, up=up, down=down).astype(np.float32)


def _frame_signal(x: np.ndarray, win: int, hop: int, pad_mode: str = "reflect") -> Tuple[np.ndarray, int]:
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


def _overlap_add(frames: np.ndarray, win: int, hop: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, L = frames.shape
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


def _bandpass(x: np.ndarray, fs: int, lo: float, hi: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    nyq = fs / 2.0
    lo = max(1e-6, float(lo))
    hi = min(float(hi), nyq - 1e-6)
    if not (0 < lo < hi < nyq):
        return x
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.nan_to_num(x, nan=float(np.nanmean(x)))).astype(np.float32)


def _robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return float(1.4826 * mad)


def _snr_rpeak_over_hf_noise(signal: np.ndarray, peaks: np.ndarray, fs: int, qrs_band=(5.0, 20.0), noise_band=(20.0, 90.0)) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(signal))]
    if len(peaks) == 0:
        return float("nan")
    qrs = _bandpass(signal, fs, qrs_band[0], qrs_band[1])
    hf = _bandpass(signal, fs, noise_band[0], noise_band[1])
    A = float(np.median(np.abs(qrs[peaks]))) + 1e-12
    N = _robust_std(hf) + 1e-12
    return float(20.0 * np.log10(A / N))


def _snr_rpeak_over_residual_noise(raw: np.ndarray, deno: np.ndarray, peaks_deno: np.ndarray, fs: int, qrs_band=(5.0, 20.0)) -> float:
    raw = np.asarray(raw, dtype=np.float32)
    deno = np.asarray(deno, dtype=np.float32)
    L = min(len(raw), len(deno))
    raw = raw[:L]
    deno = deno[:L]
    peaks = np.asarray(peaks_deno, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < L)]
    if len(peaks) == 0:
        return float("nan")
    deno_qrs = _bandpass(deno, fs, qrs_band[0], qrs_band[1])
    resid_qrs = _bandpass(raw - deno, fs, qrs_band[0], qrs_band[1])
    A = float(np.median(np.abs(deno_qrs[peaks]))) + 1e-12
    N = _robust_std(resid_qrs) + 1e-12
    return float(20.0 * np.log10(A / N))


def _compute_hrv_metrics(peaks: np.ndarray, fs: int) -> Dict[str, float]:
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return {"mean_rr": float("nan"), "sdnn": float("nan"), "rmssd": float("nan"), "pnn50": float("nan")}
    rr = np.diff(peaks) / float(fs)
    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0
    nn50 = int(np.sum(np.abs(diffs) > 0.05))
    pnn50 = float(nn50) / float(len(diffs)) if len(diffs) else 0.0
    return {"mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50}


def _qrs_amp_ratio_and_corr0(raw_seg: np.ndarray, den_seg: np.ndarray, rp_raw_seg: np.ndarray, rp_deno_seg: np.ndarray, fs: int, tol_ms: float, window_ms: float, qrs_band: Tuple[float, float]) -> Dict[str, float]:
    tol = int(round((tol_ms / 1000.0) * fs))
    rp_raw_seg = np.asarray(rp_raw_seg, dtype=int)
    rp_deno_seg = np.asarray(rp_deno_seg, dtype=int)
    if len(rp_raw_seg) == 0 or len(rp_deno_seg) == 0:
        return {"n_pairs": 0, "qrs_amp_ratio": float("nan"), "qrs_corr0": float("nan")}

    used = np.zeros(len(rp_deno_seg), dtype=bool)
    pairs: List[Tuple[int, int]] = []
    for pr in rp_raw_seg:
        d = np.abs(rp_deno_seg - pr)
        if used.any():
            d = d.copy()
            d[used] = 1_000_000_000
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            pairs.append((int(pr), int(rp_deno_seg[j])))

    if len(pairs) == 0:
        return {"n_pairs": 0, "qrs_amp_ratio": float("nan"), "qrs_corr0": float("nan")}

    half = int(round((window_ms / 1000.0) * fs / 2.0))
    raw_qrs = _bandpass(raw_seg, fs, qrs_band[0], qrs_band[1])
    den_qrs = _bandpass(den_seg, fs, qrs_band[0], qrs_band[1])

    amp_ratios: List[float] = []
    corr0s: List[float] = []
    for pr, _pd in pairs:
        if pr - half < 0 or pr + half > len(raw_seg):
            continue
        xr = raw_qrs[pr - half : pr + half]
        xd = den_qrs[pr - half : pr + half]  # center on raw time for both
        ar = float(np.max(np.abs(xr))) + 1e-12
        ad = float(np.max(np.abs(xd))) + 1e-12
        amp_ratios.append(ad / ar)
        xr2 = xr - float(np.mean(xr))
        xd2 = xd - float(np.mean(xd))
        sx = float(np.std(xr2)) + 1e-12
        sy = float(np.std(xd2)) + 1e-12
        corr0s.append(float(np.mean((xr2 / sx) * (xd2 / sy))))

    if len(amp_ratios) == 0:
        return {"n_pairs": 0, "qrs_amp_ratio": float("nan"), "qrs_corr0": float("nan")}

    return {
        "n_pairs": int(len(amp_ratios)),
        "qrs_amp_ratio": float(np.median(np.asarray(amp_ratios, dtype=np.float32))),
        "qrs_corr0": float(np.median(np.asarray(corr0s, dtype=np.float32))) if len(corr0s) else float("nan"),
    }


def _sliding_window_eval(raw: np.ndarray, deno: np.ndarray, rp_raw: np.ndarray, rp_deno: np.ndarray, cfg: RunConfig) -> Tuple[pd.DataFrame, Dict]:
    raw = np.asarray(raw, dtype=np.float32)
    deno = np.asarray(deno, dtype=np.float32)
    L = min(len(raw), len(deno))
    raw = raw[:L]
    deno = deno[:L]
    rp_raw = np.asarray(rp_raw, dtype=int)
    rp_deno = np.asarray(rp_deno, dtype=int)
    rp_raw = rp_raw[(rp_raw >= 0) & (rp_raw < L)]
    rp_deno = rp_deno[(rp_deno >= 0) & (rp_deno < L)]

    records: List[Dict] = []
    fs = int(cfg.fs_tgt)
    for W_s in cfg.window_sizes_s:
        W = int(round(W_s * fs))
        step = int(round(cfg.step_s * fs))
        for start in range(0, max(0, L - W + 1), step):
            end = start + W
            raw_seg = raw[start:end]
            den_seg = deno[start:end]
            rp_raw_w = rp_raw[(rp_raw >= start) & (rp_raw < end)] - start
            rp_den_w = rp_deno[(rp_deno >= start) & (rp_deno < end)] - start
            if len(rp_raw_w) == 0 or len(rp_den_w) == 0:
                continue

            snr_raw = _snr_rpeak_over_hf_noise(raw_seg, rp_raw_w, fs, qrs_band=cfg.qrs_band, noise_band=cfg.hf_band)
            snr_den = _snr_rpeak_over_hf_noise(den_seg, rp_den_w, fs, qrs_band=cfg.qrs_band, noise_band=cfg.hf_band)
            hf_gain = snr_den - snr_raw if (np.isfinite(snr_raw) and np.isfinite(snr_den)) else float("nan")

            morph = _qrs_amp_ratio_and_corr0(
                raw_seg,
                den_seg,
                rp_raw_w,
                rp_den_w,
                fs=fs,
                tol_ms=cfg.tol_ms,
                window_ms=cfg.qrs_window_ms,
                qrs_band=cfg.qrs_band,
            )

            # match rate as n_pairs/raw_peaks
            match_rate = float(morph["n_pairs"] / max(1, len(rp_raw_w)))

            records.append(
                {
                    "window_s": float(W_s),
                    "start_s": float(start / fs),
                    "end_s": float(end / fs),
                    "center_s": float((start + end) / 2.0 / fs),
                    "n_rpeaks_raw": int(len(rp_raw_w)),
                    "n_rpeaks_deno": int(len(rp_den_w)),
                    "n_pairs": int(morph["n_pairs"]),
                    "hf_snr_raw_db": float(snr_raw) if np.isfinite(snr_raw) else float("nan"),
                    "hf_snr_deno_db": float(snr_den) if np.isfinite(snr_den) else float("nan"),
                    "hf_gain_db": float(hf_gain) if np.isfinite(hf_gain) else float("nan"),
                    "match_rate": float(match_rate),
                    "qrs_amp_ratio": float(morph["qrs_amp_ratio"]) if np.isfinite(morph["qrs_amp_ratio"]) else float("nan"),
                    "qrs_corr0": float(morph["qrs_corr0"]) if np.isfinite(morph["qrs_corr0"]) else float("nan"),
                }
            )

    df = pd.DataFrame.from_records(records)
    if len(df) == 0:
        raise RuntimeError("No sliding-window records produced.")

    def summarize(sub: pd.DataFrame, metric: str) -> Dict[str, Optional[float]]:
        x = sub[metric].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {"median": None, "iqr": None, "min": None, "max": None}
        q1 = float(np.percentile(x, 25))
        q3 = float(np.percentile(x, 75))
        return {"median": float(np.median(x)), "iqr": float(q3 - q1), "min": float(np.min(x)), "max": float(np.max(x))}

    def worst(sub: pd.DataFrame, metric: str, mode: str = "min") -> Optional[Dict]:
        if metric not in sub.columns or len(sub) == 0:
            return None
        tmp = sub[np.isfinite(sub[metric])]
        if len(tmp) == 0:
            return None
        row = tmp.loc[tmp[metric].idxmin()] if mode == "min" else tmp.loc[tmp[metric].idxmax()]
        return {
            "metric": metric,
            "mode": mode,
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "center_s": float(row["center_s"]),
            "value": float(row[metric]),
            "n_rpeaks_raw": int(row.get("n_rpeaks_raw", 0)),
            "n_rpeaks_deno": int(row.get("n_rpeaks_deno", 0)),
            "n_pairs": int(row.get("n_pairs", 0)),
        }

    metrics = ["hf_gain_db", "match_rate", "qrs_amp_ratio", "qrs_corr0"]
    summary = {
        "signal": {"fs": int(cfg.fs_tgt), "n_samples": int(L), "duration_s": float(L / cfg.fs_tgt)},
        "window_sizes_s": list(cfg.window_sizes_s),
        "step_s": float(cfg.step_s),
        "metrics": {},
        "worst_windows": {},
    }

    for W_s in cfg.window_sizes_s:
        sub = df[df["window_s"] == float(W_s)].copy()
        summary["metrics"][str(W_s)] = {m: summarize(sub, m) for m in metrics}
        summary["worst_windows"][str(W_s)] = {
            "hf_gain_db_min": worst(sub, "hf_gain_db", mode="min"),
            "qrs_corr0_min": worst(sub, "qrs_corr0", mode="min"),
            "qrs_amp_ratio_min": worst(sub, "qrs_amp_ratio", mode="min"),
            "match_rate_min": worst(sub, "match_rate", mode="min"),
        }

    return df, summary


def _build_model_and_load(cfg: RunConfig):
    pretrained_path = cfg.pretrained_weights
    if not os.path.isabs(pretrained_path):
        pretrained_path = os.path.join(os.getcwd(), pretrained_path)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    # Keras 2.x cannot read Keras 3 `.weights.h5` files; fail early with a clear message.
    try:
        major = int(str(keras.__version__).split(".")[0])
    except Exception:
        major = 0
    if major and major < 3 and pretrained_path.endswith(".weights.h5"):
        raise RuntimeError(
            "This checkpoint is a Keras 3 `.weights.h5` file. "
            "Run this script with a Keras 3 environment (e.g. `conda run -p .../envs/ECGDENOISE`)."
        )

    custom_objects = {
        "TFPositionalEncoding1D": getattr(dl_models, "TFPositionalEncoding1D", None),
        "Conv1DTranspose": getattr(dl_models, "Conv1DTranspose", None),
        "Conv1DTranspose2": getattr(dl_models, "Conv1DTranspose2", None),
        "AddGatedNoise": getattr(dl_models, "AddGatedNoise", None),
    }
    custom_objects = {k: v for k, v in custom_objects.items() if v is not None}

    # 1) First try the project's canonical model definition.
    # This is the most likely to match the saved checkpoint naming.
    try:
        model = dl_models.Transformer_DAE(
            signal_size=cfg.win,
            head_size=64,
            num_heads=8,
            ff_dim=64,
            num_transformer_blocks=8,
            dropout=0,
        )
        model.load_weights(pretrained_path)
        return model
    except Exception:
        pass

    # 2) Fallback: attempt a compatibility build based on what we can infer.
    n_blocks, use_gated_noise = _infer_transformer_dae_arch_from_weights(pretrained_path)
    model = _build_transformer_dae_compat(
        signal_size=cfg.win,
        head_size=64,
        num_heads=8,
        ff_dim=64,
        num_transformer_blocks=int(n_blocks or 8),
        dropout=0.0,
        use_gated_noise=bool(use_gated_noise),
    )
    try:
        model.load_weights(pretrained_path)
    except Exception:
        # Load what we can, then manually patch attention q/k/v/out weights.
        model.load_weights(pretrained_path, skip_mismatch=True)
        _manual_load_attention_qkvo_from_weights(model, pretrained_path, int(n_blocks or 8))
    return model


def run_one_user(model, cfg: RunConfig, user: str) -> Dict:
    x_267, meta = _load_user_session(cfg, user)
    x_267 = _baseline_center(x_267)
    x_360 = _resample(x_267, cfg.fs_src, cfg.fs_tgt)

    trim_n = int(round(cfg.trim_s * cfg.fs_tgt))
    if trim_n > 0 and len(x_360) > trim_n:
        x_360 = x_360[trim_n:]

    frames, _padded_len = _frame_signal(x_360, cfg.win, cfg.hop)
    mu = np.mean(frames, axis=1, keepdims=True).astype(np.float32)
    sigma = (np.std(frames, axis=1, keepdims=True) + 1e-6).astype(np.float32)
    X = ((frames - mu) / sigma)[..., None]

    Y = model.predict(X, batch_size=cfg.batch_size, verbose=0)
    Y = np.squeeze(Y, axis=2).astype(np.float32)
    Y = Y * sigma + mu
    deno_full = _overlap_add(Y, cfg.win, cfg.hop)

    min_len = min(len(x_360), len(deno_full))
    raw = x_360[:min_len].astype(np.float32)
    deno = deno_full[:min_len].astype(np.float32)

    rp_raw = detect_rpeaks(
        raw,
        cfg.fs_tgt,
        trim_start_s=0.0,
        prefer_neurokit=True,
        prefer_positive=True,
        assume_stable_hr=True,
        min_rr_s=0.30,
    )
    rp_deno = detect_rpeaks(
        deno,
        cfg.fs_tgt,
        trim_start_s=0.0,
        prefer_neurokit=True,
        prefer_positive=True,
        assume_stable_hr=True,
        min_rr_s=0.30,
    )

    snr_hf_raw_db = _snr_rpeak_over_hf_noise(raw, rp_raw, cfg.fs_tgt, qrs_band=cfg.qrs_band, noise_band=cfg.hf_band)
    snr_hf_deno_db = _snr_rpeak_over_hf_noise(deno, rp_deno, cfg.fs_tgt, qrs_band=cfg.qrs_band, noise_band=cfg.hf_band)
    snr_hf_gain_db = snr_hf_deno_db - snr_hf_raw_db if (np.isfinite(snr_hf_raw_db) and np.isfinite(snr_hf_deno_db)) else float("nan")

    snr_residual_db = _snr_rpeak_over_residual_noise(raw, deno, rp_deno, cfg.fs_tgt, qrs_band=cfg.qrs_band)

    hrv_raw = _compute_hrv_metrics(rp_raw, cfg.fs_tgt)
    hrv_deno = _compute_hrv_metrics(rp_deno, cfg.fs_tgt)

    # Stability metrics from existing module
    stab = evaluate_rpeak_performance(rp_raw, rp_deno, cfg.fs_tgt, tol_ms=cfg.tol_ms)

    # Sliding-window
    df_sw, summary_sw = _sliding_window_eval(raw, deno, rp_raw, rp_deno, cfg)

    out_dir = os.path.join(cfg.out_root, f"ajou_outputs_phase_{user}_{cfg.date}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "raw_360.npy"), raw)
    np.save(os.path.join(out_dir, "deno_360.npy"), deno)

    eval_out = {
        "meta": meta,
        "n_samples": int(len(raw)),
        "fs": int(cfg.fs_tgt),
        "duration_s": float(len(raw) / cfg.fs_tgt),
        "n_rpeaks_raw": int(len(rp_raw)),
        "n_rpeaks_deno": int(len(rp_deno)),
        "snr_hf_raw_db": float(snr_hf_raw_db) if np.isfinite(snr_hf_raw_db) else None,
        "snr_hf_deno_db": float(snr_hf_deno_db) if np.isfinite(snr_hf_deno_db) else None,
        "snr_hf_gain_db": float(snr_hf_gain_db) if np.isfinite(snr_hf_gain_db) else None,
        "snr_residual_db": float(snr_residual_db) if np.isfinite(snr_residual_db) else None,
        "hrv_raw": hrv_raw,
        "hrv_deno": hrv_deno,
        "rpeak_stability_raw_vs_deno": stab,
    }

    with open(os.path.join(out_dir, "evaluation_label_free_v2.json"), "w") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)

    sw_dir = os.path.join(out_dir, "sliding_window_eval")
    os.makedirs(sw_dir, exist_ok=True)
    csv_path = os.path.join(sw_dir, f"sliding_eval_windows_{'_'.join(map(str, cfg.window_sizes_s))}s_step{cfg.step_s}s.csv")
    json_path = os.path.join(sw_dir, f"sliding_eval_windows_{'_'.join(map(str, cfg.window_sizes_s))}s_step{cfg.step_s}s.json")
    summary_path = os.path.join(sw_dir, f"sliding_eval_summary_{'_'.join(map(str, cfg.window_sizes_s))}s_step{cfg.step_s}s.json")

    df_sw.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df_sw.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    with open(summary_path, "w") as f:
        json.dump(summary_sw, f, indent=2, ensure_ascii=False)

    # Return a row for global summary
    row = {
        "user": user,
        "date": cfg.date,
        "out_dir": out_dir,
        "n_rpeaks_raw": eval_out["n_rpeaks_raw"],
        "n_rpeaks_deno": eval_out["n_rpeaks_deno"],
        "snr_hf_gain_db": eval_out["snr_hf_gain_db"],
        "snr_residual_db": eval_out["snr_residual_db"],
        "rpeak_f1": float(stab.get("f1", 0.0)),
        "rpeak_precision": float(stab.get("precision", 0.0)),
        "rpeak_recall": float(stab.get("recall", 0.0)),
    }

    # Add worst-window match_rate (10s/30s)
    for W_s in cfg.window_sizes_s:
        ww = summary_sw.get("worst_windows", {}).get(str(W_s), {}).get("match_rate_min", None)
        row[f"worst_match_rate_{W_s}s"] = ww.get("value") if isinstance(ww, dict) else None
        row[f"worst_match_center_s_{W_s}s"] = ww.get("center_s") if isinstance(ww, dict) else None

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="ajou_phase_augmented_267")
    p.add_argument("--out_root", default=".")
    p.add_argument("--date", default="260101")
    p.add_argument(
        "--users",
        nargs="*",
        default=None,
        help="Users to run. Accepts space-separated tokens (e.g., --users User02 User03) or a single comma-separated string (e.g., --users User02,User03).",
    )
    p.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Phases to run. Accepts space-separated tokens or a single comma-separated string.",
    )
    p.add_argument("--pretrained_weights", default="0221_FIXED/Transformer_DAE_weights.best.weights.h5")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    default_users = ["User01", "User02", "User03", "User04", "User05"]
    default_phases = ["Stable", "Breathe", "Walk", "Recovery"]

    def _parse_list_arg(value: Optional[List[str]], default: List[str]) -> List[str]:
        if not value:
            return list(default)
        if len(value) == 1 and "," in value[0]:
            return [x.strip() for x in value[0].split(",") if x.strip()]
        return [str(x).strip() for x in value if str(x).strip()]

    cfg = RunConfig(
        data_dir=str(args.data_dir),
        out_root=str(args.out_root),
        date=str(args.date),
        users=_parse_list_arg(args.users, default_users),
        phases=_parse_list_arg(args.phases, default_phases),
        pretrained_weights=str(args.pretrained_weights),
    )

    # Normalize paths
    cfg = RunConfig(
        **{
            **cfg.__dict__,
            "data_dir": os.path.abspath(cfg.data_dir),
            "out_root": os.path.abspath(cfg.out_root),
        }
    )

    os.makedirs(cfg.out_root, exist_ok=True)

    model = _build_model_and_load(cfg)

    rows: List[Dict] = []
    for user in cfg.users:
        print(f"=== {user} ({cfg.date}) ===")
        row = run_one_user(model, cfg, user)
        rows.append(row)
        print("  done:", row["out_dir"])

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.out_root, f"phase_batch_summary_{cfg.date}.csv")
    if os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            if "user" in prev.columns:
                prev["user"] = prev["user"].astype(str)
            if "date" in prev.columns:
                prev["date"] = prev["date"].astype(str)
            combined = pd.concat([prev, df], ignore_index=True)
            if {"user", "date"}.issubset(set(combined.columns)):
                combined["user"] = combined["user"].astype(str)
                combined["date"] = combined["date"].astype(str)
                combined = combined.drop_duplicates(subset=["user", "date"], keep="last")
            df = combined
        except Exception:
            # If prior file is malformed, fall back to overwriting.
            pass

    # Stable sort for readability (User01..User05, then date)
    if "user" in df.columns:
        def _user_sort_key(v: object) -> int:
            m = re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else 10**9

        df = df.assign(_user_sort=df["user"].map(_user_sort_key)).sort_values(
            by=["_user_sort", "user", "date"] if "date" in df.columns else ["_user_sort", "user"],
            kind="mergesort",
        )
        df = df.drop(columns=["_user_sort"], errors="ignore")

    df.to_csv(out_csv, index=False)
    print("✓ Summary saved:", out_csv)


if __name__ == "__main__":
    main()
