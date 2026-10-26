#!/usr/bin/env python3
"""Model comparison on wet electrode dataset (ajou_real).

Outputs:
- model_compare_wet.csv (long format: subject/date/phase/model -> metrics)
- figs/*.png (boxplots + heatmaps)

Models (default): Transformer_DAE, CNN_DAE, DeepFilter, FCN_DAE
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Make repo root + tools dir importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import realdata_eval_utils as U  # noqa: E402
import run_phase_batch_transformer_dae as base  # noqa: E402

import deepFilter.dl_models as dl_models  # noqa: E402


PLOT_LABEL_ALIAS = {
    "Transformer_DAE": "Proposed",
}


@dataclass
class Args:
    wet_root: str
    out_dir: str
    weights_dir: str
    lead_index: int
    models: List[str]
    fs_src: int
    fs_tgt: int
    trim_s: float
    win: int
    hop: int
    batch_size: int
    plot_seconds: float
    max_plot_cases: int


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--wet_root", default="ajou_real")
    p.add_argument("--out_dir", default="reports/model_compare_wet_260208")
    p.add_argument("--weights_dir", default="0221_FIXED")
    p.add_argument("--lead_index", type=int, default=2)
    p.add_argument(
        "--models",
        type=str,
        default="Transformer_DAE,DeepFilter,CNN_DAE,FCN_DAE,AttentionSkipDAE",
        help="Comma-separated model keys",
    )
    p.add_argument("--fs_src", type=int, default=267)
    p.add_argument("--fs_tgt", type=int, default=360)
    p.add_argument("--trim_s", type=float, default=1.0)
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--plot_seconds", type=float, default=10.0, help="Seconds to plot per subject×phase")
    p.add_argument(
        "--max_plot_cases",
        type=int,
        default=0,
        help="Limit plotted subjects (0 = all). Useful to reduce image count.",
    )
    a = p.parse_args()
    models = [m.strip() for m in str(a.models).split(",") if m.strip()]
    return Args(
        wet_root=str(a.wet_root),
        out_dir=str(a.out_dir),
        weights_dir=str(a.weights_dir),
        lead_index=int(a.lead_index),
        models=models,
        fs_src=int(a.fs_src),
        fs_tgt=int(a.fs_tgt),
        trim_s=float(a.trim_s),
        win=int(a.win),
        hop=int(a.hop),
        batch_size=int(a.batch_size),
        plot_seconds=float(a.plot_seconds),
        max_plot_cases=int(a.max_plot_cases),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _weights_for_model(weights_dir: str, model_key: str) -> str:
    fname = f"{model_key}_weights.best.weights.h5"
    cand = os.path.join(weights_dir, fname)
    if os.path.exists(cand):
        return cand

    # Transformer has a few variants in some dirs; prefer standard name
    if model_key == "Transformer_DAE":
        for alt in [
            "Transformer_DAE_weights.best.weights.h5",
            "Transformer_DAE_weights.best.keras3.weights.h5",
            "Transformer_DAE_weights.best.keras3.h5",
        ]:
            p = os.path.join(weights_dir, alt)
            if os.path.exists(p):
                return p

    raise FileNotFoundError(f"No weights found for {model_key} under {weights_dir}")


def _build_and_load_model(model_key: str, weights_path: str):
    # Transformer uses the robust compatibility loader from base runner.
    if model_key == "Transformer_DAE":
        cfg = base.RunConfig(
            data_dir=".",
            out_root=".",
            date="000000",
            users=["_"],
            phases=list(U.PHASE_ORDER),
            pretrained_weights=weights_path,
        )
        cfg = base.RunConfig(
            **{**cfg.__dict__, "data_dir": os.path.abspath(cfg.data_dir), "out_root": os.path.abspath(cfg.out_root)}
        )
        return base._build_model_and_load(cfg)

    if model_key == "CNN_DAE":
        model = dl_models.CNN_DAE(signal_size=512)
    elif model_key == "DRNN":
        model = dl_models.DRRN_denoising()
    elif model_key == "FCN_DAE":
        model = dl_models.FCN_DAE()
    elif model_key == "DeepFilter":
        model = dl_models.deep_filter_model_I_LANL_dilated()
    elif model_key == "AttentionSkipDAE":
        model = dl_models.AttentionSkipDAE(signal_size=512)
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    # Build vars
    import numpy as np

    dummy = np.zeros((1, 512, 1), dtype=np.float32)
    _ = model(dummy, training=False)
    model.load_weights(weights_path)
    return model


def _sanitize_filename(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _make_subject_id_map(cases: List[U.WetCase]) -> Dict[tuple, str]:
    keys = sorted({(c.subject, c.date) for c in cases}, key=lambda x: (str(x[0]), str(x[1])))
    return {k: f"S{idx:03d}" for idx, k in enumerate(keys, start=1)}


def _plot_waveforms_multi_model(
    *,
    raw: "np.ndarray",
    deno_by_model: Dict[str, "np.ndarray"],
    fs: int,
    seconds: float,
    title: str,
    out_path: str,
) -> None:
    import numpy as np

    n = int(min(len(raw), max(1, round(float(seconds) * fs))))
    t = np.arange(n, dtype=float) / float(fs)
    nrows = 1 + len(deno_by_model)
    plt.figure(figsize=(14, max(3.0, 1.7 * nrows)))

    ax = plt.subplot(nrows, 1, 1)
    ax.plot(t, raw[:n], lw=0.9, color="black")
    ax.set_title(title + " — RAW")
    ax.set_xlabel("s")
    ax.set_ylabel("mV (a.u.)")
    ax.grid(True, alpha=0.25)

    for i, (mk, y) in enumerate(deno_by_model.items(), start=2):
        ax = plt.subplot(nrows, 1, i)
        ax.plot(t, y[:n], lw=0.9)
        ax.set_title(PLOT_LABEL_ALIAS.get(mk, mk))
        ax.set_xlabel("s")
        ax.set_ylabel("mV (a.u.)")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_box(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str],
    title: str,
    out_path: str,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    plt.figure(figsize=(12, 4))
    ax = sns.boxplot(data=df, x=x, y=y, hue=hue)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: str, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(max(6, 0.7 * piv.shape[1]), max(4, 0.35 * piv.shape[0])))
    sns.heatmap(piv, annot=False, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main() -> None:
    args = parse_args()

    wet_root = os.path.abspath(args.wet_root)
    out_dir = os.path.abspath(args.out_dir)
    figs_dir = os.path.join(out_dir, "figs")
    sig_dir = os.path.join(figs_dir, "signals_subject_phase")
    _ensure_dir(figs_dir)
    _ensure_dir(sig_dir)

    cases = U.parse_wet_cases(wet_root)
    if not cases:
        raise SystemExit(f"No wet cases found under {wet_root}")

    sid_map = _make_subject_id_map(cases)

    weights_dir = os.path.abspath(args.weights_dir)

    # Build/cache models
    model_cache: Dict[str, object] = {}
    for mk in args.models:
        w = _weights_for_model(weights_dir, mk)
        print(f"[INFO] Loading {mk} from {os.path.basename(w)}")
        model_cache[mk] = _build_and_load_model(mk, w)

    rows: List[Dict] = []

    # Common metric config
    fs = args.fs_tgt
    tol_ms = 50.0
    qrs_band = (5.0, 20.0)
    hf_band = (20.0, 90.0)
    window_sizes_s = (10, 30)
    step_s = 1
    qrs_window_ms = 300.0

    for case_i, case in enumerate(cases):
        subject_id = sid_map.get((case.subject, case.date), "S???")
        phase_signals_267, phase_order = U.load_wet_case_signals_267(case, lead_index_1based=args.lead_index)

        if phase_order == ["Session"]:
            # Session-only wet case: treat as All
            x = U.baseline_center(phase_signals_267["Session"])
            x_360 = U.resample(x, fs_src=args.fs_src, fs_tgt=args.fs_tgt)
            trim_n = int(round(args.trim_s * args.fs_tgt))
            if trim_n > 0 and len(x_360) > trim_n:
                x_360 = x_360[trim_n:]
            raw_360 = x_360

            for mk, model in model_cache.items():
                den_360 = U.denoise_full_signal(model, raw_360, win=args.win, hop=args.hop, batch_size=args.batch_size)
                m = U.compute_overall_metrics(
                    raw_360=raw_360,
                    deno_360=den_360,
                    fs=fs,
                    tol_ms=tol_ms,
                    qrs_band=qrs_band,
                    hf_band=hf_band,
                    snr_hf_fn=base._snr_rpeak_over_hf_noise,
                    snr_residual_fn=base._snr_rpeak_over_residual_noise,
                    sliding_eval_fn=base._sliding_window_eval,
                    window_sizes_s=window_sizes_s,
                    step_s=step_s,
                    qrs_window_ms=qrs_window_ms,
                )
                rows.append(
                    {
                        "subject": case.subject,
                        "subject_id": subject_id,
                        "date": case.date,
                        "phase": "All",
                        "model": mk,
                        "duration_s": float(len(raw_360) / fs),
                        **m,
                    }
                )
            continue

        raw_360, bounds = U.build_concat_360_and_bounds(
            phase_signals_267,
            phase_order,
            fs_src=args.fs_src,
            fs_tgt=args.fs_tgt,
            trim_s=args.trim_s,
        )

        # Optionally create per-subject/phase waveform comparisons.
        do_plot_case = (args.max_plot_cases <= 0) or (case_i < args.max_plot_cases)

        deno_full_by_model: Dict[str, "np.ndarray"] = {}

        for mk, model in model_cache.items():
            den_360 = U.denoise_full_signal(model, raw_360, win=args.win, hop=args.hop, batch_size=args.batch_size)
            deno_full_by_model[mk] = den_360

            for ph, s, e in bounds:
                xraw = raw_360[s:e]
                xden = den_360[s:e]
                m = U.compute_overall_metrics(
                    raw_360=xraw,
                    deno_360=xden,
                    fs=fs,
                    tol_ms=tol_ms,
                    qrs_band=qrs_band,
                    hf_band=hf_band,
                    snr_hf_fn=base._snr_rpeak_over_hf_noise,
                    snr_residual_fn=base._snr_rpeak_over_residual_noise,
                    sliding_eval_fn=base._sliding_window_eval,
                    window_sizes_s=window_sizes_s,
                    step_s=step_s,
                    qrs_window_ms=qrs_window_ms,
                )
                rows.append(
                    {
                        "subject": case.subject,
                        "subject_id": subject_id,
                        "date": case.date,
                        "phase": ph,
                        "model": mk,
                        "duration_s": float((e - s) / fs),
                        **m,
                    }
                )

            m_all = U.compute_overall_metrics(
                raw_360=raw_360,
                deno_360=den_360,
                fs=fs,
                tol_ms=tol_ms,
                qrs_band=qrs_band,
                hf_band=hf_band,
                snr_hf_fn=base._snr_rpeak_over_hf_noise,
                snr_residual_fn=base._snr_rpeak_over_residual_noise,
                sliding_eval_fn=base._sliding_window_eval,
                window_sizes_s=window_sizes_s,
                step_s=step_s,
                qrs_window_ms=qrs_window_ms,
            )
            rows.append(
                {
                    "subject": case.subject,
                    "subject_id": subject_id,
                    "date": case.date,
                    "phase": "All",
                    "model": mk,
                    "duration_s": float(len(raw_360) / fs),
                    **m_all,
                }
            )

        if do_plot_case:
            for ph, s, e in bounds:
                xraw = raw_360[s:e]
                deno_by_model = {mk: y[s:e] for mk, y in deno_full_by_model.items()}
                fname = f"{_sanitize_filename(subject_id)}_{_sanitize_filename(case.date)}_{ph}.png"
                out_path = os.path.join(sig_dir, fname)
                _plot_waveforms_multi_model(
                    raw=xraw,
                    deno_by_model=deno_by_model,
                    fs=fs,
                    seconds=args.plot_seconds,
                    title=f"{subject_id} {case.date} {ph}",
                    out_path=out_path,
                )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "model_compare_wet.csv")
    df.to_csv(out_csv, index=False)
    print("✓ Saved:", out_csv)

    # ---- Figures ----
    # Focus on scenario differences: exclude All for phase plots
    df_phase = df[df["phase"] != "All"].copy()
    if not df_phase.empty:
        df_phase["model_plot"] = df_phase["model"].map(PLOT_LABEL_ALIAS).fillna(df_phase["model"])
        phase_order = [p for p in U.PHASE_ORDER if p in set(df_phase["phase"])]
        df_phase["phase"] = pd.Categorical(df_phase["phase"], categories=phase_order, ordered=True)

        _plot_box(
            df_phase,
            x="phase",
            y="snr_hf_gain_db",
            hue="model_plot",
            title="HF gain (dB) by phase — model comparison (wet)",
            out_path=os.path.join(figs_dir, "box_hf_gain_by_phase_model.png"),
        )
        _plot_box(
            df_phase,
            x="phase",
            y="rpeak_f1",
            hue="model_plot",
            title="R-peak stability (F1) by phase — model comparison (wet)",
            out_path=os.path.join(figs_dir, "box_rpeak_f1_by_phase_model.png"),
        )

        # Zoomed view (helps distinguish near-1.0 performance)
        _plot_box(
            df_phase,
            x="phase",
            y="rpeak_f1",
            hue="model_plot",
            title="R-peak stability (F1) by phase — zoomed (wet)",
            out_path=os.path.join(figs_dir, "box_rpeak_f1_by_phase_model_zoom.png"),
            ylim=(0.8, 1.01),
        )

        # Heatmap: model x phase (median across subjects)
        piv_hf = df_phase.pivot_table(index="model_plot", columns="phase", values="snr_hf_gain_db", aggfunc="median")
        piv_f1 = df_phase.pivot_table(index="model_plot", columns="phase", values="rpeak_f1", aggfunc="median")
        _plot_heatmap(piv_hf, "HF gain (median) — model × phase (wet)", os.path.join(figs_dir, "heat_hf_gain_model_phase.png"))
        _plot_heatmap(piv_f1, "R-peak F1 (median) — model × phase (wet)", os.path.join(figs_dir, "heat_rpeak_f1_model_phase.png"), vmin=0.0, vmax=1.0)

    print("✓ Figures saved under:", figs_dir)
    print("✓ Signal waveforms saved under:", sig_dir)


if __name__ == "__main__":
    main()
