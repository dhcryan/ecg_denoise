#!/usr/bin/env python3
"""Create wet-vs-dry label-free comparison summary + report-ready plots.

Outputs:
- wet_vs_dry_summary.csv  (long format; includes wet phases + dry All)
- figs/*.png (boxplots + heatmaps)

Design:
- Use the SAME denoiser (Transformer_DAE) on both datasets to compare upstream signal quality
  and downstream stability under the same processing.
- Wet dataset (ajou_real) is phase-split -> highlights scenario differences.
- Dry datasets (ajou_data, ajou_data2) are session-only -> treated as phase='All'.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make repo root + tools dir importable (tools/ is not a package).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import realdata_eval_utils as U  # noqa: E402

# Reuse model + metric utilities from the transformer batch runner.
import run_phase_batch_transformer_dae as base  # noqa: E402


@dataclass
class Args:
    wet_root: str
    dry_roots: List[str]
    out_dir: str
    weights: str
    lead_index: int
    fs_src: int
    fs_tgt: int
    trim_s: float
    win: int
    hop: int
    batch_size: int


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--wet_root", default="ajou_real")
    p.add_argument("--dry_roots", nargs="*", default=["ajou_data", "ajou_data2"])
    p.add_argument("--out_dir", default="reports/wet_vs_dry_260208")
    p.add_argument("--weights", default="0221_FIXED/Transformer_DAE_weights.best.weights.h5")
    p.add_argument("--lead_index", type=int, default=2)
    p.add_argument("--fs_src", type=int, default=267)
    p.add_argument("--fs_tgt", type=int, default=360)
    p.add_argument("--trim_s", type=float, default=1.0)
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    a = p.parse_args()
    return Args(
        wet_root=str(a.wet_root),
        dry_roots=[str(x) for x in (a.dry_roots or [])],
        out_dir=str(a.out_dir),
        weights=str(a.weights),
        lead_index=int(a.lead_index),
        fs_src=int(a.fs_src),
        fs_tgt=int(a.fs_tgt),
        trim_s=float(a.trim_s),
        win=int(a.win),
        hop=int(a.hop),
        batch_size=int(a.batch_size),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_box(df: pd.DataFrame, *, x: str, y: str, hue: str | None, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_heatmap(piv: pd.DataFrame, title: str, out_path: str, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(max(6, 0.6 * piv.shape[1]), max(4, 0.35 * piv.shape[0])))
    sns.heatmap(piv, annot=False, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()

    wet_root = os.path.abspath(args.wet_root)
    dry_roots = [os.path.abspath(x) for x in args.dry_roots]
    out_dir = os.path.abspath(args.out_dir)
    figs_dir = os.path.join(out_dir, "figs")
    _ensure_dir(figs_dir)

    # Build Transformer model once
    cfg_for_model = base.RunConfig(
        data_dir=wet_root,
        out_root=out_dir,
        date="000000",
        users=["_"],
        phases=list(U.PHASE_ORDER),
        fs_src=args.fs_src,
        fs_tgt=args.fs_tgt,
        trim_s=args.trim_s,
        win=args.win,
        hop=args.hop,
        batch_size=args.batch_size,
        pretrained_weights=args.weights,
    )
    cfg_for_model = base.RunConfig(
        **{**cfg_for_model.__dict__, "data_dir": os.path.abspath(cfg_for_model.data_dir), "out_root": os.path.abspath(cfg_for_model.out_root)}
    )

    print("[INFO] Building Transformer_DAE + loading weights...")
    model = base._build_model_and_load(cfg_for_model)

    rows: List[Dict] = []

    # ---- Wet (phase-wise) ----
    cases = U.parse_wet_cases(wet_root)
    if not cases:
        raise SystemExit(f"No wet cases found under {wet_root}")

    for case in cases:
        phase_signals_267, phase_order = U.load_wet_case_signals_267(case, lead_index_1based=args.lead_index)

        if phase_order == ["Session"]:
            # Session-only wet case: treat as All
            x = U.baseline_center(phase_signals_267["Session"])
            x_360 = U.resample(x, fs_src=args.fs_src, fs_tgt=args.fs_tgt)
            trim_n = int(round(args.trim_s * args.fs_tgt))
            if trim_n > 0 and len(x_360) > trim_n:
                x_360 = x_360[trim_n:]
            raw_360 = x_360
            den_360 = U.denoise_full_signal(model, raw_360, win=args.win, hop=args.hop, batch_size=args.batch_size)

            m = U.compute_overall_metrics(
                raw_360=raw_360,
                deno_360=den_360,
                fs=args.fs_tgt,
                tol_ms=cfg_for_model.tol_ms,
                qrs_band=cfg_for_model.qrs_band,
                hf_band=cfg_for_model.hf_band,
                snr_hf_fn=base._snr_rpeak_over_hf_noise,
                snr_residual_fn=base._snr_rpeak_over_residual_noise,
                sliding_eval_fn=base._sliding_window_eval,
                window_sizes_s=cfg_for_model.window_sizes_s,
                step_s=cfg_for_model.step_s,
                qrs_window_ms=cfg_for_model.qrs_window_ms,
            )
            rows.append(
                {
                    "dataset": "wet",
                    "subject": case.subject,
                    "date": case.date,
                    "phase": "All",
                    "model": "Transformer_DAE",
                    "duration_s": float(len(raw_360) / args.fs_tgt),
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
        den_360 = U.denoise_full_signal(model, raw_360, win=args.win, hop=args.hop, batch_size=args.batch_size)

        # phase-wise metrics
        for ph, s, e in bounds:
            xraw = raw_360[s:e]
            xden = den_360[s:e]
            m = U.compute_overall_metrics(
                raw_360=xraw,
                deno_360=xden,
                fs=args.fs_tgt,
                tol_ms=cfg_for_model.tol_ms,
                qrs_band=cfg_for_model.qrs_band,
                hf_band=cfg_for_model.hf_band,
                snr_hf_fn=base._snr_rpeak_over_hf_noise,
                snr_residual_fn=base._snr_rpeak_over_residual_noise,
                sliding_eval_fn=base._sliding_window_eval,
                window_sizes_s=cfg_for_model.window_sizes_s,
                step_s=cfg_for_model.step_s,
                qrs_window_ms=cfg_for_model.qrs_window_ms,
            )
            rows.append(
                {
                    "dataset": "wet",
                    "subject": case.subject,
                    "date": case.date,
                    "phase": ph,
                    "model": "Transformer_DAE",
                    "duration_s": float((e - s) / args.fs_tgt),
                    **m,
                }
            )

        # all
        m_all = U.compute_overall_metrics(
            raw_360=raw_360,
            deno_360=den_360,
            fs=args.fs_tgt,
            tol_ms=cfg_for_model.tol_ms,
            qrs_band=cfg_for_model.qrs_band,
            hf_band=cfg_for_model.hf_band,
            snr_hf_fn=base._snr_rpeak_over_hf_noise,
            snr_residual_fn=base._snr_rpeak_over_residual_noise,
            sliding_eval_fn=base._sliding_window_eval,
            window_sizes_s=cfg_for_model.window_sizes_s,
            step_s=cfg_for_model.step_s,
            qrs_window_ms=cfg_for_model.qrs_window_ms,
        )
        rows.append(
            {
                "dataset": "wet",
                "subject": case.subject,
                "date": case.date,
                "phase": "All",
                "model": "Transformer_DAE",
                "duration_s": float(len(raw_360) / args.fs_tgt),
                **m_all,
            }
        )

    # ---- Dry (session-only) ----
    for dry_root in dry_roots:
        if not os.path.isdir(dry_root):
            continue
        for fn in sorted(os.listdir(dry_root)):
            if not fn.lower().endswith(".txt"):
                continue
            fpath = os.path.join(dry_root, fn)
            arr = U.read_multicol_txt(fpath)
            sig_267 = U.select_lead_column(arr, lead_index_1based=args.lead_index)
            raw_360 = U.resample(U.baseline_center(sig_267), fs_src=args.fs_src, fs_tgt=args.fs_tgt)
            trim_n = int(round(args.trim_s * args.fs_tgt))
            if trim_n > 0 and len(raw_360) > trim_n:
                raw_360 = raw_360[trim_n:]
            den_360 = U.denoise_full_signal(model, raw_360, win=args.win, hop=args.hop, batch_size=args.batch_size)

            m = U.compute_overall_metrics(
                raw_360=raw_360,
                deno_360=den_360,
                fs=args.fs_tgt,
                tol_ms=cfg_for_model.tol_ms,
                qrs_band=cfg_for_model.qrs_band,
                hf_band=cfg_for_model.hf_band,
                snr_hf_fn=base._snr_rpeak_over_hf_noise,
                snr_residual_fn=base._snr_rpeak_over_residual_noise,
                sliding_eval_fn=base._sliding_window_eval,
                window_sizes_s=cfg_for_model.window_sizes_s,
                step_s=cfg_for_model.step_s,
                qrs_window_ms=cfg_for_model.qrs_window_ms,
            )

            rows.append(
                {
                    "dataset": "dry",
                    "subject": os.path.basename(dry_root),
                    "date": "",
                    "phase": "All",
                    "model": "Transformer_DAE",
                    "file": fn,
                    "duration_s": float(len(raw_360) / args.fs_tgt),
                    **m,
                }
            )

    df = pd.DataFrame(rows)

    # Create ASCII-safe subject ids for plotting (avoid Korean glyph issues)
    keys = sorted({(r.get("dataset"), r.get("subject"), r.get("date")) for r in rows}, key=lambda x: (str(x[0]), str(x[1]), str(x[2])))
    sid_map = {k: f"S{idx:03d}" for idx, k in enumerate(keys, start=1)}
    df["subject_id"] = [sid_map.get((d, s, dt), "S???") for d, s, dt in zip(df["dataset"], df["subject"], df["date"])]
    out_csv = os.path.join(out_dir, "wet_vs_dry_summary.csv")
    df.to_csv(out_csv, index=False)
    print("✓ Saved:", out_csv)

    # ---- Figures ----
    # All-only wet vs dry
    df_all = df[df["phase"] == "All"].copy()
    if not df_all.empty:
        _plot_box(
            df_all,
            x="dataset",
            y="snr_hf_gain_db",
            hue=None,
            title="HF gain (dB) — wet vs dry (All)",
            out_path=os.path.join(figs_dir, "box_hf_gain_wet_vs_dry_all.png"),
        )
        _plot_box(
            df_all,
            x="dataset",
            y="rpeak_f1",
            hue=None,
            title="R-peak stability (F1) — wet vs dry (All)",
            out_path=os.path.join(figs_dir, "box_rpeak_f1_wet_vs_dry_all.png"),
        )

    # Wet-only phase boxes
    df_wet_phase = df[(df["dataset"] == "wet") & (df["phase"] != "All")].copy()
    if not df_wet_phase.empty:
        phase_order = [p for p in U.PHASE_ORDER if p in set(df_wet_phase["phase"])]
        df_wet_phase["phase"] = pd.Categorical(df_wet_phase["phase"], categories=phase_order, ordered=True)
        _plot_box(
            df_wet_phase,
            x="phase",
            y="snr_hf_gain_db",
            hue=None,
            title="HF gain (dB) by phase — wet electrode",
            out_path=os.path.join(figs_dir, "box_hf_gain_by_phase_wet.png"),
        )
        _plot_box(
            df_wet_phase,
            x="phase",
            y="rpeak_f1",
            hue=None,
            title="R-peak stability (F1) by phase — wet electrode",
            out_path=os.path.join(figs_dir, "box_rpeak_f1_by_phase_wet.png"),
        )

        # Heatmaps (subject x phase)
        piv_hf = df_wet_phase.pivot_table(index="subject_id", columns="phase", values="snr_hf_gain_db", aggfunc="median")
        piv_f1 = df_wet_phase.pivot_table(index="subject_id", columns="phase", values="rpeak_f1", aggfunc="median")
        _plot_heatmap(piv_hf, "HF gain (median) — subject × phase (wet)", os.path.join(figs_dir, "heat_hf_gain_subject_phase_wet.png"))
        _plot_heatmap(piv_f1, "R-peak F1 (median) — subject × phase (wet)", os.path.join(figs_dir, "heat_rpeak_f1_subject_phase_wet.png"), vmin=0.0, vmax=1.0)

    print("✓ Figures saved under:", figs_dir)


if __name__ == "__main__":
    main()
