#!/usr/bin/env python3
"""Render shareable batch summary images from batch_overlay_summary_*.csv.

- Masks user identifiers to ASCII (S001, S002, ...) by default.
- Writes:
  - <out_dir>/batch_overlay_summary_masked.csv
  - <out_dir>/batch_overlay_summary_masked.md
  - <out_dir>/batch_summary.png
  - <out_dir>/f1_by_phase_boxplot.png

This is intended for making figures that don't require Korean fonts and don't
expose personally identifying subject names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _mask_users(values: list[str], prefix: str = "S") -> dict[str, str]:
    uniq = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    width = max(3, len(str(len(uniq))))
    return {u: f"{prefix}{i+1:0{width}d}" for i, u in enumerate(uniq)}


def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


@dataclass
class SummaryStats:
    data_dir: str
    users_processed: int
    ok_rows: int
    missing_rows: int
    error_rows: int


def _compute_stats(df: pd.DataFrame, data_dir: str) -> SummaryStats:
    users_processed = df["user"].dropna().astype(str).nunique()
    ok_rows = int((df["status"] == "ok").sum())
    missing_rows = int((df["status"] == "missing_file").sum())
    error_rows = int((df["status"] == "error").sum())
    return SummaryStats(
        data_dir=data_dir,
        users_processed=users_processed,
        ok_rows=ok_rows,
        missing_rows=missing_rows,
        error_rows=error_rows,
    )


def _phase_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    ok["f1"] = ok["f1"].map(_safe_float)
    g = ok.groupby("phase", dropna=False)["f1"]
    out = pd.DataFrame(
        {
            "count": g.count(),
            "mean": g.mean(),
            "median": g.median(),
        }
    ).reset_index()
    out = out.sort_values("phase")
    return out


def _missing_table(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    miss = df[df["status"] == "missing_file"].copy()
    if miss.empty:
        return pd.DataFrame(columns=["user", "date", "phase"])
    cols = [c for c in ["user", "date", "phase"] if c in miss.columns]
    miss = miss[cols].head(limit)
    return miss


def _write_masked_outputs(df_masked: pd.DataFrame, out_dir: Path, data_dir: str) -> None:
    out_csv = out_dir / "batch_overlay_summary_masked.csv"
    out_md = out_dir / "batch_overlay_summary_masked.md"

    stats = _compute_stats(df_masked, data_dir=data_dir)
    phase_tbl = _phase_table(df_masked)
    miss_tbl = _missing_table(df_masked)

    df_masked.to_csv(out_csv, index=False)

    md_lines = []
    md_lines.append("# Batch Overlay Summary (masked)\n")
    md_lines.append(f"- data_dir: `{data_dir}`")
    md_lines.append(f"- users processed: **{stats.users_processed}**")
    md_lines.append(
        f"- ok rows: **{stats.ok_rows}** / missing: **{stats.missing_rows}** / error: **{stats.error_rows}**\n"
    )

    md_lines.append("## Phase-wise F1 (raw vs deno R-peak match)\n")
    md_lines.append(phase_tbl.to_markdown(index=False))
    md_lines.append("\n## Missing files (first 50)\n")
    if miss_tbl.empty:
        md_lines.append("(none)")
    else:
        md_lines.append(miss_tbl.to_markdown(index=False))

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _render_images(df_masked: pd.DataFrame, out_dir: Path, data_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = _compute_stats(df_masked, data_dir=data_dir)
    phase_tbl = _phase_table(df_masked)
    miss_tbl = _missing_table(df_masked, limit=15)

    # --- Summary figure (text + table) ---
    fig = plt.figure(figsize=(12, 7), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.9,
        "Batch Overlay Summary (masked)",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_text.text(
        0.01,
        0.55,
        f"data_dir: {data_dir}\n"
        f"users processed: {stats.users_processed}\n"
        f"ok/missing/error: {stats.ok_rows}/{stats.missing_rows}/{stats.error_rows}",
        fontsize=12,
        ha="left",
        va="top",
    )

    ax_phase = fig.add_subplot(gs[1, 0])
    ax_phase.axis("off")
    if not phase_tbl.empty:
        phase_rows = []
        for _, row in phase_tbl.iterrows():
            phase_rows.append(
                [
                    str(row.get("phase", "")),
                    int(row.get("count", 0)) if pd.notna(row.get("count")) else 0,
                    float(row.get("mean")) if pd.notna(row.get("mean")) else np.nan,
                    float(row.get("median")) if pd.notna(row.get("median")) else np.nan,
                ]
            )
        cell_text = []
        for phase, count, mean, median in phase_rows:
            cell_text.append(
                [
                    phase,
                    f"{count:d}",
                    "" if np.isnan(mean) else f"{mean:.4f}",
                    "" if np.isnan(median) else f"{median:.4f}",
                ]
            )
        tbl = ax_phase.table(
            cellText=cell_text,
            colLabels=["phase", "count", "mean", "median"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
    ax_phase.set_title("Phase-wise F1", fontsize=12, pad=10)

    ax_miss = fig.add_subplot(gs[1, 1])
    ax_miss.axis("off")
    ax_miss.set_title("Missing files (up to 15)", fontsize=12, pad=10)
    if miss_tbl.empty:
        ax_miss.text(0.5, 0.5, "(none)", ha="center", va="center", fontsize=12)
    else:
        tbl2 = ax_miss.table(
            cellText=miss_tbl.values,
            colLabels=miss_tbl.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(out_dir / "batch_summary.png", bbox_inches="tight")
    plt.close(fig)

    # --- Boxplot figure ---
    ok = df_masked[df_masked["status"] == "ok"].copy()
    ok["f1"] = ok["f1"].map(_safe_float)
    ok = ok.dropna(subset=["f1", "phase"])
    if not ok.empty:
        fig2 = plt.figure(figsize=(10, 5), dpi=150)
        ax = fig2.add_subplot(1, 1, 1)
        phases = sorted(ok["phase"].astype(str).unique())
        data = [ok.loc[ok["phase"].astype(str) == p, "f1"].values for p in phases]
        ax.boxplot(data, tick_labels=phases, showmeans=True)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title("F1 Distribution by Phase (masked)")
        fig2.tight_layout()
        fig2.savefig(out_dir / "f1_by_phase_boxplot.png", bbox_inches="tight")
        plt.close(fig2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="Path to batch_overlay_summary_*.csv")
    ap.add_argument("--out_dir", default=None, help="Where to write outputs (default: csv parent)")
    ap.add_argument("--mask", action="store_true", help="Mask user identifiers to ASCII IDs")
    ap.add_argument("--mask_prefix", default="S", help="Prefix for masked IDs")
    ap.add_argument("--data_dir", default="", help="Label for data_dir in outputs")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    data_dir = args.data_dir or str(summary_csv.parent.name)

    if args.mask and "user" in df.columns:
        mapping = _mask_users(df["user"].astype(str).tolist(), prefix=args.mask_prefix)
        df_masked = df.copy()
        df_masked["user"] = df_masked["user"].astype(str).map(lambda u: mapping.get(u, u))
    else:
        df_masked = df

    _write_masked_outputs(df_masked, out_dir=out_dir, data_dir=data_dir)
    _render_images(df_masked, out_dir=out_dir, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
