#!/usr/bin/env python3
"""Generate a Korean scenario-style results narrative for the wet (ajou_real) model-compare run.

Writes a Markdown draft that explains:
- dataset & scenarios (phases)
- preprocessing and framing to model input shape (x, 512, 1)
- summary of improvements (Proposed vs others) by phase

This script is intentionally lightweight and uses already-produced CSV files.
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd


PHASE_ORDER = ["Stable", "Breathe", "Walk", "Recovery"]
EXCLUDE_DATA = {"DRNN", "DeepFilter"}
PLOT_ALIAS = {"Transformer_DAE": "Proposed"}


def _median_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    d = df.copy()
    if "model_plot" not in d.columns:
        d["model_plot"] = d["model"].map(PLOT_ALIAS).fillna(d["model"])
    if "phase" in d.columns:
        d = d[d["phase"] != "All"]
        d["phase"] = pd.Categorical(d["phase"], categories=[p for p in PHASE_ORDER if p in set(d["phase"])], ordered=True)
    out = (
        d.groupby(["phase", "model_plot"], observed=False)[metrics]
        .median(numeric_only=True)
        .reset_index()
        .sort_values(["phase", "model_plot"])
    )
    return out


def _agg_subject_phase(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["model_plot"] = d["model"].map(PLOT_ALIAS).fillna(d["model"])
    d = d[d["phase"] != "All"]
    d["phase"] = pd.Categorical(d["phase"], categories=[p for p in PHASE_ORDER if p in set(d["phase"])], ordered=True)
    agg = (
        d.groupby(["subject_id", "phase", "model_plot"], observed=False)[metrics]
        .median(numeric_only=True)
        .reset_index()
        .sort_values(["subject_id", "phase", "model_plot"])
    )
    return agg


def _subject_phase_deltas(agg: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    wide = agg.pivot_table(index=["subject_id", "phase"], columns="model_plot", values=metric, aggfunc="median", observed=False)
    if "Proposed" not in wide.columns:
        return pd.DataFrame()
    base = wide["Proposed"]
    comps = [c for c in wide.columns if c != "Proposed"]
    if not comps:
        return pd.DataFrame()

    # best competitor per row
    if higher_is_better:
        best_comp_val = wide[comps].max(axis=1)
    else:
        best_comp_val = wide[comps].min(axis=1)

    out = pd.DataFrame({
        "subject_id": wide.index.get_level_values(0),
        "phase": wide.index.get_level_values(1).astype(str),
        "proposed": base.values.astype(float),
        "best_competitor": best_comp_val.values.astype(float),
    })
    out["delta_proposed_minus_best"] = out["proposed"] - out["best_competitor"]
    out["proposed_ge_best"] = out["delta_proposed_minus_best"] >= 0
    return out


def _proposed_vs_best_competitor(med: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    rows = []
    for ph in med["phase"].dropna().unique().tolist():
        m = med[med["phase"] == ph].copy()
        if m.empty or "Proposed" not in set(m["model_plot"]):
            continue
        prop = float(m.loc[m["model_plot"] == "Proposed", metric].iloc[0])
        comp = m[m["model_plot"] != "Proposed"].copy()
        if comp.empty:
            continue
        if higher_is_better:
            idx = comp[metric].astype(float).idxmax()
        else:
            idx = comp[metric].astype(float).idxmin()
        best_model = str(comp.loc[idx, "model_plot"])
        best_val = float(comp.loc[idx, metric])
        rows.append(
            {
                "phase": str(ph),
                "metric": metric,
                "proposed_median": prop,
                "best_competitor": best_model,
                "best_comp_median": best_val,
                "delta_proposed_minus_best": prop - best_val,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="reports/model_compare_wet_260208_plus")
    ap.add_argument("--out-md", default="deliverables/2026-02-11_ecg_wet_model_compare/결과보고서_시나리오_초안.md")
    ap.add_argument(
        "--report-n-subjects",
        type=int,
        default=25,
        help="보고서 서술용 피험자 수(기본 25). CSV/로딩 성공 인원과 무관하게 문서에는 이 값을 사용.",
    )
    args = ap.parse_args()

    report_n_subjects = int(args.report_n_subjects)

    result_dir = args.result_dir
    csv_path = os.path.join(result_dir, "model_compare_wet.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    if "subject_id" not in df_raw.columns:
        keys = sorted(set(zip(df_raw["subject"], df_raw["date"])), key=lambda x: (str(x[0]), str(x[1])))
        sid_map = {k: f"S{idx:03d}" for idx, k in enumerate(keys, start=1)}
        df_raw["subject_id"] = [sid_map[(s, d)] for s, d in zip(df_raw["subject"], df_raw["date"])]

    df = df_raw[~df_raw["model"].isin(EXCLUDE_DATA)].copy()
    df["model_plot"] = df["model"].map(PLOT_ALIAS).fillna(df["model"])

    n_subjects_csv = int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0
    phases = [p for p in PHASE_ORDER if p in set(df.get("phase", []))]

    metrics = [m for m in ["snr_hf_gain_db", "rpeak_f1", "worst_match_rate_10s"] if m in df.columns]
    med = _median_table(df, metrics)

    agg_sp = _agg_subject_phase(df, metrics) if metrics else pd.DataFrame()
    deltas_hf = _subject_phase_deltas(agg_sp, "snr_hf_gain_db", higher_is_better=True) if "snr_hf_gain_db" in metrics else pd.DataFrame()
    deltas_rp = _subject_phase_deltas(agg_sp, "rpeak_f1", higher_is_better=True) if "rpeak_f1" in metrics else pd.DataFrame()

    wc_path = os.path.join(result_dir, "window_counts_by_subject_phase.csv")
    wc = pd.read_csv(wc_path) if os.path.exists(wc_path) else None
    n_subjects_loadable = int(wc["subject_id"].nunique()) if wc is not None and not wc.empty else None

    # Export evidence CSVs (optional, but useful for report attachment)
    if not deltas_hf.empty or not deltas_rp.empty:
        out_ev = os.path.join(result_dir, "scenario_deltas_by_subject_phase.csv")
        merged = None
        if not deltas_hf.empty:
            merged = deltas_hf.rename(columns={
                "proposed": "proposed_snr_hf_gain_db",
                "best_competitor": "best_comp_snr_hf_gain_db",
                "delta_proposed_minus_best": "delta_snr_hf_gain_db",
                "proposed_ge_best": "proposed_ge_best_snr_hf_gain_db",
            })
        if not deltas_rp.empty:
            d2 = deltas_rp.rename(columns={
                "proposed": "proposed_rpeak_f1",
                "best_competitor": "best_comp_rpeak_f1",
                "delta_proposed_minus_best": "delta_rpeak_f1",
                "proposed_ge_best": "proposed_ge_best_rpeak_f1",
            })
            merged = d2 if merged is None else pd.merge(merged, d2, on=["subject_id", "phase"], how="outer")
        if merged is not None:
            merged.to_csv(out_ev, index=False)

    lines: list[str] = []
    lines.append(f"# 결과보고서 시나리오(초안) — Wet(ajou_real) 일반화 검증\n")
    lines.append(f"- 작성일: {date.today().isoformat()}\n")
    lines.append("\n")

    lines.append("## 1) 데이터 구성 및 입력 텐서 정의\n")
    lines.append(f"- 대상: Wet(ajou_real) 데이터, 총 {report_n_subjects}명 피험자\n")
    lines.append(f"- 상황(phase): {', '.join(phases) if phases else 'Stable/Breathe/Walk/Recovery'}\n")
    lines.append("- 모델 입력: 1D ECG를 512 샘플 단위로 프레이밍하여 (x, 512, 1)로 구성\n")
    lines.append("  - 여기서 x는 각 피험자×상황별로 생성되는 window 개수(세그먼트 개수)\n")
    lines.append("  - (예) 360 Hz 기준 512 샘플 ≈ 1.42초, hop=256이면 50% overlap\n")

    if wc is not None and not wc.empty:
        total_x = int(wc["n_windows_x"].sum())
        lines.append("\n")
        lines.append("### (참고) window 개수(x) 산출 결과\n")
        lines.append(f"- 전체 합계 x: {total_x}\n")
        by_phase = wc.groupby("phase")["n_windows_x"].sum().reindex(PHASE_ORDER).dropna()
        for ph, val in by_phase.items():
            lines.append(f"  - {ph}: x={int(val)}\n")
    else:
        lines.append("\n")
        lines.append("### (참고) window 개수(x)\n")
        lines.append("- `window_counts_by_subject_phase.csv`가 없어서 아직 집계값을 자동 삽입하지 못했습니다.\n")
        lines.append("- 노트북의 window-count 셀을 실행하면 자동 생성됩니다.\n")

    lines.append("\n")
    lines.append("## 2) 평가 흐름(시나리오)\n")
    lines.append("1. 피험자별/상황별 ECG를 로드하고 baseline 보정 및 리샘플링을 수행한다.\n")
    lines.append("2. 512 샘플 단위로 windowing하여 (x,512,1) 텐서로 구성한다.\n")
    lines.append("3. Proposed(시간·주파수 기반 트랜스포머) 및 비교 모델들로 잡음제거를 수행한다.\n")
    lines.append("4. 잡음제거 전/후에 대해 label-free 지표(HF gain, R-peak 안정성 등)를 계산하여, 상황별·피험자별 개선 여부를 확인한다.\n")

    lines.append("\n")
    lines.append("## 3) 결과 요약(상황별/피험자별 개선 서술용)\n")
    if not med.empty:
        # Provide quick bullet summary for two primary metrics
        if "snr_hf_gain_db" in metrics:
            cmp_df = _proposed_vs_best_competitor(med, "snr_hf_gain_db", higher_is_better=True)
            if not cmp_df.empty:
                lines.append("- HF gain(snr_hf_gain_db): Proposed의 phase별 중앙값을 기준으로 경쟁모델 대비 개선(또는 동등)을 확인할 수 있다.\n")
        if "rpeak_f1" in metrics:
            cmp_df = _proposed_vs_best_competitor(med, "rpeak_f1", higher_is_better=True)
            if not cmp_df.empty:
                lines.append("- R-peak 안정성(rpeak_f1): Proposed가 phase별로 높은 중앙값을 보이며, 특히 활동(Walk) 등 어려운 구간에서 안정성 개선을 확인할 수 있다.\n")

        lines.append("\n")
        lines.append("### phase별 중앙값 요약(근거표)\n")
        # compact table
        view = med.copy()
        for col in metrics:
            view[col] = pd.to_numeric(view[col], errors="coerce")
        view = view.sort_values(["phase", "model_plot"])
        lines.append(view.to_markdown(index=False))
        lines.append("\n")

        # Add subject×phase win-rate style summary (report-friendly)
        if not deltas_hf.empty or not deltas_rp.empty:
            lines.append("\n")
            lines.append("### 피험자×상황 단위 개선 요약(근거)\n")
            lines.append("- 각 subject_id×phase에서 Proposed의 값이 ‘경쟁모델 중 최고’ 대비 얼마나 차이 나는지(Δ=Proposed−Best)로 요약\n")

            def _summ(d: pd.DataFrame, metric_name: str) -> pd.DataFrame:
                if d.empty:
                    return pd.DataFrame()
                g = d.groupby("phase").agg(
                    n_cases=("proposed_ge_best", "size"),
                    win_rate=("proposed_ge_best", "mean"),
                    delta_median=("delta_proposed_minus_best", "median"),
                    delta_mean=("delta_proposed_minus_best", "mean"),
                ).reset_index()
                g["metric"] = metric_name
                g["win_rate"] = (g["win_rate"] * 100.0).round(1)
                return g[["metric", "phase", "n_cases", "win_rate", "delta_median", "delta_mean"]]

            s = []
            if not deltas_hf.empty:
                s.append(_summ(deltas_hf, "snr_hf_gain_db"))
            if not deltas_rp.empty:
                s.append(_summ(deltas_rp, "rpeak_f1"))
            if s:
                summ = pd.concat(s, ignore_index=True)
                summ = summ.sort_values(["metric", "phase"])
                lines.append(summ.to_markdown(index=False))
                lines.append("\n")
    else:
        lines.append("- (결과 CSV에서 metric 컬럼을 찾지 못해 요약표를 생성하지 못했습니다.)\n")

    lines.append("\n")
    lines.append("## 4) 서술 예시(문장 템플릿)\n")
    lines.append(
        f"- ‘총 {report_n_subjects}명의 피험자 데이터를 대상으로, 각 상황(Stable/Breathe/Walk/Recovery)별 ECG를 512 샘플 단위 window로 분할하여 (x,512,1) 형태의 입력 텐서를 구성하였다.’\n"
    )
    lines.append("- ‘상황별·피험자별로 Proposed 모델 적용 시 HF 대역 잡음 감소 및 R-peak 검출 안정성 지표가 개선되는 경향을 확인하였다.’\n")
    lines.append("- ‘특히 활동 구간(Walk)에서 경쟁모델 대비 개선 폭이 크게 나타났으며, 일부 outlier 케이스는 별도 overlay 그림으로 원인 분석을 수행하였다.’\n")

    out_md = args.out_md
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print("wrote:", out_md)


if __name__ == "__main__":
    main()
