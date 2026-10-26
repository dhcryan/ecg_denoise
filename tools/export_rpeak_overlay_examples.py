import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Args:
    result_dir: str = "reports/model_compare_wet_260208_plus"
    wet_root: str = "ajou_real"
    weights_dir: str = "0221_FIXED"
    lead_index: int = 2  # 1-based (Lead II default)
    seconds: int = 12
    top_n: int = 5


def _ensure_on_syspath(path: str) -> None:
    ap = os.path.abspath(path)
    if ap not in sys.path:
        sys.path.insert(0, ap)


def parse_args() -> Args:
    import argparse

    p = argparse.ArgumentParser(description="Export static R-peak overlay PNG examples for delivery/report.")
    p.add_argument("--result-dir", default=Args.result_dir)
    p.add_argument("--wet-root", default=Args.wet_root)
    p.add_argument("--weights-dir", default=Args.weights_dir)
    p.add_argument("--lead-index", type=int, default=Args.lead_index)
    p.add_argument("--seconds", type=int, default=Args.seconds)
    p.add_argument("--top-n", type=int, default=Args.top_n)
    ns = p.parse_args()
    return Args(
        result_dir=ns.result_dir,
        wet_root=ns.wet_root,
        weights_dir=ns.weights_dir,
        lead_index=ns.lead_index,
        seconds=ns.seconds,
        top_n=ns.top_n,
    )


def main() -> None:
    args = parse_args()

    csv_path = os.path.join(args.result_dir, "model_compare_wet.csv")
    worst_path = os.path.join(args.result_dir, "proposed_worst_cases_top20.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")
    if not os.path.exists(worst_path):
        raise FileNotFoundError(f"Missing: {worst_path}")

    # Import repo-local modules
    _ensure_on_syspath(".")
    _ensure_on_syspath("tools")
    import realdata_eval_utils as U  # noqa: E402
    import run_phase_batch_transformer_dae as base  # noqa: E402
    from ajou_rpeak_utils import detect_rpeaks, evaluate_rpeak_performance  # noqa: E402

    weights = os.path.join(args.weights_dir, "Transformer_DAE_weights.best.weights.h5")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Missing weight: {weights}")

    cfg = base.RunConfig(
        data_dir=".",
        out_root=".",
        date="000000",
        users=["_"],
        phases=list(getattr(U, "PHASE_ORDER", ["Stable", "Breathe", "Walk", "Recovery"])),
        pretrained_weights=weights,
    )
    model = base._build_model_and_load(cfg)

    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        keys = sorted(set(zip(df["subject"], df["date"])), key=lambda x: (str(x[0]), str(x[1])))
        sid_map = {k: f"S{idx:03d}" for idx, k in enumerate(keys, start=1)}
        df["subject_id"] = [sid_map[(s, d)] for s, d in zip(df["subject"], df["date"])]

    # Map subject_id -> (subject, date)
    lu = df[["subject_id", "subject", "date"]].drop_duplicates().sort_values(["subject_id", "date"])
    sid_to_case = {r.subject_id: (r.subject, str(r.date)) for r in lu.itertuples(index=False)}

    worst = pd.read_csv(worst_path).head(args.top_n)

    out_dir = os.path.join(args.result_dir, "figs", "rpeak_overlay_examples")
    os.makedirs(out_dir, exist_ok=True)

    def load_phase_signal_360(subject: str, date: str, phase: str):
        cases = U.parse_wet_cases(args.wet_root)
        target = None
        for c in cases:
            if c.subject == subject and str(c.date) == str(date):
                target = c
                break
        if target is None:
            raise RuntimeError(f"Case not found for subject={subject}, date={date}")
        sigs_267, _ = U.load_wet_case_signals_267(target, lead_index_1based=args.lead_index)
        if phase not in sigs_267:
            raise ValueError(f"Phase not found: {phase}; available={list(sigs_267.keys())}")
        x = U.baseline_center(sigs_267[phase])
        x = U.resample(x, fs_src=267, fs_tgt=360)
        tn = int(round(1.0 * 360))
        if tn > 0 and len(x) > tn:
            x = x[tn:]
        n = min(len(x), int(args.seconds * 360))
        return x[:n]

    for row in worst.itertuples(index=False):
        sid = str(row.subject_id)
        phase = str(row.phase)
        if sid not in sid_to_case:
            print(f"[skip] missing sid mapping: {sid}")
            continue
        subject, date = sid_to_case[sid]

        raw = load_phase_signal_360(subject, date, phase)
        deno = U.denoise_full_signal(model, raw, win=512, hop=256, batch_size=128)

        rp_raw = detect_rpeaks(raw, 360, prefer_neurokit=True, prefer_positive=True, assume_stable_hr=True, min_rr_s=0.30)
        rp_deno = detect_rpeaks(deno, 360, prefer_neurokit=True, prefer_positive=True, assume_stable_hr=True, min_rr_s=0.30)
        stab = evaluate_rpeak_performance(rp_raw, rp_deno, 360, tol_ms=50.0)

        t = np.arange(len(raw)) / 360.0
        plt.figure(figsize=(12.0, 4.0))
        plt.plot(t, raw, color="black", lw=1.0, label="raw")
        plt.plot(t, deno, color="tab:blue", lw=1.0, label="deno(Proposed)")
        if len(rp_raw) > 0:
            plt.scatter(rp_raw / 360.0, raw[rp_raw], s=18, marker="x", color="black", label="R(raw)")
        if len(rp_deno) > 0:
            plt.scatter(rp_deno / 360.0, deno[rp_deno], s=18, marker="o", color="red", label="R(deno)")

        f1 = float(stab.get("f1", np.nan))
        prec = float(stab.get("precision", np.nan))
        rec = float(stab.get("recall", np.nan))
        plt.title(f"R-peak overlay | {sid} {phase} | Proposed")
        plt.xlabel("seconds")
        plt.ylabel("ECG (a.u.)")
        plt.legend(ncol=4, fontsize=9, loc="upper right")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"rpeak_overlay_{sid}_{phase}_Proposed.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("saved:", out_png)


if __name__ == "__main__":
    main()
