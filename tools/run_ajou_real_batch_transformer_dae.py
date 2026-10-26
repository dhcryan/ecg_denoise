#!/usr/bin/env python3
"""Run Transformer_DAE generalization eval on `ajou_real/` (multi-channel) recordings.

This script:
- Scans `ajou_real/<name>_<date>/` folders
- Extracts a single ECG channel from each multi-column txt
- Writes a temporary single-column staging dataset in the naming convention expected by
  `tools/run_phase_batch_transformer_dae.py`
- Reuses that evaluator to run denoising + label-free metrics + sliding-window worst-window

Typical use:
  python tools/run_ajou_real_batch_transformer_dae.py \
    --real_root ajou_real \
    --out_root . \
    --weights 0221_FIXED/Transformer_DAE_weights.best.weights.h5 \
    --lead_index 2

Notes on `lead_index`:
- The real files usually look like: [time_or_index, lead1, lead2, lead3, aVR, aVL, aVF, (optional NaN col)]
- `lead_index` is 1-based over the *lead columns* (time/index is automatically removed when detected).
- Default `2` corresponds to Lead II in the common ordering above.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make repo root + tools dir importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

# Import the existing evaluator (build model once, run many)
import run_phase_batch_transformer_dae as base


PHASE_ORDER = ["Stable", "Breathe", "Walk", "Recovery"]


def _safe_user_id(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "", name)
    # keep Hangul, ASCII, digits, underscore
    name = re.sub(r"[^0-9A-Za-z_가-힣]", "_", name)
    return name


def _parse_subject_dir(dirname: str) -> Optional[Tuple[str, str]]:
    """Return (name, date) from a folder such as 'Subject01_260121'."""
    m = re.match(r"^(?P<name>.+)_(?P<date>\d{6})$", dirname)
    if not m:
        return None
    return (m.group("name"), m.group("date"))


def _infer_phase_from_filename(fname: str) -> Optional[str]:
    """Map various naming conventions to Stable/Breathe/Walk/Recovery/Session."""
    s = os.path.splitext(os.path.basename(fname))[0]

    # Explicit session
    if re.search(r"session|세션", s, flags=re.IGNORECASE):
        return "Session"

    # Numeric prefix like 01안정, 02호흡, 03운동, 04회복
    m = re.search(r"_(\d{2})", s)
    if m:
        idx = int(m.group(1))
        if idx == 1:
            return "Stable"
        if idx == 2:
            return "Breathe"
        if idx == 3:
            return "Walk"
        if idx == 4:
            return "Recovery"

    # Keyword based
    if re.search(r"stable|안정", s, flags=re.IGNORECASE):
        return "Stable"
    if re.search(r"breathe|호흡", s, flags=re.IGNORECASE):
        return "Breathe"
    if re.search(r"walk|운동", s, flags=re.IGNORECASE):
        return "Walk"
    if re.search(r"recovery|회복", s, flags=re.IGNORECASE):
        return "Recovery"

    return None


def _read_multicol_txt(path: str) -> np.ndarray:
    """Reads whitespace separated numeric table. Returns shape (N, C) float array."""
    arr = np.genfromtxt(path, dtype=np.float64, invalid_raise=False)
    if arr.ndim == 1:
        # single row or single col
        arr = np.atleast_2d(arr)
    if arr.shape[0] < 2:
        raise RuntimeError(f"Too few samples in {path}")
    return arr


def _looks_like_time_index(col: np.ndarray) -> bool:
    x = col[np.isfinite(col)]
    if x.size < 10:
        return False
    # Check monotonic increasing and median step close to 1
    dx = np.diff(x[: min(2000, x.size)])
    if not np.all(dx > 0):
        return False
    med = float(np.median(dx))
    return abs(med - 1.0) < 1e-3 or abs(med - 1.0) < 1e-2


def _select_lead_column(arr: np.ndarray, lead_index_1based: int) -> np.ndarray:
    """Select a lead column from multi-column array.

    - Drops columns with too many NaNs
    - Drops the first column if it looks like time/index
    - lead_index_1based is applied to the remaining columns
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    n, c = arr.shape
    # Keep columns with sufficient finite values
    finite_fracs = []
    keep_cols = []
    for j in range(c):
        frac = float(np.isfinite(arr[:, j]).mean())
        finite_fracs.append(frac)
        if frac >= 0.5:
            keep_cols.append(j)

    if not keep_cols:
        raise RuntimeError("No usable columns (too many NaNs)")

    arr2 = arr[:, keep_cols]

    # Drop time/index-like first col
    if arr2.shape[1] >= 2 and _looks_like_time_index(arr2[:, 0]):
        arr2 = arr2[:, 1:]

    if arr2.shape[1] < 1:
        raise RuntimeError("No lead columns left after dropping time/index")

    lead_idx0 = int(lead_index_1based) - 1
    if lead_idx0 < 0 or lead_idx0 >= arr2.shape[1]:
        raise ValueError(
            f"lead_index={lead_index_1based} out of range; available lead cols: 1..{arr2.shape[1]}"
        )

    sig = np.asarray(arr2[:, lead_idx0], dtype=np.float32)
    # Replace NaNs by median (rare) to keep pipeline stable
    if not np.all(np.isfinite(sig)):
        med = float(np.nanmedian(sig))
        sig = np.nan_to_num(sig, nan=med)
    return sig


def _write_single_column_txt(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    with open(path, "w") as f:
        for v in x:
            f.write(f"{float(v)}\n")


@dataclass
class SubjectCase:
    user: str
    date: str
    src_dir: str
    phase_files: Dict[str, str]  # phase -> filepath


def _collect_cases(real_root: str) -> List[SubjectCase]:
    cases: List[SubjectCase] = []
    for entry in sorted(os.listdir(real_root)):
        full = os.path.join(real_root, entry)
        if not os.path.isdir(full):
            continue
        parsed = _parse_subject_dir(entry)
        if not parsed:
            continue
        name, date = parsed
        phase_files: Dict[str, str] = {}
        for fn in sorted(os.listdir(full)):
            if not fn.lower().endswith(".txt"):
                continue
            ph = _infer_phase_from_filename(fn)
            if not ph:
                continue
            phase_files[ph] = os.path.join(full, fn)
        if not any(p in phase_files for p in PHASE_ORDER) and "Session" not in phase_files:
            continue
        cases.append(
            SubjectCase(user=_safe_user_id(name), date=str(date), src_dir=full, phase_files=phase_files)
        )
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real_root", default="ajou_real", help="Root folder like ajou_real/")
    p.add_argument("--out_root", default=".")
    p.add_argument("--stage_root", default=".cache/ajou_real_singlecol_267")
    p.add_argument("--weights", default="0221_FIXED/Transformer_DAE_weights.best.weights.h5")
    p.add_argument(
        "--lead_index",
        type=int,
        default=2,
        help="1-based lead index after dropping time/index column; default 2 ~ Lead II",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run first N subjects (for quick smoke test).",
    )
    p.add_argument(
        "--clean_stage",
        action="store_true",
        help="Delete stage_root before running (rebuild from scratch).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    real_root = os.path.abspath(str(args.real_root))
    out_root = os.path.abspath(str(args.out_root))
    stage_root = os.path.abspath(str(args.stage_root))

    if args.clean_stage and os.path.exists(stage_root):
        shutil.rmtree(stage_root)

    cases = _collect_cases(real_root)
    if args.limit and args.limit > 0:
        cases = cases[: int(args.limit)]

    if not cases:
        raise SystemExit(f"No valid cases found under: {real_root}")

    # Build a model once
    first = cases[0]
    dummy_cfg = base.RunConfig(
        data_dir=stage_root,
        out_root=out_root,
        date=first.date,
        users=[first.user],
        phases=list(PHASE_ORDER),
        pretrained_weights=str(args.weights),
    )
    dummy_cfg = base.RunConfig(
        **{**dummy_cfg.__dict__, "data_dir": os.path.abspath(dummy_cfg.data_dir), "out_root": os.path.abspath(dummy_cfg.out_root)}
    )
    os.makedirs(dummy_cfg.out_root, exist_ok=True)

    print("Building model + loading weights...")
    model = base._build_model_and_load(dummy_cfg)

    rows = []
    for case in cases:
        print(f"=== {case.user} ({case.date}) from {os.path.basename(case.src_dir)} ===")

        stage_dir = os.path.join(stage_root, f"{case.user}_{case.date}")
        os.makedirs(stage_dir, exist_ok=True)

        # Prefer session if available
        if "Session" in case.phase_files:
            arr = _read_multicol_txt(case.phase_files["Session"])
            sig = _select_lead_column(arr, lead_index_1based=int(args.lead_index))
            _write_single_column_txt(os.path.join(stage_dir, f"{case.user}_Session_{case.date}.txt"), sig)
        else:
            for ph in PHASE_ORDER:
                if ph not in case.phase_files:
                    raise FileNotFoundError(f"Missing phase {ph} for {case.user} {case.date}")
                arr = _read_multicol_txt(case.phase_files[ph])
                sig = _select_lead_column(arr, lead_index_1based=int(args.lead_index))
                _write_single_column_txt(os.path.join(stage_dir, f"{case.user}_{ph}_{case.date}.txt"), sig)

        cfg = base.RunConfig(
            data_dir=stage_dir,
            out_root=out_root,
            date=case.date,
            users=[case.user],
            phases=list(PHASE_ORDER),
            pretrained_weights=str(args.weights),
        )
        cfg = base.RunConfig(
            **{**cfg.__dict__, "data_dir": os.path.abspath(cfg.data_dir), "out_root": os.path.abspath(cfg.out_root)}
        )

        row = base.run_one_user(model, cfg, case.user)
        rows.append(row)
        print("  done:", row.get("out_dir"))

    # Write a consolidated summary for convenience
    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_root, "ajou_real_batch_summary.csv")
    df.to_csv(out_csv, index=False)
    print("✓ Summary saved:", out_csv)


if __name__ == "__main__":
    main()
