from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from Data_Preparation.data_preparation import Data_Preparation
from utils.visualization import generate_table

from baselines_1221_fixed.common import ensure_dir, save_test_results
from baselines_1221_fixed.metrics_1221 import compute_metrics
from baselines_1221_fixed.fully_gated_dae import FullyGatedDAEConfig, train_and_predict as run_fgdae


def _maybe_run_scorediff(*args, **kwargs):
    from baselines_1221_fixed.score_diffusion import train_and_predict as run_sd

    return run_sd(*args, **kwargs)


def _maybe_run_cyclegan(*args, **kwargs):
    from baselines_1221_fixed.cycle_gan_oc import train_and_predict as run_cg

    return run_cg(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test only the 1221_FIXED baselines under the unchanged QTDB+NSTDB-noise protocol.")
    parser.add_argument("--out-dir", type=str, default="1221_FIXED")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["CycleGAN_OC", "ScoreDiffusion", "FullyGatedDAE"],
        choices=["CycleGAN_OC", "ScoreDiffusion", "FullyGatedDAE"],
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for PyTorch baselines")

    # FullyGatedDAE knobs
    parser.add_argument("--fgdae-q", type=int, default=3)
    parser.add_argument(
        "--fgdae-qs",
        type=int,
        nargs="+",
        default=None,
        help="If provided, run FullyGatedDAE for each q in this list (e.g., --fgdae-qs 1 2 3 4). Overrides --fgdae-q.",
    )
    parser.add_argument("--fgdae-batch", type=int, default=128)
    parser.add_argument("--fgdae-lr", type=float, default=1e-3)
    parser.add_argument("--fgdae-epochs", type=int, default=100000)
    parser.add_argument("--fgdae-patience", type=int, default=20)
    parser.add_argument("--fgdae-min-delta", type=float, default=1e-4)

    # ScoreDiffusion knobs
    parser.add_argument("--sd-epochs", type=int, default=400)
    parser.add_argument("--sd-batch", type=int, default=96)
    parser.add_argument("--sd-feats", type=int, default=80)
    parser.add_argument("--sd-steps", type=int, default=50)
    parser.add_argument("--sd-early-stop-patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--sd-early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--sd-valid-epoch-interval", type=int, default=5)
    parser.add_argument("--sd-valid-split", type=float, default=0.1, help="Validation fraction for early stopping (0 disables)")
    parser.add_argument("--sd-min-epochs", type=int, default=0, help="Minimum epochs to run before early stopping can trigger")

    # CycleGAN knobs
    parser.add_argument("--cg-epochs", type=int, default=50)
    parser.add_argument("--cg-batch", type=int, default=32)

    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)

    # 1) Dataset: MUST remain identical to the existing protocol
    Dataset, _valid_train_indices, _valid_test_indices = Data_Preparation(samples=512)
    x_train, y_train, x_test, y_test = Dataset

    exp_names: List[str] = []
    metric_arrays = {"RMSE": [], "PRD": [], "COS_SIM": [], "SNR": []}

    for exp in args.experiments:
        if exp == "FullyGatedDAE":
            qs = args.fgdae_qs if args.fgdae_qs is not None else [args.fgdae_q]
            for q in qs:
                exp_variant = f"FullyGatedDAE_q{q}"
                print(f"\n[1221_FIXED] Running: {exp_variant}")

                cfg = FullyGatedDAEConfig(
                    q=q,
                    batch_size=args.fgdae_batch,
                    lr=args.fgdae_lr,
                    max_epochs=args.fgdae_epochs,
                    patience=args.fgdae_patience,
                    min_delta=args.fgdae_min_delta,
                )

                fg_out_dir = os.path.join(out_dir, exp_variant)
                y_pred = run_fgdae(x_train, y_train, x_test, out_dir=fg_out_dir, cfg=cfg)

                save_path = save_test_results(out_dir, exp_variant, x_test, y_test, y_pred)
                print(f"[1221_FIXED] Saved: {save_path}")

                m = compute_metrics(y_test, y_pred)
                exp_names.append(exp_variant)
                for k in metric_arrays.keys():
                    metric_arrays[k].append(m[k])

            continue

        elif exp == "ScoreDiffusion":
            from baselines_1221_fixed.score_diffusion import ScoreDiffusionConfig

            cfg = ScoreDiffusionConfig(
                feats=args.sd_feats,
                epochs=args.sd_epochs,
                batch_size=args.sd_batch,
                diffusion_num_steps=args.sd_steps,
                valid_split=args.sd_valid_split,
                early_stop_patience=args.sd_early_stop_patience,
                early_stop_min_delta=args.sd_early_stop_min_delta,
                valid_epoch_interval=args.sd_valid_epoch_interval,
                min_epochs=args.sd_min_epochs,
            )
            y_pred = _maybe_run_scorediff(x_train, y_train, x_test, out_dir=out_dir, cfg=cfg, device=args.device)

        elif exp == "CycleGAN_OC":
            from baselines_1221_fixed.cycle_gan_oc import CycleGANConfig

            cfg = CycleGANConfig(epochs=args.cg_epochs, batch_size=args.cg_batch)
            y_pred = _maybe_run_cyclegan(x_train, y_train, x_test, out_dir=out_dir, cfg=cfg, device=args.device)

        else:
            raise ValueError(f"Unknown experiment: {exp}")

        print(f"\n[1221_FIXED] Running: {exp}")

        # Save test results exactly like the existing runs
        save_path = save_test_results(out_dir, exp, x_test, y_test, y_pred)
        print(f"[1221_FIXED] Saved: {save_path}")

        # Metrics
        m = compute_metrics(y_test, y_pred)
        exp_names.append(exp)
        for k in metric_arrays.keys():
            metric_arrays[k].append(m[k])

    # 3) Print the requested table
    metrics = ["RMSE", "PRD", "COS_SIM", "SNR"]
    metric_values = [metric_arrays[m] for m in metrics]
    generate_table(metrics, metric_values, exp_names)


if __name__ == "__main__":
    main()
