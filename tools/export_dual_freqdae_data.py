#!/usr/bin/env python3
"""Export prepared Dual_FreqDAE dataset arrays (train/test + Fourier) to .npy files.

This helps with external inference without re-running the full training script.

Usage:
  python tools/export_dual_freqdae_data.py --samples 512 --out-dir exported/dual_freqdae_ds --reuse-cache

Files written:
  X_train.npy, y_train.npy, X_test.npy, y_test.npy
  F_train_x.npy, F_train_y.npy, F_test_x.npy, F_test_y.npy

If --reuse-cache is set and a cached npz exists, it is reused.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier  # noqa: E402

def prepare_dataset(samples: int, reuse_cache: bool, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = cache_dir / f"dual_freqdae_dataset_{samples}.npz"
    if reuse_cache and bundle.exists():
        data = np.load(bundle)
        return [
            data['X_train'], data['y_train'], data['X_test'], data['y_test'],
            data['F_train_x'], data['F_train_y'], data['F_test_x'], data['F_test_y']
        ]
    Dataset, _, _ = Data_Preparation_with_Fourier(samples=samples, fs=360)
    np.savez_compressed(
        bundle,
        X_train=Dataset[0], y_train=Dataset[1], X_test=Dataset[2], y_test=Dataset[3],
        F_train_x=Dataset[4], F_train_y=Dataset[5], F_test_x=Dataset[6], F_test_y=Dataset[7]
    )
    return Dataset

def main():
    ap = argparse.ArgumentParser(description="Export Dual_FreqDAE dataset arrays")
    ap.add_argument('--samples', type=int, default=512)
    ap.add_argument('--out-dir', type=str, default='exported/dual_freqdae_ds')
    ap.add_argument('--reuse-cache', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / 'cache'
    Dataset = prepare_dataset(args.samples, args.reuse_cache, cache_dir)

    names = [
        'X_train', 'y_train', 'X_test', 'y_test',
        'F_train_x', 'F_train_y', 'F_test_x', 'F_test_y'
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in zip(names, Dataset):
        np.save(out_dir / f"{name}.npy", arr)
        print(f"Saved {name}.npy shape={arr.shape}")
    print(f"Export complete in {out_dir}")

if __name__ == '__main__':
    main()
