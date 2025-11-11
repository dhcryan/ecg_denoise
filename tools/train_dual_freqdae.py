#!/usr/bin/env python3
"""Dedicated training & evaluation script for Dual_FreqDAE.

Replicates the original hyperparameters / callbacks logic from `deepFilter/dl_pipeline.py`
but scoped only to Dual_FreqDAE with explicit data preparation (Fourier branch) and
post-training metrics computation (SSD, MAD, PRD, COS_SIM, SNR).

Outputs
  <exp_dir>/Dual_FreqDAE_weights.best.weights.h5   (best weights)
  <exp_dir>/history.json                          (training history)
  <exp_dir>/metrics_summary.json                  (aggregate metrics)
  <exp_dir>/metrics_per_sample.csv                (per-sample metrics)

Usage examples
  python tools/train_dual_freqdae.py --data-prep-samples 512 --epochs 100000 \
	  --exp-dir experiments/dual_run --patience 10 --min-delta 0.05

  (Resume / reuse existing weights)
  python tools/train_dual_freqdae.py --exp-dir experiments/dual_run --resume

  (Evaluate only)
  python tools/train_dual_freqdae.py --exp-dir experiments/dual_run --evaluate-only
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import losses

# Add repo root to sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from deepFilter.dl_models import Dual_FreqDAE  # noqa: E402
from utils.metrics import PRD, COS_SIM, SNR, RMSE  # noqa: E402
from Data_Preparation.data_preparation_with_fourier import Data_Preparation_with_Fourier  # noqa: E402

def combined_ssd_mad_loss(y_true, y_pred):
	# same as original pipeline
	return tf.reduce_max(tf.square(y_true - y_pred), axis=-2) * 50 + tf.reduce_sum(tf.square(y_true - y_pred), axis=-2)

def build_model(signal_size: int = 512):
	model = Dual_FreqDAE(signal_size=signal_size)
	lr = 1e-3
	model.compile(
		loss=combined_ssd_mad_loss,
		optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
		# Use stable metric identifiers compatible with current Keras
		metrics=["mse", "mae"],
	)
	return model

def prepare_dataset(samples: int, reuse_cache: bool, cache_dir: Path):
	cache_dir.mkdir(parents=True, exist_ok=True)
	npy_bundle = cache_dir / f"dual_freqdae_dataset_{samples}.npz"
	if reuse_cache and npy_bundle.exists():
		data = np.load(npy_bundle)
		return [
			data['X_train'], data['y_train'], data['X_test'], data['y_test'],
			data['F_train_x'], data['F_train_y'], data['F_test_x'], data['F_test_y']
		]
	Dataset, _, _ = Data_Preparation_with_Fourier(samples=samples, fs=360)
	# Persist for reuse (avoid repeated heavy preprocessing)
	np.savez_compressed(
		npy_bundle,
		X_train=Dataset[0], y_train=Dataset[1], X_test=Dataset[2], y_test=Dataset[3],
		F_train_x=Dataset[4], F_train_y=Dataset[5], F_test_x=Dataset[6], F_test_y=Dataset[7]
	)
	return Dataset

def train(model, Dataset, exp_dir: Path, epochs: int, patience: int, min_delta: float, resume: bool):
	[X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset
	# split train/val identically to original (30%)
	from sklearn.model_selection import train_test_split
	X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
	F_tr_x, F_val_x, F_tr_y, F_val_y = train_test_split(F_train_x, F_train_y, test_size=0.3, shuffle=True, random_state=1)

	exp_dir.mkdir(parents=True, exist_ok=True)
	weights_path = exp_dir / "Dual_FreqDAE_weights.best.weights.h5"
	history_json = exp_dir / "history.json"

	callbacks = []
	checkpoint = ModelCheckpoint(
		str(weights_path), monitor="val_loss", verbose=1, save_best_only=True,
		mode='min', save_weights_only=True
	)
	callbacks.append(checkpoint)
	reduce_lr = ReduceLROnPlateau(
		monitor="val_loss", factor=0.5, min_delta=min_delta, mode='min',
		patience=2, min_lr=1e-10, verbose=1
	)
	callbacks.append(reduce_lr)
	early_stop = EarlyStopping(
		monitor="val_loss", min_delta=min_delta, mode='min', patience=patience, verbose=1
	)
	callbacks.append(early_stop)
	log_dir = exp_dir / f"runs_{datetime.now().strftime('%m%d')}_Dual_FreqDAE"
	tboard = TensorBoard(log_dir=str(log_dir), histogram_freq=0, write_graph=False, write_images=False)
	callbacks.append(tboard)

	if resume and weights_path.exists():
		print(f"[INFO] Resuming from weights: {weights_path}")
		model.load_weights(str(weights_path))

	print("[INFO] Starting training...")
	hist = model.fit(
		x=[X_tr, F_tr_x], y=y_tr,
		validation_data=([X_val, F_val_x], y_val),
		batch_size=128,
		epochs=epochs,
		verbose=1,
		callbacks=callbacks,
	)
	with history_json.open('w') as f:
		json.dump(hist.history, f, indent=2)
	print(f"[INFO] Saved training history to {history_json}")
	return weights_path

def evaluate(model, Dataset, weights_path: Path, exp_dir: Path):
	[X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y] = Dataset
	if weights_path.exists():
		model.load_weights(str(weights_path))
		print(f"[INFO] Loaded best weights for evaluation: {weights_path}")
	else:
		print("[WARN] Weights path does not exist, evaluating random-initialized model.")

	print("[INFO] Running inference on test set...")
	y_pred = model.predict([X_test, F_test_x], batch_size=128, verbose=1)

	# New metric set: RMSE, PRD, COS_SIM, SNR
	rmse = RMSE(y_test, y_pred)
	prd = PRD(y_test, y_pred)
	cos = COS_SIM(y_test, y_pred).reshape(-1)
	snr = SNR(y_test, y_pred)

	summary = {
		'n_test': int(y_test.shape[0]),
		'RMSE_mean': float(np.mean(rmse)), 'RMSE_std': float(np.std(rmse)),
		'PRD_mean': float(np.mean(prd)), 'PRD_std': float(np.std(prd)),
		'COS_SIM_mean': float(np.mean(cos)), 'COS_SIM_std': float(np.std(cos)),
		'SNR_mean': float(np.mean(snr)), 'SNR_std': float(np.std(snr)),
	}
	metrics_json = exp_dir / "metrics_summary.json"
	with metrics_json.open('w') as f:
		json.dump(summary, f, indent=2)
	print(f"[INFO] Saved metrics summary to {metrics_json}")

	metrics_csv = exp_dir / "metrics_per_sample.csv"
	with metrics_csv.open('w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["idx", "RMSE", "PRD", "COS_SIM", "SNR"])
		for i in range(y_test.shape[0]):
			writer.writerow([i, rmse[i], prd[i], cos[i], snr[i]])
	print(f"[INFO] Saved per-sample metrics to {metrics_csv}")

def parse_args():
	p = argparse.ArgumentParser(description="Train/Evaluate Dual_FreqDAE model")
	p.add_argument('--data-prep-samples', type=int, default=512, help='Signal length (must match model)')
	p.add_argument('--exp-dir', type=str, default='experiments/dual_freqdae', help='Output experiment directory')
	p.add_argument('--epochs', type=int, default=int(1e5), help='Max epochs (original pipeline default)')
	p.add_argument('--patience', type=int, default=10, help='Early stopping patience (original)')
	p.add_argument('--min-delta', type=float, default=0.05, help='Early stopping & LR reduce min_delta (original)')
	p.add_argument('--reuse-cache', action='store_true', help='Reuse cached preprocessed dataset npz')
	p.add_argument('--resume', action='store_true', help='Resume training from existing weights')
	p.add_argument('--evaluate-only', action='store_true', help='Skip training and only evaluate')
	p.add_argument('--no-train', action='store_true', help='Alias of --evaluate-only (backward friendly)')
	p.add_argument('--dry-run', action='store_true', help='Build dataset (if needed) and model summary, then exit without training/eval')
	return p.parse_args()

def main():
	args = parse_args()
	exp_dir = Path(args.exp_dir)

	print("[STEP] Preparing dataset...")
	Dataset = prepare_dataset(samples=args.data_prep_samples, reuse_cache=args.reuse_cache, cache_dir=exp_dir / 'cache')
	print(f"[INFO] Dataset ready: train {Dataset[0].shape}, test {Dataset[2].shape}")

	print("[STEP] Building model...")
	model = build_model(signal_size=args.data_prep_samples)
	model.summary()

	weights_path = exp_dir / "Dual_FreqDAE_weights.best.weights.h5"

	if args.dry_run:
		print("[INFO] Dry-run requested. Exiting after model build & dataset prep.")
		return

	if args.evaluate_only or args.no_train:
		evaluate(model, Dataset, weights_path, exp_dir)
		return

	print("[STEP] Training...")
	weights_path = train(model, Dataset, exp_dir, epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, resume=args.resume)
	print("[STEP] Evaluation...")
	evaluate(model, Dataset, weights_path, exp_dir)

if __name__ == '__main__':
	main()

