# Dual_FreqDAE Repository

### Version
This is a stripped-down standalone extraction of the ECG denoising Dual_FreqDAE pipeline: model architecture, data preparation, training, evaluation, and inference utilities.

## 1. Overview
Dual_FreqDAE is a dual-branch denoising autoencoder for single-beat ECG segments (length 512 samples at 360 Hz). It processes:
- Time-domain branch: gated Conv1D encoder with multiplicative feature interaction (sigmoid × linear / elu pairs).
- Frequency-domain branch: FFT magnitude (half-spectrum duplicated), processed by a parallel convolutional encoder.
- Fusion: Concatenation of the latent outputs + positional encoding + transformer (FANformer) blocks.
- Decoder: Residual Conv1DTranspose chain reconstructing a cleaned beat.

Target use case: Remove mixed baseline/wideband noise while preserving morphological fidelity (P-QRS-T) for downstream tasks.

## 2. Directory Structure
```
dual_freqdae_repo/
	deepFilter/
		dl_models.py              # Dual_FreqDAE and building blocks (FANLayer, AddGatedNoise, positional encoding)
		__init__.py
	Data_Preparation/
		Prepare_QTDatabase.py     # Build QTDatabase.pkl (beat-separated, resampled to 360 Hz)
		data_preparation_with_fourier.py  # Creates noisy beats + FFT magnitude arrays
	tools/
		train_dual_freqdae.py     # Train + evaluate (RMSE, PRD, COS_SIM, SNR)
		run_dual_freqdae.py       # Inference for beats or segmented long signal
		export_dual_freqdae_data.py # Optional dataset export to .npy files
	utils/
		metrics.py                # Metrics functions (extendable)
	data/
		QTDatabase.pkl            # Source beats (already prepared)
		CombinedNoise_Train.pkl   # Noise for training augmentation
		CombinedNoise_Test.pkl    # Noise for test augmentation
	environment.yml             # Conda environment spec
	README.md                   # This document
	docs/
		TECH_TRANSFER.md          # Detailed technology transfer & operations guide
```

## 3. Environment Setup
```bash
conda env create -f environment.yml
conda activate ecgdenoise   # or your chosen env name
```
If you rebuild QTDatabase from raw PhysioNet files, install `wfdb` (already listed) and download QT DB under `data/qt-database-1.0.0/` then run `Prepare_QTDatabase.py`.

## 4. Data Preparation Pipeline
`data_preparation_with_fourier.py` steps:
1. Load `QTDatabase.pkl` (dict: signal_id -> list[beat])
2. Select test signal set (predefined list) → split into train/test beats.
3. Add noise segments sliced sequentially from `CombinedNoise_Train.pkl` / `CombinedNoise_Test.pkl` (wrapping when index overflows).
4. For each noisy beat: compute FFT magnitude of half spectrum, duplicate to length 512.
5. Expand dims to `(N,512,1)` for both time (`X_*`) and frequency (`F_*`).
6. Return dataset list:
	 `[X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y]`.

Re-export (optional):
```bash
python tools/export_dual_freqdae_data.py --samples 512 --out-dir exported --reuse-cache
```

## 5. Model Architecture (High-Level)
Time branch (encoder):
```
Conv1D(16, stride 2) -> gated pair -> BN
Conv1D(32, stride 2) -> gated pair -> BN
Conv1D(64, stride 2) -> gated pair -> BN
```
Frequency branch: mirrored multi-scale Conv1D gated stack producing tensor f2.
Fusion:
```
concat([time_latent, freq_latent]) + positional encoding
for block in range(num_transformer_blocks):
		FANformer_encoder (MHA + FAN layers + residual)
```
Decoder (residual upsampling):
```
Conv1DTranspose(64) + skip
Conv1DTranspose(32) + skip
Conv1DTranspose(16) + skip
Conv1DTranspose(1)  -> output (512 samples)
```
Activation mix: linear + sigmoid gating; `AddGatedNoise` can inject training-time multiplicative noise for robustness.

## 6. Training
```bash
python tools/train_dual_freqdae.py \
	--exp-dir experiments/run1 \
	--data-prep-samples 512 \
	--epochs 100000 \
	--patience 10 \
	--min-delta 0.05 \
	--reuse-cache
```
Key callbacks: EarlyStopping, ReduceLROnPlateau (factor=0.5, min_delta=0.05), ModelCheckpoint(best by `val_loss`), TensorBoard.
Optimizer: Adam(lr=1e-3). Loss: custom combined SSD/MAD style (`combined_ssd_mad_loss`) as originally used; can be swapped.

### Dry Run
```bash
python tools/train_dual_freqdae.py --dry-run --exp-dir experiments/dry
```
Prints model summary and exits.

## 7. Evaluation Metrics
Post-training evaluation computes per-beat vectors:
- RMSE (derived from MSE): `sqrt(mean((y - y_pred)^2))`
- PRD: Percentage Root-mean-square Difference
- COS_SIM: Cosine similarity (averaged)
- SNR (dB): `10 * log10(signal_power / noise_power)`

Aggregate statistics saved to `metrics_summary.json`; per-sample data to `metrics_per_sample.csv`.

## 8. Inference
Pretrained weights path: `experiments/run1/Dual_FreqDAE_weights.best.weights.h5` (or legacy `0221_FIXED/...`).

Beat-level inference:
```bash
python tools/run_dual_freqdae.py \
	--input exported/X_test.npy \
	--weights experiments/run1/Dual_FreqDAE_weights.best.weights.h5 \
	--output denoised_X_test.npy
```
Continuous 1D signal segmentation:
```bash
python tools/run_dual_freqdae.py \
	--input long_signal.npy \
	--segment --hop 512 --pad \
	--weights experiments/run1/Dual_FreqDAE_weights.best.weights.h5 \
	--output denoised_segments.npy
```

## 9. Extending / Customization
- Transformer depth: modify `num_transformer_blocks` argument in `Dual_FreqDAE()` factory.
- Replace gating/noise: swap `AddGatedNoise` with `ParametricNoiseInjection` in `dl_models.py`.
- Alternative metrics: add functions to `utils/metrics.py` and hook inside `evaluate()` section of `train_dual_freqdae.py`.
- Mixed precision: add `tf.keras.mixed_precision.set_global_policy('mixed_float16')` early in `train_dual_freqdae.py`.

## 10. Performance Notes
- Typical early stopping occurs << 100000 epochs (val_loss stabilizes; LR reduces to ~1e-6–1e-7).
- GPU memory: Dual_FreqDAE ~3.1M parameters; fits easily in consumer GPUs (test time ~16s for 13k beats on RTX 4090 pair).

## 11. Known Limitations
- Beat segmentation assumes QTDatabase beat boundaries; performance may degrade on raw unsegmented Holter data.
- Frequency branch duplicates half-spectrum magnitude; phase information discarded (could be extended).
