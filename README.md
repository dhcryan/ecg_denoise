# ECG Denoising · Dual_FreqDAE

Research workspace for ECG denoising experiments, datasets, trained weights, and results. The 2026-10-26 cleanup groups notebooks, figures, documents, and historical experiments by purpose while retaining research artifacts and saved notebook outputs.

## How it works

The root `Dual_FreqDAE` model reconstructs a clean, single-channel ECG beat from a noisy **512-sample segment**, using data prepared at **360 Hz**.

1. **Two views of the input.** The time branch receives the waveform. The frequency branch receives `abs(FFT(x) / 512)[:256]`, concatenated with a second copy to give 512 features.
2. **Gated convolutional encoders.** Each branch compresses its input through three stages to 64 positions with 64 features. The time encoder also applies multiplicative noise to gate activations during training.
3. **Feature fusion and FAN attention.** Concatenated features have shape `(batch, 64, 128)`. Positional encoding and eight FANformer blocks combine self-attention with learned cosine/sine and GELU features.
4. **Waveform reconstruction.** A transposed-convolution decoder adds matching time-encoder skip features and returns `(batch, 512, 1)` clean-waveform predictions. The CLI saves these as `(batch, 512)` arrays.

The training loss is `sum(error²) + 50 × max(error²)` over the signal samples. The model predicts the time-domain waveform; it does not reconstruct an FFT target. See [Model and data flow](docs/MODEL_AND_DATA.md) for the architecture diagram, tensor shapes, data splits, FAN equations, and metric definitions.

## Quick start

Run these commands from the repository root. They use the existing, case-sensitive `ECGDENOISE` environment and local checkpoint:

```bash
conda activate ECGDENOISE

# One synthetic sample, using existing trained weights on CPU.
CUDA_VISIBLE_DEVICES=-1 python tools/run_dual_freqdae.py \
  --demo --demo-batch 1 --batch-size 1 \
  --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5 \
  --output results/demo_dual_freqdae.npy
```

Expected output: a finite `float32` array of shape `(1, 512)`. This exact inference path passed a CPU smoke test; it is not a paper-performance evaluation.

For training, first provide `data/QTDatabase.pkl`, `data/CombinedNoise_Train.pkl`, and `data/CombinedNoise_Test.pkl`. Use a new run directory to preserve previous results:

```bash
ECG_RUN_DIR=experiments/dual_freqdae_new_run
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" --data-prep-samples 512 \
  --epochs 100000 --patience 10 --min-delta 0.05 --reuse-cache

# Evaluate the run's checkpoint; the file check prevents random-weight evaluation.
test -f "$ECG_RUN_DIR/Dual_FreqDAE_weights.best.weights.h5" && \
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" --reuse-cache --evaluate-only
```

Training writes the best checkpoint, history, test metrics, a dataset cache, and TensorBoard logs into the run directory. The script also evaluates automatically after training; the second command is for a later evaluation-only run. Training was not rerun during this documentation update.

Follow [the execution guide](docs/README_Dual_FreqDAE.md) for environment setup, data preparation, array export, resume, inference on saved arrays or long signals, output files, and troubleshooting. In particular, `--fs` does not resample recordings, `--segment` returns separate windows, and `--dry-run` still prepares the full dataset.

## Find files

| Location | Contents |
| --- | --- |
| [notebooks/](notebooks/README.md) | 16 notebooks for QTDB, MIT-BIH, FAN, and real-data analysis |
| [figures/](figures/README.md) | Existing `fig*` folders and PSD / R-peak visualizations |
| [paper/](paper/README.md) | Existing manuscript PDF; accepted-version status unconfirmed |
| [docs/MODEL_AND_DATA.md](docs/MODEL_AND_DATA.md) | Architecture diagram, Fourier/FAN features, data flow, loss, and metrics |
| [docs/README_Dual_FreqDAE.md](docs/README_Dual_FreqDAE.md) | Data preparation, training, and inference guide |
| [docs/history/](docs/history/) | Previous README, development notes, and interim reports; local archive |
| [archive/](archive/README.md) | Historical experiments, noise sweeps, logs, and cleanup backups |
| [deliverables/](deliverables/) | Project reports and technology-transfer packages; private local material |
| [reports/](reports/) | Real-data wet/dry evaluation tables and figures; private local material |
| [dual_freqdae_repo/](dual_freqdae_repo/README.md) | Standalone extraction and fusion ablations; differs from the root implementation |

## Main notebooks

- [qtdb.ipynb](notebooks/qtdb.ipynb): QTDB experiments, evaluation, and figures.
- [mitbih.ipynb](notebooks/mitbih.ipynb): MIT-BIH experiments and evaluation.
- [FAN_reviewer_response_experiments.ipynb](notebooks/FAN_reviewer_response_experiments.ipynb): FAN experiments for reviewer responses.
- [Arrhythmia_Classification_Full_and_Final_Code.ipynb](notebooks/Arrhythmia_Classification_Full_and_Final_Code.ipynb): Arrhythmia classification experiments.
- [ajou_ecg_deepfilter_beataug.ipynb](notebooks/ajou_ecg_deepfilter_beataug.ipynb): Ajou real-data ECG processing and analysis; private, local only.
- [wet_model_compare_report.ipynb](notebooks/wet_model_compare_report.ipynb): Wet-data model comparison report; private, local only.

## Code and saved data

See [environment_full.yml](environment_full.yml) for the newer environment snapshot and [environment.yml](environment.yml) for the historical Python 3.7 configuration. The execution guide explains their compatibility limits; the historical snapshot is not the setup recipe for the modern CLI.

Open notebooks from inside the project and **run the first code cell before other cells**. It sets the working directory and Python import path to the project root so existing relative data and weight paths continue to work. Saved outputs are from the previous experiments; training and evaluation were not rerun for this cleanup.

| Location | Role |
| --- | --- |
| [deepFilter/](deepFilter/) | Model definitions and training pipeline |
| [Data_Preparation/](Data_Preparation/) | Dataset preparation |
| [digitalFilters/](digitalFilters/), [utils/](utils/) | Filters, metrics, and visualization |
| [tools/](tools/) | Training, inference, evaluation, and report scripts |
| [baselines_1221_fixed/](baselines_1221_fixed/) | Additional baseline implementations |

Directories referenced by existing code retain their root paths, including `data*`, `0221_FIXED`, `0221_snr*`, `0301_final`, `0302`, `0908`, `1220`, `1221_FIXED`, `exp`, `experiments`, and `ajou_*`. The `tmp` directory contains weights and results and is also retained.

## Local material and cleanup records

Raw participant recordings, real-data outputs and reports, selected notebooks containing sensitive material, and local archival backups are covered by the project's `.gitignore` rules. They remain available locally. Ignore rules do not erase Git history or automatically remove content that is already tracked.

Navigation documents are in English. Historical documents retain their original contents. Saved research values and figures are preserved; personal machine paths were redacted from six public notebooks. Disposable caches and empty files were removed.

See the [cleanup record](docs/CLEANUP_2026-10-26.md) for changes and recovery instructions, and [manifest.json](archive/cleanup_2026-10-26/manifest.json) for the old-to-new path mapping. The original README is preserved at [docs/history/README_before_cleanup.md](docs/history/README_before_cleanup.md).
