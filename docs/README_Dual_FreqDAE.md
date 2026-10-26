# Running Dual_FreqDAE

This guide covers the **root implementation** in `deepFilter/`, `Data_Preparation/`, and `tools/`. Run every command from the repository root. For the model diagram, data construction, loss, and exact metric definitions, see [Model and data flow](MODEL_AND_DATA.md).

## 1. Environment and prerequisites

Use the existing environment in this workspace:

```bash
conda activate ECGDENOISE
python -c "import sys, numpy, scipy, tensorflow as tf, keras; print(sys.version); print('NumPy', numpy.__version__, 'SciPy', scipy.__version__, 'TensorFlow', tf.__version__, 'Keras', keras.__version__)"
```

The environment name is case-sensitive. The command checks actual imports, which can differ from package-manager metadata.

| Specification | Role |
| --- | --- |
| `environment_full.yml` | Snapshot of the newer `ECGDENOISE` environment; Python 3.12 / TensorFlow 2.18 / Keras 3.7 entries |
| `environment.yml` | Historical `DeepFilter` environment; Python 3.7 and older TensorFlow entries |

For a separate installation, the exported specification can be used as a starting point:

```bash
conda env create -n ecgdenoise_rebuild -f environment_full.yml
conda activate ecgdenoise_rebuild
```

A fresh installation was not tested. The modern CLI uses Python 3.10+ features, so the Python 3.7 snapshot is not its setup recipe. The installed newer environment emits a NumPy/SciPy version-range warning; the full export also contains conflicting NumPy records. Resolve that environment inconsistency before treating the snapshot as a reproducible dependency lock.

List the actual command options without preparing data or running a model:

```bash
python tools/run_dual_freqdae.py --help
python tools/train_dual_freqdae.py --help
python tools/export_dual_freqdae_data.py --help
```

## 2. Quick inference with existing weights

This generates one synthetic input, loads the existing checkpoint, and saves a denoised array. It does not train a model or measure paper performance.

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/run_dual_freqdae.py \
  --demo --demo-batch 1 --batch-size 1 \
  --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5 \
  --output results/demo_dual_freqdae.npy
```

`CUDA_VISIBLE_DEVICES=-1` selects CPU execution. Omit it to allow TensorFlow's configured GPU devices. `--demo` changes the input source; it still requires a checkpoint.

Inspect the output:

```bash
python - <<'PY'
import numpy as np
result = np.load('results/demo_dual_freqdae.npy')
print('shape:', result.shape, 'dtype:', result.dtype)
print('all finite:', np.isfinite(result).all())
PY
```

Expected shape: `(1, 512)`, dtype: `float32`. The checkpoint above is the one tested for the root runner. Do not assume every similarly named checkpoint or the independent `dual_freqdae_repo/` model has the same architecture.

## 3. Input data and preparation

Training and dataset export require these existing files:

| File | Contents |
| --- | --- |
| `data/QTDatabase.pkl` | Record-to-beat dictionary prepared at 360 Hz |
| `data/CombinedNoise_Train.pkl` | One-dimensional training noise sequence |
| `data/CombinedNoise_Test.pkl` | One-dimensional test noise sequence |

Check their presence without loading the dataset:

```bash
python - <<'PY'
from pathlib import Path
for name in ('QTDatabase.pkl', 'CombinedNoise_Train.pkl', 'CombinedNoise_Test.pkl'):
    path = Path('data') / name
    assert path.is_file(), f'Missing: {path}'
    print(path, path.stat().st_size, 'bytes')
PY
```

If rebuilding the QT dictionary, place the raw records and annotations under `data/qt-database-1.0.0/`, then call the preparation function explicitly:

```bash
python - <<'PY'
from pathlib import Path
from Data_Preparation.Prepare_QTDatabase import prepare
assert any(Path('data/qt-database-1.0.0').glob('*.dat')), 'Raw QT records are missing'
prepare()
PY
```

This **overwrites `data/QTDatabase.pkl`**. Running `python Data_Preparation/Prepare_QTDatabase.py` alone only defines the function and does not prepare data.

The train/test noise-building workflow is in [qtdb.ipynb](../notebooks/qtdb.ipynb), around `generate_combined_noise()` and the prerequisite `*_Noise_SNR_*.pkl` generation cells. Review those cells and their settings before regenerating accepted-experiment inputs. `Prepare_NSTDB.py` alone does not generate the required train/test pickle pair. This repository does not provide a single complete raw-data regeneration command.

The preparation function centers each beat by its endpoint average, pads it to 512 samples, adds stored noise, and creates the Fourier input. These steps are detailed in [Model and data flow](MODEL_AND_DATA.md#4-training-data-and-split).

## 4. Export arrays

Use a new output directory to preserve existing exports:

```bash
python tools/export_dual_freqdae_data.py \
  --samples 512 \
  --out-dir exported/dual_freqdae_new_run \
  --reuse-cache
```

It creates eight arrays of shape `(N, 512, 1)`:

```text
X_train.npy    y_train.npy    X_test.npy    y_test.npy
F_train_x.npy  F_train_y.npy  F_test_x.npy  F_test_y.npy
cache/dual_freqdae_dataset_512.npz
```

`X` is noisy time-domain input, `y` is clean time-domain reference, and `F` is the duplicated FFT-magnitude representation. The model consumes `[X, F_x]` and predicts `y`; it does not predict `F_y`.

Both export and training rebuild data on a cache miss, even with `--reuse-cache`. Rebuilding also writes the legacy root file `rnd_test.npy`. Cache reuse checks existence only: after changing input pickles or preprocessing, use a fresh experiment/export directory.

## 5. Train, resume, and evaluate

### Train in a new directory

Keep the same `ECG_RUN_DIR` value for subsequent commands in this section:

```bash
ECG_RUN_DIR=experiments/dual_freqdae_new_run
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" \
  --data-prep-samples 512 \
  --epochs 100000 \
  --patience 10 \
  --min-delta 0.05 \
  --reuse-cache
```

This trains and then evaluates the best validation-loss checkpoint. The current script fixes batch size at 128 and initial Adam learning rate at `1e-3`; these are not CLI flags. The 100,000 epochs are a cap, with early stopping enabled. See [the training objective](MODEL_AND_DATA.md#5-tensor-contracts-and-objective) for the exact loss.

For a preparation/model-build check:

```bash
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" --reuse-cache --dry-run
```

`--dry-run` still prepares or loads the **full dataset**, writes a cache on a miss, and builds the model. It is not a lightweight replacement for `--help`.

### Resume from weights

```bash
test -f "$ECG_RUN_DIR/Dual_FreqDAE_weights.best.weights.h5" && \
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" --reuse-cache --resume
```

This restores model weights, not optimizer state or epoch/history progress. Without the file check, `--resume` silently starts fresh when no checkpoint exists. The run updates the checkpoint and replaces `history.json`.

### Evaluate the trained checkpoint

```bash
test -f "$ECG_RUN_DIR/Dual_FreqDAE_weights.best.weights.h5" && \
python tools/train_dual_freqdae.py \
  --exp-dir "$ECG_RUN_DIR" --reuse-cache --evaluate-only
```

Keep the file check: the script otherwise evaluates **randomly initialized weights** when the checkpoint is absent. `--no-train` is an alias for `--evaluate-only`. Evaluation overwrites the metric files in the run directory.

To evaluate another checkpoint, use a separate run directory containing that file with the exact basename `Dual_FreqDAE_weights.best.weights.h5`, then point `--exp-dir` there. There is no separate `--weights` option on the training CLI. `tools/eval_dual_freqdae.py` is an empty legacy placeholder and is not an evaluator.

### Outputs

| Run-relative path | Contents |
| --- | --- |
| `Dual_FreqDAE_weights.best.weights.h5` | Best checkpoint by validation loss |
| `history.json` | Loss and metric history from the latest training invocation |
| `metrics_summary.json` | Test count and mean/std of RMSE, PRD, cosine similarity, and output SNR |
| `metrics_per_sample.csv` | Per-beat metrics |
| `cache/dual_freqdae_dataset_512.npz` | Prepared dataset for this run |
| `runs_MMDD_Dual_FreqDAE/` | TensorBoard logs |

```bash
python -m json.tool "$ECG_RUN_DIR/metrics_summary.json"
tensorboard --logdir "$ECG_RUN_DIR"
```

Training/evaluation does not save prediction arrays; use the inference command below for those. SNR is output reconstruction SNR, not input-to-output improvement. PRD follows the repository's particular formula. The CSV currently serializes RMSE, PRD, and SNR as bracketed one-element arrays; the JSON summary contains scalar values. See [metric definitions](MODEL_AND_DATA.md#6-metrics-and-checkpoint-scope).

## 6. Infer on exported beats or a long signal

### Existing exported test beats

```bash
python tools/run_dual_freqdae.py \
  --input exported/dual_freqdae_ds/X_test.npy \
  --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5 \
  --batch-size 64 \
  --output results/denoised_X_test.npy
```

For a new export, change `--input` to `exported/dual_freqdae_new_run/X_test.npy`. For your own trained model, change `--weights` to its checkpoint.

| Input | Behavior |
| --- | --- |
| `(512,)` | One beat |
| `(N, 512)` | Batch of beats |
| `(N, 512, 1)` | Batch with a singleton channel dimension |
| Directory of `.npy` files | Reads every `.npy` in sorted order and combines them |

The output is always `(N, 512)` `float32`. Point the runner at an individual input file, not the full export directory: that directory also contains clean targets and Fourier arrays.

The runner casts and reshapes input and constructs FFT features. It does **not** resample, normalize amplitude, remove baseline, or detect beats. Prepare recordings at the expected sampling rate and in the same signal representation as the trained data. `--fs 360` is used only for an unused frequency grid; changing it does not resample the signal.

### Segment one continuous one-dimensional array

Replace `path/to/long_signal_360hz.npy` with your prepared recording:

```bash
python tools/run_dual_freqdae.py \
  --input path/to/long_signal_360hz.npy \
  --segment --hop 256 --pad \
  --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5 \
  --output results/denoised_windows.npy
```

This creates overlapping 512-sample windows, pads the final window, and returns independent denoised windows. It does not reconstruct a continuous trace with overlap-add. `--segment` applies only to a one-dimensional input; use a positive hop no larger than 512.

## 7. Analysis and additional workflows

```bash
python -m jupyter lab notebooks/
```

Run a notebook's first setup cell before its analysis cells. See the [notebook index](../notebooks/README.md) for QTDB, MIT-BIH, FAN, classification, and local real-data analyses. These research notebooks contain preparation and training cells; choose the intended analysis section before rerunning historical experiments.

| Workflow | Entry point |
| --- | --- |
| Reviewer baselines | `python run_1221_fixed.py --help` |
| External SimEMG evaluation | `python tools/eval_simemg_compare_synthesis.py --help` |
| Local wet-data comparison | `python tools/run_model_compare_wet.py --help` |
| Local real-data batch processing | `python tools/run_ajou_real_batch_transformer_dae.py --help` |

These are additional workflows with their own dependencies and data requirements, not substitutes for the main Dual_FreqDAE commands. Private recordings, reports, and real-data notebooks remain covered by `.gitignore`.

## 8. Troubleshooting and validation scope

| Symptom | Check |
| --- | --- |
| Missing module or wrong file paths | Activate the intended environment and run from the repository root |
| Missing noise/QT pickle | Complete the prerequisites in section 3 |
| Unexpected input shape | Supply 512-sample beats or use `--segment` on one one-dimensional recording |
| Missing or incompatible checkpoint | Confirm the file and use the matching model implementation; do not substitute a different architecture |
| Results do not reflect changed data | Use a fresh cache/run directory; cache reuse does not validate source files |
| Memory pressure | Reduce inference `--batch-size`; training still uses a fixed batch size of 128 and loads the full dataset |
| NumPy/SciPy version warning | Reconcile the installed package versions; the environment export contains conflicting NumPy records |

On 2026-10-26, the three primary `--help` commands succeeded in the installed `ECGDENOISE` environment. A CPU smoke test loaded `0221_FIXED/Dual_FreqDAE_weights.best.weights.h5`, inferred on one synthetic sample, and saved a finite `(1, 512)` `float32` result to a temporary location. Fresh environment installation, GPU execution, full training, data regeneration, and paper-metric reproduction were not run for this documentation update.

[Project guide](../README.md) · [Model and data flow](MODEL_AND_DATA.md) · [Cleanup record](CLEANUP_2026-10-26.md)
