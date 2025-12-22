from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class FullyGatedDAEConfig:
    q: int = 3
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 100000
    # NOTE: In our wrapper we train with MSE, so a large min_delta (e.g., 0.05)
    # would cause premature early-stopping. Use a small default.
    patience: int = 20
    min_delta: float = 1e-4
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-10


def _import_fgdae():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ext_path = os.path.join(repo_root, "external", "fully_gated_DAE")
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)

    import tensorflow as tf  # noqa: F401
    import dl_models as fg_models  # type: ignore

    return fg_models


def train_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    out_dir: str,
    cfg: Optional[FullyGatedDAEConfig] = None,
) -> np.ndarray:
    """Train FGDAE on (x_train->y_train) and return predictions on x_test.

    Keeps the QTDB+noise preparation unchanged; only trains the model.
    """
    if cfg is None:
        cfg = FullyGatedDAEConfig()

    fg_models = _import_fgdae()

    import tensorflow as tf

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.3, shuffle=True, random_state=1
    )

    model = fg_models.GatedONNDAE(signal_size=x_train.shape[1], q=cfg.q)

    model_path = os.path.join(out_dir, f"FullyGatedDAE_q{cfg.q}_weights.best.weights.h5")

    os.makedirs(out_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=cfg.reduce_lr_factor,
        min_delta=cfg.min_delta,
        mode="min",
        patience=cfg.reduce_lr_patience,
        min_lr=cfg.min_lr,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.min_delta,
        mode="min",
        patience=cfg.patience,
        verbose=1,
        restore_best_weights=False,
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        metrics=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.MAE],
    )

    model.fit(
        x=x_tr,
        y=y_tr,
        validation_data=(x_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.max_epochs,
        verbose=1,
        callbacks=[early_stop, reduce_lr, checkpoint],
    )

    # Load best weights (if any were saved)
    if os.path.exists(model_path):
        model.load_weights(model_path)

    y_pred = model.predict(x_test, batch_size=cfg.batch_size, verbose=1)

    # Ensure (N,512,1)
    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, axis=-1)

    return y_pred
