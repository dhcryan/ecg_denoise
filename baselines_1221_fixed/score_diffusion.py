from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .common import to_ncl, to_nlc


@dataclass
class ScoreDiffusionConfig:
    feats: int = 80
    epochs: int = 400
    batch_size: int = 96
    lr: float = 1e-3

    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.5
    diffusion_num_steps: int = 50
    diffusion_schedule: str = "quad"

    shots: int = 1

    # Validation / early-stopping (disabled by default to preserve original behavior)
    valid_split: float = 0.1
    valid_epoch_interval: int = 5
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    min_epochs: int = 0


def _import_scorediff():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ext_path = os.path.join(repo_root, "external", "Score-based-ECG-Denoising")
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)

    from denoising_model_small import ConditionalModel  # type: ignore
    from main_model import DDPM  # type: ignore

    return ConditionalModel, DDPM


class _NumpyECGDataset:
    def __init__(self, clean_ncl: np.ndarray, noisy_ncl: np.ndarray):
        self.clean = clean_ncl
        self.noisy = noisy_ncl

    def __len__(self):
        return int(self.clean.shape[0])

    def __getitem__(self, idx: int):
        import torch

        return torch.from_numpy(self.clean[idx]), torch.from_numpy(self.noisy[idx])


def train_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    out_dir: str,
    cfg: Optional[ScoreDiffusionConfig] = None,
    device: str = "cuda:0",
) -> np.ndarray:
    """Train DDPM (conditional score-based diffusion) and return predictions."""

    if cfg is None:
        cfg = ScoreDiffusionConfig()

    ConditionalModel, DDPM = _import_scorediff()

    import torch
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)

    train_noisy = to_ncl(x_train)
    train_clean = to_ncl(y_train)
    test_noisy = to_ncl(x_test)

    # Train/valid split
    n = int(train_noisy.shape[0])
    rng = np.random.RandomState(1)
    perm = rng.permutation(n)

    valid_split = float(cfg.valid_split)
    valid_split = 0.0 if valid_split < 0.0 else valid_split
    valid_split = 0.5 if valid_split > 0.5 else valid_split

    if valid_split <= 0.0 or n < 2:
        tr_idx = perm
        va_idx = None
    else:
        split = int(n * (1.0 - valid_split))
        split = max(1, min(split, n - 1))
        tr_idx, va_idx = perm[:split], perm[split:]

    train_ds = _NumpyECGDataset(train_clean[tr_idx], train_noisy[tr_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    valid_loader = None
    if va_idx is not None and int(va_idx.size) > 0:
        valid_ds = _NumpyECGDataset(train_clean[va_idx], train_noisy[va_idx])
        valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)

    test_ds = _NumpyECGDataset(np.zeros_like(test_noisy), test_noisy)  # clean is unused in denoising()
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    config = {
        "train": {"feats": cfg.feats, "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr},
        "diffusion": {
            "beta_start": cfg.diffusion_beta_start,
            "beta_end": cfg.diffusion_beta_end,
            "num_steps": cfg.diffusion_num_steps,
            "schedule": cfg.diffusion_schedule,
        },
    }

    base_model = ConditionalModel(config["train"]["feats"]).to(device)
    model = DDPM(base_model, config, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    foldername = os.path.join(out_dir, "ScoreDiffusion")
    os.makedirs(foldername, exist_ok=True)
    best_path = os.path.join(foldername, "best.pth")
    final_path = os.path.join(foldername, "final.pth")

    best_valid_loss = float("inf")
    wait = 0
    valid_epoch_interval = max(int(cfg.valid_epoch_interval), 1)
    patience = max(int(cfg.early_stop_patience), 0)
    min_delta = float(cfg.early_stop_min_delta)
    min_epochs = max(int(cfg.min_epochs), 0)

    for epoch_no in range(int(config["train"]["epochs"])):
        model.train()
        avg_loss = 0.0
        n_batches = 0

        for clean_batch, noisy_batch in train_loader:
            clean_batch = clean_batch.to(device)
            noisy_batch = noisy_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(clean_batch, noisy_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()

            avg_loss += float(loss.item())
            n_batches += 1

        lr_scheduler.step()
        print(
            f"[ScoreDiffusion] epoch {epoch_no+1}/{config['train']['epochs']} "
            f"train_loss={avg_loss / max(n_batches, 1):.6f}"
        )

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            n_val = 0
            with torch.no_grad():
                for clean_batch, noisy_batch in valid_loader:
                    clean_batch = clean_batch.to(device)
                    noisy_batch = noisy_batch.to(device)
                    loss_valid = model(clean_batch, noisy_batch)
                    avg_loss_valid += float(loss_valid.item())
                    n_val += 1

            valid_loss = avg_loss_valid / max(n_val, 1)
            print(f"[ScoreDiffusion] epoch {epoch_no+1} valid_loss={valid_loss:.6f}")

            if valid_loss < (best_valid_loss - min_delta):
                best_valid_loss = valid_loss
                wait = 0
                torch.save(model.state_dict(), best_path)
                print(f"[ScoreDiffusion] best updated: {best_valid_loss:.6f}")
            else:
                wait += 1
                print(f"[ScoreDiffusion] no-improve count: {wait}/{patience}")
                if patience > 0 and wait >= patience and (epoch_no + 1) >= min_epochs:
                    print(
                        f"[ScoreDiffusion] Early stopping at epoch {epoch_no+1} "
                        f"(best_valid_loss={best_valid_loss:.6f}, min_delta={min_delta})."
                    )
                    break

    torch.save(model.state_dict(), final_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()
    preds = []
    with torch.no_grad():
        for _clean_dummy, noisy_batch in test_loader:
            noisy_batch = noisy_batch.to(device)
            if int(cfg.shots) > 1:
                out = 0
                for _ in range(int(cfg.shots)):
                    out = out + model.denoising(noisy_batch)
                out = out / float(cfg.shots)
            else:
                out = model.denoising(noisy_batch)
            preds.append(out.detach().cpu().numpy())

    preds_ncl = np.concatenate(preds, axis=0)
    return to_nlc(preds_ncl)
