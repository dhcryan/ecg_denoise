from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .common import to_ncl, to_nlc


@dataclass
class CycleGANConfig:
    epochs: int = 50
    batch_size: int = 32
    # Original CycleGAN commonly uses ~2e-4; 1e-5 was too low and tended to underfit.
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    # Cycle consistency and identity losses (kept for CycleGAN flavor)
    reconstr_w: float = 10.0
    id_w: float = 5.0
    # Our protocol provides paired (noisy, clean) samples; adding supervised L1 helps a lot.
    sup_w: float = 50.0
    seed: int = 42


def _build_models(filter_ch: int = 16):
    import torch
    from torch import nn

    class Downsample(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=5, stride=2, padding=2, apply_instancenorm=True):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")
            self.norm = nn.InstanceNorm1d(out_ch) if apply_instancenorm else nn.Identity()
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

    class Upsample(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=5, stride=2, padding=2, dropout=False):
            super().__init__()
            self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)
            self.norm = nn.InstanceNorm1d(out_ch)
            self.act = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        def forward(self, x, shortcut=None):
            x = self.act(self.norm(self.deconv(x)))
            x = self.dropout(x)
            if shortcut is not None:
                # crop/pad to match if needed
                if x.shape[-1] != shortcut.shape[-1]:
                    min_len = min(x.shape[-1], shortcut.shape[-1])
                    x = x[..., :min_len]
                    shortcut = shortcut[..., :min_len]
                x = torch.cat([x, shortcut], dim=1)
            return x

    class Generator(nn.Module):
        def __init__(self, filt=filter_ch):
            super().__init__()
            self.down = nn.ModuleList([
                Downsample(1, filt, apply_instancenorm=False),
                Downsample(filt, filt * 2),
                Downsample(filt * 2, filt * 4),
                Downsample(filt * 4, filt * 8),
                Downsample(filt * 8, filt * 8),
            ])
            self.up = nn.ModuleList([
                Upsample(filt * 8, filt * 8),
                Upsample(filt * 16, filt * 4),
                Upsample(filt * 8, filt * 2),
                Upsample(filt * 4, filt),
            ])
            self.last = nn.Sequential(
                nn.ConvTranspose1d(filt * 2, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
                # No final tanh: our beats are not explicitly scaled to [-1, 1] in Data_Preparation.
            )

        def forward(self, x):
            skips = []
            for layer in self.down:
                x = layer(x)
                skips.append(x)
            skips = list(reversed(skips[:-1]))
            for layer, s in zip(self.up, skips):
                x = layer(x, s)
            return self.last(x)

    class Discriminator(nn.Module):
        def __init__(self, filt=filter_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(1, filt, kernel_size=4, stride=4, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(filt, filt * 2, kernel_size=4, stride=4, padding=1, padding_mode="reflect"),
                nn.InstanceNorm1d(filt * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(filt * 2, filt * 4, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.InstanceNorm1d(filt * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(filt * 4, filt * 8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.InstanceNorm1d(filt * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(filt * 8, filt * 16, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
                nn.InstanceNorm1d(filt * 16),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.last = nn.Conv1d(filt * 16, 1, kernel_size=4, stride=2, padding=1, padding_mode="reflect")

        def forward(self, x):
            return self.last(self.block(x))

    return Generator(), Generator(), Discriminator(), Discriminator()


class _PairedDataset:
    def __init__(self, noisy_ncl: np.ndarray, clean_ncl: np.ndarray):
        if noisy_ncl.shape != clean_ncl.shape:
            raise ValueError(f"Paired dataset requires same shapes, got {noisy_ncl.shape} vs {clean_ncl.shape}")
        self.noisy = noisy_ncl
        self.clean = clean_ncl

    def __len__(self):
        return self.noisy.shape[0]

    def __getitem__(self, idx):
        import torch

        return torch.from_numpy(self.noisy[idx]), torch.from_numpy(self.clean[idx])


def train_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    out_dir: str,
    cfg: Optional[CycleGANConfig] = None,
    device: str = "cuda:0",
) -> np.ndarray:
    """Train a CycleGAN-style model on noisy<->clean domains and predict denoised outputs.

    Notes:
    - This wrapper uses a lightweight 1D CycleGAN architecture compatible with length-512 beats.
    - If you need the exact Operational/SelfONN layers from the original repo, you will need the
      FastONN dependency and may need to adapt padding/length handling.
    """
    if cfg is None:
        cfg = CycleGANConfig()

    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(out_dir, exist_ok=True)

    base_ncl = to_ncl(x_train)  # noisy domain
    style_ncl = to_ncl(y_train)  # clean domain

    # Our protocol provides paired (noisy, clean) samples. Using paired batches stabilizes training
    # and avoids learning a "random style transfer" that hurts denoising metrics.
    ds = _PairedDataset(base_ncl, style_ncl)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    G_basestyle, G_stylebase, D_base, D_style = _build_models()
    G_basestyle.to(device)
    G_stylebase.to(device)
    D_base.to(device)
    D_style.to(device)

    mae = nn.L1Loss()
    mse = nn.MSELoss()  # LSGAN

    opt_g1 = optim.Adam(G_basestyle.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_g2 = optim.Adam(G_stylebase.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d1 = optim.Adam(D_base.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
    opt_d2 = optim.Adam(D_style.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    for epoch in range(cfg.epochs):
        G_basestyle.train(); G_stylebase.train(); D_base.train(); D_style.train()
        for base_img, style_img in loader:
            base_img = base_img.to(device)
            style_img = style_img.to(device)

            # dynamic target shapes
            valid_base = torch.ones_like(D_base(base_img))
            valid_style = torch.ones_like(D_style(style_img))
            fake_base = torch.zeros_like(valid_base)
            fake_style = torch.zeros_like(valid_style)

            # --- Generators ---
            opt_g1.zero_grad(set_to_none=True)
            opt_g2.zero_grad(set_to_none=True)

            fake_style_img = G_basestyle(base_img)   # noisy -> clean
            fake_base_img = G_stylebase(style_img)   # clean -> noisy

            # adversarial (LSGAN)
            val_base = mse(D_base(fake_base_img), valid_base)
            val_style = mse(D_style(fake_style_img), valid_style)
            val_loss = 0.5 * (val_base + val_style)

            # supervised mapping (paired noisy->clean)
            sup_loss = mae(fake_style_img, style_img)

            # cycle consistency
            recon_base = mae(G_stylebase(fake_style_img), base_img)
            recon_style = mae(G_basestyle(fake_base_img), style_img)
            recon_loss = 0.5 * (recon_base + recon_style)

            # identity
            id_base = mae(G_stylebase(base_img), base_img)
            id_style = mae(G_basestyle(style_img), style_img)
            id_loss = 0.5 * (id_base + id_style)

            g_loss = val_loss + cfg.sup_w * sup_loss + cfg.reconstr_w * recon_loss + cfg.id_w * id_loss
            g_loss.backward()
            opt_g1.step(); opt_g2.step()

            # --- Discriminators ---
            opt_d1.zero_grad(set_to_none=True)
            opt_d2.zero_grad(set_to_none=True)

            # Standard LSGAN discriminator losses
            d_base_fake = mse(D_base(fake_base_img.detach()), fake_base)
            d_base_real = mse(D_base(base_img), valid_base)
            d_style_fake = mse(D_style(fake_style_img.detach()), fake_style)
            d_style_real = mse(D_style(style_img), valid_style)

            d_base_loss = 0.5 * (d_base_real + d_base_fake)
            d_style_loss = 0.5 * (d_style_real + d_style_fake)
            d_loss = 0.5 * (d_base_loss + d_style_loss)
            d_loss.backward()
            opt_d1.step(); opt_d2.step()

        print(f"[CycleGAN_OC] epoch {epoch+1}/{cfg.epochs}  g_loss={g_loss.item():.4f}  d_loss={d_loss.item():.4f}")

    # Save generator weights (noisy->clean)
    gen_path = os.path.join(out_dir, "CycleGAN_OC_G_basestyle.pth")
    torch.save(G_basestyle.state_dict(), gen_path)

    # Predict
    G_basestyle.eval()
    test_noisy = to_ncl(x_test)
    preds = []
    with torch.no_grad():
        for i in range(0, test_noisy.shape[0], cfg.batch_size):
            batch = torch.from_numpy(test_noisy[i : i + cfg.batch_size]).to(device)
            out = G_basestyle(batch)
            preds.append(out.detach().cpu().numpy())

    preds_ncl = np.concatenate(preds, axis=0)
    y_pred = to_nlc(preds_ncl)
    return y_pred
