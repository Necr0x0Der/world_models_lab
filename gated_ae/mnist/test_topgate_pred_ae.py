"""Autoencoder with unit-norm encoder + *top-level hard top-k gate* + predictor loss.

As requested:
- Encoder: plain conv stack with UnitKernelConv2d (kernel-normalized), NO gates inside.
- Decoder: plain ConvTranspose2d stack (ConvDecoder), reconstruction uses ALL features.
- A separate top-level gate selects k channels from the top latent feature map.
- Predictor sees ONLY gated features; adds a JEPA-like prediction loss.
- No teacher/student EMA.

Loss:
  recon = BCE(dec( z_full(clean) ), clean)
  pred  = MSE( pred( gate(z_full(clean)) ), gate(z_full(corrupt)) )
  total = recon + pred_weight * pred

To make the hard top-k gate trainable, we use a straight-through estimator (STE):
  mask_hard = topk_binary(logits)
  mask_soft = sigmoid(logits / T)
  mask = (mask_hard - mask_soft).detach() + mask_soft

Usage:
  python3 test_topgate_pred_ae.py --epochs 20 --topk 16 --pred-weight 1.0

Requires: torch, torchvision.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common_mnist import ClassifierEvalCfg, eval_classifier, get_mnist_loaders, corrupt_batch, set_seed  # noqa: E402
from common_conv import ConvDecoder, ConvEncoder, ConvPredictor, ConvStackCfg, UnitKernelConv2d  # noqa: E402


class TopKGate(nn.Module):
    """Channel-wise hard top-k gate with STE gradients."""

    def __init__(self, channels: int, k: int, temperature: float = 1.0, init: float = 0.0):
        super().__init__()
        assert 1 <= k <= channels
        self.channels = channels
        self.k = k
        self.temperature = temperature
        self.logits = nn.Parameter(torch.full((channels,), float(init)))

    def mask(self, device=None, dtype=None) -> torch.Tensor:
        logits = self.logits
        if device is not None:
            logits = logits.to(device)
        if dtype is not None:
            logits = logits.to(dtype)

        # hard top-k
        idx = torch.topk(logits, self.k).indices
        hard = torch.zeros_like(logits)
        hard[idx] = 1.0

        # soft for gradients
        soft = torch.sigmoid(logits / float(self.temperature))

        # straight-through
        m = (hard - soft).detach() + soft
        return m

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,C,H,W)
        m = self.mask(device=z.device, dtype=z.dtype)
        return z * m.view(1, -1, 1, 1)


class TopGatePredAE(nn.Module):
    """AE with a separate top-level gate used only for prediction."""

    def __init__(
        self,
        cfg: ConvStackCfg,
        topk: int,
        gate_temperature: float = 1.0,
        gate_init: float = 0.0,
        pred_hidden: int | None = None,
    ):
        super().__init__()
        self.encoder = ConvEncoder(UnitKernelConv2d, cfg) #UnitKernelConv2d
        self.decoder = ConvDecoder(cfg, out_channels=cfg.in_channels)
        self.gate = TopKGate(cfg.channels[-1], k=topk, temperature=gate_temperature, init=gate_init)
        self.predictor = ConvPredictor(cfg.channels[-1], hidden=pred_hidden, nonlinearity=cfg.nonlinearity)

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_gated(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.encode_full(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reconstruction uses ALL features
        z_full = self.encode_full(x)
        return self.decoder(z_full)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # For eval_classifier: by default probe on full top-level features.
        return self.encode_full(x)


def train(
    model: TopGatePredAE,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    pred_weight: float,
    corrupt_mode: str,
    corrupt_max_frac: float,
    corrupt_noise_std: float,
    stopgrad_target: bool,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def eval_recon(loader) -> float:
        model.eval()
        loss_sum = 0.0
        n = 0
        for x, _y in loader:
            x = x.to(device)
            x_hat = model(x)
            if x_hat.shape != x.shape:
                raise RuntimeError(f"shape mismatch: x_hat={tuple(x_hat.shape)} vs x={tuple(x.shape)}")
            loss_sum += float(F.binary_cross_entropy(x_hat, x, reduction="sum").item())
            n += int(x.numel())
        return loss_sum / max(1, n)

    for ep in range(1, epochs + 1):
        model.train()
        recon_avg = 0.0
        pred_avg = 0.0
        n = 0

        for x, _y in train_loader:
            x = x.to(device)
            x_cor = corrupt_batch(x, mode=corrupt_mode, max_frac=corrupt_max_frac, noise_std=corrupt_noise_std)
            # reconstruction (all features)
            x_hat = model(x)
            if x_hat.shape != x.shape:
                raise RuntimeError(f"shape mismatch: x_hat={tuple(x_hat.shape)} vs x={tuple(x.shape)}")
            loss_recon = F.binary_cross_entropy(x_hat, x)

            # prediction (gated features only)
            z_g = model.encode_gated(x)
            z_tgt = model.encode_gated(x_cor)
            if stopgrad_target:
                z_tgt = z_tgt.detach()
            z_hat = model.predictor(z_g)
            loss_pred = F.mse_loss(z_hat, z_tgt)

            loss = loss_recon + pred_weight * loss_pred

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            recon_avg += float(loss_recon.item())
            pred_avg += float(loss_pred.item())
            n += 1

        te = eval_recon(test_loader)
        print(f"epoch {ep:03d} | recon(bce)={recon_avg/max(1,n):.6f} | pred(mse)={pred_avg/max(1,n):.6f} | recon_eval/test={te:.6f}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"], help="dataset source")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])

    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--gate-temperature", type=float, default=1.0)
    ap.add_argument("--gate-init", type=float, default=0.0)

    ap.add_argument("--pred-weight", type=float, default=0.3)
    ap.add_argument("--corrupt-mode", type=str, default="mask", choices=["mask", "noise"])
    ap.add_argument("--corrupt-max-frac", type=float, default=0.25)
    ap.add_argument("--corrupt-noise-std", type=float, default=0.2)
    ap.add_argument("--stopgrad-target", type=int, default=1)

    ap.add_argument("--pred-hidden", type=int, default=1)

    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("k-Gated Predictive Autoencoders")
    print("device:", device)

    data_dir = os.path.join(HERE, "_data")
    train_loader, test_loader = get_mnist_loaders(data_dir, args.batch_size, dataset=args.dataset, seed=args.seed)

    cfg = ConvStackCfg(
        in_channels=1,
        channels=tuple(int(x) for x in args.channels.split(",") if x.strip()),
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        nonlinearity=args.nonlinearity,
    )
    print("config: ", cfg)

    if not (1 <= args.topk <= cfg.channels[-1]):
        raise ValueError(f"topk must be in [1, {cfg.channels[-1]}]")

    pred_hidden = args.pred_hidden if args.pred_hidden > 0 else None

    model = TopGatePredAE(
        cfg,
        topk=args.topk,
        gate_temperature=args.gate_temperature,
        gate_init=args.gate_init,
        pred_hidden=pred_hidden,
    )

    train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        pred_weight=args.pred_weight,
        corrupt_mode=args.corrupt_mode,
        corrupt_max_frac=args.corrupt_max_frac,
        corrupt_noise_std=args.corrupt_noise_std,
        stopgrad_target=bool(args.stopgrad_target),
    )

    probe_cfg = ClassifierEvalCfg(epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, print_every_epoch=True)
    acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"last linear probe acc: {acc:.4f}")


if __name__ == "__main__":
    main()
