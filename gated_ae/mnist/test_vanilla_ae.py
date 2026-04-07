"""Vanilla convolutional autoencoder pretraining + linear probe on MNIST.

No aeblock usage.

Pipeline:
1) Pretrain an autoencoder (one deep encoder + one deep decoder) with reconstruction loss.
2) Freeze encoder and evaluate representation quality with eval_classifier(..., frozen=True).

Usage:
  python3 test_vanilla_ae.py --epochs-ae 20 --probe-epochs 10

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

from common_mnist import ClassifierEvalCfg, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402
from common_conv import ConvDecoder, ConvEncoder, ConvStackCfg  # noqa: E402


class VanillaAE(nn.Module):
    def __init__(self, cfg: ConvStackCfg):
        super().__init__()
        self.encoder = ConvEncoder(nn.Conv2d, cfg)
        self.decoder = ConvDecoder(cfg, out_channels=cfg.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_features(x)


def train_autoencoder(ae: VanillaAE, train_loader, test_loader, device: torch.device, epochs: int, lr: float, wd: float):
    ae.to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def eval_recon(loader) -> float:
        ae.eval()
        loss_sum = 0.0
        n = 0
        for x, _y in loader:
            x = x.to(device)
            x_hat = ae(x)
            if x_hat.shape != x.shape:
                raise RuntimeError(f"shape mismatch: x_hat={tuple(x_hat.shape)} vs x={tuple(x.shape)}")
            loss_sum += float(F.binary_cross_entropy(x_hat, x, reduction="sum").item())
            n += int(x.numel())
        return loss_sum / max(1, n)

    for ep in range(1, epochs + 1):
        ae.train()
        for x, _y in train_loader:
            x = x.to(device)
            x_hat = ae(x)
            if x_hat.shape != x.shape:
                raise RuntimeError(f"shape mismatch: x_hat={tuple(x_hat.shape)} vs x={tuple(x.shape)}")
            loss = F.binary_cross_entropy(x_hat, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        tr = eval_recon(train_loader)
        te = eval_recon(test_loader)
        print(f"[AE] epoch {ep:03d} | bce/train={tr:.6f} | bce/test={te:.6f}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"], help="dataset source")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--epochs-ae", type=int, default=20)
    ap.add_argument("--lr-ae", type=float, default=4e-4)
    ap.add_argument("--wd-ae", type=float, default=1e-4)

    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)

    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("basic autoencoders")
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

    ae = VanillaAE(cfg)
    train_autoencoder(ae, train_loader, test_loader, device, epochs=args.epochs_ae, lr=args.lr_ae, wd=args.wd_ae)

    # linear probe on frozen encoder
    probe_cfg = ClassifierEvalCfg(epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, print_every_epoch=True)
    acc = eval_classifier(ae, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"final linear probe acc: {acc:.4f}")


if __name__ == "__main__":
    main()
