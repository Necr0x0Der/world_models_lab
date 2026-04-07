"""JEPA-style self-supervised pretraining on MNIST (conv student/teacher + predictor).

Modular version using:
- common_mnist.get_mnist_loaders, common_mnist.eval_classifier
- common_conv.ConvEncoder

Training:
- Student encoder + predictor are optimized to match teacher encoder latents.
- Teacher is EMA of student.
- No reconstruction loss.

Evaluation:
- Linear probe (frozen feature extractor) using eval_classifier(..., frozen=True).

Usage:
  python3 test_jepa.py --epochs-jepa 20 --probe-epochs 10

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

from common_mnist import ClassifierEvalCfg, corrupt_batch, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402
from common_conv import ConvEncoder, ConvPredictor, ConvStackCfg  # noqa: E402


class ConvJEPA(nn.Module):
    def __init__(self, cfg: ConvStackCfg, pred_hidden: int | None = None):
        super().__init__()
        # One can try with UnitKernelConv2d
        # from common_conv import UnitKernelConv2d
        self.student = ConvEncoder(nn.Conv2d, cfg) # UnitKernelConv2d
        self.teacher = ConvEncoder(nn.Conv2d, cfg) # UnitKernelConv2d
        self.predictor = ConvPredictor(channels=cfg.channels[-1], hidden=pred_hidden, nonlinearity=cfg.nonlinearity,
                                       kernel_size=3,padding=1)
        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        self.teacher.load_state_dict(self.student.state_dict())

    @torch.no_grad()
    def ema_update(self, tau: float):
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.mul_(tau).add_(p_s.data, alpha=1 - tau)

    def forward(self, x: torch.Tensor, x_cor: torch.Tensor):
        z_s = self.student(x)
        z_hat = self.predictor(z_s)
        with torch.no_grad():
            z_t = self.teacher(x_cor)
        return z_hat, z_t


def train_jepa(
    model: ConvJEPA,
    train_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    tau: float,
    corrupt_mode: str,
    corrupt_max_frac: float,
    corrupt_noise_std: float,
):
    model.to(device)
    opt = torch.optim.AdamW(list(model.student.parameters()) + list(model.predictor.parameters()), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        mse_avg = 0.0
        n = 0
        for x, _y in train_loader:
            x = x.to(device)
            x_cor = corrupt_batch(x, mode=corrupt_mode, max_frac=corrupt_max_frac, noise_std=corrupt_noise_std)
            z_hat, z_t = model(x, x_cor)
            loss = F.mse_loss(z_hat, z_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.ema_update(tau=tau)

            mse_avg += float(loss.item())
            n += 1

        print(f"[JEPA] epoch {ep:03d} | mse={mse_avg / max(1, n):.6f}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"], help="dataset source")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--epochs-jepa", type=int, default=15)
    ap.add_argument("--lr-jepa", type=float, default=4e-4)
    ap.add_argument("--tau", type=float, default=0.99)

    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])
    ap.add_argument("--pred-hidden", type=int, default=0)

    ap.add_argument("--corrupt-mode", type=str, default="mask", choices=["mask", "noise"])
    ap.add_argument("--corrupt-max-frac", type=float, default=0.25)
    ap.add_argument("--corrupt-noise-std", type=float, default=0.2)

    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("JEPA")
    print("device:", device)

    cfg = ConvStackCfg(
        in_channels=1,
        channels=tuple(int(x) for x in args.channels.split(",") if x.strip()),
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        nonlinearity=args.nonlinearity,
    )
    print("config: ", cfg)

    data_dir = os.path.join(HERE, "_data")
    train_loader, test_loader = get_mnist_loaders(data_dir, args.batch_size, dataset=args.dataset, seed=args.seed)

    pred_hidden = args.pred_hidden if args.pred_hidden > 0 else None
    model = ConvJEPA(cfg, pred_hidden=pred_hidden)

    train_jepa(
        model,
        train_loader,
        device,
        epochs=args.epochs_jepa,
        lr=args.lr_jepa,
        tau=args.tau,
        corrupt_mode=args.corrupt_mode,
        corrupt_max_frac=args.corrupt_max_frac,
        corrupt_noise_std=args.corrupt_noise_std,
    )

    # linear probe using unified eval_classifier
    probe_cfg = ClassifierEvalCfg(epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, print_every_epoch=True)
    acc = eval_classifier(model.student, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"last linear probe acc: {acc:.4f}")


if __name__ == "__main__":
    main()
