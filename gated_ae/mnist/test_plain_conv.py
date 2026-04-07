"""Plain supervised conv baseline on MNIST.

No aeblock, no JEPA, no pretraining.
Just a small conv feature extractor + linear head trained end-to-end via eval_classifier.

Usage:
  python3 test_plain_conv.py --epochs 10

Requires: torch, torchvision.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common_mnist import ClassifierEvalCfg, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402


class PlainConvNet(nn.Module):
    """A simple conv feature extractor with get_features()."""

    def __init__(self, in_channels: int = 1, channels=(16, 32, 64), kernerl_size=3, stride=1, padding=1, nonlinearity: str = "relu"):
        super().__init__()
        act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }[nonlinearity]

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernerl_size, stride=stride, padding=padding, bias=False),
                act,
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"], help="dataset source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh"])
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, test_loader = get_mnist_loaders(os.path.join(HERE, "_data"), args.batch_size, dataset=args.dataset, seed=args.seed)

    feat = PlainConvNet(in_channels=1, channels=tuple(int(x) for x in args.channels.split(",") if x.strip()), nonlinearity=args.nonlinearity)

    cfg = ClassifierEvalCfg(epochs=args.epochs, lr=args.lr, wd=args.wd, print_every_epoch=True)
    acc = eval_classifier(feat, train_loader, test_loader, device, cfg=cfg, frozen=False)
    print(f"final acc: {acc:.4f}")


if __name__ == "__main__":
    main()
