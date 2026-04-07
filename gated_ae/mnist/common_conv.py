"""Common conv feature extractors for MNIST experiments.

A feature extractor must implement:
  get_features(x) -> tensor (B, ...)

This module provides conv backbones used by AE/JEPA experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConvStackCfg:
    in_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 6
    stride: int = 2
    padding: int = 2
    nonlinearity: str = "relu"  # relu|gelu|sigmoid|tanh


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class UnitKernelConv2d(nn.Module):
    """Conv2d with per-output-channel unit L2-norm kernels (parametrized).

    Weight has shape (out_ch, in_ch, kH, kW). We normalize each kernel
    W[o,:,:,:] to ||.||_2 = 1 at every forward pass.

    Purpose: prevent trivial feature shrinkage via kernel scaling.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = False, eps: float = 1e-8):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.V = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def weight(self) -> torch.Tensor:
        denom = torch.linalg.vector_norm(self.V, dim=(1, 2, 3), keepdim=True) + self.eps
        return self.V / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight(), self.bias, stride=self.stride, padding=self.padding)


class UnitKernelConvTranspose2d(nn.Module):
    """ConvTranspose2d with per-input-channel unit L2-norm kernels.

    Weight has shape (in_ch, out_ch, kH, kW). We normalize each kernel
    W[i,:,:,:] to ||.||_2 = 1.

    Note: this is not a strict inverse; it is a scale-control mechanism.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = False, eps: float = 1e-8):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.V = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def weight(self) -> torch.Tensor:
        denom = torch.linalg.vector_norm(self.V, dim=(1, 2, 3), keepdim=True) + self.eps
        return self.V / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(x, self.weight(), self.bias, stride=self.stride, padding=self.padding)


class ConvEncoder(nn.Module):
    """A simple conv encoder producing a spatial feature map."""

    def __init__(self, conv2d_cls, cfg: ConvStackCfg):
        super().__init__()
        self.cfg = cfg
        blocks = []
        c_in = cfg.in_channels
        for c_out in cfg.channels:
            blocks.append(
                nn.Sequential(
                    conv2d_cls(
                        c_in,
                        c_out,
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=cfg.padding,
                        bias=False,
                    ),
                    _act(cfg.nonlinearity),
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ConvPredictor(nn.Module):
    """Predictor in representation space"""

    def __init__(self, channels: int, hidden: int | None = None, nonlinearity: str = "relu", kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        if hidden is None:
            hidden = max(32, channels)
        act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[nonlinearity if nonlinearity in ("relu", "gelu") else "relu"]
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            act,
            nn.Conv2d(hidden, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder using ConvTranspose2d blocks.

    This is a convenient building block for vanilla autoencoders.

    Note: geometry (kernel/stride/padding) must match the encoder.
    With MNIST padded to 32x32 and the default (k=6,s=2,p=2), shapes match.
    """

    def __init__(self, cfg: ConvStackCfg, out_channels: int | None = None):
        super().__init__()
        if out_channels is None:
            out_channels = cfg.in_channels

        ch = list(cfg.channels)
        blocks = []

        # deconv: c_L -> ... -> c_1
        for i in range(len(ch) - 1, 0, -1):
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch[i],
                        ch[i - 1],
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=cfg.padding,
                        bias=False,
                    ),
                    _act(cfg.nonlinearity),
                )
            )

        # final: c_1 -> image
        blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    ch[0],
                    out_channels,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for b in self.blocks:
            x = b(x)
        return x
