"""Common MNIST utilities + unified classifier evaluation.

This module provides:
- get_mnist_loaders(): MNIST -> (train_loader, test_loader), padded to 32x32.
- eval_classifier(): train/eval a linear classifier head on features produced by a feature extractor.

Design goal:
- The evaluation/training code must not depend on how features are produced.
- A feature extractor must implement get_features(x) -> tensor (B, ...).

You can run eval_classifier in different regimes:
- frozen=True: classic linear probe (no fine-tuning of feature extractor)
- frozen=False: end-to-end supervised training (fine-tuning feature extractor + head)

Requires: torch, torchvision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Best-effort reproducibility helper."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    dataset: str = "mnist",
    seed: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST-like datasets.

    dataset:
      - "mnist": torchvision.datasets.MNIST
      - "a-mnist": HuggingFace dataset gorar/A-MNIST

    All images are padded to 32x32 and converted to float tensors in [0,1].
    """

    from torchvision import transforms

    # Note: transforms.Pad expects PIL image for grayscale; numpy arrays may error.
    # We therefore convert to tensor first, then pad.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),  # 28x28 -> 32x32
    ])

    ds = dataset.lower()

    if ds == "mnist":
        from torchvision import datasets

        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    elif ds in ("a-mnist", "amnist", "a_mnist"):
        # HuggingFace datasets recently dropped support for script-based datasets.
        # A-MNIST is script-based, but the underlying data are just MNIST-style IDX gz files.
        # So we download the 4 gz files and parse them directly.
        import gzip
        import struct
        import urllib.request

        def _download(url: str, out_path: str):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return
            urllib.request.urlretrieve(url, out_path)

        def _parse_images_gz(path: str):
            with gzip.open(path, "rb") as f:
                magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError(f"Bad magic for images: {magic}")
                buf = f.read(n * rows * cols)
            import numpy as np

            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
            return arr

        def _parse_labels_gz(path: str):
            with gzip.open(path, "rb") as f:
                magic, n = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError(f"Bad magic for labels: {magic}")
                buf = f.read(n)
            import numpy as np

            arr = np.frombuffer(buf, dtype=np.uint8)
            return arr

        base = "https://huggingface.co/datasets/gorar/A-MNIST/resolve/main/"
        files = {
            "train_images": "data/train-images-idx3-ubyte.gz",
            "train_labels": "data/train-labels-idx1-ubyte.gz",
            "test_images": "data/t10k-images-idx3-ubyte.gz",
            "test_labels": "data/t10k-labels-idx1-ubyte.gz",
        }

        cache_dir = os.path.join(data_dir, "a_mnist")
        local = {}
        for k, rel in files.items():
            out_path = os.path.join(cache_dir, os.path.basename(rel))
            _download(base + rel, out_path)
            local[k] = out_path

        train_images = _parse_images_gz(local["train_images"])
        train_labels = _parse_labels_gz(local["train_labels"])
        test_images = _parse_images_gz(local["test_images"])
        test_labels = _parse_labels_gz(local["test_labels"])

        class _AMNIST(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                n_img = len(images)
                n_lab = len(labels)

                if n_img != n_lab:
                    # A-MNIST train split is often stored as 120k images but only 60k labels
                    # (labels correspond to original MNIST and should be repeated for augmented copies).
                    if n_lab > 0 and (n_img % n_lab == 0):
                        rep = n_img // n_lab
                        print(f"[A-MNIST] Info: image/label mismatch: images={n_img} labels={n_lab}. Repeating labels x{rep}.")
                        import numpy as np

                        labels = np.tile(labels, rep)
                    else:
                        n = min(n_img, n_lab)
                        print(
                            f"[A-MNIST] Warning: image/label mismatch: images={n_img} labels={n_lab}. Truncating to {n}."
                        )
                        images = images[:n]
                        labels = labels[:n]

                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                img = self.images[idx]  # uint8 HxW
                # Ensure array is writable to avoid torchvision warning
                img = img.copy()
                x = tfm(img)
                y = int(self.labels[idx])
                return x, y

        train_ds = _AMNIST(train_images, train_labels)
        test_ds = _AMNIST(test_images, test_labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Reproducible shuffling / workers
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def _wif(worker_id: int):
            import random

            s = int(seed) + worker_id
            random.seed(s)
            try:
                import numpy as np

                np.random.seed(s)
            except Exception:
                pass
            torch.manual_seed(s)

        worker_init_fn = _wif

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


@dataclass
class ClassifierEvalCfg:
    epochs: int = 10
    lr: float = 1e-3
    wd: float = 1e-4
    num_classes: int = 10
    print_every_epoch: bool = True
    amp: bool = False  # optional autocast


def corrupt_batch(x: torch.Tensor, mode: str = "mask", max_frac: float = 0.25, noise_std: float = 0.2) -> torch.Tensor:
    if mode == "noise":
        return torch.clamp(x + torch.randn_like(x) * noise_std, 0.0, 1.0)

    if mode != "mask":
        raise ValueError(f"Unknown corruption mode: {mode}")

    B, C, H, W = x.shape
    x2 = x.clone()

    max_area = int(max_frac * H * W)
    max_side = int(max(1, (max_area ** 0.5)))

    for i in range(B):
        side = int(torch.randint(1, max_side + 1, (1,), device=x.device).item())
        top = int(torch.randint(0, H - side + 1, (1,), device=x.device).item())
        left = int(torch.randint(0, W - side + 1, (1,), device=x.device).item())
        x2[i, :, top:top + side, left:left + side] = 0.0

    return x2


def eval_classifier(
    feature_extractor: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: ClassifierEvalCfg = ClassifierEvalCfg(),
    frozen: bool = True,
) -> float:
    """Train a classifier on top of a feature extractor.

    feature_extractor: nn.Module implementing get_features(x)->(B, ...)

    frozen=True  => train only classifier head (linear probe)
    frozen=False => train feature_extractor + head end-to-end

    Returns: final test accuracy.
    """

    if not hasattr(feature_extractor, "get_features"):
        raise TypeError("feature_extractor must implement get_features(x)")

    feature_extractor.to(device)

    # freeze/unfreeze
    feature_extractor.train(not frozen)
    for p in feature_extractor.parameters():
        p.requires_grad_(not frozen)

    # infer feat dim
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        z0 = feature_extractor.get_features(x0.to(device)).flatten(start_dim=1)
        feat_dim = int(z0.shape[1])

    head = nn.Linear(feat_dim, cfg.num_classes).to(device)

    params = list(head.parameters()) + ([] if frozen else list(feature_extractor.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)

    scaler = torch.amp.GradScaler(device=device, enabled=cfg.amp and device.type == "cuda")

    @torch.no_grad()
    def acc(loader: DataLoader) -> float:
        feature_extractor.eval()
        head.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z = feature_extractor.get_features(x).flatten(start_dim=1)
            logits = head(z)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        return correct / max(1, total)

    acc_te_best = 0
    for ep in range(1, cfg.epochs + 1):
        feature_extractor.train(not frozen)
        head.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(enabled=scaler.is_enabled(), device_type=device.type):
                z = feature_extractor.get_features(x).flatten(start_dim=1)
                logits = head(z)
                loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        if cfg.print_every_epoch:
            acc_tr = acc(train_loader)
            acc_te = acc(test_loader)
            if acc_te > acc_te_best:
                acc_te_best = acc_te
            mode = "probe" if frozen else "finetune"
            print(f"[{mode}] epoch {ep:03d} | acc/train={acc_tr:.4f} | acc/test={acc_te:.4f}")
    if cfg.print_every_epoch:
        print("Best test accuracy: ", acc_te_best)
    return acc(test_loader)
