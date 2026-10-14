"""MobileNetV3-Small definition and checkpoint I/O."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARCH = "mobilenet_v3_small"
CHECKPOINT_FORMAT = 1


class ModelLoadError(RuntimeError):
    pass


def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights, dropout=dropout)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta = json.loads(json.dumps(meta, default=str))  # plain types only, so weights_only=True loading works
    torch.save({"format": CHECKPOINT_FORMAT, "arch": ARCH, "state_dict": state, "meta": meta}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"model file not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"could not read model file {path.name}: {exc}") from exc
    if ckpt.get("format") != CHECKPOINT_FORMAT or ckpt.get("arch") != ARCH:
        raise ModelLoadError(f"{path.name} is not a LeafScan {ARCH} checkpoint")
    meta = ckpt["meta"]
    model = build_model(len(meta["class_keys"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
