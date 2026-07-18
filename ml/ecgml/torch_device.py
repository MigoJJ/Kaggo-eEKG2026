"""Torch device selection with a real CUDA smoke check."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device: {requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        x = torch.ones(1, device="cuda")
        _ = (x + 1).cpu().item()
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA is visible but unusable; falling back to CPU: {e}")
        return torch.device("cpu")
