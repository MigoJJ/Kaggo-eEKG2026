import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(train_frame):
    labels = torch.as_tensor(train_frame["labels"].tolist(), dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / torch.clamp(positives, min=1.0)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        modulating = (1.0 - pt).pow(self.gamma)
        loss = bce * modulating
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_factor
        return loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pos = probs
        neg = 1.0 - probs
        if self.clip > 0.0:
            neg = torch.clamp(neg + self.clip, max=1.0)
        pos_loss = targets * torch.log(torch.clamp(pos, min=self.eps)) * (1.0 - pos).pow(self.gamma_pos)
        neg_loss = (1.0 - targets) * torch.log(torch.clamp(neg, min=self.eps)) * neg.pow(self.gamma_neg)
        return (-pos_loss - neg_loss).mean()


def build_multilabel_loss(loss_config, pos_weight):
    name = loss_config["name"].lower()
    if name == "weighted_bce":
        return WeightedBCEWithLogitsLoss(
            pos_weight=pos_weight,
            label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
        )
    if name == "focal":
        alpha = pos_weight / torch.clamp(pos_weight + 1.0, min=1.0)
        return FocalLoss(alpha=alpha, gamma=float(loss_config.get("gamma", 2.0)))
    if name == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=float(loss_config.get("gamma_neg", 4.0)),
            gamma_pos=float(loss_config.get("gamma_pos", 1.0)),
            clip=float(loss_config.get("clip", 0.05)),
        )
    raise ValueError(f"Unsupported loss: {name}")
