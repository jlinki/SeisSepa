"""Editable training losses. Labels may supervise latent geometry, never model inputs."""
import torch
from torch import nn
from torch.nn import functional as F


def separation(z, labels, margin):
    z = F.normalize(z, dim=1)
    sim = z @ z.T
    off_diagonal = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & off_diagonal
    negative = labels[:, None].ne(labels[None, :]) & off_diagonal
    zero = z.sum() * 0.0
    return ((1 - sim[positive]).mean() if positive.any() else zero) + (
        F.relu(sim[negative] - margin).mean() if negative.any() else zero)


def supervised_contrastive(z, labels, temperature):
    """Multi-positive supervised contrastive loss for a training mini-batch."""
    z = F.normalize(z, dim=1)
    if len(z) < 2:
        return z.sum() * 0.0
    logits = (z @ z.T) / temperature
    off_diagonal = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(~off_diagonal, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive = labels[:, None].eq(labels[None, :]) & off_diagonal
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return z.sum() * 0.0
    positive_log_prob = log_prob.masked_fill(~positive, 0.0)
    per_anchor = -positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[valid].mean()


class SeparationLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, event_weight=0.05, station_weight=0.05, margin=0.2):
        super().__init__()
        self.weights = reconstruction_weight, event_weight, station_weight
        self.margin = margin

    def forward(self, output, batch):
        reconstruction = F.mse_loss(output["reconstruction"], batch["target"])
        event = separation(output["ze"], batch["event"], self.margin)
        station = separation(output["zs"], batch["station"], self.margin)
        total = sum(w * loss for w, loss in zip(self.weights, (reconstruction, event, station)))
        return total, {"reconstruction": reconstruction, "event": event, "station": station}


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, event_weight=0.05, station_weight=0.05,
                 temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.weights = reconstruction_weight, event_weight, station_weight
        self.temperature = temperature

    def forward(self, output, batch):
        reconstruction = F.mse_loss(output["reconstruction"], batch["target"])
        event = supervised_contrastive(output["ze"], batch["event"], self.temperature)
        station = supervised_contrastive(output["zs"], batch["station"], self.temperature)
        total = sum(w * loss for w, loss in zip(self.weights, (reconstruction, event, station)))
        return total, {"reconstruction": reconstruction, "event": event, "station": station}
