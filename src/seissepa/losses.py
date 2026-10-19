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
