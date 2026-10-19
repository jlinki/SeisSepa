"""Editable models. forward(inputs) returns reconstruction, ze, zs; inputs contain no IDs."""
import torch
from torch import nn
from torch.nn import functional as F

from seissepa.data import TARGET_LENGTH, canonical_waveform


class NotebookAE(nn.Module):
    """Reference AE1D_Conv_MLP: separate normalized heads; convolutional decoder."""
    def __init__(self, in_channels, channels=(32, 64, 128), latent_dim=128):
        super().__init__()
        if latent_dim < 2 or latent_dim % 2:
            raise ValueError("latent_dim must be a positive even number")
        down, previous = [], in_channels
        for width in channels:
            down.extend([nn.Conv1d(previous, width, 4, 2, 1), nn.GELU(),
                         nn.Conv1d(width, width, 3, padding=1), nn.GELU()])
            previous = width
        self.encoder = nn.Sequential(*down)
        with torch.no_grad():
            size = self.encoder(torch.zeros(1, in_channels, TARGET_LENGTH)).numel()
        self.event_head = nn.Linear(size, latent_dim // 2)
        self.station_head = nn.Linear(size, latent_dim // 2)
        up, previous = [], channels[-1]
        for width in [*reversed(channels[:-1]), channels[0]]:
            up.extend([nn.ConvTranspose1d(previous, width, 4, 2, 1), nn.GELU(),
                       nn.Conv1d(width, width, 3, padding=1), nn.GELU()])
            previous = width
        self.decoder = nn.Sequential(*up, nn.Conv1d(previous, in_channels, 1))

    def forward(self, inputs):
        x = canonical_waveform(inputs["waveform"])
        h = self.encoder(x)
        reconstruction = self.decoder(h)
        reconstruction = F.pad(reconstruction, (0, max(0, TARGET_LENGTH - reconstruction.shape[-1])))
        return {"reconstruction": reconstruction[..., :TARGET_LENGTH],
                "ze": F.normalize(self.event_head(h.flatten(1)), dim=1),
                "zs": F.normalize(self.station_head(h.flatten(1)), dim=1)}


def spectral_features(waveform, bins=256):
    x = canonical_waveform(waveform)
    power = torch.fft.rfft(x).abs().square()
    return F.adaptive_avg_pool1d(torch.log1p(power), bins).flatten(1)


class SpectralMLP(nn.Module):
    """Fast alternative illustrating the same model interface."""
    def __init__(self, in_channels, hidden_dim=256, latent_dim=128, bins=256):
        super().__init__()
        if latent_dim < 2 or latent_dim % 2:
            raise ValueError("latent_dim must be a positive even number")
        self.bins = bins
        self.encoder = nn.Sequential(nn.Linear(in_channels * bins, hidden_dim), nn.GELU())
        self.event_head = nn.Linear(hidden_dim, latent_dim // 2)
        self.station_head = nn.Linear(hidden_dim, latent_dim // 2)
        self.decoder = nn.Linear(hidden_dim, in_channels * TARGET_LENGTH)
        self.in_channels = in_channels

    def forward(self, inputs):
        h = self.encoder(spectral_features(inputs["waveform"], self.bins))
        return {"reconstruction": self.decoder(h).reshape(-1, self.in_channels, TARGET_LENGTH),
                "ze": F.normalize(self.event_head(h), dim=1),
                "zs": F.normalize(self.station_head(h), dim=1)}
