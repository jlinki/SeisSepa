"""Protected benchmark input and reconstruction target. No labels reach forward()."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
MANIFEST = ROOT / "benchmark" / "manifest.json"
LOCK = ROOT / "benchmark" / "lock.json"
TARGET_LENGTH = 8192


def canonical_waveform(waveform):
    """Notebook center crop/pad, then channel mean/std (sample std)."""
    x = torch.nan_to_num(waveform.float(), nan=0.0, posinf=0.0, neginf=0.0)
    length = x.shape[-1]
    if length > TARGET_LENGTH:
        start = (length - TARGET_LENGTH) // 2
        x = x[..., start:start + TARGET_LENGTH]
    elif length < TARGET_LENGTH:
        left = (TARGET_LENGTH - length) // 2
        x = torch.nn.functional.pad(x, (left, TARGET_LENGTH - length - left))
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)


class FrozenDataset(Dataset):
    def __init__(self, manifest, split):
        self.waveforms = np.load(FROZEN / "waveforms.npy", mmap_mode="r", allow_pickle=False)
        self.rows = manifest["records"]
        if self.waveforms.shape != (len(self.rows), *manifest.get("waveform_shape", self.waveforms.shape[1:])):
            raise ValueError("Frozen waveform shape and manifest disagree")
        self.indices = manifest["split"][split]
        self.event_labels = {v: i for i, v in enumerate(manifest["events"])}
        self.station_labels = {v: i for i, v in enumerate(manifest["stations"])}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_index = self.indices[index]
        row = self.rows[global_index]
        waveform = torch.from_numpy(np.array(self.waveforms[global_index], copy=True))
        # IDs, names, labels and split membership are deliberately separate from inputs.
        return {
            "inputs": {"waveform": waveform, "sampling_rate": torch.tensor(row["sampling_rate"])},
            "target": canonical_waveform(waveform),
            "event": self.event_labels[row["event"]],
            "station": self.station_labels[row["station"]],
        }
