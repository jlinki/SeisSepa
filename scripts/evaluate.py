"""Recompute every metric from a saved checkpoint, without fitting model weights."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import torch
from hydra.utils import instantiate

from seissepa.contract import read_verified, sha256
from seissepa.data import ROOT, FrozenDataset
from seissepa.evaluate import extract
from seissepa.metrics import full_metrics, markdown_report
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    manifest, lock = read_verified()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["benchmark_id"] != lock["benchmark_id"]:
        raise ValueError("Checkpoint belongs to a different benchmark")
    for relative, expected in checkpoint["source"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Source changed: {relative}; restore the checkpoint's source.tar.gz in a separate checkout")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    datasets = [FrozenDataset(manifest, split) for split in ("train", "validation")]
    model = instantiate(checkpoint["config"]["model"], in_channels=datasets[0].waveforms.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    results = [extract(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), device, include_spectra=True)
               for dataset in datasets]
    report = full_metrics(*results, {"event": len(manifest["events"]), "station": len(manifest["stations"])})
    report.update(benchmark_id=lock["benchmark_id"], best_epoch=checkpoint["epoch"], seed=checkpoint["config"]["seed"])
    report["git_commit"] = checkpoint.get("git_commit")
    path = args.checkpoint.with_suffix(".evaluation.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))


if __name__ == "__main__":
    main()
