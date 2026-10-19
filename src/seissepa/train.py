"""Editable training loop; protected evaluation always scores every frozen query."""
import importlib.metadata
import json
import platform
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from seissepa.contract import read_verified, sha256
from seissepa.data import ROOT, FrozenDataset
from seissepa.evaluate import extract, move_batch
from seissepa.metrics import EVALUATION, full_metrics, markdown_report, retrieval_scores


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def loaders(manifest, batch_size, seed):
    train = FrozenDataset(manifest, "train")
    validation = FrozenDataset(manifest, "validation")
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0),
            DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0),
            DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0))


def experiment_files():
    files = [p for folder in ("src", "configs", "scripts") for p in (ROOT / folder).rglob("*")
             if p.suffix in {".py", ".yaml", ".cpp"}]
    files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "AGENTS.md"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()}


def scalar_metrics(value, prefix="final"):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(scalar_metrics(item, f"{prefix}/{key}"))
    elif isinstance(value, (int, float)):
        result[prefix] = value
    return result


def run(config, output_dir):
    if config.train.epochs < 1 or config.train.batch_size < 1 or config.train.patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    manifest, lock = read_verified()
    seed_everything(config.seed, config.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "provenance.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}; choose a new Hydra output directory")
    resolved = OmegaConf.to_container(config, resolve=True)
    OmegaConf.save(config, output_dir / "config.yaml", resolve=True)
    files = experiment_files()
    with tarfile.open(output_dir / "source.tar.gz", "w:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "h5py", "scikit-learn", "hydra-core", "wandb")}
    provenance = {"benchmark_id": lock["benchmark_id"], "source": files, "versions": versions,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "device": str(device), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "evaluation": EVALUATION}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    train_loader, train_eval, val_loader = loaders(manifest, config.train.batch_size, config.seed)
    model = instantiate(config.model, in_channels=train_loader.dataset.waveforms.shape[1]).to(device)
    criterion = instantiate(config.loss).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    start = time.monotonic()
    best, best_mse, stale = -1.0, float("inf"), 0
    with wandb.init(project=config.wandb.project, entity=config.wandb.entity, mode=config.wandb.mode,
                    name=config.wandb.name, config={**resolved, "benchmark_id": lock["benchmark_id"]},
                    dir=str(output_dir)) as tracking:
        tracking.define_metric("val/retrieval_accuracy", summary="max")
        for epoch in range(1, config.train.epochs + 1):
            model.train()
            totals, count = {}, 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["inputs"])
                loss, components = criterion(output, batch)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                size = len(batch["target"])
                for key, value in {"loss": loss, **components}.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
                count += size
            train_embeddings = extract(model, train_eval, device)
            validation = extract(model, val_loader, device)
            metrics = {"epoch": epoch, **{f"train/{k}": v / count for k, v in totals.items()},
                       **retrieval_scores(train_embeddings, validation),
                       "val/mse": validation["reconstruction"]["mse"],
                       "elapsed_seconds": time.monotonic() - start}
            with (output_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            tracking.log(metrics, step=epoch)
            score, mse = metrics["val/retrieval_accuracy"], metrics["val/mse"]
            print(f"epoch={epoch} retrieval={score:.6f} event={metrics['val/event/retrieval_top1']:.4f} "
                  f"station={metrics['val/station/retrieval_top1']:.4f} mse={mse:.5f}", flush=True)
            # Exact score ties prefer smaller canonical validation MSE; otherwise keep the earlier epoch.
            improved = score > best or (score == best and mse < best_mse)
            if improved:
                best, best_mse, stale = score, mse, 0
                torch.save({"state_dict": model.state_dict(), "config": resolved,
                            "benchmark_id": lock["benchmark_id"], "epoch": epoch,
                            "score": best, "source": files}, output_dir / "best.pt")
            else:
                stale += 1
            if stale >= config.train.patience:
                break
        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        train_embeddings = extract(model, train_eval, device, include_spectra=True)
        validation = extract(model, val_loader, device, include_spectra=True)
        report = full_metrics(train_embeddings, validation,
                              {"event": len(manifest["events"]), "station": len(manifest["stations"])})
        report.update(benchmark_id=lock["benchmark_id"], best_epoch=checkpoint["epoch"], seed=config.seed,
                      elapsed_seconds=time.monotonic() - start)
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output_dir / "report.md").write_text(markdown_report(report))
        tracking.summary.update({**report["objective"], "best_epoch": checkpoint["epoch"],
                                 "benchmark_id": lock["benchmark_id"],
                                 **scalar_metrics(report),
                                 "report": wandb.Table(columns=["report"], data=[[markdown_report(report)]])})
    print(f"Results: {output_dir}", flush=True)
    return report
