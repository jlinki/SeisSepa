"""Editable training loop; protected evaluation always scores every frozen query."""
import importlib.metadata
import json
from math import cos, pi
import platform
import random
import subprocess
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


@torch.inference_mode()
def extract_retrieval_gallery(model, loader, device):
    """Extract only retrieval latents; the decoder is unnecessary for gallery scoring."""
    model.eval()
    parts = {key: [] for key in ("ze", "zs", "event", "station")}
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
        output = model.encode(inputs) if hasattr(model, "encode") else model(inputs)
        size = len(batch["event"])
        for key in ("ze", "zs"):
            if key not in output or output[key].ndim != 2 or len(output[key]) != size or output[key].shape[1] < 1:
                raise ValueError("Expected one nonempty gallery embedding per trace")
            if not torch.isfinite(output[key]).all():
                raise ValueError(f"Non-finite gallery embedding: {key}")
            parts[key].append(output[key].cpu().numpy())
        for key in ("event", "station"):
            parts[key].append(batch[key].numpy())
    return {key: np.concatenate(values) for key, values in parts.items()}


@torch.inference_mode()
def canonical_validation_mse(model, loader, device):
    """Compute the evaluator's exact MSE only when a retrieval score can be selected."""
    model.eval()
    weighted_mse, count = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["inputs"])
        target = batch["target"]
        if output["reconstruction"].shape != target.shape:
            raise ValueError("Model must reconstruct the fixed canonical waveform target")
        for key in ("reconstruction", "ze", "zs"):
            if not torch.isfinite(output[key]).all():
                raise ValueError(f"Non-finite model output: {key}")
        target_array = target.cpu().numpy().astype(np.float64, copy=False)
        reconstruction_array = output["reconstruction"].cpu().numpy().astype(np.float64, copy=False)
        batch_mse = float(np.square(target_array - reconstruction_array).mean())
        size = len(target)
        weighted_mse += batch_mse * size
        count += size
    return weighted_mse / count


def experiment_files():
    files = [p for folder in ("src", "configs", "scripts", "tests") for p in (ROOT / folder).rglob("*")
             if p.suffix in {".py", ".yaml"}]
    files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "AGENTS.md",
                                          "benchmark/manifest.json", "benchmark/lock.json"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files) if p.is_file()}


def git_provenance():
    def query(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    try:
        commit = query("rev-parse", "HEAD")
        if query("status", "--porcelain"):
            raise RuntimeError("Commit and push all changes before training; Git must be clean.")
        upstream = query("rev-parse", "--abbrev-ref", "@{upstream}")
        upstream_commit = query("rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Training requires a Git commit and a pushed upstream branch.") from exc
    if commit != upstream_commit:
        raise RuntimeError("Push the current Git commit to its upstream before training.")
    return {"commit": commit, "branch": query("branch", "--show-current"),
            "upstream": upstream, "dirty": False}


def final_diagnostic_scalars(report):
    """Flatten the final report into explicit W&B chart and summary scalars."""
    values = {"final/objective/val/retrieval_accuracy": report["objective"]["val/retrieval_accuracy"]}
    for split, metrics in report["reconstruction"].items():
        for name in ("mse", "r2", "mean_correlation"):
            values[f"final/reconstruction/{split}/{name}"] = metrics[name]
    for scope, item in report["scopes"].items():
        for split, metrics in item.items():
            if split not in {"training", "validation"}:
                continue
            for name, record in metrics.items():
                prefix = f"final/{scope}/{split}/{name}"
                if isinstance(record, dict) and "accuracy" in record:
                    values[f"{prefix}/accuracy"] = record["accuracy"]
                    low, high = record["wilson95"]
                    values[f"{prefix}/wilson95_low"] = low
                    values[f"{prefix}/wilson95_high"] = high
                elif isinstance(record, dict) and record.get("score") is not None:
                    values[f"{prefix}/score"] = record["score"]
    return values


def tracking_options(config, benchmark_id, git_commit):
    if config.wandb.mode != "online":
        raise ValueError("Training requires wandb.mode=online; do not fall back to offline logging.")
    if (config.wandb.entity, config.wandb.project) != ("jlinki", "seissepa-autoresearch"):
        raise ValueError("The approved W&B destination is jlinki/seissepa-autoresearch.")
    resolved = OmegaConf.to_container(config, resolve=True)
    # Send selected hyperparameters, never the full resolved configuration or local paths.
    def parameters(value):
        if isinstance(value, dict):
            return {key: parameters(item) for key, item in value.items()
                    if not key.startswith("_") and isinstance(item, (dict, list, int, float, bool, type(None)))}
        if isinstance(value, list):
            return [parameters(item) for item in value if isinstance(item, (dict, list, int, float, bool, type(None)))]
        return value
    payload = {"seed": resolved["seed"], "threads": resolved["threads"],
               "benchmark_id": benchmark_id, "git_commit": git_commit}
    for section in ("model", "loss", "optimizer", "train"):
        payload[section] = parameters(resolved[section])
        if "_target_" in resolved[section]:
            payload[section]["class"] = resolved[section]["_target_"].rsplit(".", 1)[-1]
    settings = wandb.Settings(base_url="https://api.wandb.ai", save_code=False,
                              disable_code=True, disable_git=True, disable_job_creation=True,
                              console="off", x_disable_stats=True, x_disable_meta=True,
                              x_disable_machine_info=True, x_save_requirements=False)
    return payload, settings


def run(config, output_dir):
    if config.train.epochs < 1 or config.train.batch_size < 1 or config.train.patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    manifest, lock = read_verified()
    git = git_provenance()
    online_config, online_settings = tracking_options(config, lock["benchmark_id"], git["commit"])
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
    provenance = {"benchmark_id": lock["benchmark_id"], "git": git, "source": files, "versions": versions,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "device": str(device), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "evaluation": EVALUATION}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    train_loader, train_eval, val_loader = loaders(manifest, config.train.batch_size, config.seed)
    model = instantiate(config.model, in_channels=train_loader.dataset.waveforms.shape[1]).to(device)
    criterion = instantiate(config.loss).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = None
    if config.train.lr_schedule_cosine:
        schedule_epochs = int(config.train.lr_schedule_epochs)
        min_ratio = float(config.train.lr_min_ratio)
        if schedule_epochs < 1 or not 0.0 <= min_ratio <= 1.0:
            raise ValueError("Cosine schedule epochs and minimum learning-rate ratio are invalid")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min_ratio + (1.0 - min_ratio) * 0.5
            * (1.0 + cos(pi * min(step / schedule_epochs, 1.0))),
        )
    start = time.monotonic()
    best, best_mse, stale = -1.0, float("inf"), 0
    with wandb.init(project=config.wandb.project, entity=config.wandb.entity, mode=config.wandb.mode,
                    name=config.wandb.name, config=online_config, settings=online_settings,
                    force=True, dir=str(output_dir)) as tracking:
        tracking_info = {"id": tracking.id, "url": tracking.url, "mode": "online",
                         "entity": config.wandb.entity, "project": config.wandb.project}
        (output_dir / "wandb.json").write_text(json.dumps(tracking_info, indent=2) + "\n")
        provenance["wandb"] = tracking_info
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
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
            train_embeddings = extract_retrieval_gallery(model, train_eval, device)
            validation = extract_retrieval_gallery(model, val_loader, device)
            metrics = {"epoch": epoch, **{f"train/{k}": v / count for k, v in totals.items()},
                       **retrieval_scores(train_embeddings, validation),
                       "train/learning_rate": float(optimizer.param_groups[0]["lr"])}
            score = metrics["val/retrieval_accuracy"]
            candidate_mse = None
            if score >= best:
                candidate_mse = canonical_validation_mse(model, val_loader, device)
                metrics["val/mse"] = candidate_mse
            metrics["elapsed_seconds"] = time.monotonic() - start
            with (output_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            tracking.log(metrics, step=epoch)
            mse = candidate_mse
            mse_text = f"{mse:.5f}" if mse is not None else "not-needed"
            print(f"epoch={epoch} retrieval={score:.6f} event={metrics['val/event/retrieval_top1']:.4f} "
                  f"station={metrics['val/station/retrieval_top1']:.4f} mse={mse_text}", flush=True)
            # Exact score ties prefer smaller canonical validation MSE; otherwise keep the earlier epoch.
            improved = score > best or (score == best and mse < best_mse)
            if improved:
                best, best_mse, stale = score, mse, 0
                torch.save({"state_dict": model.state_dict(), "config": resolved,
                            "git_commit": git["commit"],
                            "benchmark_id": lock["benchmark_id"], "epoch": epoch,
                            "score": best, "source": files}, output_dir / "best.pt")
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            if stale >= config.train.patience:
                break
        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        train_embeddings = extract(model, train_eval, device, include_spectra=True)
        validation = extract(model, val_loader, device, include_spectra=True)
        report = full_metrics(train_embeddings, validation,
                              {"event": len(manifest["events"]), "station": len(manifest["stations"])})
        report.update(benchmark_id=lock["benchmark_id"], best_epoch=checkpoint["epoch"], seed=config.seed,
                      git_commit=git["commit"],
                      wandb_url=tracking.url,
                      elapsed_seconds=time.monotonic() - start)
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output_dir / "report.md").write_text(markdown_report(report))
        diagnostics = final_diagnostic_scalars(report)
        tracking.log(diagnostics, step=epoch + 1)
        tracking.summary.update({**report["objective"], "best_epoch": checkpoint["epoch"],
                                 "benchmark_id": lock["benchmark_id"],
                                 "git_commit": git["commit"], **diagnostics})
    print(f"Results: {output_dir}", flush=True)
    return report
