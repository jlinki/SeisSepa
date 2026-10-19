"""Protected, label-blind model inference and fixed benchmark evaluation."""
import torch
import numpy as np

from seissepa.metrics import reconstruction_metrics


def move_batch(batch, device):
    return {k: ({a: b.to(device) for a, b in v.items()} if isinstance(v, dict) else v.to(device))
            for k, v in batch.items()}


@torch.inference_mode()
def extract(model, loader, device, include_spectra=False):
    model.eval()
    parts = {k: [] for k in ("ze", "zs", "event", "station", "spectra")}
    reconstruction = {"mse": 0.0, "r2": 0.0, "mean_correlation": 0.0}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["inputs"])
        target = batch["target"]
        if output["reconstruction"].shape != target.shape:
            raise ValueError("Model must reconstruct the fixed canonical waveform target")
        for key in ("reconstruction", "ze", "zs"):
            if not torch.isfinite(output[key]).all():
                raise ValueError(f"Non-finite model output: {key}")
        for key in ("ze", "zs"):
            if output[key].ndim != 2 or len(output[key]) != len(target) or output[key].shape[1] < 1:
                raise ValueError("Expected one nonempty embedding per trace")
        rec = reconstruction_metrics(target.cpu().numpy(), output["reconstruction"].cpu().numpy())
        for key, value in rec.items():
            reconstruction[key] += value * len(target)
        n += len(target)
        for key in ("ze", "zs"):
            parts[key].append(output[key].cpu().numpy())
        for key in ("event", "station"):
            parts[key].append(batch[key].cpu().numpy())
        if include_spectra:
            # Kept in the protected evaluator, independent of the editable model's features.
            power = torch.fft.rfft(target).abs().square()
            spectra = torch.nn.functional.adaptive_avg_pool1d(torch.log1p(power), 256).flatten(1)
            parts["spectra"].append(spectra.cpu().numpy())
    result = {k: np.concatenate(v) for k, v in parts.items() if v}
    result["reconstruction"] = {k: v / n for k, v in reconstruction.items()}
    return result
