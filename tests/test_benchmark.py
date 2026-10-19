import copy
import itertools
import json
import shutil
import subprocess
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from seissepa import contract, data
from seissepa.contract import validate_manifest
from seissepa.data import canonical_waveform
from seissepa.evaluate import extract
from seissepa.losses import SeparationLoss
from seissepa.metrics import accuracy, full_metrics, nearest, reconstruction_metrics, retrieval_scores, wilson
from seissepa.models import NotebookAE, SpectralMLP


def manifest(n=5):
    rows = [{"trace": f"s{s}-e{e}", "station": str(s), "event": str(e), "sampling_rate": 100.0}
            for s in range(n) for e in range(n)]
    val = [i * n + i for i in range(n)]
    return {"stations": [str(i) for i in range(n)], "events": [str(i) for i in range(n)],
            "records": rows, "split": {"train": [i for i in range(n*n) if i not in val], "validation": val}}


def test_rectangle_and_split_reject_leakage():
    original = manifest()
    validate_manifest(original)
    bad = copy.deepcopy(original)
    bad["split"]["train"].append(bad["split"]["validation"][0])
    with pytest.raises(ValueError, match="disjoint"):
        validate_manifest(bad)
    bad = copy.deepcopy(original)
    bad["records"][1] = bad["records"][0]
    with pytest.raises(ValueError, match="one distinct trace"):
        validate_manifest(bad)
    bad = copy.deepcopy(original)
    bad["split"]["validation"] = list(range(5))
    bad["split"]["train"] = list(range(5, 25))
    with pytest.raises(ValueError, match="training reference"):
        validate_manifest(bad)


def test_lock_rejects_changed_frozen_files(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"benchmark_id": "test", "files": {"manifest.json": contract.sha256(path)}}))
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "MANIFEST", path)
    monkeypatch.setattr(contract, "LOCK", lock)
    contract.read_verified()
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="Frozen benchmark changed"):
        contract.read_verified()


def test_training_query_is_excluded_and_vote_ties_are_fixed():
    z = np.eye(6)
    labels = np.arange(6)
    assert np.all(nearest(z, labels, z) == labels)
    assert np.all(nearest(z, labels, z, leave_one_out=True) != labels)
    # Equal similarities follow gallery order; equal class votes choose smallest label.
    reference = np.ones((6, 2))
    prediction = nearest(reference, np.array([9, 3, 9, 3, 7, 7]), reference[:1], k=5)
    assert prediction[0] == 3


def test_retrieval_objective_averages_both_tasks():
    train = {"ze": np.eye(2), "zs": np.eye(2), "event": np.array([0, 1]), "station": np.array([0, 1])}
    validation = {"ze": np.eye(2), "zs": np.eye(2)[::-1], "event": np.array([0, 1]), "station": np.array([0, 1])}
    result = retrieval_scores(train, validation)
    assert result == {"val/event/retrieval_top1": 1.0, "val/station/retrieval_top1": 0.0, "val/retrieval_accuracy": 0.5}


def test_reconstruction_and_wilson_known_values():
    target = np.array([[[1.0, 2.0, 3.0]]])
    assert reconstruction_metrics(target, target) == {"mse": 0.0, "r2": 1.0, "mean_correlation": 1.0}
    result = reconstruction_metrics(target, target + 1)
    assert result["mse"] == 1.0
    assert result["r2"] == -0.5
    assert result["mean_correlation"] == 1.0
    assert np.allclose(wilson(9, 26), [0.194, 0.538], atol=0.002)
    assert accuracy([0, 1], np.array([0, 0]))["n"] == 2
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0


def test_canonical_transform_matches_notebook():
    x = torch.arange(3 * 12000, dtype=torch.float32).reshape(3, 12000)
    crop = x[:, 1904:10096]
    expected = (crop - crop.mean(-1, keepdim=True)) / crop.std(-1, keepdim=True).clamp_min(1e-6)
    assert torch.equal(canonical_waveform(x), expected)
    assert torch.isfinite(canonical_waveform(torch.ones(3, 100))).all()


def test_fast_rectangle_is_rectangular_and_independent_of_metadata_order():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    import sys
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("prepare_data", scripts / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [[f"{s}-{e}", str(s), str(e), "100"] for s in range(6) for e in range(11)]
    a = module.greedy_rectangle(rows)
    b = module.greedy_rectangle(rows[::-1])
    assert a == b
    assert len(a[0]) == 6 and len(a[1]) == 11 and a[2]["area"] == 66
    assert a[2]["status"] == "HEURISTIC_LARGEST_FOUND"


def test_inference_interface_does_not_receive_identity(tmp_path, monkeypatch):
    np.save(tmp_path / "waveforms.npy", np.ones((25, 3, 32), dtype=np.float32))
    monkeypatch.setattr(data, "FROZEN", tmp_path)
    dataset = data.FrozenDataset(manifest(), "validation")
    class IdentityBlind(torch.nn.Module):
        def forward(self, inputs):
            assert set(inputs) == {"waveform", "sampling_rate"}
            target = canonical_waveform(inputs["waveform"])
            z = target.mean(-1)
            return {"reconstruction": target, "ze": z, "zs": z}
    result = extract(IdentityBlind(), DataLoader(dataset, batch_size=2), "cpu", include_spectra=True)
    assert len(result["event"]) == len(dataset)
    assert result["reconstruction"]["mse"] == 0.0
    assert result["spectra"].shape == (5, 3 * 256)


@pytest.mark.parametrize("model", [lambda: NotebookAE(3, channels=[4, 8], latent_dim=8),
                                  lambda: SpectralMLP(3, hidden_dim=8, latent_dim=8, bins=8)])
def test_models_and_latent_loss_have_finite_gradients(model):
    torch.set_num_threads(1)
    inputs = {"waveform": torch.randn(4, 3, 12000), "sampling_rate": torch.full((4,), 100.0)}
    output = model()(inputs)
    batch = {"target": canonical_waveform(inputs["waveform"]), "event": torch.tensor([0, 0, 1, 1]),
             "station": torch.arange(4)}
    loss, _ = SeparationLoss()(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert output["reconstruction"].shape == (4, 3, 8192)


@pytest.fixture(scope="module")
def selector(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("Exact selector tests require the preparation C++17 compiler")
    path = tmp_path_factory.mktemp("native") / "selector"
    source = Path(__file__).resolve().parents[1] / "scripts/rectangle_search.cpp"
    subprocess.run([compiler, "-O3", "-std=c++17", str(source), "-o", str(path)], check=True)
    return path


@pytest.mark.parametrize("seed", range(12))
def test_exact_selector_matches_exhaustive_search(selector, seed):
    rng = np.random.default_rng(seed)
    graph = rng.random((9, 11)) < 0.78
    graph[:5, :5] = True
    best = 0
    for size in range(5, 10):
        for stations in itertools.combinations(range(9), size):
            events = graph[list(stations)].all(0).sum()
            if events >= 5:
                best = max(best, size * int(events))
    s, e = np.nonzero(graph)
    text = f"9 11 {len(s)}\n" + "".join(f"{i} {j}\n" for i, j in zip(s, e))
    process = subprocess.run([str(selector), "10"], input=text, capture_output=True, text=True, check=True)
    result = json.loads(process.stdout)
    assert result["status"] == "OPTIMAL"
    assert result["area"] == best
    assert graph[np.ix_(result["stations"], result["events"])].all()
