# seisSepa autoresearch environment

Script-based successor to `AE_1D/run_new_reproducible.ipynb`. The original notebook remains at that path. Superseded code, notebooks and generated results are preserved under `legacy/`. The original HDF5 is never modified.

The objective is the mean of event and station validation **top-1 cosine retrieval accuracy**, using training traces as the reference gallery. A separate centroid/probe report follows the supplied table. Read [AGENTS.md](AGENTS.md) for the precise research rules.

## Setup and run

Run from the repository root. Python is pinned to the notebook environment's 3.9.25; `uv.lock` pins transitive dependencies. CUDA 12.6 PyTorch also runs on CPU. No Jupyter or solver package is required.

```bash
uv sync --frozen --extra dev

# Owner-only, ONCE on a new checkout; needs the source HDF5.
# Skip this when benchmark/lock.json and data/frozen/ already exist.
uv run --frozen python scripts/prepare_data.py

uv run --frozen python scripts/verify_benchmark.py
uv run --frozen pytest -q
uv run --frozen python scripts/train.py
```

Preparation selects the **largest-found complete square: N stations × N events**, using a fixed deterministic greedy search in both orientations (up to 512 starts per side). Every pair must exist, with one lexicographically first trace per cell. It scores the smaller side at every prefix and crops the larger side to make a square; the minimum side is five. Like the notebook's search, it **does not prove global optimality**. It works on the entire metadata graph and freezes a 90/10 split with seed 42. Preparation refuses to overwrite an existing benchmark. It hashes the full original HDF5 once, then stores all selected raw waveform samples in `data/frozen/waveforms.npy`. Normal training verifies the smaller frozen cache and protected source hashes, without rereading the full HDF5. No compiler or solver is needed.

The saved reference notebook reports 34×34. The current full-data greedy search finds 35×35; additional bounded seeded checks did not verify 37×37. The manifest records the actual found square and labels it `HEURISTIC_LARGEST_SQUARE_FOUND`; do not assume a remembered side length or silently substitute an incomplete block.

The frozen benchmark contains **1,225 traces: 1,102 training and 123 validation**, covering 35 stations and 35 events. Its SHA-256 benchmark ID is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`.

`benchmark/manifest.json` and `benchmark/lock.json` must travel with the frozen waveform cache. Keep the small manifest and lock in version control; keep the large source HDF5, cache, environments and run outputs outside Git. Commit and push all code/configuration changes before training; the runner requires a clean tree and a HEAD matching its upstream tracking commit. Each run records that commit locally. Commit and push the experiment journal after each batch.

## Experiments

```bash
# One fast pipeline check on the same full frozen dataset and split
uv run --frozen python scripts/train.py model=mlp train.epochs=1

# Train / sweep; Hydra creates a separate directory for every run
uv run --frozen python scripts/train.py model=notebook train.epochs=100 device=cuda:0
uv run --frozen python scripts/train.py --multirun loss.event_weight=0.05,0.1 optimizer.lr=0.0002,0.001

# Inspect and independently re-evaluate the selected checkpoint
uv run --frozen python scripts/leaderboard.py
uv run --frozen python scripts/evaluate.py outputs/PATH/best.pt --device cuda:0 --batch-size 128
```

The baseline copies the notebook's convolutional architecture, separate normalized latent heads, 8192-sample normalized center crop and AdamW defaults. Its stable separation loss is the notebook's cosine-margin variant; the notebook contains multiple incompatible training cells, so this is a defined baseline rather than a promise to reproduce a particular displayed number. `model=mlp` demonstrates a fast spectral alternative. Models accept `forward({waveform, sampling_rate})` and return `{reconstruction, ze, zs}`. The full raw waveform is available to alternative models; identity labels are passed separately to the training loss. There is no dataset or split override in Hydra.

Each epoch evaluates the actual current model against the full fixed training gallery and validation queries. The best checkpoint maximizes retrieval accuracy; exact ties prefer lower validation MSE, then the earlier epoch. Early stopping uses this same ordering. Full diagnostics and probes run once on the best checkpoint. Training uses deterministic PyTorch operations and explicit Python, NumPy, PyTorch and DataLoader seeds. A rerun on the same runtime/device/batch size is the reproducibility target; cross-device/library floating point differences can change nearest-neighbor ties. Checkpoints restart evaluation, not mid-training optimizer resumption.

Every run saves resolved Hydra configuration, source archive and SHA-256 digests, package/device provenance, per-epoch JSONL, best checkpoint, JSON metrics and a Markdown table. Standalone evaluation verifies the benchmark ID and source hashes; to evaluate an older model after editing source, restore its `source.tar.gz` into a separate copy with the same benchmark files and cache. Run outputs live in `outputs/`; the journal schema and workflow are in [experiments/README.md](experiments/README.md).

The included `outputs/smoke/` runs use one epoch and validate the pipeline; they are not the full-budget research baseline. Their commands and results are recorded in `experiments/journal.jsonl`, with verification evidence in [experiments/smoke_validation.json](experiments/smoke_validation.json). The superseded rectangular setup is preserved under `legacy/rectangle_benchmark/` and is excluded from the active leaderboard. Start the default seed-42 baseline before tuning.

The [online integration check](https://wandb.ai/jlinki/seissepa-autoresearch/runs/773yaub2) completed on the same full benchmark. Online history and final metrics match the local report; its approved configuration and Git/benchmark identifiers were verified through the W&B API. No source or artifact files were uploaded. See [experiments/wandb_validation.json](experiments/wandb_validation.json) for the verification record.

## Metrics

| Scope | Metrics | Definition |
|---|---|---|
| Reconstruction | MSE, R², mean correlation | Fixed normalized 8192-sample center crop; no scale alignment. MSE over all elements; mean per-trace `1−SSE/SST` and Pearson correlation over flattened channels/time. |
| Event on `ze`; station on `zs` | Top-1 retrieval | L2-normalized cosine query/gallery similarity. Every validation query uses only training references. Training diagnostic excludes the query itself. |
| Both latent spaces | Nearest-centroid accuracy | Mean of normalized training embeddings per class, then normalized for cosine comparison. Training centroid diagnostic includes its own query. |
| Both latent spaces | k-NN (k=5) | Five cosine-nearest training traces, uniform vote. Similarity ties follow canonical gallery order; vote ties use the smallest fixed class index. |
| Both latent spaces | Linear probe | Train-only `StandardScaler` and `LogisticRegression(C=1, solver=lbfgs, max_iter=2000)`; validation accuracy. |
| Raw spectra | Linear probe reference | Same fixed probe on channelwise log power spectra of the canonical target, mean-pooled to 256 bins/channel. |
| Both latent spaces | Cosine silhouette | Each split separately; at most 2,048 samples, fixed seed 42. Undefined cases are `null` with a reason. |
| Individual accuracies | 95% Wilson interval | Correct/total counts are saved; chance is `1 / number of gallery classes`. |

The report prints actual train/validation trace and class counts. Some classes may be absent from the random validation subset; all must be in training and their counts are reported. Constant-target R² is 1 for an exact reconstruction and 0 otherwise; a zero-variance Pearson vector contributes 0. No aggregate Wilson interval is assigned to the two-task mean. The supplied example's 16×16 values are not substituted for measured results.

W&B defaults to **online** in the approved project `jlinki/seissepa-autoresearch`. Authenticate using your usual W&B setup. Training rejects offline/disabled modes and other destinations; authentication/logging failures do not trigger an offline fallback. Record the run URL from `wandb.json` in the journal. Online records contain numeric training hyperparameters, model/loss/optimizer class names, metrics, the benchmark fingerprint and Git commit ID. The complete configuration, source, datasets, checkpoints and artifact files stay local. W&B code/Git capture, console capture and system metadata/statistics are disabled. The earlier offline smoke checks predate this policy. The integration uses the documented [W&B run API](https://docs.wandb.ai/ref/python/experiments/run/) and [Hydra configuration targets](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/).
