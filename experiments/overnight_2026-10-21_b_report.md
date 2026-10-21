# Overnight 2026-10-21 — lane B report

## Invocation status

This bounded invocation is complete after four attempts (slots 6–9), the maximum for one invocation. Nine of 64 cumulative lane slots have been consumed; 55 remain, including at least 12 reserved for future confirmations. The branch is `research/overnight-20261021-b`. All journal/report changes are being pushed with a clean tree. No source or model configuration files changed in this invocation; NotebookAE architecture remained fixed.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training, 123 validation; 35 event and 35 station classes). Preflight passed: 24 tests and `scripts/verify_benchmark.py`. Every attempt used the guarded runner, GPU 0, online W&B in `jlinki/seissepa-autoresearch`, batch size 128, at most 200 epochs, patience 40, and a 1,200-second cap. All four completed with required local artifacts and no W&B artifact uploads.

## Hypothesis and budget

Hypothesis: increasing supervised-contrastive `event_weight` from 0.05 to 0.10 while keeping `station_weight=0.05` will improve event retrieval, which had stayed at 0.032520 across the prior temperature-0.1 confirmation seeds, without losing station retrieval.

The seed-42 screen scored 0.150407, with event retrieval rising to 0.081301 and station retrieval unchanged at 0.219512 versus the prior seed-42 SupCon run. It passed the guarded standalone evaluator, so the predeclared repeat and seed-43/44 confirmations followed. The repeat exactly matched the screen. The confirmation batch held all configuration and training budget fields fixed.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.event_weight=0.10 loss.station_weight=0.05 seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_event010_seed42 hydra.run.dir=outputs/overnight_2026-10-21/06_supcon_event010_seed42
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.event_weight=0.10 loss.station_weight=0.05 seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_event010_seed42_repeat hydra.run.dir=outputs/overnight_2026-10-21/07_supcon_event010_seed42_repeat
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.event_weight=0.10 loss.station_weight=0.05 seed=43 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_event010_seed43 hydra.run.dir=outputs/overnight_2026-10-21/08_supcon_event010_seed43
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.event_weight=0.10 loss.station_weight=0.05 seed=44 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_event010_seed44 hydra.run.dir=outputs/overnight_2026-10-21/09_supcon_event010_seed44
```

AdamW used learning rate 0.0002 and weight decay 0.0001; reconstruction weight 1.0; SupCon temperature 0.1. Each exact command and outcome is in `experiments/journal.jsonl`. Each journal update was pushed before the next training launch.

## Run inventory

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 1 | Baseline | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 42 / 1 | 41 / 296.09 s | `01_baseline_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/9ja0fw0w) |
| 2 | SupCon screen | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 | 42 / 5 | 45 / 293.15 s | `02_supcon_t010_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/a8r87xmw) |
| 3 | Exact seed-42 repeat | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 | 42 / 5 | 45 / 290.85 s | `03_supcon_t010_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vvc2uob8) |
| 4 | SupCon seed 43 | 0.032520 | 0.260163 | 0.146341 | 0.5637014239 | 43 / 66 | 106 / 373.73 s | `04_supcon_t010_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/pe1d8vr5) |
| 5 | SupCon seed 44 | 0.032520 | 0.235772 | 0.134146 | 0.6621046324 | 44 / 47 | 87 / 351.75 s | `05_supcon_t010_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/cog72e5x) |
| 6 | Event-weight screen | 0.081301 | 0.219512 | 0.150407 | 0.9800989762 | 42 / 4 | 44 / 220.09 s | `06_supcon_event010_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/04a5mje5) |
| 7 | Exact seed-42 repeat | 0.081301 | 0.219512 | 0.150407 | 0.9800989762 | 42 / 4 | 44 / 215.98 s | `07_supcon_event010_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/biuf5il0) |
| 8 | Event-weight seed 43 | 0.016260 | 0.252033 | 0.134146 | 0.7381186025 | 43 / 24 | 64 / 222.53 s | `08_supcon_event010_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/bkq81wkc) |
| 9 | Event-weight seed 44 | 0.056911 | 0.260163 | **0.158537** | 0.7094152145 | 44 / 31 | 71 / 280.59 s | `09_supcon_event010_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/otw9w92w) |

Elapsed values are guarded runner wall times. All run directories are under `outputs/overnight_2026-10-21/`; each has `config.yaml`, `provenance.json`, `source.tar.gz`, `history.jsonl`, `best.pt`, `metrics.json`, and `report.md`.

## Three-seed evidence and evaluation

| Seed | Event top-1 | Station top-1 | Combined score | Canonical val MSE |
|---:|---:|---:|---:|---:|
| 42 (screen and exact repeat) | 0.081301 | 0.219512 | 0.150407 | 0.9800989762 |
| 43 | 0.016260 | 0.252033 | 0.134146 | 0.7381186025 |
| 44 | 0.056911 | 0.260163 | 0.158537 | 0.7094152145 |
| Three-seed mean | 0.051490 | 0.243902 | **0.147696** | — |
| Three-seed sample SD | 0.032857 | 0.021510 | **0.012419** | — |

The seed-42 repeat matched all 86 shared scientific fields with zero mismatches. The best checkpoint, slot 9 seed 44, passed guarded evaluation on GPU 0 with batch size 128; all 86 shared scientific fields matched exactly. Its benchmark ID is the frozen ID above and its training source commit is `e3c34c20d607e8697ce1316c0ecbfbc00a614ad6`.

The new configuration’s three-seed mean is 0.025745 above the full-budget seed-42 baseline (0.121951) and 0.012195 above the previous SupCon configuration’s three-seed mean (0.135501; sample SD 0.010230). The best single-seed result is 0.158537, above the prior best seed-42 score 0.130081. The gain is not uniform by task: event retrieval varies across seeds, while station retrieval is stronger at seeds 43 and 44. Validation was repeatedly used for selection, so these are validation results, not an untouched test result. No confidence interval is assigned to the combined score.

## Current best and next hypothesis

Current best valid configuration: `model=notebook loss=supervised_contrastive`, reconstruction/event/station weights `1.0/0.10/0.05`, temperature 0.1, AdamW lr 0.0002, weight decay 0.0001, batch 128, max 200 epochs, patience 40. Best checkpoint: `outputs/overnight_2026-10-21/09_supcon_event010_seed44/best.pt`; canonical MSE 0.7094152145; best epoch 31; standalone evaluation matched exactly. Keep its three-seed mean alongside the best single-seed score.

Next bounded hypothesis: raise `event_weight` from 0.10 to 0.15 while keeping `station_weight=0.05` and all other settings fixed; one seed-42 screen can test whether the event improvement continues without reducing station retrieval. Run it only in a new authorized invocation after journaling and pushing its exact command and budget. Do not start another attempt in this invocation.
