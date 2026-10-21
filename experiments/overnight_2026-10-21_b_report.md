# Overnight 2026-10-21 — lane B report

## Invocation status

This bounded invocation is complete after three confirmation attempts (slots 3–5), below the four-attempt invocation cap. Five of 64 cumulative lane slots have been consumed, leaving 59; at least 12 remain reserved for future confirmation runs. The worktree is on `research/overnight-20261021-b` and the journal/report are being committed and pushed.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training and 123 validation; 35 event and 35 station classes). Preflight passed: 24 tests, `scripts/verify_benchmark.py`, and a clean/upstream-aligned branch. The guarded runner verified the benchmark before each attempt. All five runs used online W&B in `jlinki/seissepa-autoresearch`, GPU 0, batch size 128, maximum 200 epochs and patience 40. No source or configuration changed during this confirmation batch.

## Batch hypothesis and budget

Hypothesis: the seed-42 supervised-contrastive gain over the fresh full-budget baseline reflects a reproducible retrieval improvement. An exact seed-42 repeat and unchanged seeds 43/44 test repeatability and seed sensitivity.

The batch used three attempts, each capped at 200 epochs, patience 40 and 1,200 seconds. NotebookAE, supervised-contrastive temperature 0.1, reconstruction/event/station weights 1.0/0.05/0.05, AdamW learning rate 0.0002 and weight decay 0.0001 were held fixed.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_t010_seed42_repeat hydra.run.dir=outputs/overnight_2026-10-21/03_supcon_t010_seed42_repeat
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive seed=43 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_t010_seed43 hydra.run.dir=outputs/overnight_2026-10-21/04_supcon_t010_seed43
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive seed=44 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_t010_seed44 hydra.run.dir=outputs/overnight_2026-10-21/05_supcon_t010_seed44
```

## Run inventory

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs | Guard elapsed | Run / W&B |
|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 | Fresh full-budget separation baseline | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 42 / 1 | 41 | 296.09 s | [`01_baseline_seed42`](outputs/overnight_2026-10-21/01_baseline_seed42) · [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/9ja0fw0w) |
| 2 | SupCon screen | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 | 42 / 5 | 45 | 293.15 s | [`02_supcon_t010_seed42`](outputs/overnight_2026-10-21/02_supcon_t010_seed42) · [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/a8r87xmw) |
| 3 | Exact seed-42 repeat | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 | 42 / 5 | 45 | 290.85 s | [`03_supcon_t010_seed42_repeat`](outputs/overnight_2026-10-21/03_supcon_t010_seed42_repeat) · [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vvc2uob8) |
| 4 | Seed-43 confirmation; best single seed | 0.032520 | 0.260163 | 0.146341 | 0.5637014239 | 43 / 66 | 106 | 373.73 s | [`04_supcon_t010_seed43`](outputs/overnight_2026-10-21/04_supcon_t010_seed43) · [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/pe1d8vr5) |
| 5 | Seed-44 confirmation | 0.032520 | 0.235772 | 0.134146 | 0.6621046324 | 44 / 47 | 87 | 351.75 s | [`05_supcon_t010_seed44`](outputs/overnight_2026-10-21/05_supcon_t010_seed44) · [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/cog72e5x) |

Each run has the required local `config.yaml`, `provenance.json`, `source.tar.gz`, `history.jsonl`, `best.pt`, `metrics.json` and `report.md`. W&B uploaded only the approved scalar payload; local source and artifact files remain in the lane worktree.

## Evaluation and seed comparison

The seed-42 repeat matches slot 2 exactly on all scientific metrics. Guarded standalone evaluation at batch size 128 passed for the seed-42 repeat, seed 43 and seed 44; each evaluation matched the saved scientific fields with zero mismatches. Their outputs are saved as each run's `best.evaluation.json`.

| Seed | Event top-1 | Station top-1 | Combined score | Canonical val MSE |
|---:|---:|---:|---:|---:|
| 42 (screen and exact repeat) | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 |
| 43 | 0.032520 | 0.260163 | 0.146341 | 0.5637014239 |
| 44 | 0.032520 | 0.235772 | 0.134146 | 0.6621046324 |
| Three-seed mean | — | — | **0.135501** | — |
| Three-seed sample SD | — | — | **0.010230** | — |

The three-seed mean is 0.013550 above the fresh full-budget seed-42 baseline (0.121951) and 0.005420 above the prior round's seed-42 best (0.130081). Station retrieval drives the gain: event retrieval is unchanged at 0.032520 for all three seeds. The mean is evidence for this validation setup, not an untouched test result; no confidence interval is assigned to the combined score.

## Current best candidate

The best valid single-seed checkpoint is [`04_supcon_t010_seed43/best.pt`](outputs/overnight_2026-10-21/04_supcon_t010_seed43/best.pt), with standalone evaluation at [`best.evaluation.json`](outputs/overnight_2026-10-21/04_supcon_t010_seed43/best.evaluation.json). It was trained with source commit `e90363234caa63bef8a20675562085056b5d39e0`. Its config is `model=notebook loss=supervised_contrastive`, temperature 0.1, reconstruction/event/station weights 1.0/0.05/0.05, AdamW at 0.0002, weight decay 0.0001, batch 128, maximum 200 epochs and patience 40. Keep the three-seed mean alongside the seed-43 result when comparing future candidates.

## Journal and next hypothesis

The exact batch commands, per-attempt outcomes, benchmark fingerprint, source commits, elapsed times, W&B URLs, evaluator comparisons and three-seed summary are recorded in `experiments/journal.jsonl`.

Next bounded hypothesis: event retrieval stayed flat across all three seeds while station retrieval improved, so raise the supervised-contrastive event weight from 0.05 to 0.10 while keeping station weight at 0.05 and all other settings fixed. Start with one seed-42 screen at the same full budget; only schedule confirmation runs if it improves the primary score without a material event/station tradeoff. No further attempt is part of this invocation.
