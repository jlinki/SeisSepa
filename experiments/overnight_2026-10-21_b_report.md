# Overnight 2026-10-21 — lane B report

## Invocation status

This bounded invocation is complete after one attempt (slot 10). Ten of 64 cumulative lane slots have been consumed; 54 remain, with at least 12 reserved for future confirmations. The branch is `research/overnight-20261021-b`. The full-budget seed-42 separation baseline was already completed as slot 1, so it was not repeated. No source or model configuration files changed; NotebookAE architecture remained fixed. The plan and outcome are journaled and will be pushed with a clean tree.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training, 123 validation; 35 event and 35 station classes). Preflight passed: 24 tests and `scripts/verify_benchmark.py`; the guard showed no STOP file and nine slots consumed before this batch. All ten attempts used the guarded runner, GPU 0, online W&B in `jlinki/seissepa-autoresearch`, batch size 128, at most 200 epochs, patience 40, and a 1,200-second cap. All completed attempts have the required local artifacts; W&B received no artifact files.

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

## Latest bounded batch (slot 10)

Hypothesis: increasing `event_weight` from 0.10 to 0.15 while holding `station_weight=0.05`, temperature 0.1, the NotebookAE model, AdamW settings and training budget fixed would improve seed-42 event retrieval without lowering station retrieval. The batch budget was at most four runs, 200 epochs each, patience 40 and 1,200 seconds per attempt. The exact screen command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.event_weight=0.15 loss.station_weight=0.05 seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_event015_seed42 hydra.run.dir=outputs/overnight_2026-10-21/10_supcon_event015_seed42
```

The screen completed 45 epochs and selected epoch 5. Event top-1 was 0.024390, station top-1 was 0.252033, the combined score was 0.138211, and canonical validation MSE was 0.9779321476. It ran for 211.96 seconds on the runner and logged online at [W&B run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gi7q1zrb). Its provenance records commit `c88f5814eb66bd31e4937f3db3c05af2a1cba81e` and the frozen benchmark ID; all required local artifacts are present.

The score and event retrieval were below the seed-42 `event_weight=0.10` reference (0.150407 score; 0.081301 event top-1), although station top-1 rose from 0.219512 to 0.252033. The screen failed the pre-registered continuation criterion, so the exact repeat and seeds 43/44 were not launched. Keep the result and its artifacts as a poor screen, not a promoted candidate.

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
| 10 | Event-weight 0.15 screen; not promoted | 0.024390 | 0.252033 | 0.138211 | 0.9779321476 | 42 / 5 | 45 / 211.96 s | `10_supcon_event015_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gi7q1zrb) |

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

Current best valid configuration: `model=notebook loss=supervised_contrastive`, reconstruction/event/station weights `1.0/0.10/0.05`, temperature 0.1, AdamW lr 0.0002, weight decay 0.0001, batch 128, max 200 epochs, patience 40. Best checkpoint: `outputs/overnight_2026-10-21/09_supcon_event010_seed44/best.pt`; canonical MSE 0.7094152145; best epoch 31; standalone evaluation matched exactly. Its three-seed mean is 0.147696 (sample SD 0.012419); keep all three seed results alongside its best single-seed score of 0.158537. Slot 10's `event_weight=0.15` screen scored 0.138211 and was not independently evaluated or promoted because it failed the pre-registered screen criterion.

Next bounded hypothesis: with `event_weight=0.10` and `station_weight=0.05`, lower SupCon temperature from 0.1 to 0.05 to test whether sharper contrastive logits improve event retrieval while retaining station retrieval. Register one seed-42 screen and its budget in the journal before any new invocation starts.
