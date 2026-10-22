# Overnight 2026-10-21 — lane B report

## Invocation status

The latest bounded invocation finished after one seed-42 screen (slot 26). Twenty-six of 64 cumulative lane attempts are consumed; 38 remain, with at least 12 kept for exact repeats and seed confirmations. The branch is `research/overnight-20261021-b`. The train-only auxiliary-head implementation and pre-registered batch plan were pushed before training in commit `1d2e52440f1e72dc8a30195bc40cb9557abfca26`. The screen was below the seed-42 comparator, so no standalone evaluation or follow-up confirmations were run. NotebookAE architecture and inference outputs remained fixed.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training and 123 validation; 35 event and 35 station classes). Before the latest batch, benchmark verification passed, all 24 tests passed, the leaderboard showed slot 18 as incumbent, the branch was clean and synchronized, and STOP was absent. The new classifier heads use training labels only, are part of the loss optimizer, and are absent from the model and inference checkpoint. Slot 26 used the guarded GPU-0 runner and online W&B in `jlinki/seissepa-autoresearch`; all required local artifacts are present and no artifact files were uploaded. The hypothesis, command, result, and decision are recorded in `experiments/journal.jsonl`.

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

## Latest bounded batch (slots 11–14)

Hypothesis: lowering supervised-contrastive temperature from 0.10 to 0.05, with event/station weights 0.10/0.05 and the NotebookAE, optimizer, sampling, and training budget fixed, may sharpen latent supervision and improve retrieval. The batch allowed at most four attempts, 200 epochs, patience 40, and 1,200 seconds per attempt. Slot 11 was the seed-42 screen; after it beat the seed-42 incumbent and passed guarded evaluation, slots 12–14 repeated seed 42 and measured seeds 43/44.

| Slot | Role | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 11 | Seed-42 screen | 0.048780 | 0.260163 | 0.154472 | 0.3941654605 | 42 / 88 | 128 / 288.59 s | `11_supcon_temp005_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/5h8fs4j4) |
| 12 | Exact seed-42 repeat | 0.048780 | 0.260163 | 0.154472 | 0.3941654605 | 42 / 88 | 128 / 294.16 s | `12_supcon_temp005_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ufogzcoe) |
| 13 | Seed 43 | 0.016260 | 0.284553 | 0.150407 | 0.3888945942 | 43 / 78 | 118 / 305.11 s | `13_supcon_temp005_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/yklmg9gf) |
| 14 | Seed 44 | 0.040650 | 0.284553 | **0.162602** | 0.6499847232 | 44 / 59 | 99 / 247.62 s | `14_supcon_temp005_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/kurfujxe) |

Every attempt completed online and has the required local artifacts. The seed-42 screen passed guarded standalone evaluation with 101 shared scientific fields and zero mismatches. The exact seed-42 repeat matched all 100 shared scientific fields; only elapsed time, source commit, and W&B URL differed. The best score is slot 14, seed 44; its guarded standalone evaluation matched all 101 shared scientific fields exactly with zero mismatches using GPU 0 and batch size 128. Its benchmark ID is the frozen fingerprint above and its training source commit is `a54eb97239b41599ab5e6b1b1c922d60a502370e`.

### Three-seed evidence

| Seed | Event top-1 | Station top-1 | Score | Canonical val MSE |
|---:|---:|---:|---:|---:|
| 42 (screen and exact repeat) | 0.048780 | 0.260163 | 0.154472 | 0.3941654605 |
| 43 | 0.016260 | 0.284553 | 0.150407 | 0.3888945942 |
| 44 | 0.040650 | 0.284553 | 0.162602 | 0.6499847232 |
| Three-seed mean | 0.035230 | 0.276423 | **0.155827** | — |
| Three-seed sample SD | 0.016924 | 0.014082 | **0.006209** | — |

The primary three-seed mean is 0.033875 above the fresh full-budget seed-42 baseline (0.121951) and 0.008130 above the previous temperature-0.10 SupCon configuration’s mean (0.147696; sample SD 0.012419). The seed-42 score improves by 0.004065 over that configuration’s seed-42 reference (0.150407). The best single-seed score is 0.162602. The gain is mainly in station retrieval: station mean rose relative to the previous SupCon configuration, while event mean fell from 0.051491 to 0.035230. Validation was repeatedly used for selection, so these are validation results, not an untouched test result. The sample SD describes the three observed seed scores; no confidence interval is assigned to the combined objective.

## Latest bounded batch (slots 15–18)

Hypothesis: increasing supervised-contrastive `event_weight` from 0.10 to 0.125 at temperature 0.05, while holding `station_weight=0.05`, NotebookAE, optimizer, sampling, and training budget fixed, may improve event retrieval while retaining the station gain. The four-attempt maximum, 200-epoch cap, patience 40, 1,200-second per-attempt timeout, and exact screen/repeat/seed-43/44 commands were registered in the journal and pushed before training. The seed-42 screen exceeded the previous seed-42 score/MSE ordering and passed guarded standalone evaluation, so all three-seed evidence was collected.

| Slot | Role | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 15 | Seed-42 screen | 0.040650 | 0.284553 | 0.162602 | 0.7314721880 | 42 / 28 | 68 / 175.07 s | `15_supcon_event0125_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/kzmz6cvx) |
| 16 | Exact seed-42 repeat | 0.040650 | 0.284553 | 0.162602 | 0.7314721880 | 42 / 28 | 68 / 178.61 s | `16_supcon_event0125_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/1t2wad1m) |
| 17 | Seed 43 | 0.016260 | 0.276423 | 0.146341 | 0.6888329101 | 43 / 42 | 82 / 209.84 s | `17_supcon_event0125_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/c3dbvw9f) |
| 18 | Seed 44 | 0.065041 | 0.276423 | **0.170732** | 0.4306792109 | 44 / 78 | 118 / 365.87 s | `18_supcon_event0125_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/8u3wk30z) |

The seed-42 repeat matched all 86 shared scientific fields exactly. Slots 15 and 18 passed guarded standalone evaluation on GPU 0 at batch size 128: all 87 shared scientific fields matched with zero mismatches. Slot 18’s evaluation records the frozen benchmark ID above and training source commit `368c2fccf8fdd115d86e2fd64479c28a6286a672`.

### Three-seed evidence

| Seed | Event top-1 | Station top-1 | Score | Canonical val MSE |
|---:|---:|---:|---:|---:|
| 42 (screen and exact repeat) | 0.040650 | 0.284553 | 0.162602 | 0.7314721880 |
| 43 | 0.016260 | 0.276423 | 0.146341 | 0.6888329101 |
| 44 | 0.065041 | 0.276423 | 0.170732 | 0.4306792109 |
| Three-seed mean | 0.040650 | 0.279133 | **0.159892** | — |
| Three-seed sample SD | 0.024390 | 0.004694 | **0.012419** | — |

The new three-seed mean is 0.037940 above the fresh full-budget seed-42 separation baseline (0.121951) and 0.004065 above the preceding temperature-0.05 SupCon configuration at event/station weights 0.10/0.05 (mean 0.155827; sample SD 0.006209). The best single-seed score is 0.170732, above the previous best 0.162602. Event retrieval varies by seed; validation was repeatedly used for selection, so these are validation results, not an untouched test result. Sample SD describes these three observed seed scores; no confidence interval is assigned to the combined objective.

## Latest bounded invocation (slots 19–22)

This invocation completed four attempts under the cumulative lane budget. The frozen benchmark remains `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. Before the batch, benchmark verification passed, all 24 tests passed, the branch was clean and synchronized, 18 slots were consumed, and STOP was absent. Every run used the guarded runner on GPU 0, online W&B in `jlinki/seissepa-autoresearch`, batch size 128, maximum 200 epochs, patience 40, and a 1,200-second per-attempt cap. All four completed runs have the required local artifacts; W&B received no artifact files.

### Hypothesis and registered budget

Hypothesis: at temperature 0.05 and station weight 0.05, increasing supervised-contrastive event weight from 0.125 to 0.15 may improve event retrieval while retaining station retrieval. The NotebookAE architecture, reconstruction weight 1.0, AdamW learning rate 0.0002, weight decay 0.0001, sampling, and training budget stayed fixed. The batch allowed at most four attempts. Slot 19 was the seed-42 screen; the exact repeat and seeds 43/44 were conditional on a better seed-42 score/MSE ordering and an exact guarded standalone evaluation.

The exact commands were registered in the journal and pushed before training:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.15 loss.station_weight=0.05 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event015_seed42 hydra.run.dir=outputs/overnight_2026-10-21/19_supcon_event015_seed42
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.15 loss.station_weight=0.05 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event015_seed42_repeat hydra.run.dir=outputs/overnight_2026-10-21/20_supcon_event015_seed42_repeat
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.15 loss.station_weight=0.05 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=43 wandb.name=overnight_b_supcon_event015_seed43 hydra.run.dir=outputs/overnight_2026-10-21/21_supcon_event015_seed43
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.15 loss.station_weight=0.05 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=44 wandb.name=overnight_b_supcon_event015_seed44 hydra.run.dir=outputs/overnight_2026-10-21/22_supcon_event015_seed44
~~~

### Run inventory

| Slot | Role | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 19 | Seed-42 screen | 0.048780 | 0.276423 | 0.162602 | 0.4190212875 | 42 / 74 | 114 / 413.78 s | `19_supcon_event015_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/7k8uuzov) |
| 20 | Exact seed-42 repeat | 0.048780 | 0.276423 | 0.162602 | 0.4190212875 | 42 / 74 | 114 / 371.44 s | `20_supcon_event015_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/3ssvs767) |
| 21 | Seed 43 | 0.008130 | 0.284553 | 0.146341 | 0.6838424690 | 43 / 47 | 87 / 435.48 s | `21_supcon_event015_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/dsyq53ff) |
| 22 | Seed 44 | 0.048780 | 0.292683 | 0.170732* | 0.4235681187 | 44 / 76 | 116 / 408.82 s | `22_supcon_event015_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/tzcwggj2) |

All required local artifacts are present. Slot 19 passed guarded standalone GPU evaluation at batch size 128: all 101 shared scientific fields matched exactly, with zero mismatches. The result is saved at `outputs/overnight_2026-10-21/19_supcon_event015_seed42/best.evaluation.json`. Slot 20 exactly reproduced slot 19 across all 100 shared scientific fields. Its software versions, CUDA, GPU and device provenance also match; both runs used batch size 128. No candidate from slots 21–22 replaced the incumbent.

### Three-seed evidence

| Seed | Event top-1 | Station top-1 | Score | Canonical val MSE | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 42 (screen and exact repeat) | 0.048780 | 0.276423 | 0.162602 | 0.4190212875 | 74 |
| 43 | 0.008130 | 0.284553 | 0.146341 | 0.6838424690 | 47 |
| 44 | 0.048780 | 0.292683 | 0.170732* | 0.4235681187 | 76 |
| Three-seed mean | 0.035230 | 0.284553 | **0.159892** | — | — |
| Three-seed sample SD | 0.023470 | 0.008130 | **0.012419** | — | — |

The primary three-seed mean is 0.037940 above the fresh full-budget seed-42 separation baseline (0.121951). It matches the prior event-weight-0.125 configuration’s mean to displayed precision (0.159892; sample SD 0.012419). The event mean is lower (0.035230 versus 0.040650), while the station mean is higher (0.284553 versus 0.279133). Validation was repeatedly used for selection; these are validation results, not an untouched test result. The sample SD describes the three observed combined scores. No confidence interval is assigned to the combined objective.

*Slot 22’s full-precision objective is 0.17073170731707316. Slot 18’s is 0.17073170731707318. The displayed scores round alike, but the leaderboard orders slot 18 higher by the exact evaluator value; slot 22’s lower MSE therefore does not trigger the exact-score tie-break. Slot 18 remains the best valid single-run result.

## Current best and next hypothesis

Best valid configuration and checkpoint remain slot 18: model=notebook, supervised-contrastive loss, reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002, weight decay 0.0001, batch size 128, maximum 200 epochs, patience 40. Checkpoint: `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`; event/station top-1 0.065041/0.276423; exact evaluator score 0.17073170731707318; canonical validation MSE 0.4306792109; best epoch 78. Its guarded standalone evaluation matched all 87 shared scientific fields. Its three-seed mean/sample SD are 0.1598915989/0.0124189043.

For the seed-42 comparison, slot 19/20 ties the previous event-weight-0.125 seed-42 score at 0.16260162601626016 and has lower canonical MSE (0.4190212875 versus 0.7314721880). Its evaluation passed exactly, but the event-weight-0.15 three-seed mean does not improve.

Next bounded hypothesis: test temperature 0.075 with event/station weights 0.125/0.05 and all other settings fixed. A seed-42 screen can test whether a softer contrastive temperature improves event retrieval while retaining station retrieval. Continue to the exact repeat and seeds 43/44 only if the seed-42 score/MSE ordering beats its current incumbent (score 0.16260162601626016; canonical validation MSE 0.419021287532869) and guarded standalone evaluation matches exactly. Use at most four attempts, 200 epochs, patience 40 and 1,200 seconds per attempt. No additional attempt was started in this invocation.

## Latest bounded invocation (slot 23)

This invocation completed one of one registered attempts. Before launch, 22 of 64 lane slots were reserved and STOP was absent. Benchmark verification passed for the frozen ID below, all 24 tests passed, and the leaderboard confirmed slot 18 as the incumbent. The prelaunch plan was committed and pushed before training in commit `c0e19ab764e5ab6dcdd6f86d52b9aeddfb902635`. No source or configuration files changed.

### Hypothesis, command and budget

Hypothesis: a fresh seed-42 run of the default NotebookAE separation objective at the overnight budget provides the requested fair 200-epoch baseline anchor for this round. The batch allowed one attempt, up to 200 epochs, patience 40, and 1,200 seconds. The actual run used GPU 0, batch size 128, AdamW learning rate 0.0002, and online W&B in `jlinki/seissepa-autoresearch`.

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=separation loss.reconstruction_weight=1.0 loss.event_weight=0.05 loss.station_weight=0.05 loss.margin=0.2 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_fresh_baseline_seed42_round2 hydra.run.dir=outputs/overnight_2026-10-21/23_fresh_baseline_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 23 | Completed baseline | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 42 / 1 | 41 / 134.72 s | `outputs/overnight_2026-10-21/23_fresh_baseline_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/fsf4ti9z) |

The benchmark ID is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; the run records commit `c0e19ab764e5ab6dcdd6f86d52b9aeddfb902635`. All required local files are present (`config.yaml`, `provenance.json`, `source.tar.gz`, `history.jsonl`, `best.pt`, `metrics.json`, `report.md`, and `wandb.json`). W&B completed online with no artifact files uploaded.

Slot 23 matches slot 1's scientific metrics exactly, including score and canonical MSE. The recorded software and hardware provenance also matches at batch size 128. Source hashes differ only for the later-added `configs/loss/supervised_contrastive.yaml` and `src/seissepa/losses.py`. Slot 23's score is below the lane incumbent, so it is retained as the fresh baseline and is not promoted.

### Current best and next step

The current best valid checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: supervised-contrastive NotebookAE, reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, maximum 200 epochs, patience 40. Its event/station top-1 are 0.065041/0.276423, exact score 0.17073170731707318, canonical validation MSE 0.4306792109, and best epoch 78. Guarded standalone evaluation matched all 87 shared scientific fields.

Its seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; the three-seed mean is 0.1598915989 and sample SD is 0.0124189043. The complete run inventory and W&B links for slots 1–22 are above; slot 23 is added here. This invocation stops after slot 23. The next hypothesis remains the previously proposed temperature 0.075 screen at event/station weights 0.125/0.05, with the rest of the configuration fixed. There are 41 of 64 slots remaining, with at least 12 kept for confirmations.

## Latest bounded invocation (slot 24)

This invocation completed one attempt and stops here. Before the batch, slot 23 had been reserved, the worktree was clean and synchronized, STOP was absent, benchmark verification passed, all 24 tests passed, and the leaderboard showed slot 18 as the incumbent. The batch plan and exact conditional commands were committed and pushed before training in commit 0de43a757b696090acdf1e7d088b9678ca1760b0. The screen used the guarded runner on physical GPU 0, online W&B in jlinki/seissepa-autoresearch, batch size 128, maximum 200 epochs, patience 40, and a 1,200-second per-attempt cap.

### Hypothesis and registered budget

Hypothesis: raising supervised-contrastive temperature from 0.05 to 0.075 may soften the event latent geometry enough to improve seed-42 retrieval while preserving station retrieval. NotebookAE, reconstruction/event/station weights 1.0/0.125/0.05, AdamW learning rate 0.0002, weight decay 0.0001, sampling, and training budget stayed fixed. The batch allowed at most four attempts. Slot 24 was the seed-42 screen; the exact repeat and seeds 43/44 were conditional on improving the seed-42 comparator (score 0.16260162601626016, canonical validation MSE 0.419021287532869) and passing guarded standalone evaluation.

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.075 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_temp0075_seed42 hydra.run.dir=outputs/overnight_2026-10-21/24_supcon_temp0075_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 24 | Completed | 0.048780 | 0.227642 | 0.138211 | 0.9779287941 | 42 / 5 | 45 / 144.31 s | 24_supcon_temp0075_seed42 | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/bwogdcpl) |

The benchmark ID remained abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26; the run records commit 0de43a757b696090acdf1e7d088b9678ca1760b0. All required local artifacts are present. W&B completed online and synced no artifact files. The score is below the seed-42 comparator and canonical MSE is higher, so slot 24 was not standalone-evaluated and slots 25–27 were not launched. Preserve the run and its artifacts; no promotion is made.

## Current best and next hypothesis

Best valid checkpoint remains slot 18: outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt, with NotebookAE, supervised-contrastive loss, reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, maximum 200 epochs, and patience 40. Its event/station top-1 are 0.065041/0.276423, exact score is 0.17073170731707318, canonical validation MSE is 0.4306792109, and best epoch is 78. Its guarded standalone evaluation matched all 87 shared scientific fields.

The seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; their three-seed mean and sample SD are 0.1598915989 and 0.0124189043. The full previous run inventory, paths, W&B URLs, and exact repeatability/evaluation evidence remain above; slot 23's fresh separation baseline and slot 24 are added in their respective sections. At the end of slot 24, 40 of 64 training slots remained. After slot 25, 39 remain, with at least 12 kept for confirmations. The slot-24 invocation used one of its four-attempt maximum.

The balanced-sampler hypothesis was screened in slot 25 and did not improve on the seed-42 comparator. Next hypothesis: train auxiliary linear classification heads on `ze` and `zs` using training event/station labels and cross-entropy, alongside the current SupCon objective. The heads would be used only during training and excluded from inference, leaving NotebookAE's architecture and model inputs unchanged. Register exact loss weights, commands, and budget before a later invocation. Validation has been used repeatedly for selection; these are validation results, not an untouched test result.


## Latest bounded invocation (slot 25)

### Hypothesis and registered budget

Hypothesis: with NotebookAE and supervised-contrastive weights fixed at reconstruction/event/station 1.0/0.125/0.05 and temperature 0.05, drawing each minibatch from 14 training event classes and 14 training station classes, with at least two distinct traces per selected class, could strengthen multi-positive supervision in both latent spaces compared with uniform shuffle. The batch pre-registered four attempts (200 epochs, patience 40, 1,200 seconds each): seed-42 screen, exact seed-42 repeat, and seeds 43/44. Confirmations were conditional on the screen beating the seed-42 score/MSE ordering and passing guarded standalone evaluation. The screen failed that continuation rule, so unused slots 26–28 were not launched.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.balanced_batches=true train.balanced_groups=14 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_balancedgrid14_seed42 hydra.run.dir=outputs/overnight_2026-10-21/25_supcon_balancedgrid14_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 25 | Completed screen; not promoted | 0.048780487805 | 0.235772357724 | 0.142276422764 | 0.715555823775 | 42 / 36 | 76 / 183.41 s | `25_supcon_balancedgrid14_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/n3rswfaz) |

The run used commit `dca129118969de9956d125f350fa6905f5a1ee63` and benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. All required local artifacts are present. It completed online and uploaded zero artifact files. Compared with the best prior seed-42 ordering (score 0.16260162601626016, MSE 0.419021287532869), slot 25 has both a lower score and higher MSE. It was not evaluated standalone and no confirmations followed.

The best valid single-run checkpoint remains slot 18: `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`, with event/station top-1 0.065041/0.276423, exact score 0.17073170731707318, canonical validation MSE 0.4306792109, and best epoch 78. Its guarded standalone evaluation matched all 87 shared scientific fields. Its seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; mean 0.1598915989, sample SD 0.0124189043. All runs 1–25 and their W&B links are listed above.

The next hypothesis is to add training-only linear heads on `ze` and `zs`, supervised by event/station cross-entropy alongside SupCon. Keep the heads out of inference and keep all IDs out of model inputs.

## Latest bounded invocation (slot 26)

This invocation completed one seed-42 screen and stops here. Before launch, 25 of 64 slots were reserved, STOP was absent, the benchmark verification passed, all 24 tests passed, and the leaderboard showed slot 18 as incumbent. The source/config change and registered plan were committed and pushed before training in `1d2e52440f1e72dc8a30195bc40cb9557abfca26`. Slot 26 used the guarded runner on GPU 0, online W&B in `jlinki/seissepa-autoresearch`, batch size 128, maximum 200 epochs, patience 40, and the 1,200-second attempt cap.

### Hypothesis and budget

Hypothesis: adding training-only linear classifiers on the normalized `ze` and `zs` latents, with event and station cross-entropy weights 0.05 alongside the existing SupCon weights 0.125/0.05, could improve latent class supervision. The head weights were included in the optimizer and gradient clipping. Labels were consumed only by the training loss; the heads were absent from NotebookAE and its inference checkpoint. The batch allowed up to four attempts: slot 26 was the seed-42 screen; slots 27–29 were conditional on beating the best seed-42 ordering (score 0.16260162601626016, canonical MSE 0.419021287532869) and passing guarded standalone evaluation with an exact scientific-field match.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.05 loss.station_head_weight=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads005_seed42 hydra.run.dir=outputs/overnight_2026-10-21/26_supcon_heads005_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 26 | Completed screen; not promoted | 0.048780487805 | 0.203252032520 | 0.126016260163 | 0.736037642590 | 42 / 29 | 69 / 158.08 s | `26_supcon_heads005_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/5nspzdb2) |

The run records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and commit `1d2e52440f1e72dc8a30195bc40cb9557abfca26`. Its training-reported elapsed time was 147.76 seconds; guarded wall time was 158.08 seconds. All required local artifacts are present. W&B completed online and synced zero artifact files. The score is below the seed-42 comparator, and canonical MSE is higher; standalone evaluation and slots 27–29 were therefore not run. Preserve the checkpoint and logs without promotion.

The best valid checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`, at score 0.17073170731707318, event/station top-1 0.065041/0.276423, canonical validation MSE 0.4306792109, and best epoch 78. The seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; the three-seed mean is 0.1598915989 and sample SD is 0.0124189043. The new outcome and W&B URL are recorded in `experiments/journal.jsonl`; all runs 1–26 now have local outcomes and W&B links in this report and the earlier inventory.

Next hypothesis: lower both auxiliary cross-entropy weights from 0.05 to 0.01 while keeping NotebookAE, SupCon weights, temperature, optimizer, sampling, and training budget fixed. Run a seed-42 screen, then require the same score/MSE ordering and exact standalone evaluation before repeating seed 42 or collecting seeds 43/44.


## Latest bounded invocation (slot 27)

This invocation ran one seed-42 screen and stops here. The registered batch allowed at most four attempts, up to 200 epochs, patience 40, and a 1,200-second attempt limit. Before launch, 26 of 64 attempt slots were consumed; after slot 27, 37 remain and the minimum 12-slot confirmation reserve is intact. The plan was committed and pushed before training as `4ffe0c2ed33163b9b5f55005c4e74d35a7cfcfd4`. Benchmark verification passed and all 24 tests passed before the batch. Training used the guarded runner on GPU 0 with online W&B in the approved project; the run synced zero artifact files.

### Hypothesis and registered budget

Hypothesis: reducing both training-only auxiliary cross-entropy weights from 0.05 to 0.01 may preserve useful direct class supervision while reducing its interference with the supervised-contrastive latent geometry. NotebookAE, SupCon weights 0.125/0.05, temperature 0.05, optimizer, sampling, and training budget stayed fixed. The auxiliary heads use only training event/station labels and remain outside the model inputs and inference checkpoint. The exact seed-42 repeat and seeds 43/44 were conditional on improving the seed-42 score/MSE ordering and passing exact guarded standalone evaluation.

Exact command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.01 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads001_seed42 hydra.run.dir=outputs/overnight_2026-10-21/27_supcon_heads001_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 27 | Completed screen; not promoted | 0.065040650407 | 0.227642276423 | 0.146341463415 | 0.972801635858 | 42 / 7 | 47 / 337.84 s | `27_supcon_heads001_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ng4tzq8t) |

The run records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and training commit `4ffe0c2ed33163b9b5f55005c4e74d35a7cfcfd4`. Training-reported elapsed time was 326.30 seconds; guarded wall time was 337.84 seconds. All required local artifacts are present. The screen is below the seed-42 comparator (score 0.16260162601626016; canonical MSE 0.419021287532869), so it was not evaluated standalone and slots 28–30 were not launched. Preserve its checkpoint and logs without promotion.

The best valid checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`, with event/station top-1 0.065041/0.276423, exact score 0.17073170731707318, canonical validation MSE 0.4306792109, and best epoch 78. Its seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; their mean is 0.1598915989 and sample SD is 0.0124189043. Its guarded standalone evaluation matched all 87 shared scientific fields. The earlier inventory retains the paths and W&B URLs for slots 1–26; slot 27 is linked above. All results are validation results after repeated selection, not untouched test evidence.

Next hypothesis: keep the event head weight at 0.01 and increase only the station head weight to 0.025, aiming to recover station retrieval while retaining the event improvement seen in slot 27. Hold NotebookAE, SupCon objective, optimizer, sampling, and budget fixed; screen seed 42 before any confirmation runs.

## Latest bounded invocation (slot 28)

This invocation completed one seed-42 screen and stops here. Before launch, 27 of 64 slots were consumed; STOP was absent, the worktree was clean, HEAD matched upstream, benchmark verification passed, all 24 tests passed, and the leaderboard showed slot 18 as the best observed run. The exact plan was committed and pushed before training as `e7743ee54e6ff35c0deb6acc5d1e3a2316483c76`. The guarded runner reverified the benchmark, used GPU 0 and online W&B, and enforced the 1,200-second attempt limit.

### Hypothesis and registered budget

Hypothesis: raising only `station_head_weight` from 0.01 to 0.025 may recover station retrieval after slot 27's event-head screen while retaining `event_head_weight=0.01`. NotebookAE, SupCon weights and temperature, optimizer, sampling, and training budget stayed fixed. The batch registered one attempt only: slot 28, seed 42, up to 200 epochs, patience 40. Confirmation runs were not registered for this invocation.

Exact command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.025 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads001_station0025_seed42 hydra.run.dir=outputs/overnight_2026-10-21/28_supcon_heads001_station0025_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 28 | Completed screen; not promoted | 0.040650406504 | 0.260162601626 | 0.150406504065 | 0.971185763571 | 42 / 9 | 49 / 188.21 s | `28_supcon_heads001_station0025_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vm02ut6r) |

Training-reported elapsed time was 178.35 seconds. The run records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and commit `e7743ee54e6ff35c0deb6acc5d1e3a2316483c76`. All required local artifacts are present. W&B completed online and synced zero artifact files.

The result is below the best seed-42 comparator, slot 19, which scored 0.16260162601626016 with canonical validation MSE 0.419021287532869. Slot 28 has both lower event and station retrieval and a higher MSE. It was therefore not standalone-evaluated and no repeat or other-seed attempts followed. Preserve the checkpoint and logs without promotion.

### Current best and next hypothesis

The best valid single-run checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`, with score 0.17073170731707318, event/station top-1 0.0650406504065/0.276422764228, canonical validation MSE 0.430679210870, and best epoch 78. Its guarded standalone evaluation matched all 87 shared scientific fields. Its seed-42/43/44 scores remain 0.1626016260/0.1463414634/0.1707317073; mean 0.1598915989 and sample SD 0.0124189043.

Slot 28 brings cumulative consumption to 28 of 64 attempts, leaving 36, with the minimum 12-slot confirmation reserve intact. The next hypothesis is to remove the event auxiliary head while retaining `station_head_weight=0.025`, then screen seed 42 to test whether event retrieval recovers without losing the station-head gain. Preregister that as a separate bounded batch. This invocation ends after slot 28 with no new training; validation results remain subject to repeated-selection optimism.

## Latest bounded invocation (slot 29)

This invocation preregistered one seed-42 screen and up to three conditional confirmations. The screen failed the predeclared seed-42 score/MSE criterion, so slots 30–32 were not launched and this invocation stops after slot 29. Before the batch, 28 of 64 attempts were consumed; the stop flag and lane block marker were absent; benchmark verification passed; all 24 tests passed; and the branch was clean and synchronized. The batch plan was pushed before training as commit f1b9c0a607c94abbec6a7dedc9302e4b8b989c54.

### Hypothesis, commands, and budget

Hypothesis: removing the training-only event cross-entropy head while keeping station_head_weight=0.025 may recover event retrieval without losing the station-head gain. NotebookAE, SupCon weights 1.0/0.125/0.05, temperature 0.05, AdamW settings, sampling, and the 200-epoch/patience-40 budget stayed fixed. The batch budget was at most four attempts (slots 29–32), each capped at 1,200 seconds. The seed-42 repeat and seeds 43/44 were conditional on the screen beating the seed-42 ordering (score 0.16260162601626016; canonical validation MSE 0.419021287532869) and passing exact guarded standalone evaluation.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.0 loss.station_head_weight=0.025 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_station0025_eventoff_seed42 hydra.run.dir=outputs/overnight_2026-10-21/29_supcon_heads_station0025_eventoff_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 29 | Completed screen; not promoted | 0.040650406504 | 0.252032520325 | 0.146341463415 | 0.971199064029 | 42 / 9 | 49 / 179.12 s | 29_supcon_heads_station0025_eventoff_seed42 | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/dh7hjt4b) |

The run records benchmark ID abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26 and commit f1b9c0a607c94abbec6a7dedc9302e4b8b989c54. Training-reported elapsed time was 169.08 seconds; guarded elapsed time was 179.12 seconds. All required local artifacts are present. W&B completed online in jlinki/seissepa-autoresearch and uploaded zero artifact files.

The screen is below the best seed-42 comparator (score 0.16260162601626016; canonical validation MSE 0.419021287532869), with lower event and station retrieval and higher MSE. It was not evaluated standalone or promoted. Keep its checkpoint and logs. The best valid single-run checkpoint remains slot 18, outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt: score 0.17073170731707318, event/station top-1 0.065040650407/0.276422764228, canonical validation MSE 0.430679210870, best epoch 78. Its seed-42/43/44 scores and mean/sample SD remain 0.1626016260/0.1463414634/0.1707317073 and 0.1598915989/0.0124189043.

Slot 29 brings cumulative use to 29 of 64 attempts, leaving 35; at least 12 slots remain reserved for confirmations. The next hypothesis is to test an event-only auxiliary classifier at event_head_weight=0.025 and station_head_weight=0.0, holding all other settings fixed and screening seed 42 before any confirmation. These are validation results after repeated selection, not untouched test evidence.

## Latest bounded invocation (slot 30)

This invocation started with 29 of 64 slots consumed and completed one seed-42 screen, then stopped under its preregistered continuation rule. The STOP file and lane BLOCKED marker were absent before launch; the worktree was clean and synchronized. Benchmark verification passed and all 24 tests passed before the batch. The event-only auxiliary-head plan was committed and pushed before training as `a2d832ba1f437f1d6e3cc71802b9290a9093edeb`. The screen used the guarded runner on GPU 0 with online W&B, batch size 128, maximum 200 epochs, patience 40, and a 1,200-second attempt cap. The registered batch allowed at most four attempts: slot 30 as the seed-42 screen, followed conditionally by an exact repeat and seeds 43/44.

### Hypothesis, command, and budget

Hypothesis: an event-only training classifier on `ze` at cross-entropy weight 0.025, with station-head weight 0, may improve event retrieval without station-classifier gradients interfering with event geometry. NotebookAE, SupCon reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, sampling, and training budget stayed fixed. Identity labels supervised training latents only; the model inputs and retrieval ranking were unchanged.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.025 loss.station_head_weight=0.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_event0025_stationoff_seed42 hydra.run.dir=outputs/overnight_2026-10-21/30_supcon_heads_event0025_stationoff_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 30 | Completed screen; not promoted | 0.065040650407 | 0.243902439024 | 0.154471544715 | 0.977925246704 | 42 / 5 | 45 / 200.27 s | `30_supcon_heads_event0025_stationoff_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/3su0c0e5) |

The run records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and commit `a2d832ba1f437f1d6e3cc71802b9290a9093edeb` in both metrics and provenance. Training-reported elapsed time was 189.98 seconds; guarded wall time was 200.27 seconds. All required local artifacts are present. W&B completed online and synced zero artifact files.

Slot 30 is below the seed-42 comparator, slot 19/20 (score 0.16260162601626016; canonical validation MSE 0.419021287532869), on both score and MSE. It was not standalone-evaluated or promoted; slots 31–33 were not launched. Preserve its checkpoint and logs. No source or config files were changed for this run.

### Current best and lane state

The best valid checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: supervised-contrastive NotebookAE, reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, maximum 200 epochs, patience 40. Its event/station top-1 are 0.065040650407/0.276422764228, exact score 0.17073170731707318, canonical validation MSE 0.430679210870, and best epoch 78. Guarded standalone evaluation matched all 87 shared scientific fields.

| Seed | Event top-1 | Station top-1 | Score |
|---:|---:|---:|---:|
| 42 | 0.048780487805 | 0.276422764228 | 0.162601626016 |
| 43 | 0.008130081301 | 0.284552845528 | 0.146341463415 |
| 44 | 0.048780487805 | 0.292682926829 | 0.170731707317 |
| Three-seed mean | — | — | 0.1598915989 |
| Three-seed sample SD | — | — | 0.0124189043 |

The fresh overnight-budget separation baseline remains slot 23 at score 0.121951219512 (event 0.040650406504, station 0.203252032520; MSE 0.985821658387). Slot 30 brings total lane consumption to 30 of 64, leaving 34 slots; 20 seed-42 attempts have been used, below the 48-screen cap, and the 12-slot confirmation reserve remains intact. All run paths and W&B URLs for slots 1–30 are retained in this report and journal.

Next bounded hypothesis: reduce the event-only auxiliary cross-entropy weight to 0.01 while keeping the station-head weight at 0, to test whether lighter direct event supervision avoids disrupting the SupCon geometry. This invocation ends after slot 30. Validation has been used repeatedly for selection; these are validation results, not untouched test evidence.


## Latest bounded invocation (slots 31–34)

This invocation started with 30 consumed attempts and completed the four preregistered attempts, then stopped. The worktree was clean and synchronized before launch; the frozen benchmark verification passed and all 24 tests passed. GPU 0 was used only. The slot-31 plan and exact commands were committed and pushed before training as `07ce1a1ee2290f7131af19459377a114c6f623f2`. Slots 32–34 were conditional confirmations specified in the pushed plan. Every run completed with online W&B in the approved project, produced all required local artifacts, and uploaded zero artifact files.

### Hypothesis and budget

Hypothesis: lowering the training-only event auxiliary cross-entropy weight from 0.025 to 0.01, with the station head disabled, may preserve event supervision while reducing interference with SupCon geometry. NotebookAE, SupCon weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW (learning rate 0.0002, weight decay 0.0001), batch size 128, gradient clipping 1.0, and unbalanced sampling stayed fixed. Each attempt allowed up to 200 epochs, patience 40, and a 1,200-second guard timeout. Slots 31–34 were the seed-42 screen, exact seed-42 repeat, and seeds 43/44.

Exact seed-42 screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_event001_stationoff_seed42 hydra.run.dir=outputs/overnight_2026-10-21/31_supcon_heads_event001_stationoff_seed42
~~~

### Outcomes

| Slot | Seed | Event top-1 | Station top-1 | Score | Canonical val MSE | Best epoch / epochs | Guarded elapsed | Run | W&B |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 31 | 42 | 0.065040650407 | 0.268292682927 | 0.166666666667 | 0.740670498947 | 26 / 66 | 210.23 s | `outputs/overnight_2026-10-21/31_supcon_heads_event001_stationoff_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/349tz3xu) |
| 32 | 42 repeat | 0.065040650407 | 0.268292682927 | 0.166666666667 | 0.740670498947 | 26 / 66 | 216.54 s | `outputs/overnight_2026-10-21/32_supcon_heads_event001_stationoff_seed42_repeat` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/99uem2dv) |
| 33 | 43 | 0.032520325203 | 0.260162601626 | 0.146341463415 | 0.374979653736 | 87 / 127 | 306.55 s | `outputs/overnight_2026-10-21/33_supcon_heads_event001_stationoff_seed43` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/384iqrcs) |
| 34 | 44 | 0.024390243902 | 0.300813008130 | 0.162601626016 | 0.658428433532 | 56 / 96 | 292.76 s | `outputs/overnight_2026-10-21/34_supcon_heads_event001_stationoff_seed44` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/3ounqz16) |

The screen's guarded standalone evaluation used batch size 128 and exactly matched all 101 shared scientific fields, including event/station retrieval, combined retrieval, and canonical MSE. The training-only metadata fields were elapsed time and W&B URL. The exact seed-42 repeat also matched all 100 shared scientific fields with no differences.

| Seed | Event top-1 | Station top-1 | Score |
|---:|---:|---:|---:|
| 42 | 0.065040650407 | 0.268292682927 | 0.166666666667 |
| 43 | 0.032520325203 | 0.260162601626 | 0.146341463415 |
| 44 | 0.024390243902 | 0.300813008130 | 0.162601626016 |
| Three-seed mean | — | — | 0.1585365854 |
| Three-seed sample SD | — | — | 0.0107550866 |

The seed-42 screen improves on the prior best seed-42 comparator, slot 19 (0.162601626016), by 0.004065040650, and its repeat is exact. Across seeds, however, this configuration's mean is below the existing best configuration's mean of 0.1598915989 (sample SD 0.0124189043). Do not promote it as the lane's best configuration. Preserve its four run directories, checkpoints, logs, and W&B records.

### Current best and lane state

The best validated checkpoint and configuration remain slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with the SupCon loss (reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05), AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, and the 200-epoch/patience-40 budget. Its event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its three-seed mean/sample SD are 0.1598915989/0.0124189043; its guarded standalone evaluation matched all shared scientific fields. The fresh overnight separation baseline remains slot 23 at score 0.121951219512, event/station 0.040650406504/0.203252032520, and MSE 0.985821658387.

This batch brings lane use to 34 of 64 attempts, with 30 slots remaining and 22 seed-42 attempts used against the 48-screen cap. The 12-slot confirmation reserve remains intact. Run inventories and W&B URLs for slots 1–30 remain in the sections above; slots 31–34 are listed here and all exact commands and provenance are in `experiments/journal.jsonl`.

Next hypothesis: test class-balanced training batches with the event-only head at cross-entropy weight 0.01 to see whether broader class coverage stabilizes event retrieval across seeds, holding NotebookAE, SupCon weights, optimizer, and budget fixed. Any follow-up should be preregistered as a new bounded batch. Validation has been used repeatedly for selection; these remain validation results, not untouched test evidence.


## Latest bounded invocation (slot 35)

This invocation registered at most four attempts and completed only the seed-42 screen. Before launch, 34 of 64 slots were consumed, STOP and BLOCKED were absent, the worktree was clean and synchronized, the frozen benchmark verified, all 24 tests passed, and GPU 0 was idle. The plan and exact conditional commands were committed and pushed as `7d5ac0a04391a71e1ff8e05918e6bbc844ea87ef`. Slot 35 used the guarded runner with a 1,200-second limit and online W&B. The screen failed its predeclared ranking criterion, so slots 36–38 were not launched.

### Hypothesis and registered budget

Hypothesis: class-balanced training batches over 14 event and 14 station groups, with at least two examples per group, may make SupCon positive pairs more consistent and stabilize event retrieval while the event-only auxiliary CE weight remains 0.01. NotebookAE, SupCon weights (1.0/0.125/0.05), temperature 0.05, AdamW (learning rate 0.0002, weight decay 0.0001), batch size 128, clipping 1.0, and the 200-epoch/patience-40 budget stayed fixed. Labels were used only by the existing train-only sampler and loss. The maximum batch was four attempts (slots 35–38); the repeat and seeds 43/44 were conditional on slot 35 outranking comparator slot 31 and passing exact guarded standalone evaluation.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=true train.balanced_groups=14 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_event001_balanced_seed42 hydra.run.dir=outputs/overnight_2026-10-21/35_supcon_heads_event001_balanced_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 35 | Completed screen; not promoted | 0.048780487805 | 0.227642276423 | 0.138211382114 | 0.737150221771 | 42 / 25 | 65 / 198.47 s | `outputs/overnight_2026-10-21/35_supcon_heads_event001_balanced_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/wcopid7d) |

Training-reported elapsed time was 185.13 seconds; guarded wall time was 198.47 seconds. The run records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and training commit `7d5ac0a04391a71e1ff8e05918e6bbc844ea87ef`. All required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch` and uploaded zero artifact files.

Slot 35 is below seed-42 comparator slot 31 (score 0.166666666667; MSE 0.740670498947). Its MSE is lower, but the frozen ordering prioritizes score. Therefore it was not standalone-evaluated and slots 36–38 were not launched. Preserve its checkpoint and logs. The read-only leaderboard still ranks slot 18 first at score 0.170731707317, followed by slot 22 at the same displayed score.

### Current best and lane state

The best validated configuration/checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with supervised contrastive weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Its event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its three-seed scores are 0.162601626016 / 0.146341463415 / 0.170731707317 (seeds 42/43/44), mean 0.1598915989 and sample SD 0.0124189043. Its standalone evaluation matched all shared scientific fields. The fresh overnight separation baseline remains slot 23 at score 0.121951219512 (event/station 0.040650406504/0.203252032520; MSE 0.985821658387).

Slot 35 brings cumulative use to 35 of 64 attempts, leaving 29. There are 23 seed-42 attempts against the 48-screen cap; at least 12 remaining slots are preserved for confirmations. Slots 1–34 and their W&B URLs remain inventoried in the preceding report sections and journal; slot 35 is listed above. The next hypothesis is to reduce `train.balanced_groups` from 14 to 8 while holding event-only CE 0.01 and all other settings fixed, increasing positive-pair density per sampled group. This invocation ends after slot 35. These are validation results after repeated selection, not untouched test evidence.


## Latest bounded invocation (slot 36)

This invocation started with 35 of 64 slots consumed. The branch was clean and synced, STOP and BLOCKED were absent, benchmark verification passed, all 24 existing tests passed, and GPU 0 was idle. The groups-8 batch plan and four exact conditional commands were committed and pushed before training as `c2b8ac95d614b4bf867457adfbba0bfb9aeeb059`. The invocation budget was at most four attempts (slots 36–39), each capped at 200 epochs, patience 40, and 1,200 seconds.

### Hypothesis and registered budget

Hypothesis: reducing balanced training groups from 14 to 8 while keeping the event-only auxiliary CE weight at 0.01 may increase positive-pair density and improve retrieval stability. NotebookAE, SupCon weights (1.0/0.125/0.05), temperature 0.05, AdamW (learning rate 0.0002, weight decay 0.0001), batch size 128, clipping 1.0, and the epoch/patience budget stayed fixed. Identity labels were used only by the existing train-only sampler and latent losses.

Exact seed-42 screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=true train.balanced_groups=8 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_event001_balanced_groups08_seed42 hydra.run.dir=outputs/overnight_2026-10-21/36_supcon_heads_event001_balanced_groups08_seed42
~~~

The exact seed-42 repeat and seeds 43/44 were conditional on slot 36 outranking seed-42 comparator slot 31 under score/MSE/epoch ordering and passing guarded standalone evaluation.

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 36 | Failed before first epoch | — | — | — | — | 42 / — | 0 / 11.35 s | `outputs/overnight_2026-10-21/36_supcon_heads_event001_balanced_groups08_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/hhw6yhq6) |

The train-only `BalancedGridBatchSampler` exhausted its candidate draws before forming the first 128-example batch. With eight event and eight station groups, the grid can contain at most 64 cells, so it cannot supply the requested batch size. No retrieval metrics, checkpoint, history, or report were produced. The failed run's local config, provenance, source archive, W&B metadata, console log, and training log are preserved. Provenance records the benchmark ID and plan commit. W&B initialized in online mode; zero artifact files were uploaded. This was a sampler feasibility failure, not an authentication, benchmark-integrity, upstream-push, or online-logging failure.

Slots 37–39 were not launched because the screen did not produce a candidate to confirm. Preserve slot 36 and its W&B run. No source or protected files were changed.

### Current best and lane state

The best validated checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with supervised contrastive weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Its event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its guarded standalone evaluation matched the shared scientific fields. Its seed-42/43/44 scores are 0.162601626016 / 0.146341463415 / 0.170731707317, with mean 0.1598915989 and sample SD 0.0124189043. The fresh overnight separation baseline remains slot 23 at score 0.121951219512 (event/station 0.040650406504/0.203252032520; MSE 0.985821658387).

Slot 36 brings cumulative use to 36 of 64 attempts, leaving 28. There are 24 seed-42 attempts against the 48-screen cap; at least 12 slots remain for confirmations. Slots 1–35 and their W&B URLs remain inventoried in the preceding report sections and journal; slot 36 is listed above.

Next hypothesis: keep batch size 128 and test balanced_groups=12. A 12-by-12 train grid has at most 144 cells, enough in principle to form a 128-example batch while using fewer groups than 14. First confirm sampler feasibility, then compare retrieval using the same frozen score and confirmation rules. This invocation ends after slot 36. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence.

## Latest bounded invocation (slot 37)

This invocation ran one seed-42 screen and stops here. Before the batch, 36 of 64 attempts were consumed; the branch was clean and synchronized, STOP and BLOCKED were absent, benchmark verification passed, all 24 tests passed, and the leaderboard showed slot 18 as the best valid single-run checkpoint. The train-only sampler feasibility preflight at 12 groups formed all nine seed-42 batches (eight of size 128 and a final size 78). The plan and four exact conditional commands were committed and pushed before training as `5a9fba74eeff3bddf0f03e99fec272ab323ed1f7`. Slot 37 used the guarded runner on GPU 0 with online W&B, up to 200 epochs, patience 40, batch size 128, and a 1,200-second attempt limit.

### Hypothesis and registered budget

Hypothesis: with event-only auxiliary CE weight 0.01 and SupCon weights fixed, reducing train-only balanced groups from 14 to 12 may provide denser positive groups while retaining feasible 128-example batches, improving retrieval over the current seed-42 comparator. NotebookAE, temperature 0.05, AdamW learning rate 0.0002, weight decay 0.0001, clipping 1.0, batch size 128, and the 200-epoch/patience-40 budget stayed fixed. The batch registered up to four attempts: slot 37 was the seed-42 screen; slots 38-40 (exact seed-42 repeat and seeds 43/44) required a better seed-42 score/MSE/epoch ordering than slot 31 and an exact guarded standalone evaluation.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive_heads loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.05 loss.embedding_dim=64 loss.event_classes=35 loss.station_classes=35 loss.event_head_weight=0.01 loss.station_head_weight=0.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=true train.balanced_groups=12 device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_heads_event001_balanced_groups12_seed42 hydra.run.dir=outputs/overnight_2026-10-21/37_supcon_heads_event001_balanced_groups12_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---|---|---|
| 37 | Completed screen; not promoted | 0.065041 | 0.203252 | 0.134146 | 0.9577550397 | 42 / 11 | 51 / 178.29 s | `37_supcon_heads_event001_balanced_groups12_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/fu96y98j) |

Training-reported elapsed time was 168.52 seconds. The benchmark ID remained `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; provenance records commit `5a9fba74eeff3bddf0f03e99fec272ab323ed1f7`. All required local artifacts are present. W&B completed online and synced zero artifact files.

Slot 37 scores 0.032520 below the seed-42 comparator slot 31 (0.166667; event/station 0.065041/0.268293; canonical MSE 0.7406704989). Its canonical MSE is also higher. It was not evaluated standalone and slots 38-40 were not launched. Preserve its checkpoint and logs without promotion.

### Current best and lane state

The best validated checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with supervised-contrastive weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Its event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its guarded standalone evaluation matched all shared scientific fields. The seed-42/43/44 scores remain 0.162601626016/0.146341463415/0.170731707317; mean 0.1598915989, sample SD 0.0124189043.

Slot 37 brings cumulative use to 37 of 64 attempts, leaving 27. There are 25 seed-42 attempts against the 48-screen cap; at least 12 confirmation slots remain reserved. Slots 1-36 and their W&B URLs remain inventoried above and in the journal; slot 37 is listed here.

Next hypothesis: test a colder supervised-contrastive temperature of 0.025 with event/station weights 0.125/0.05 and uniform sampling. Sharper logits may improve positive-versus-negative separation. Register a seed-42 screen against the current seed-42 comparator before any launch. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence.



## Latest bounded invocation (slot 38)

This invocation registered up to four attempts and completed one seed-42 screen. Before training, 37 of 64 lane slots were consumed, 25 seed-42 attempts had been used against the 48-screen cap, and 27 slots remained. STOP and BLOCKED were absent; the branch was clean and synchronized; benchmark verification passed; all 24 tests passed; and GPU 0 was idle. The colder-temperature plan was committed and pushed before launch as `1b689b9d11ce6995916d6cc7f5dcd392c233d3ea`. The guarded runner verified the frozen benchmark and ran slot 38 on GPU 0 with online W&B.

### Hypothesis and registered budget

Hypothesis: reducing the supervised-contrastive temperature from 0.05 to 0.025 at event/station weights 0.125/0.05 may sharpen positive-versus-negative separation and improve seed-42 retrieval. NotebookAE, uniform training sampling, AdamW (learning rate 0.0002, weight decay 0.0001), batch size 128, gradient clipping 1.0, and the 200-epoch/patience-40 budget stayed fixed.

The batch registered at most four attempts (slots 38–41). Slot 38 was the seed-42 screen. The exact seed-42 repeat and seeds 43/44 were conditional on outranking seed-42 comparator slot 31 under the frozen score/MSE/earlier-epoch ordering and matching the guarded standalone evaluator.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.025 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_temp0025_event0125_seed42 hydra.run.dir=outputs/overnight_2026-10-21/38_supcon_temp0025_event0125_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 38 | Completed screen; not promoted | 0.056911 | 0.235772 | 0.146341 | 0.770837568285 | 42 / 21 | 61 / 234.72 s | `outputs/overnight_2026-10-21/38_supcon_temp0025_event0125_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ltl1lxg3) |

Training-reported elapsed time was 224.30 seconds. The benchmark ID remained `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; provenance records commit `1b689b9d11ce6995916d6cc7f5dcd392c233d3ea`. The required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch` and uploaded zero artifact files.

Slot 38 is below seed-42 comparator slot 31 (score 0.166667; event/station 0.065041/0.268293; canonical MSE 0.740670498947). Its canonical MSE is also higher. It was not standalone-evaluated, and slots 39–41 were not launched. Preserve its checkpoint and logs.

### Current best and lane state

The best valid overall checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`, with score 0.170731707317, event/station 0.065040650407/0.276422764228, canonical validation MSE 0.430679210870, and best epoch 78. Its seed-42/43/44 scores remain 0.162601626016 / 0.146341463415 / 0.170731707317; mean 0.1598915989 and sample SD 0.0124189043. Its guarded standalone evaluation matched all shared scientific fields. The best seed-42 comparator remains slot 31 at score 0.166666666667. The fresh overnight separation baseline remains slot 23 at score 0.121951219512.

Slot 38 brings lane use to 38 of 64 attempts, leaving 26. There are 26 seed-42 attempts against the 48-screen cap; the 12-slot confirmation reserve remains intact. Slots 1–37 and their W&B URLs remain inventoried above and in the journal; slot 38 is listed here.

Next hypothesis: test temperature 0.075 with event/station weights 0.125/0.05 and uniform sampling. A modestly warmer softmax may recover retrieval after temperature 0.025 underperformed. Keep NotebookAE and all other settings fixed. This invocation ends after slot 38. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence.


## Latest bounded invocation (slot 39)

This invocation ended after the seed-42 screen. Before the batch, 38 attempts were reserved, STOP and BLOCKED were absent, the lane worktree was clean and synchronized, the frozen benchmark verified, all 24 tests passed, and GPU 0 was idle. The four-attempt maximum and exact screen/repeat/seeds 43/44 commands were preregistered and pushed as commit `994938ac00604dfa73f2190cb4f17d661eb90290`. Slot 39 used the guarded runner on physical GPU 0 with online W&B. The screen did not clear its predeclared continuation criterion, so the repeat and seed confirmations were not launched.

### Hypothesis and registered budget

Hypothesis: raising supervised-contrastive temperature from 0.05 to 0.075 at event/station weights 0.125/0.05 may recover seed-42 retrieval after temperature 0.025 underperformed. NotebookAE, reconstruction weight 1.0, uniform sampling, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, gradient clipping 1.0, and the 200-epoch/patience-40 budget stayed fixed. The maximum batch was four attempts (slots 39–42); slots 40–42 were conditional on slot 39 outranking seed-42 comparator slot 31 and passing guarded standalone evaluation exactly.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.075 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event0125_temp0075_seed42 hydra.run.dir=outputs/overnight_2026-10-21/39_supcon_event0125_temp0075_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 39 | Completed screen; not promoted | 0.048780487805 | 0.227642276423 | 0.138211382114 | 0.977928794119 | 42 / 5 | 45 / 187.39 s | `outputs/overnight_2026-10-21/39_supcon_event0125_temp0075_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/r7gxm9g5) |

The run's metrics report 174.11 seconds elapsed; guarded wall time was 187.39 seconds. Provenance records benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` and training commit `994938ac00604dfa73f2190cb4f17d661eb90290`. All required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch` and uploaded zero artifact files.

Slot 39 is below seed-42 comparator slot 31 (score 0.166666666667; event/station 0.065040650407/0.268292682927; canonical MSE 0.740670498947), and its MSE is also higher. It was not standalone-evaluated. Slots 40–42 were not launched. Preserve the run and its logs.

### Current best and lane state

The best valid checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with SupCon reconstruction/event/station weights 1.0/0.125/0.05, temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Its score is 0.17073170731707318 (event/station 0.06504065040650407/0.2764227642276423), canonical validation MSE 0.43067921087042654, and best epoch 78. Its guarded standalone evaluation matched all scientific fields. The seed-42/43/44 scores are 0.162601626016 / 0.146341463415 / 0.170731707317, with mean 0.1598915989 and sample SD 0.0124189043. Slot 22 rounds to the same displayed score but has a slightly lower exact score and was not promoted. The fresh separation baseline remains slot 23 at 0.121951219512.

Slot 39 brings cumulative use to 39 of 64 attempts, leaving 25. There are 27 seed-42 attempts against the 48-attempt screen cap, and the minimum 12-slot confirmation reserve remains intact. Slots 1–38 and their W&B URLs remain inventoried above and in `experiments/journal.jsonl`; slot 39 is listed here. The next hypothesis is to test temperature 0.10 at weights 0.125/0.05 with uniform sampling, keeping NotebookAE and other settings fixed. This invocation ends here. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence.

## Latest bounded invocation (slot 40)

This invocation ran one seed-42 screen and stopped under its preregistered continuation rule. Before training, 39 slots were reserved, 27 seed-42 runs had been used, the worktree was clean and synchronized on `research/overnight-20261021-b`, STOP/BLOCKED were absent, GPU 0 was idle, benchmark verification passed, and all 24 tests passed. The four-attempt plan was committed and pushed before launch as `990507fb1f6345aa050189b1c2efd503a0f076d6`. The guarded runner passed the frozen benchmark check and ran slot 40 on GPU 0 with online W&B.

### Hypothesis and registered budget

Hypothesis: raising supervised-contrastive temperature from 0.075 to 0.10 at event/station weights 0.125/0.05 may recover seed-42 retrieval after the temperature-0.075 screen underperformed. NotebookAE, reconstruction weight 1.0, uniform sampling, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, gradient clipping 1.0, 200-epoch maximum and patience 40 stayed fixed. The batch allowed at most four attempts, with 1,200 seconds per attempt. The exact seed-42 repeat and seeds 43/44 were conditional on slot 40 outranking seed-42 comparator slot 31 and passing exact guarded standalone evaluation.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.05 loss.temperature=0.10 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event0125_temp010_seed42 hydra.run.dir=outputs/overnight_2026-10-21/40_supcon_event0125_temp010_seed42
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 40 | Completed screen; not promoted | 0.024390243902 | 0.227642276423 | 0.126016260163 | 0.970525191916 | 42 / 9 | 49 / 189.48 s | `outputs/overnight_2026-10-21/40_supcon_event0125_temp010_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/5xcxahcs) |

The training-reported elapsed time was 178.58 seconds. Slot 40 used commit `990507fb1f6345aa050189b1c2efd503a0f076d6` and benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. All required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch`; zero artifact files were uploaded.

Slot 40 is below seed-42 comparator slot 31 (score 0.166666666667, canonical MSE 0.740670498947) on both score and MSE. It was not standalone-evaluated; conditional slots 41–43 were not launched. Preserve the checkpoint and logs.

### Current best and lane state

The best validated checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with supervised-contrastive weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its standalone evaluation matched the shared scientific fields. Scores for seeds 42/43/44 were 0.162601626016/0.146341463415/0.170731707317, mean 0.1598915989 and sample SD 0.0124189043. The fresh separation baseline remains slot 23 at score 0.121951219512 (event/station 0.040650406504/0.203252032520; MSE 0.985821658387).

Slot 40 brings cumulative lane use to 40 of 64 attempts, leaving 24. There are 28 seed-42 attempts against the 48-screen cap; at least 12 slots remain reserved for confirmations. This invocation's only run and W&B URL are listed above; all earlier runs remain inventoried above and in the journal.

Next hypothesis: at temperature 0.05 and event weight 0.125, raise station weight from 0.05 to 0.075 to test whether stronger station-latent supervision improves the balanced objective while retaining event retrieval. Keep NotebookAE, optimizer, uniform sampling, and training budget fixed. Validation was repeatedly used for selection; these are validation results, not untouched test evidence.

## Latest bounded invocation (slot 41)

This invocation preregistered a maximum of four attempts and completed one seed-42 screen. Before the batch, 40 of 64 attempts were consumed, 28 seed-42 screens had been used, STOP and BLOCKED were absent, the branch was clean and synchronized, benchmark verification passed, all 24 tests passed, and GPU 0 was idle. The plan and exact conditional commands were committed and pushed before training as `904a8db6637522d58ea2adcf826e702e87554d0b`. The guarded runner verified the frozen benchmark and ran slot 41 on physical GPU 0 with online W&B.

### Hypothesis and registered budget

Hypothesis: at the best observed SupCon temperature 0.05 and event weight 0.125, raising station weight from 0.05 to 0.075 may improve station geometry and the balanced retrieval objective while retaining event retrieval. NotebookAE, reconstruction weight, AdamW learning rate 0.0002 and weight decay 0.0001, uniform sampling, batch size 128, gradient clipping 1.0, and the 200-epoch/patience-40 budget stayed fixed. The batch preregistered slots 41–44 (at most four runs; 1,200 seconds each). The exact seed-42 repeat and seeds 43/44 were conditional on outranking seed-42 comparator slot 31 and matching the guarded standalone evaluator exactly.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.075 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event0125_station0075_seed42 hydra.run.dir=outputs/overnight_2026-10-21/41_supcon_event0125_station0075_seed42 threads=4
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 41 | Completed screen; not promoted | 0.073171 | 0.260163 | 0.166667 | 0.741104390130 | 42 / 26 | 66 / 230.96 s | `outputs/overnight_2026-10-21/41_supcon_event0125_station0075_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/86s5jgtz) |

Training-reported elapsed time was 220.70 seconds. Provenance records training commit `904a8db6637522d58ea2adcf826e702e87554d0b` and benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. All required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch` and uploaded zero artifact files.

Slot 41 ties seed-42 comparator slot 31 on the exact primary score (0.166666666667), but its canonical MSE is higher by 0.000433891183 (0.741104390130 versus 0.740670498947); the best epochs also tie at 26. Slot 31 therefore remains ahead under the frozen score/MSE/epoch ordering. Slot 41 improved event retrieval relative to slot 31 (0.073171 versus 0.065041) while station retrieval fell (0.260163 versus 0.268293). The preregistered continuation criterion was not met: slot 41 was not standalone-evaluated and slots 42–44 were not launched. Preserve the run and logs.

### Current best and lane state

The best validated checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with SupCon weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Its event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its guarded standalone evaluation matched all shared scientific fields. The seed-42/43/44 scores remain 0.162601626016/0.146341463415/0.170731707317; mean 0.1598915989 and sample SD 0.0124189043. The best seed-42 comparator remains slot 31. The fresh overnight separation baseline remains slot 23 at score 0.121951219512 (event/station 0.040650406504/0.203252032520; MSE 0.985821658387).

Slot 41 brings cumulative lane use to 41 of 64 attempts, leaving 23. There are 29 seed-42 attempts against the 48-screen cap; at least 12 slots remain reserved for confirmations. Slots 1–40 and their W&B URLs remain inventoried in the preceding report sections and `experiments/journal.jsonl`; slot 41 is recorded above and in the journal.

Next hypothesis: at SupCon temperature 0.05 and event weight 0.125, reduce station weight to 0.025 to test whether weakening the station loss recovers station retrieval while preserving the event gain seen at station weight 0.075. Keep NotebookAE and all other settings fixed. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence.

## Latest bounded invocation (slot 42)

This invocation completed one seed-42 screen from a pre-registered batch of at most four attempts. Before training, 41 of 64 cumulative lane slots were reserved, STOP and BLOCKED were absent, the branch was clean and synchronized, benchmark verification passed, all 24 tests passed, the leaderboard was inspected, and GPU 0 was idle. The plan and exact conditional commands were committed and pushed before training as `a674c6c330d40a8304cc9234ef2b3e5fa3e95663`. The guarded runner verified the frozen benchmark and ran slot 42 on GPU 0 with online W&B.

### Hypothesis and registered budget

Hypothesis: at SupCon temperature 0.05 and event weight 0.125, lowering station weight from 0.075 to 0.025 may recover station retrieval while retaining the event gain seen at station weight 0.075, improving the balanced objective. NotebookAE, reconstruction weight 1.0, AdamW learning rate 0.0002 and weight decay 0.0001, uniform sampling, batch size 128, gradient clipping 1.0, and the 200-epoch/patience-40 budget stayed fixed.

The batch allowed at most four attempts, up to 200 epochs each, patience 40, and 1,200 seconds per attempt. It reserved slots 42–45 conditionally. Exact seed-42 repeat and seeds 43/44 required the screen to outrank seed-42 comparator slot 31 under score/MSE/epoch ordering and pass guarded standalone evaluation exactly; seeds 43/44 also required an exact seed-42 repeat.

Exact screen command:

~~~bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive loss.reconstruction_weight=1.0 loss.event_weight=0.125 loss.station_weight=0.025 loss.temperature=0.05 optimizer.lr=0.0002 optimizer.weight_decay=0.0001 train.epochs=200 train.patience=40 train.batch_size=128 train.grad_clip=1.0 train.balanced_batches=false device=cuda:0 wandb.mode=online wandb.entity=jlinki wandb.project=seissepa-autoresearch seed=42 wandb.name=overnight_b_supcon_event0125_station0025_seed42 hydra.run.dir=outputs/overnight_2026-10-21/42_supcon_event0125_station0025_seed42 threads=4
~~~

### Outcome

| Slot | Status | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed / best epoch | Epochs / guarded elapsed | Run | W&B |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 42 | Completed screen; not promoted | 0.065041 | 0.219512 | 0.142276 | 0.985821598505 | 42 / 1 | 41 / 172.25 s | `outputs/overnight_2026-10-21/42_supcon_event0125_station0025_seed42` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/p8y3aemo) |

Training-reported elapsed time was 161.29 seconds. The training commit is `a674c6c330d40a8304cc9234ef2b3e5fa3e95663`; the benchmark ID is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. All required local artifacts are present. W&B completed online in `jlinki/seissepa-autoresearch` and uploaded zero artifact files.

Slot 42 is below seed-42 comparator slot 31 (score 0.166666666667; event/station 0.065041/0.268293; MSE 0.740670498947; best epoch 26) on the primary score and has higher canonical MSE. Station top-1 fell to 0.219512 while event top-1 matched the comparator at 0.065041. It does not qualify for standalone evaluation or promotion; preserve the run and logs, and leave conditional slots 43–45 unlaunched.

### Current best and lane state

The best validated checkpoint remains slot 18, `outputs/overnight_2026-10-21/18_supcon_event0125_seed44/best.pt`: NotebookAE with supervised-contrastive weights (reconstruction/event/station 1.0/0.125/0.05), temperature 0.05, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, up to 200 epochs and patience 40. Event/station top-1 are 0.065040650407/0.276422764228, score 0.170731707317, canonical validation MSE 0.430679210870, and best epoch 78. Its guarded standalone evaluation matched all shared scientific fields.

| Seed | Run | Score |
|---:|---|---:|
| 42 | slot 15 | 0.162601626016 |
| 43 | slot 17 | 0.146341463415 |
| 44 | slot 18 | 0.170731707317 |
| Three-seed mean | — | 0.159891598916 |
| Three-seed sample SD | — | 0.012418904322 |

Slot 42 brings cumulative lane use to 42 of 64 attempts, leaving 22. There are 30 seed-42 screens against the 48-screen cap; at least 12 slots remain available for exact repeats and seed confirmations. Slots 1–41 and their W&B links remain in the preceding report sections and `experiments/journal.jsonl`; slot 42 is recorded above and in the journal.

Next hypothesis: use the train-only balanced grid batch sampler with the best observed SupCon weights (1.0/0.125/0.05, temperature 0.05). Requiring at least two training examples for each selected event and station class per batch may provide more positive pairs and improve latent supervision. Keep NotebookAE, optimizer, and budget fixed. Validation has been repeatedly used for selection; these are validation results, not untouched test evidence. Do not assign a confidence interval to the combined objective.
