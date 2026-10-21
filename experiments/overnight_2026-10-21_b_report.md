# Overnight 2026-10-21 — lane B report

## Invocation status

The latest bounded invocation finished after one seed-42 screen (slot 25). Twenty-five of 64 cumulative lane attempts are consumed; 39 remain, with at least 12 kept for exact repeats and seed confirmations. The branch is `research/overnight-20261021-b`. The balanced-grid code/config and pre-registered batch plan were pushed before training in commit `dca129118969de9956d125f350fa6905f5a1ee63`. The screen was below the seed-42 comparator, so no standalone evaluation or follow-up confirmations were run. NotebookAE architecture remained fixed.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training and 123 validation; 35 event and 35 station classes). Before the latest batch, benchmark verification passed, all 24 tests passed, the leaderboard showed slot 18 as incumbent, the branch was clean and synchronized, and STOP was absent. The new sampler uses training labels only and leaves model inputs unchanged. Slot 25 used the guarded GPU-0 runner and online W&B in `jlinki/seissepa-autoresearch`; all required local artifacts are present and no artifact files were uploaded. Slot 25 and its report update are recorded on this branch.

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
