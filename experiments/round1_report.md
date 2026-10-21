# Round 1 report

## Status

Run 3 is the completed seed-42 baseline. Its first process exited with a `NameError` after writing the local checkpoint, metrics, and report; the typo was fixed and the same online W&B run was resumed. W&B has all 53 final diagnostic scalars and is marked finished. The original exception remains preserved at `outputs/round1/03_baseline_after_perf_fix/failure.log`.

Run 4 completed the one predeclared doubled-separation-weight candidate. It improved the primary validation retrieval score from 0.121951 to 0.130081 (+0.008130), so the candidate proceeds unchanged to the predeclared seed-42 repeat and seeds 43/44. Five of seven slots have been used; seed 44 remains. No other candidate is authorized.

## Frozen benchmark and baseline

- Benchmark ID: `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training, 123 validation; 35 event and 35 station classes).
- Configuration: `model=notebook loss=separation`, seed 42; maximum 50 epochs, patience 12; physical GPU 0; online project `jlinki/seissepa-autoresearch`.
- Baseline run commit: `e37e8d7c282d5ec1c17ae2c39c24523b8d552a69`.
- Current candidate configuration: `model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10`; its three-seed results are pending.

## Run inventory

| Slot | Status and outcome | Event retrieval | Station retrieval | Combined score | Canonical validation MSE | Seed / best epoch | Elapsed | Artifacts and W&B |
|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | Owner-stopped partial after 8 epochs; not a valid completed baseline | 0.040650 | 0.203252 | 0.121951 | 0.985821658 | 42 / 1 | 165.99 s at the last recorded epoch | `outputs/round1/01_baseline`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/dv8yv261) |
| 2 | Failed detached retry; no valid metrics | — | — | — | — | 42 / — | — | `outputs/round1/02_baseline_retry`; failure and console logs preserved |
| 3 | Completed; early stopped at epoch 13; final W&B diagnostics recovered on the same run ID | 0.040650 (95% Wilson 0.017–0.092) | 0.203252 (95% Wilson 0.142–0.283) | 0.121951 | 0.985821658387 | 42 / 1 | 127.32 s reported by the run | `outputs/round1/03_baseline_after_perf_fix`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/fwi5w1h5) |
| 4 | Completed; doubled both separation weights; primary score improved by 0.008130 | 0.024390 | 0.235772 | 0.130081 | 0.961848411743 | 42 / 10 | 104.86 s | `outputs/round1/04_double_separation`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/o9y68io8) |
| 5 | Completed; exact seed-42 training repeat matched run 4 on all scientific metrics | 0.024390 | 0.235772 | 0.130081 | 0.961848411743 | 42 / 10 | 100.83 s | `outputs/round1/05_double_separation_seed42_repeat`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/jlca5tii) |
| 6 | Completed; same candidate at seed 43 | 0.056911 | 0.170732 | 0.113821 | 0.953893602915 | 43 / 10 | 268.87 s | `outputs/round1/06_double_separation_seed43`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/mn4d177n) |
| 7 | Reserved for same candidate at seed 44 | — | — | — | — | 44 / — | — | — |

The round-3 exact command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation seed=42 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed42_baseline_perf_fix hydra.run.dir=outputs/round1/03_baseline_after_perf_fix
```

Run 3 wrote all required local files. `src/seissepa/train.py` originally referenced an undefined epoch variable in the final W&B step; the fix was pushed in commit `d35758b`, then the approved diagnostic scalars were logged to the same run ID. Read-back from W&B confirmed the run is finished, the primary score is 0.121951, and all 53 `final/...` scalar keys are present. No offline fallback or artifact upload was used.

The exact run-4 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10 seed=42 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed42_double_separation hydra.run.dir=outputs/round1/04_double_separation
```

Run 4 used commit `6b77ee6f056b3fbeaf087346910b68b1bf9fce3f` and benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. It early-stopped after 22 epochs, selecting epoch 10. Validation event retrieval was 0.024390 (−0.016260 relative to baseline), station retrieval was 0.235772 (+0.032520), and their required mean was 0.130081 (+0.008130). Canonical validation MSE was 0.9618484117428208 (baseline 0.9858216583872177). Local artifacts are complete, online W&B is finished with all 53 final diagnostic scalar keys, and its URL is [run 4](https://wandb.ai/jlinki/seissepa-autoresearch/runs/o9y68io8). The protected standalone evaluator was run with batch size 128 and matched all 97 shared scientific fields with zero mismatches; see `outputs/round1/04_double_separation/best.evaluation.json`.

Run 5 repeated seed 42 with the same candidate and budget, using commit `58ed80c6ab2bbfb292c96bd487b155bb0f01f623`. It also stopped after 22 epochs at epoch 10. All scientific fields in the local metrics JSON matched run 4 exactly, including both retrieval tasks and canonical MSE. Its elapsed time was 100.829 seconds; all required local artifacts and 53 final W&B diagnostic scalars are present at [run 5](https://wandb.ai/jlinki/seissepa-autoresearch/runs/jlca5tii).

The exact run-6 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10 seed=43 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed43_double_separation hydra.run.dir=outputs/round1/06_double_separation_seed43
```

Run 6 used commit `6524a6dedf57a2d53291b61e84a927e5e030182f` and the fixed benchmark ID above. It stopped after 22 epochs, selecting epoch 10. Event retrieval was 0.056910569105691054, station retrieval 0.17073170731707318, and the primary mean 0.11382113821138212. Canonical validation MSE was 0.9538936029151842 and elapsed time was 268.87348535098135 seconds. The score is below both run 4 and the seed-42 baseline; preserve the outcome and its artifacts. Online W&B has all 53 final diagnostic scalars at [run 6](https://wandb.ai/jlinki/seissepa-autoresearch/runs/mn4d177n).

## Evaluation and comparisons

The protected standalone evaluator was run on the run-3 baseline and run-4 candidate on GPU 0 with batch size 128. It exactly matched all 86 shared scientific fields for run 3 and all 97 shared fields for run 4, with zero mismatches in either case. This confirms repeatability of checkpoint evaluation on the same runtime/device/batch size; it is not a repeated training run.

Run 3 matches the epoch-1 result from the owner-stopped partial run 1: event, station, combined retrieval, and canonical MSE deltas are all 0. Run 1 remains partial and is not used as a completed research baseline. Run 4 improves the primary score by 0.008130; event retrieval fell by 0.016260 while station retrieval rose by 0.032520. Its canonical validation MSE fell by 0.023973246644. Run 5 exactly reproduces run 4's scientific metrics under the same seed and runtime. Seed 43 produced 0.113821, below the seed-42 score of 0.130081 and baseline score of 0.121951; seed 44 is pending. The combined score has no confidence interval; the individual event and station intervals use Wilson bounds.

## Journal and next hypothesis

The run-3 through run-6 commands, observed metrics, commits, benchmark IDs, W&B URLs, recovery details and decisions are recorded in `experiments/journal.jsonl`. Run 5 exactly reproduced run 4. Run 6 is lower, but preserve it and complete the final predeclared seed-44 run 7 with the unchanged candidate. Use the same architecture, weights, epoch/patience budget, physical GPU and online W&B project, with unique run paths. Commit and push each outcome before starting the next. Stop by 2026-10-21 17:13:52 UTC and at seven total slots.
