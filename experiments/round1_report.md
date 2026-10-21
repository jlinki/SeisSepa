# Round 1 report

## Status

Run 3 is the completed seed-42 baseline. Its first process exited with a `NameError` after writing the local checkpoint, metrics, and report; the typo was fixed and the same online W&B run was resumed. W&B has all 53 final diagnostic scalars and is marked finished. The original exception remains preserved at `outputs/round1/03_baseline_after_perf_fix/failure.log`.

All seven training slots are complete. The doubled-weight candidate reached 0.130081 on seed 42, and its required exact repeat matched. Seeds 43 and 44 scored 0.113821 each; the three-seed mean is 0.119241 (sample SD 0.009388), below the baseline score 0.121951. The seed-42 repeat checkpoint is selected as the best observed single-seed result and has passed the protected standalone evaluator. No additional candidate or run was started.

## Frozen benchmark and baseline

- Benchmark ID: `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training, 123 validation; 35 event and 35 station classes).
- Configuration: `model=notebook loss=separation`, seed 42; maximum 50 epochs, patience 12; physical GPU 0; online project `jlinki/seissepa-autoresearch`.
- Baseline run commit: `e37e8d7c282d5ec1c17ae2c39c24523b8d552a69`.
- Candidate configuration tested: `model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10`. The best observed single-seed checkpoint is `outputs/round1/05_double_separation_seed42_repeat/best.pt`; the three-seed average is below the seed-42 baseline.

## Run inventory

| Slot | Status and outcome | Event retrieval | Station retrieval | Combined score | Canonical validation MSE | Seed / best epoch | Elapsed | Artifacts and W&B |
|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | Owner-stopped partial after 8 epochs; not a valid completed baseline | 0.040650 | 0.203252 | 0.121951 | 0.985821658 | 42 / 1 | 165.99 s at the last recorded epoch | `outputs/round1/01_baseline`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/dv8yv261) |
| 2 | Failed detached retry; no valid metrics | — | — | — | — | 42 / — | — | `outputs/round1/02_baseline_retry`; failure and console logs preserved |
| 3 | Completed; early stopped at epoch 13; final W&B diagnostics recovered on the same run ID | 0.040650 (95% Wilson 0.017–0.092) | 0.203252 (95% Wilson 0.142–0.283) | 0.121951 | 0.985821658387 | 42 / 1 | 127.32 s reported by the run | `outputs/round1/03_baseline_after_perf_fix`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/fwi5w1h5) |
| 4 | Completed; doubled both separation weights; primary score improved by 0.008130 | 0.024390 | 0.235772 | 0.130081 | 0.961848411743 | 42 / 10 | 104.86 s | `outputs/round1/04_double_separation`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/o9y68io8) |
| 5 | Completed; exact seed-42 training repeat matched run 4 on all scientific metrics | 0.024390 | 0.235772 | 0.130081 | 0.961848411743 | 42 / 10 | 100.83 s | `outputs/round1/05_double_separation_seed42_repeat`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/jlca5tii) |
| 6 | Completed; same candidate at seed 43 | 0.056911 | 0.170732 | 0.113821 | 0.953893602915 | 43 / 10 | 268.87 s | `outputs/round1/06_double_separation_seed43`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/mn4d177n) |
| 7 | Completed; same candidate at seed 44 | 0.065041 | 0.162602 | 0.113821 | 0.816304484157 | 44 / 13 | 282.65 s | `outputs/round1/07_double_separation_seed44`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/swi75han) |

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

The exact run-5 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10 seed=42 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed42_double_separation_repeat hydra.run.dir=outputs/round1/05_double_separation_seed42_repeat
```

Run 5 repeated seed 42 with the same candidate and budget, using commit `58ed80c6ab2bbfb292c96bd487b155bb0f01f623`. It also stopped after 22 epochs at epoch 10. All scientific fields in the local metrics JSON matched run 4 exactly, including both retrieval tasks and canonical MSE. Its elapsed time was 100.829 seconds; all required local artifacts and 53 final W&B diagnostic scalars are present at [run 5](https://wandb.ai/jlinki/seissepa-autoresearch/runs/jlca5tii).

The exact run-6 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10 seed=43 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed43_double_separation hydra.run.dir=outputs/round1/06_double_separation_seed43
```

Run 6 used commit `6524a6dedf57a2d53291b61e84a927e5e030182f` and the fixed benchmark ID above. It stopped after 22 epochs, selecting epoch 10. Event retrieval was 0.056910569105691054, station retrieval 0.17073170731707318, and the primary mean 0.11382113821138212. Canonical validation MSE was 0.9538936029151842 and elapsed time was 268.87348535098135 seconds. The score is below both run 4 and the seed-42 baseline; preserve the outcome and its artifacts. Online W&B has all 53 final diagnostic scalars at [run 6](https://wandb.ai/jlinki/seissepa-autoresearch/runs/mn4d177n).

The exact run-7 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation loss.event_weight=0.10 loss.station_weight=0.10 seed=44 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed44_double_separation hydra.run.dir=outputs/round1/07_double_separation_seed44
```

Run 7 used commit `799623835303b577d2973af066aaa8ad0583a704` and the fixed benchmark ID above. It completed 25 epochs and selected epoch 13 after a score tie was resolved by lower canonical validation MSE. Event retrieval was 0.06504065040650407, station retrieval 0.16260162601626016, and the primary mean 0.11382113821138212. Canonical validation MSE was 0.8163044841565271; elapsed time was 282.64696878939867 seconds. W&B has all 53 final diagnostic scalars at [run 7](https://wandb.ai/jlinki/seissepa-autoresearch/runs/swi75han).

## Evaluation and comparisons

The protected standalone evaluator was run on the run-3 baseline, run-4 candidate and final selected run-5 seed-42 repeat on physical GPU 0 with batch size 128. It matched all 86, 97 and 97 shared scientific fields respectively, with zero mismatches. The final evaluation is saved at `outputs/round1/05_double_separation_seed42_repeat/best.evaluation.json`. The command was `CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/evaluate.py outputs/round1/05_double_separation_seed42_repeat/best.pt --device cuda:0 --batch-size 128`.

| Candidate seed | Event retrieval | Station retrieval | Primary score |
|---|---:|---:|---:|
| 42 (run 4; run 5 exact repeat) | 0.024390 | 0.235772 | 0.130081 |
| 43 (run 6) | 0.056911 | 0.170732 | 0.113821 |
| 44 (run 7) | 0.065041 | 0.162602 | 0.113821 |
| Three-seed mean | — | — | 0.119241 |
| Three-seed sample standard deviation | — | — | 0.009388 |

Run 3 matches the epoch-1 result from the owner-stopped partial run 1, which remains an invalid partial attempt. Run 4 improves the primary score over the baseline by 0.008130; event retrieval falls by 0.016260 while station retrieval rises by 0.032520. Run 5 exactly reproduces run 4's scientific metrics, and the model tensors are bitwise identical. Across the predeclared seeds 42, 43 and 44, the primary scores are 0.130081, 0.113821 and 0.113821; their mean is 0.119241 and sample standard deviation is 0.009388. The mean is 0.002710 below the baseline score 0.121951. The doubled-weight configuration is the best observed single-seed result, but the three-seed mean does not improve on the baseline. Validation was used repeatedly for selection, so these are validation results. No confidence interval is assigned to the combined score; Wilson intervals apply only to individual accuracies.

## Journal and next hypothesis

Commands, outcomes, commits, benchmark IDs and W&B URLs for runs 3–7 are recorded in `experiments/journal.jsonl`. All seven training slots were used within the deadline. Preserve the doubled-weight candidate and its seed-42 repeat as the best single-seed validation checkpoint; report the lower three-seed mean alongside it. The seed-42 repeat has bitwise-identical model tensors to run 4 and passed standalone evaluation. No further experiments are authorized in this round.
