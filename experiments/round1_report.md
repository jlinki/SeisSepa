# Round 1 report

## Status

The seed-42 baseline trained and produced its local checkpoint, metrics, and report. The process then exited with a `NameError` before sending the final diagnostic scalars to online W&B. Per the online-logging rule, the round stops here: no candidate or seed-repeat run was started. The failure is preserved at `outputs/round1/03_baseline_after_perf_fix/failure.log`.

## Frozen benchmark and baseline

- Benchmark ID: `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training, 123 validation; 35 event and 35 station classes).
- Configuration: `model=notebook loss=separation`, seed 42; maximum 50 epochs, patience 12; physical GPU 0; online project `jlinki/seissepa-autoresearch`.
- Baseline run commit: `e37e8d7c282d5ec1c17ae2c39c24523b8d552a69`.
- Best configuration available from this round: the default seed-42 baseline command below. No candidate was evaluated or promoted.

## Run inventory

| Slot | Status and outcome | Event retrieval | Station retrieval | Combined score | Canonical validation MSE | Seed / best epoch | Elapsed | Artifacts and W&B |
|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | Owner-stopped partial after 8 epochs; not a valid completed baseline | 0.040650 | 0.203252 | 0.121951 | 0.985821658 | 42 / 1 | 165.99 s at the last recorded epoch | `outputs/round1/01_baseline`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/dv8yv261) |
| 2 | Failed detached retry; no valid metrics | — | — | — | — | 42 / — | — | `outputs/round1/02_baseline_retry`; failure and console logs preserved |
| 3 | Training and local report completed through early stop at epoch 13; process failed during final W&B diagnostic logging | 0.040650 (95% Wilson 0.017–0.092) | 0.203252 (95% Wilson 0.142–0.283) | 0.121951 | 0.985821658387 | 42 / 1 | 127.32 s reported by the run | `outputs/round1/03_baseline_after_perf_fix`; [W&B](https://wandb.ai/jlinki/seissepa-autoresearch/runs/fwi5w1h5) |
| 4 | Not started; candidate allocation stopped after the logging failure | — | — | — | — | — | — | — |
| 5–7 | Not started; seed-42 repeat and seeds 43/44 were conditional on a promising candidate | — | — | — | — | — | — | — |

The round-3 exact command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/train.py model=notebook loss=separation seed=42 device=cuda:0 train.epochs=50 train.patience=12 wandb.name=round1_seed42_baseline_perf_fix hydra.run.dir=outputs/round1/03_baseline_after_perf_fix
```

Run 3 wrote `best.pt`, `metrics.json`, and `report.md` before the final W&B call. Its online run contains synced per-epoch metrics, but the local W&B summary has 0 of the expected 53 final diagnostic scalars. `src/seissepa/train.py:262` references undefined `last_epoch`; this is recorded in the run's `failure.log`. No offline fallback was used.

## Evaluation and comparisons

The protected standalone evaluator was run on the round-3 checkpoint on GPU 0 with batch size 128. It exactly matched all 86 shared scientific fields in `metrics.json` (zero mismatches). This confirms repeatability of checkpoint evaluation on the same runtime/device/batch size; it is not a repeated training run.

Run 3 matches the epoch-1 result from the owner-stopped partial run 1: event, station, combined retrieval, and canonical MSE deltas are all 0. Run 1 remains partial and is not used as a completed research baseline. There is no candidate-relative delta, and there are no three-seed mean/std results because runs 4–7 were not launched. The combined score has no confidence interval; the individual event and station intervals above use Wilson bounds.

## Journal and next hypothesis

The run-3 command, observed metrics, commit, benchmark ID, W&B URL, failure, and decision are recorded in `experiments/journal.jsonl`. The next bounded hypothesis from the allocation remains that doubling both separation weights (`loss.event_weight=0.10`, `loss.station_weight=0.10`) may improve retrieval. It was not tested in this round because final online diagnostic logging failed. A future continuation must repair and push the undefined logging step before starting another online run, then preserve the remaining run and deadline limits.
