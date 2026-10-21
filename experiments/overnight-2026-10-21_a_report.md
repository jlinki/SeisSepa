# Lane A overnight report

## Status

This invocation completed one seed-42 schedule screen at slot 32 and stopped at its predeclared gate. The cosine screen scored 0.182927, below the verified best at slot 22 (0.211382), so no standalone evaluation or conditional confirmations were run. The attempt, W&B record, journal entry, source changes, and this report are preserved locally and pushed on the lane branch.

- Attempts consumed: **32 / 64**; attempts remaining: **32**.
- Seed-42 screens: **26 / 48**; confirmation runs: **6**.
- At least 12 remaining slots are reserved for confirmations; the attempt cap therefore permits at most 20 more seed-42 screens.
- All 32 slot records are complete; all runs use online W&B in `jlinki/seissepa-autoresearch` and the frozen benchmark.
- No STOP or BLOCKED marker was present; no integrity, upstream, authentication, or online-logging failure occurred.
- Training cutoff: 2026-10-22 05:00 UTC; worker reporting cutoff: 2026-10-22 05:30 UTC.

Pre-research checks in this invocation: `scripts/verify_benchmark.py` verified benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` with 1,225 traces (1,102 train, 123 validation); `pytest -q` passed 24 tests.

The optional cosine schedule is controlled by config values and defaults off. It does not change NotebookAE. The source and config were committed and pushed before training as `3f58b27e5e15c56fbbed4de45a66785827e7d837`.

## Best valid checkpoint

- Run: `outputs/overnight_2026-10-21/a_22_batch32_seed43/best.pt`
- Configuration: NotebookAE; separation loss; reconstruction/event/station weights `1.0/0.10/0.10`; margin `0.2`; AdamW `lr=0.0002`, `weight_decay=0`; batch size 32; gradient clip 1.0; constant learning rate; seed 43; 200 epoch maximum; patience 40.
- Validation event top-1: 0.065041; station top-1: 0.357724; required mean score: **0.211382**.
- Canonical validation reconstruction MSE: 0.076475924231; best epoch: 79; run commit: `c19ef9a05f0aa589de1fb7b77d956b8f4316a128`.
- W&B: [slot 22](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gtb11jjt).
- Existing `best.evaluation.json` matches the training metrics exactly for all shared scientific fields: **True**.

The fresh seed-42 default baseline at slot 1 scored 0.121951 (event 0.040650, station 0.203252; MSE 0.985821658). The highest seed-42 screen across all configs was slot 30 at 0.199187 (event 0.048780, station 0.349593; run `outputs/overnight_2026-10-21/a_30_lr0003_batch32_seed42`).

## Three-seed evidence for the best configuration

The batch-32 balanced configuration was repeated with seed 42 and confirmed with seeds 43 and 44. Slot 21 is the exact seed-42 repeat of slot 20; the seed-42/43/44 comparison is:

| Seed | Slot | Event top-1 | Station top-1 | Combined score | Canonical val MSE | Best epoch | W&B |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 21 | 0.056911 | 0.325203 | 0.191057 | 0.343089505 | 36 | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/b6gjrlva) |
| 43 | 22 | 0.065041 | 0.357724 | 0.211382 | 0.076475924 | 79 | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gtb11jjt) |
| 44 | 23 | 0.065041 | 0.308943 | 0.186992 | 0.108797490 | 52 | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/qwaj8efb) |

- Mean combined score: **0.196477**; sample standard deviation: **0.013067**.
- Difference from the fresh seed-42 default baseline: **+0.074526**.
- Validation was used for research and selection; this is not an untouched test result. No confidence interval is assigned to the mean of the two retrieval tasks.

## Slot 32 schedule screen

- Hypothesis: At batch size 32, cosine-decaying AdamW from 0.0002 to a 10% learning-rate floor over 120 epochs may reduce late overfitting and improve retrieval over the constant-rate run. Keep NotebookAE, reconstruction/event/station weights 1.0/0.10/0.10, margin 0.2, zero weight decay, gradient clip 1.0, sampling, seed, and 200-epoch/patience-40 budget fixed.
- Exact command: `CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=separation loss.reconstruction_weight=1.0 loss.event_weight=0.10 loss.station_weight=0.10 loss.margin=0.2 train.epochs=200 train.patience=40 train.batch_size=32 train.grad_clip=1.0 train.lr_schedule_cosine=true train.lr_schedule_epochs=120 train.lr_min_ratio=0.1 optimizer.lr=0.0002 optimizer.weight_decay=0.0 device=cuda:0 threads=4 wandb.mode=online seed=42 wandb.name=overnight_a_32_cosine120_seed42 hydra.run.dir=outputs/overnight_2026-10-21/a_32_cosine120_seed42`
- Result: event 0.065041; station 0.300813; score **0.182927**; canonical MSE 0.371695985; seed 42; best epoch 36; elapsed 172.92 s training / 184.32 s guarded.
- Benchmark ID: `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; commit: `3f58b27e5e15c56fbbed4de45a66785827e7d837`; W&B: [slot 32](https://wandb.ai/jlinki/seissepa-autoresearch/runs/wzoqkpj1). Required local artifacts are present.
- Decision: preserve the result; it ranks below slot 22, so no standalone evaluation, seed-42 repeat, or seeds 43/44.

## All attempts

All per-run commands, hypotheses, slot outcomes, and decisions are in `experiments/journal.jsonl`. `Elapsed` shows training time and guarded wall time.

| Slot | Status | Seed | Event | Station | Score | Canonical MSE | Epoch | Elapsed train / guard (s) | Commit | W&B | Run path |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | completed | 42 | 0.040650 | 0.203252 | 0.121951 | 0.985821658 | 1 | 289.71 / 300.89 | `b0140623` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/0p7zsyx6) | `outputs/overnight_2026-10-21/a_01_baseline` |
| 2 | completed | 42 | 0.048780 | 0.211382 | 0.130081 | 0.985821631 | 1 | 229.42 / 240.41 | `d0135a6d` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/xhho1tww) | `outputs/overnight_2026-10-21/a_02_event010` |
| 3 | completed | 42 | 0.040650 | 0.211382 | 0.126016 | 0.985821702 | 1 | 250.69 / 261.96 | `fb6f533c` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/uspv5ol8) | `outputs/overnight_2026-10-21/a_03_station010` |
| 4 | completed | 42 | 0.024390 | 0.235772 | 0.130081 | 0.961848412 | 10 | 248.63 / 258.29 | `49a125d5` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/thjw54j9) | `outputs/overnight_2026-10-21/a_04_balanced010` |
| 5 | completed | 42 | 0.024390 | 0.235772 | 0.130081 | 0.961848412 | 10 | 239.61 / 249.96 | `5b89290c` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/kt60c7or) | `outputs/overnight_2026-10-21/a_05_balanced010_seed42_repeat` |
| 6 | completed | 43 | 0.056911 | 0.260163 | 0.158537 | 0.656984617 | 48 | 353.14 / 363.86 | `de03a8ff` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gh8ijgbx) | `outputs/overnight_2026-10-21/a_06_balanced010_seed43` |
| 7 | completed | 44 | 0.040650 | 0.227642 | 0.134146 | 0.515269530 | 64 | 245.33 / 254.95 | `63b287c5` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gx3l247a) | `outputs/overnight_2026-10-21/a_07_balanced010_seed44` |
| 8 | completed | 42 | 0.048780 | 0.203252 | 0.126016 | 0.985822225 | 1 | 168.12 / 177.44 | `38476423` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/wr2mip6z) | `outputs/overnight_2026-10-21/a_08_reconstruction050_seed42` |
| 9 | completed | 42 | 0.048780 | 0.203252 | 0.126016 | 0.985821684 | 1 | 153.45 / 164.50 | `e72ccbb9` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4nifno2a) | `outputs/overnight_2026-10-21/a_09_margin030_seed42` |
| 10 | completed | 42 | 0.048780 | 0.203252 | 0.126016 | 0.985821684 | 1 | 208.19 / 217.68 | `9392cf63` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/z5uuvz9g) | `outputs/overnight_2026-10-21/a_10_margin010_seed42` |
| 11 | completed | 42 | 0.056911 | 0.211382 | 0.134146 | 0.968788238 | 16 | 355.31 / 373.07 | `24261af5` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4m7r1pwz) | `outputs/overnight_2026-10-21/a_11_batch256_seed42` |
| 12 | completed | 42 | 0.056911 | 0.219512 | 0.138211 | 0.986826038 | 1 | 158.31 / 168.34 | `8782a6af` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/st0aibk7) | `outputs/overnight_2026-10-21/a_12_lr0001_seed42` |
| 13 | completed | 42 | 0.032520 | 0.227642 | 0.130081 | 0.979239570 | 3 | 159.03 / 168.82 | `edef2bc2` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4zdma9fo) | `outputs/overnight_2026-10-21/a_13_lr0003_seed42` |
| 14 | completed | 42 | 0.056911 | 0.243902 | 0.150407 | 0.110022325 | 105 | 309.20 / 318.66 | `fd442303` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/w8p1wson) | `outputs/overnight_2026-10-21/a_14_batch64_seed42` |
| 15 | completed | 42 | 0.065041 | 0.211382 | 0.138211 | 0.435700860 | 51 | 213.71 / 224.16 | `97e6785f` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/rtev7ole) | `outputs/overnight_2026-10-21/a_15_reconstruction075_batch64_seed42` |
| 16 | completed | 42 | 0.024390 | 0.235772 | 0.130081 | 0.384051890 | 55 | 193.80 / 202.78 | `4d8d5a53` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nizxqasz) | `outputs/overnight_2026-10-21/a_16_station015_batch64_seed42` |
| 17 | completed | 42 | 0.040650 | 0.235772 | 0.138211 | 0.151000115 | 77 | 228.57 / 237.99 | `07f4af70` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/19m2bufm) | `outputs/overnight_2026-10-21/a_17_event015_batch64_seed42` |
| 18 | completed | 42 | 0.073171 | 0.186992 | 0.130081 | 0.971477956 | 7 | 222.22 / 232.65 | `abf5993d` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/60uyru99) | `outputs/overnight_2026-10-21/a_18_lr0001_batch64_balanced_seed42` |
| 19 | completed | 42 | 0.056911 | 0.243902 | 0.150407 | 0.110022325 | 105 | 434.84 / 445.22 | `04974d56` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nlie1gun) | `outputs/overnight_2026-10-21/a_19_weightdecay0_batch64_seed42` |
| 20 | completed | 42 | 0.056911 | 0.325203 | 0.191057 | 0.343089505 | 36 | 238.77 / 248.68 | `56dff997` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/7xrhy2hl) | `outputs/overnight_2026-10-21/a_20_batch32_seed42` |
| 21 | completed | 42 | 0.056911 | 0.325203 | 0.191057 | 0.343089505 | 36 | 209.96 / 221.05 | `8b5a3940` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/b6gjrlva) | `outputs/overnight_2026-10-21/a_21_batch32_seed42_repeat` |
| 22 | completed | 43 | 0.065041 | 0.357724 | 0.211382 | 0.076475924 | 79 | 261.47 / 271.18 | `c19ef9a0` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gtb11jjt) | `outputs/overnight_2026-10-21/a_22_batch32_seed43` |
| 23 | completed | 44 | 0.065041 | 0.308943 | 0.186992 | 0.108797490 | 52 | 201.54 / 211.90 | `ed9a5ca6` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/qwaj8efb) | `outputs/overnight_2026-10-21/a_23_batch32_seed44` |
| 24 | completed | 42 | 0.040650 | 0.325203 | 0.182927 | 0.309674976 | 41 | 165.00 / 175.17 | `d5a80178` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/h1pxqf1c) | `outputs/overnight_2026-10-21/a_24_reconstruction075_batch32_seed42` |
| 25 | completed | 42 | 0.081301 | 0.308943 | 0.195122 | 0.082754719 | 76 | 217.68 / 227.89 | `855fe0b5` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/eipjg9re) | `outputs/overnight_2026-10-21/a_25_reconstruction125_batch32_seed42` |
| 26 | completed | 42 | 0.040650 | 0.341463 | 0.191057 | 0.155274947 | 51 | 172.16 / 182.23 | `f9d6c65f` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/313eahzn) | `outputs/overnight_2026-10-21/a_26_station015_rec125_batch32_seed42` |
| 27 | completed | 42 | 0.048780 | 0.300813 | 0.174797 | 0.076112420 | 73 | 392.12 / 402.25 | `c2946974` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/s5n6x4qu) | `outputs/overnight_2026-10-21/a_27_event015_rec125_batch32_seed42` |
| 28 | completed | 42 | 0.048780 | 0.325203 | 0.186992 | 0.075624627 | 82 | 278.33 / 287.94 | `91bd0ca3` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gxjhyom1) | `outputs/overnight_2026-10-21/a_28_margin010_batch32_seed42` |
| 29 | completed | 42 | 0.032520 | 0.276423 | 0.154472 | 0.178188113 | 73 | 223.71 / 233.11 | `51bb8443` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ddmfehkr) | `outputs/overnight_2026-10-21/a_29_lr0001_batch32_seed42` |
| 30 | completed | 42 | 0.048780 | 0.349593 | 0.199187 | 0.102193265 | 48 | 195.97 / 205.07 | `10d42011` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vsb52wez) | `outputs/overnight_2026-10-21/a_30_lr0003_batch32_seed42` |
| 31 | completed | 42 | 0.065041 | 0.317073 | 0.191057 | 0.090235976 | 47 | 206.64 / 216.22 | `f9977046` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/qajoldnt) | `outputs/overnight_2026-10-21/a_31_lr0004_batch32_seed42` |
| 32 | completed | 42 | 0.065041 | 0.300813 | 0.182927 | 0.371695985 | 36 | 172.92 / 184.32 | `3f58b27e` | [run](https://wandb.ai/jlinki/seissepa-autoresearch/runs/wzoqkpj1) | `outputs/overnight_2026-10-21/a_32_cosine120_seed42` |

## Next hypothesis

The 120-epoch cosine schedule scored below both the best checkpoint and its constant-rate seed-42 comparison. A next bounded screen could keep the same architecture, loss balance, batch size, and optimizer rate but spread cosine decay over the full 200-epoch budget; this may retain useful learning rate longer while still reducing late training. No further run was started in this invocation.

The lane branch is clean and synchronized with `origin/research/overnight-20261021-a` after the final journal and report commit.
