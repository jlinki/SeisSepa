# Lane A overnight report

## Status

This invocation completed one seed-42 AdamW learning-rate screen at batch size 32 (slot 30). It scored 0.199187, below the verified slot-22 best at 0.211382. The screen improved on the balanced batch-32 seed-42 result of 0.191057, but it did not outrank the verified candidate, so the standalone evaluator and conditional confirmations were not run. Lane A has consumed **30 of 64 attempts**: 24 seed-42 screens and 6 confirmation runs. There are 34 attempts left; at least 12 remain reserved for confirmations. The 48-screen cap allows 24 more seed-42 screens, while preserving the future confirmation reserve limits additional screens to 22. All 30 slots completed successfully. No STOP/BLOCKED file, authentication failure, benchmark-integrity failure, upstream push failure, or online W&B failure occurred.

The lane is `research/overnight-20261021-a`. The batch plan, attempt record, and this report are committed and pushed. No source, model, loss, configuration, evaluator, benchmark, or test files were changed; research used Hydra overrides.

Preflight passed in this invocation: `scripts/verify_benchmark.py` verified benchmark `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` with 1,225 traces (1,102 train, 123 validation); `pytest -q` passed all 24 tests. The leaderboard confirmed slot 22 remains top-ranked before the screen.

## Confirmation hypothesis and result

Hypothesis: balanced separation weights `reconstruction/event/station = 1.0/0.10/0.10` reproduce the seed-42 result and improve the three-seed mean over the fresh overnight default baseline. The confirmation held NotebookAE, margin 0.2, AdamW learning rate 0.0002 and weight decay 0.0001, batch size 128, gradient clip 1.0, sampling, and the 200-epoch/patience-40 budget fixed.

| Slot | Run | Seed | Event top-1 | Station top-1 | Score | Canonical val MSE | Best epoch / epochs | Guard elapsed | Training commit | W&B | Standalone evaluation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `outputs/overnight_2026-10-21/a_01_baseline` | 42 | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 1 / 41 | 300.9 s | `b0140623c71e49021ed594dccc8524f674ded1a2` | [0p7zsyx6](https://wandb.ai/jlinki/seissepa-autoresearch/runs/0p7zsyx6) | — |
| 2 | `outputs/overnight_2026-10-21/a_02_event010` | 42 | 0.048780 | 0.211382 | 0.130081 | 0.9858216307 | 1 / 41 | 240.4 s | `d0135a6da852577a3bb0aa07abf269aade75a0b9` | [xhho1tww](https://wandb.ai/jlinki/seissepa-autoresearch/runs/xhho1tww) | 101 fields, 0 mismatches |
| 3 | `outputs/overnight_2026-10-21/a_03_station010` | 42 | 0.040650 | 0.211382 | 0.126016 | 0.9858217018 | 1 / 41 | 262.0 s | `fb6f533c70dc263ccc82519f83578e5cfbf7a3ab` | [uspv5ol8](https://wandb.ai/jlinki/seissepa-autoresearch/runs/uspv5ol8) | — |
| 4 | `outputs/overnight_2026-10-21/a_04_balanced010` | 42 | 0.024390 | 0.235772 | 0.130081 | 0.9618484117 | 10 / 50 | 258.3 s | `49a125d59ba35c2df1ac40d8623fd0066a9230fd` | [thjw54j9](https://wandb.ai/jlinki/seissepa-autoresearch/runs/thjw54j9) | 101 fields, 0 mismatches |
| 5 | `outputs/overnight_2026-10-21/a_05_balanced010_seed42_repeat` | 42 | 0.024390 | 0.235772 | 0.130081 | 0.9618484117 | 10 / 50 | 250.0 s | `5b89290cc918dc81260bf928414cfd73497e593e` | [kt60c7or](https://wandb.ai/jlinki/seissepa-autoresearch/runs/kt60c7or) | 100 fields, 0 mismatches; exact repeat of slot 4 on all 100 scientific metrics |
| 6 | `outputs/overnight_2026-10-21/a_06_balanced010_seed43` | 43 | 0.056911 | 0.260163 | **0.158537** | 0.6569846171 | 48 / 88 | 363.9 s | `de03a8ff9a1186c0b2a4753cf82f44d00a45ac99` | [gh8ijgbx](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gh8ijgbx) | 100 fields, 0 mismatches |
| 7 | `outputs/overnight_2026-10-21/a_07_balanced010_seed44` | 44 | 0.040650 | 0.227642 | 0.134146 | 0.5152695297 | 64 / 104 | 255.0 s | `63b287c50c783b03762883556dddf088087e2754` | [gx3l247a](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gx3l247a) | Not run; lower score than the evaluated slot-6 candidate |
| 8 | `outputs/overnight_2026-10-21/a_08_reconstruction050_seed42` | 42 | 0.048780 | 0.203252 | 0.126016 | 0.9858222247 | 1 / 41 | 177.4 s | `3847642340e196573dcf36910b29d810a93471e3` | [wr2mip6z](https://wandb.ai/jlinki/seissepa-autoresearch/runs/wr2mip6z) | Not run; below verified best |
| 9 | `outputs/overnight_2026-10-21/a_09_margin030_seed42` | 42 | 0.048780 | 0.203252 | 0.126016 | 0.9858216842 | 1 / 41 | 164.5 s | `e72ccbb9fead57eb4aff5282c6192ac963c86d65` | [4nifno2a](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4nifno2a) | Not run; below prior seed-42 best |
| 10 | `outputs/overnight_2026-10-21/a_10_margin010_seed42` | 42 | 0.048780 | 0.203252 | 0.126016 | 0.9858216842 | 1 / 41 | 217.7 s | `9392cf63fd3ba73ab472ddbbc6a848594b10aefd` | [z5uuvz9g](https://wandb.ai/jlinki/seissepa-autoresearch/runs/z5uuvz9g) | Not run; below verified best |
| 11 | `outputs/overnight_2026-10-21/a_11_batch256_seed42` | 42 | 0.056911 | 0.211382 | 0.134146 | 0.9687882376 | 16 / 56 | 373.1 s | `24261af50e7bfd26a67113ad94ecd6c6b38fdd7a` | [4m7r1pwz](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4m7r1pwz) | Not run; below predeclared best-score threshold |
| 12 | `outputs/overnight_2026-10-21/a_12_lr0001_seed42` | 42 | 0.056911 | 0.219512 | 0.138211 | 0.9868260379 | 1 / 41 | 168.3 s | `8782a6afd3cfc6ec3225fc5c9879ce000a384675` | [st0aibk7](https://wandb.ai/jlinki/seissepa-autoresearch/runs/st0aibk7) | Not run; below verified best |
| 13 | `outputs/overnight_2026-10-21/a_13_lr0003_seed42` | 42 | 0.032520 | 0.227642 | 0.130081 | 0.9792395698 | 3 / 43 | 168.8 s | `edef2bc2ce006231b01bca594d0baaac9cbe624f` | [4zdma9fo](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4zdma9fo) | Not run; below verified best |
| 14 | `outputs/overnight_2026-10-21/a_14_batch64_seed42` | 42 | 0.056911 | 0.243902 | 0.150407 | 0.1100223245 | 105 / 145 | 318.7 s | `fd442303da9568bc0d6b46405f7af636ee0717b9` | [w8p1wson](https://wandb.ai/jlinki/seissepa-autoresearch/runs/w8p1wson) | Not run; below verified best |
| 15 | `outputs/overnight_2026-10-21/a_15_reconstruction075_batch64_seed42` | 42 | 0.065041 | 0.211382 | 0.138211 | 0.4357008604 | 51 / 91 | 224.2 s | `97e6785fb93ac6427036c0b7376e0b6f40f7592b` | [rtev7ole](https://wandb.ai/jlinki/seissepa-autoresearch/runs/rtev7ole) | Not run; below verified best |
| 16 | `outputs/overnight_2026-10-21/a_16_station015_batch64_seed42` | 42 | 0.024390 | 0.235772 | 0.130081 | 0.3840518899 | 55 / 95 | 202.8 s | `4d8d5a534186588d3cd7d0270273968befc6fe98` | [nizxqasz](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nizxqasz) | Not run; below verified best |
| 17 | `outputs/overnight_2026-10-21/a_17_event015_batch64_seed42` | 42 | 0.040650 | 0.235772 | 0.138211 | 0.1510001148 | 77 / 117 | 238.0 s | `07f4af7079335df608fef91f183876d04a247e41` | [19m2bufm](https://wandb.ai/jlinki/seissepa-autoresearch/runs/19m2bufm) | Not run; below verified best |
| 18 | `outputs/overnight_2026-10-21/a_18_lr0001_batch64_balanced_seed42` | 42 | 0.073171 | 0.186992 | 0.130081 | 0.9714779560 | 7 / 47 | 232.6 s | `abf5993d50aa93e6edae41cc746991483d545080` | [60uyru99](https://wandb.ai/jlinki/seissepa-autoresearch/runs/60uyru99) | Not run; below verified best |
| 19 | `outputs/overnight_2026-10-21/a_19_weightdecay0_batch64_seed42` | 42 | 0.056911 | 0.243902 | 0.150407 | 0.1100223245 | 105 / 145 | 445.2 s (298.5 s train) | `04974d56fd075190b4494e32c6463aaf293f6c4a` | [nlie1gun](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nlie1gun) | Not run; below slot-6 best |
| 20 | `outputs/overnight_2026-10-21/a_20_batch32_seed42` | 42 | 0.056911 | 0.325203 | 0.191057 | 0.3430895047 | 36 / 76 | 238.8 s | `56dff99769a79141ebdf5ead5135a069941769c6` | [7xrhy2hl](https://wandb.ai/jlinki/seissepa-autoresearch/runs/7xrhy2hl) | 101 fields, 0 mismatches |
| 21 | `outputs/overnight_2026-10-21/a_21_batch32_seed42_repeat` | 42 | 0.056911 | 0.325203 | 0.191057 | 0.3430895047 | 36 / 76 | 210.0 s | `8b5a3940dd8c3906c60be67ab3676ab996b5c323` | [b6gjrlva](https://wandb.ai/jlinki/seissepa-autoresearch/runs/b6gjrlva) | 100 fields, 0 mismatches; exact repeat of slot 20 |
| 22 | `outputs/overnight_2026-10-21/a_22_batch32_seed43` | 43 | 0.065041 | 0.357724 | **0.211382** | 0.0764759242 | 79 / 119 | 261.5 s | `c19ef9a05f0aa589de1fb7b77d956b8f4316a128` | [gtb11jjt](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gtb11jjt) | 101 fields, 0 mismatches |
| 23 | `outputs/overnight_2026-10-21/a_23_batch32_seed44` | 44 | 0.065041 | 0.308943 | 0.186992 | 0.1087974898 | 52 / 92 | 201.5 s | `ed9a5ca647511f1b3c01ec9f289fc91b2148c24c` | [qwaj8efb](https://wandb.ai/jlinki/seissepa-autoresearch/runs/qwaj8efb) | Not run; below verified slot-22 candidate |
| 24 | `outputs/overnight_2026-10-21/a_24_reconstruction075_batch32_seed42` | 42 | 0.040650 | 0.325203 | 0.182927 | 0.3096749757 | 41 / 81 | 175.2 s | `d5a801783ddbf78b639365de244842c6820d4bf4` | [h1pxqf1c](https://wandb.ai/jlinki/seissepa-autoresearch/runs/h1pxqf1c) | Not run; below verified slot-22 candidate |
| 25 | `outputs/overnight_2026-10-21/a_25_reconstruction125_batch32_seed42` | 42 | 0.081301 | 0.308943 | 0.195122 | 0.0827547192 | 76 / 116 | 227.9 s | `855fe0b59de169e49a51feefc65920ba6f82ec20` | [eipjg9re](https://wandb.ai/jlinki/seissepa-autoresearch/runs/eipjg9re) | Not run; below verified slot-22 candidate |
| 26 | `outputs/overnight_2026-10-21/a_26_station015_rec125_batch32_seed42` | 42 | 0.040650 | 0.341463 | 0.191057 | 0.1552749467 | 51 / 91 | 182.2 s | `f9d6c65fd15061de0ac1b24439c72c072aa2e553` | [313eahzn](https://wandb.ai/jlinki/seissepa-autoresearch/runs/313eahzn) | Not run; below verified slot-22 candidate |
| 27 | `outputs/overnight_2026-10-21/a_27_event015_rec125_batch32_seed42` | 42 | 0.048780 | 0.300813 | 0.174797 | 0.0761124201 | 73 / 113 | 402.3 s | `c2946974a2c8db9f41259b1374e460f7b3c6e2a7` | [s5n6x4qu](https://wandb.ai/jlinki/seissepa-autoresearch/runs/s5n6x4qu) | Not run; below verified slot-22 candidate |
| 28 | `outputs/overnight_2026-10-21/a_28_margin010_batch32_seed42` | 42 | 0.048780 | 0.325203 | 0.186992 | 0.0756246267 | 82 / 122 | 287.9 s | `91bd0ca3d17ed4c050dc1f1561506f84424916cc` | [gxjhyom1](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gxjhyom1) | Not run; below verified slot-22 candidate |
| 29 | `outputs/overnight_2026-10-21/a_29_lr0001_batch32_seed42` | 42 | 0.032520 | 0.276423 | 0.154472 | 0.1781881134 | 73 / 113 | 233.1 s | `51bb844339f8b3739716f6cf3687db58846d1894` | [ddmfehkr](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ddmfehkr) | Not run; below verified slot-22 candidate |
| 30 | `outputs/overnight-2026-10-21/a_30_lr0003_batch32_seed42` | 42 | 0.048780 | 0.349593 | 0.199187 | 0.1021932648 | 48 / 88 | 205.1 s (195.97 s train) | `10d42011f3312fc71940a9383f2e43d04a559f94` | [vsb52wez](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vsb52wez) | Not run; below verified slot-22 candidate |

All runs used online W&B in `jlinki/seissepa-autoresearch` and have the required local files: `config.yaml`, `provenance.json`, `source.tar.gz`, `history.jsonl`, `best.pt`, `metrics.json`, and `report.md`. The guarded runner verified the frozen benchmark before each training attempt. Exact commands, run outcomes, and commit IDs are in `experiments/journal.jsonl`.

## Previous best checkpoint and three-seed evidence (before slots 20–23)

Before slots 20–23, the best valid single checkpoint was `outputs/overnight_2026-10-21/a_06_balanced010_seed43/best.pt`, trained from commit `de03a8ff9a1186c0b2a4753cf82f44d00a45ac99`. It scored 0.158537 (event 0.056911, station 0.260163) with canonical validation MSE 0.6569846171 at epoch 48. The protected standalone evaluator ran on GPU 0 at batch size 128 and matched all 100 shared scientific fields with zero mismatches. Slot 22 superseded it as the current best; see the final section.

Prior best configuration: fixed `model=notebook` (NotebookAE); `loss=separation`; reconstruction/event/station weights `1.0/0.10/0.10`; margin `0.2`; AdamW learning rate `0.0002`, weight decay `0.0001`; gradient clip `1.0`; batch size `128`; maximum 200 epochs; patience `40`. The checkpoint's training seed is 43.

| Confirmation seed | Event top-1 | Station top-1 | Combined score | Canonical val MSE |
|---:|---:|---:|---:|---:|
| 42 (slot 5; exact repeat of slot 4) | 0.024390 | 0.235772 | 0.130081 | 0.9618484117 |
| 43 (slot 6) | 0.056911 | 0.260163 | 0.158537 | 0.6569846171 |
| 44 (slot 7) | 0.040650 | 0.227642 | 0.134146 | 0.5152695297 |
| Three-seed mean | — | — | **0.140921** | — |
| Three-seed sample standard deviation | — | — | **0.015390** | — |

At that point, the three-seed mean was 0.018970 above the fresh seed-42 baseline score of 0.121951, and each confirmation score exceeded that baseline. These earlier validation results were used for selection. The final section reports the later batch-size-32 results and the current best; no confidence interval is assigned to the mean of the two retrieval tasks.

## Reconstruction, margin, batch, and learning-rate screens

Hypothesis: reducing the reconstruction weight from `1.0` to `0.5` while retaining event/station weights `0.10/0.10` could strengthen the latent separation signal. The single seed-42 screen completed at slot 8, selecting epoch 1 after 41 epochs. Event retrieval was 0.048780, station retrieval 0.203252, the combined score 0.126016, and canonical validation MSE 0.985822. This is above the fresh default baseline 0.121951 but below the balanced configuration's seed-42 score 0.130081 and its best verified checkpoint at 0.158537. W&B online logging succeeded and all required artifacts are local. The checkpoint was not separately evaluated because it did not improve the verified candidate. Preserve the outcome and keep the balanced configuration as best.

The predeclared margin screen raised the balanced configuration's margin from `0.2` to `0.3`, leaving NotebookAE, weights `1.0/0.10/0.10`, AdamW settings, batch size 128, gradient clip, and 200-epoch/patience-40 budget fixed. Slot 9 selected epoch 1 after 41 epochs: event retrieval 0.048780, station retrieval 0.203252, score 0.126016, canonical validation MSE 0.9858216842, and guarded elapsed time 164.5 s (training elapsed 153.45 s). Online W&B completed and all required local artifacts are present. This is below the balanced margin-0.2 seed-42 result 0.130081 and the best verified score 0.158537, so it was not separately evaluated and does not replace the slot-6 checkpoint.

The predeclared margin-0.1 screen kept the same balanced weights, optimizer, batch size, gradient clip, and budget. Slot 10 selected epoch 1 after 41 epochs: event retrieval 0.048780, station retrieval 0.203252, score 0.126016, canonical validation MSE 0.9858216842, and guarded elapsed time 217.7 s (training elapsed 208.2 s). Online W&B completed and all required local artifacts are present. The score is below both the balanced margin-0.2 seed-42 result 0.130081 and the best verified result 0.158537, so it was not separately evaluated and does not replace slot 6.

The predeclared batch-size-256 screen kept NotebookAE, weights `1.0/0.10/0.10`, margin `0.2`, AdamW learning rate `0.0002`, weight decay `0.0001`, gradient clip `1.0`, and the 200-epoch/patience-40 budget fixed. Slot 11 selected epoch 16 after 56 epochs, scoring 0.134146 (event 0.056911, station 0.211382) with canonical validation MSE 0.9687882376. Guard elapsed time was 373.1 s (training elapsed 355.3 s). Online W&B completed, the benchmark fingerprint matched, and all required local artifacts are present. The screen exceeds the balanced batch-128 seed-42 score 0.130081 by 0.004065, but is below the best verified score 0.158537 and the balanced configuration's three-seed mean 0.140921. Under the predeclared decision rule it is not promoted, was not separately evaluated, and does not warrant confirmation runs. Keep the balanced batch-128 seed-43 checkpoint as best.

The predeclared learning-rate screen kept NotebookAE, weights `1.0/0.10/0.10`, margin `0.2`, weight decay `0.0001`, gradient clip `1.0`, batch size 128, and the 200-epoch/patience-40 budget fixed, while lowering AdamW learning rate from `0.0002` to `0.0001`. Slot 12 completed 41 epochs and selected epoch 1. It scored 0.138211 (event 0.056911, station 0.219512) with canonical validation MSE 0.9868260379. Guard elapsed time was 168.3 s (training elapsed 158.3 s). Online W&B completed, the benchmark fingerprint matched, and all required local artifacts are present. The score is 0.008130 above the balanced seed-42 result 0.130081, but 0.020325 below the best verified score 0.158537. It does not rank ahead of the current best, so it was not separately evaluated and does not warrant confirmation runs. Keep the balanced batch-128 seed-43 checkpoint as best.

The predeclared learning-rate-0.0003 screen held NotebookAE, separation weights `1.0/0.10/0.10`, margin `0.2`, weight decay `0.0001`, gradient clip `1.0`, batch size 128, and the 200-epoch/patience-40 budget fixed. Slot 13 completed 43 epochs and selected epoch 3. It scored 0.130081 (event 0.032520, station 0.227642) with canonical validation MSE 0.9792395698. Guard elapsed time was 168.8 s (training elapsed 159.0 s). Online W&B completed at [4zdma9fo](https://wandb.ai/jlinki/seissepa-autoresearch/runs/4zdma9fo), the benchmark fingerprint matched, and all required local artifacts are present. It is below the verified best score 0.158537; it also ties slot 4 at 0.130081 but loses on MSE (0.9792395698 versus 0.9618484117). It was not separately evaluated and does not warrant confirmation. Keep the slot-6 checkpoint as best.

## Batch-size-64 screen

The predeclared screen held NotebookAE, balanced separation weights `1.0/0.10/0.10`, margin `0.2`, AdamW learning rate `0.0002` and weight decay `0.0001`, gradient clip `1.0`, and the 200-epoch/patience-40 budget fixed while reducing batch size from 128 to 64. Slot 14 completed 145 epochs and selected epoch 105. It scored 0.150407 (event 0.056911, station 0.243902) with canonical validation MSE 0.1100223245. Guard elapsed time was 318.7 s (training elapsed 309.2 s). Online W&B completed at [w8p1wson](https://wandb.ai/jlinki/seissepa-autoresearch/runs/w8p1wson), the benchmark fingerprint matched, and all required local artifacts are present.

This is the strongest seed-42 screen so far and improves on the fresh baseline (0.121951) and the balanced batch-128 seed-42 result (0.130081). Its score remains below the current best verified checkpoint, slot 6 at 0.158537. The fixed ordering compares score before MSE, so the lower MSE does not change that ranking. The candidate did not meet the predeclared standalone-evaluation or confirmation threshold. Preserve its complete run artifacts and keep the evaluated slot-6 checkpoint as best valid.

## Reconstruction-weight-0.75 batch-size-64 screen

The predeclared screen kept NotebookAE, event/station weights `0.10/0.10`, margin `0.2`, AdamW learning rate `0.0002` and weight decay `0.0001`, gradient clip `1.0`, batch size 64, and the 200-epoch/patience-40 budget fixed while reducing the reconstruction weight from `1.0` to `0.75`. Slot 15 completed 91 epochs and selected epoch 51. It scored 0.138211 (event 0.065041, station 0.211382) with canonical validation MSE 0.4357008604. Guard elapsed time was 224.2 s (training elapsed 213.7 s). Online W&B completed at [rtev7ole](https://wandb.ai/jlinki/seissepa-autoresearch/runs/rtev7ole), the benchmark fingerprint matched, and all required local artifacts are present.

Relative to the batch-size-64 screen with reconstruction weight `1.0` (slot 14), event retrieval rose by 0.008130 while station retrieval fell by 0.032520; the combined score fell by 0.012195, from 0.150407 to 0.138211. The canonical MSE was also higher. Slot 15 remains below the verified slot-6 checkpoint at 0.158537, so it was not evaluated or promoted. Preserve its run artifacts and keep slot 6 as best valid.

## Station-weight-0.15 batch-size-64 screen

The predeclared screen raised station separation weight from `0.10` to `0.15` while holding NotebookAE, reconstruction/event weights `1.0/0.10`, margin `0.2`, AdamW learning rate `0.0002` and weight decay `0.0001`, gradient clip `1.0`, batch size 64, and the 200-epoch/patience-40 budget fixed. Slot 16 completed 95 epochs and selected epoch 55. It scored 0.130081 (event 0.024390, station 0.235772) with canonical validation MSE 0.3840518899. Guard elapsed time was 202.8 s (training elapsed 193.8 s). Online W&B completed at [nizxqasz](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nizxqasz); the benchmark fingerprint matched, all seven required local artifacts are present, and the run's source commit is recorded in provenance, checkpoint, metrics, and its local report.

The score is below both the balanced batch-size-64 screen (0.150407) and the best verified checkpoint (0.158537). Event and station retrieval were both lower than the batch-size-64 balanced result; the lower MSE does not change ranking because the primary score is lower. Do not evaluate or spend confirmation slots on this candidate. Keep the verified slot-6 seed-43 checkpoint as best.

## Event-weight-0.15 batch-size-64 screen

The predeclared screen raised event separation weight from `0.10` to `0.15` while holding NotebookAE, reconstruction/station weights `1.0/0.10`, margin `0.2`, AdamW learning rate `0.0002` and weight decay `0.0001`, gradient clip `1.0`, batch size 64, and the 200-epoch/patience-40 budget fixed. Slot 17 completed 117 epochs and selected epoch 77. It scored 0.138211 (event 0.040650, station 0.235772) with canonical validation MSE 0.1510001148. Guard elapsed time was 238.0 s. Online W&B completed at [19m2bufm](https://wandb.ai/jlinki/seissepa-autoresearch/runs/19m2bufm), the benchmark fingerprint matched, and all seven required local artifacts are present.

Event and station retrieval were both below the balanced batch-64 screen (0.056911 and 0.243902), and the score is below the verified slot-6 best of 0.158537. Its lower MSE than slot 6 does not affect ranking because the primary scores are not tied. Do not evaluate or spend confirmation attempts on this candidate. Preserve slot 17 and keep the evaluated slot-6 seed-43 checkpoint as best.

## Learning-rate-0.0001 batch-size-64 screen

The predeclared screen held NotebookAE, balanced separation weights `1.0/0.10/0.10`, margin `0.2`, AdamW weight decay `0.0001`, gradient clip `1.0`, batch size 64, and the 200-epoch/patience-40 budget fixed while lowering AdamW learning rate from `0.0002` to `0.0001`. Slot 18 completed 47 epochs and selected epoch 7. It scored 0.130081 (event 0.073171, station 0.186992) with canonical validation MSE 0.9714779560. Guard elapsed time was 232.6 s (training elapsed 222.2 s). Online W&B completed at [60uyru99](https://wandb.ai/jlinki/seissepa-autoresearch/runs/60uyru99); the benchmark fingerprint matched, all seven required local artifacts are present, and commit `abf5993d50aa93e6edae41cc746991483d545080` is recorded in provenance and metrics.

Relative to the balanced batch-size-64 screen at learning rate `0.0002` (slot 14), event retrieval rose by 0.016260 while station retrieval fell by 0.056911; the combined score fell by 0.020325, from 0.150407 to 0.130081. The candidate is below the verified slot-6 best at 0.158537, so it was not evaluated and does not warrant confirmation runs. Preserve its artifacts and keep the evaluated slot-6 seed-43 checkpoint as best valid.

This proposed zero-weight-decay screen was tested as slot 19; see the following section.

## Zero-weight-decay batch-size-64 screen

The predeclared screen kept NotebookAE, balanced separation weights `1.0/0.10/0.10`, margin `0.2`, AdamW learning rate `0.0002`, gradient clip `1.0`, batch size 64, seed 42, and the 200-epoch/patience-40 budget fixed while lowering AdamW weight decay from `0.0001` to `0.0`. Slot 19 completed 145 epochs and selected epoch 105. It scored 0.150407 (event 0.056911, station 0.243902) with canonical validation MSE 0.1100223245. Guard elapsed time was 445.2 s; training elapsed time was 298.5 s. The online W&B run completed at [nlie1gun](https://wandb.ai/jlinki/seissepa-autoresearch/runs/nlie1gun), the benchmark fingerprint matched, and all required local artifacts are present. Training used commit `04974d56fd075190b4494e32c6463aaf293f6c4a`.

The score and canonical MSE match slot 14's batch-size-64 balanced screen, and the score remains below the verified slot-6 best at 0.158537. It did not meet the predeclared standalone-evaluation or confirmation threshold. Preserve the run and keep the evaluated slot-6 seed-43 checkpoint as best valid.

The exact command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=separation loss.reconstruction_weight=1.0 loss.event_weight=0.10 loss.station_weight=0.10 loss.margin=0.2 seed=42 train.epochs=200 train.patience=40 train.batch_size=64 train.grad_clip=1.0 optimizer.lr=0.0002 optimizer.weight_decay=0.0 device=cuda:0 threads=4 wandb.mode=online wandb.name=overnight_a_19_weightdecay0_batch64_seed42 hydra.run.dir=outputs/overnight_2026-10-21/a_19_weightdecay0_batch64_seed42
```

The batch-size-32 hypothesis proposed here was tested in slots 20–23; see the final section below.


## Batch-size-32 screen and seed confirmations

Hypothesis: at batch size 32, more optimizer updates per epoch may improve retrieval over batch size 64. Slot 20 isolated batch size against slot 19 while keeping NotebookAE, separation weights `reconstruction/event/station = 1.0/0.10/0.10`, margin `0.2`, AdamW learning rate `0.0002`, zero weight decay, gradient clip `1.0`, seed 42, and the 200-epoch/patience-40 budget fixed. The exact screen command and the repeat/seed-43/44 commands are in `experiments/journal.jsonl`.

The seed-42 screen scored 0.191057 (event 0.056911, station 0.325203), with canonical validation MSE 0.3430895047 at epoch 36 after 76 epochs. Its online W&B run completed at [7xrhy2hl](https://wandb.ai/jlinki/seissepa-autoresearch/runs/7xrhy2hl), and its standalone evaluator matched all 101 shared scientific fields exactly. The exact seed-42 repeat also selected epoch 36 and matched all 100 shared scientific metrics with zero mismatches; its W&B run is [b6gjrlva](https://wandb.ai/jlinki/seissepa-autoresearch/runs/b6gjrlva).

Seed 43 scored 0.211382 (event 0.065041, station 0.357724), with canonical validation MSE 0.0764759242 at epoch 79 after 119 epochs. Its standalone evaluator matched all 101 shared scientific fields exactly, and the checkpoint is the best valid candidate in this lane. Seed 44 scored 0.186992 (event 0.065041, station 0.308943), with canonical validation MSE 0.1087974898 at epoch 52 after 92 epochs. It was not evaluated because it ranks below the verified seed-43 checkpoint.

| Seed | Slot | Event top-1 | Station top-1 | Score | Canonical val MSE | Best epoch / epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 20 (repeat 21 matched exactly) | 0.056911 | 0.325203 | 0.191057 | 0.3430895047 | 36 / 76 |
| 43 | 22 | 0.065041 | 0.357724 | **0.211382** | 0.0764759242 | 79 / 119 |
| 44 | 23 | 0.065041 | 0.308943 | 0.186992 | 0.1087974898 | 52 / 92 |
| Three-seed mean | — | — | — | **0.196477** | — | — |
| Three-seed sample SD | — | — | — | **0.013067** | — | — |

All three seed scores exceed the fresh overnight default seed-42 baseline of 0.121951. The three-seed mean is 0.074526 above that baseline and 0.055556 above the prior balanced batch-size-128 three-seed mean of 0.140921. The best valid checkpoint is `outputs/overnight_2026-10-21/a_22_batch32_seed43/best.pt`, trained from commit `c19ef9a05f0aa589de1fb7b77d956b8f4316a128`, with the configuration above. Its benchmark ID is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`. The batch consumed slots 20–23; all attempts and evaluator outcomes are in the pushed journal. These are validation results after repeated selection, not untouched test results; no confidence interval is assigned to the two-task mean.

After slot 24, the next hypothesis was to raise reconstruction weight to 1.25 at batch size 32. Slot 25 tested that screen; see the final section.

## Reconstruction-weight-0.75 batch-size-32 seed-42 screen

Hypothesis: at batch size 32, reducing the reconstruction weight from 1.0 to 0.75 while keeping separation weights 0.10/0.10 may free the latent representation to improve retrieval. Slot 24 isolated this weight against the verified batch-size-32 balanced configuration: NotebookAE, margin 0.2, AdamW learning rate 0.0002 and zero weight decay, gradient clip 1.0, seed 42, and the 200-epoch/patience-40 budget remained fixed. The exact command and bounded conditional confirmation commands are in `experiments/journal.jsonl`.

Slot 24 completed 81 epochs and selected epoch 41. It scored 0.182927 (event retrieval 0.040650, station retrieval 0.325203) with canonical validation MSE 0.3096749757. Guard elapsed time was 175.2 s (metrics elapsed time 165.0 s). The run used commit `d5a801783ddbf78b639365de244842c6820d4bf4`, verified benchmark ID `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`, and online W&B completed at [h1pxqf1c](https://wandb.ai/jlinki/seissepa-autoresearch/runs/h1pxqf1c). All required local artifacts are present.

The score is below both slot 20 at 0.191057 and the verified slot-22 best at 0.211382. The guarded evaluator and conditional exact seed-42 repeat/seeds 43/44 were therefore not run. The leaderboard confirms slot 22 remains the best valid checkpoint. Preserve slot 24 and keep the current slot-22 configuration and checkpoint.

Next bounded hypothesis: at batch size 32, raise reconstruction weight to 1.25 while keeping separation weights 0.10/0.10, margin 0.2, AdamW `lr=0.0002, weight_decay=0.0`, clipping 1.0, seed-42 screening, and the 200-epoch/patience-40 budget fixed. This tests whether retaining more reconstruction pressure improves on both the 0.75 screen and the current 1.0 configuration. Run the exact seed-42 repeat and seeds 43/44 only if the screen ranks ahead of the current verified candidate and its guarded evaluation matches.

## Reconstruction-weight-1.25 batch-size-32 seed-42 screen

Hypothesis: at batch size 32, raising NotebookAE reconstruction weight from 1.0 to 1.25 while keeping event and station separation weights at 0.10 may retain useful waveform structure and improve retrieval over both the balanced 1.0 and 0.75 screens. Architecture, margin 0.2, AdamW learning rate 0.0002 and zero weight decay, gradient clip 1.0, sampling, and the 200-epoch/patience-40 budget stayed fixed. The exact screen and conditional confirmation commands were journaled and pushed before training in commit `855fe0b59de169e49a51feefc65920ba6f82ec20`.

Slot 25 completed 116 epochs and selected epoch 76. It scored 0.195122 (event retrieval 0.081301, station retrieval 0.308943) with canonical validation MSE 0.0827547192. Guard elapsed time was 227.9 seconds; training metrics elapsed time was 217.7 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`, online W&B completed at [eipjg9re](https://wandb.ai/jlinki/seissepa-autoresearch/runs/eipjg9re), and all seven required local artifacts are present.

Slot 25 is 0.004065 above the balanced seed-42 batch-size-32 screen at slot 20 (0.191057), with higher event retrieval and lower station retrieval. It remains 0.016260 below verified slot 22 (0.211382), and its MSE is higher than slot 22's 0.0764759. Under the predeclared ranking rule it was not evaluated and did not qualify for confirmations. Preserve its run and keep `outputs/overnight_2026-10-21/a_22_batch32_seed43/best.pt` as the best valid checkpoint. The existing balanced batch-size-32 three-seed evidence remains 0.191057, 0.211382, and 0.186992 (mean 0.196477; sample SD 0.013067).

## Station-weight-0.15 screen at reconstruction weight 1.25, batch size 32

Hypothesis: with reconstruction weight 1.25 and batch size 32, raising station separation weight from 0.10 to 0.15 while keeping event weight 0.10 might recover station retrieval without losing the event gain from slot 25. The exact command and conditional confirmation commands were journaled and pushed before training in commit `f9d6c65fd15061de0ac1b24439c72c072aa2e553`.

Slot 26 completed 91 epochs and selected epoch 51. It scored 0.191057 (event retrieval 0.040650, station retrieval 0.341463) with canonical validation MSE 0.1552749467. Guard elapsed time was 182.2 seconds; training metrics elapsed time was 172.2 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; online W&B completed at [313eahzn](https://wandb.ai/jlinki/seissepa-autoresearch/runs/313eahzn), and all seven required local artifacts are present. Training provenance records commit `f9d6c65fd15061de0ac1b24439c72c072aa2e553`.

Compared with slot 25, station retrieval rose from 0.308943 to 0.341463 while event retrieval fell from 0.081301 to 0.040650. The combined score fell from 0.195122 to 0.191057, and canonical MSE rose from 0.0827547192 to 0.1552749467. Slot 26 remains below the verified slot-22 best score of 0.211382, so the standalone evaluator and conditional confirmation runs were not performed. Preserve the run and keep slot 22 as the best valid checkpoint.

## Event-weight-0.15 screen at reconstruction weight 1.25, batch size 32

Hypothesis: at batch size 32 and reconstruction weight 1.25, raising event separation weight from 0.10 to 0.15 while holding station weight at 0.10 might recover event retrieval without losing too much station accuracy. The exact screen and conditional confirmation commands were journaled and pushed before training in commit `c2946974a2c8db9f41259b1374e460f7b3c6e2a7`.

Slot 27 completed 113 epochs and selected epoch 73. It scored 0.174797 (event retrieval 0.048780, station retrieval 0.300813) with canonical validation MSE 0.0761124201. Guard elapsed time was 402.3 seconds; metrics elapsed time was 392.1 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`, online W&B completed at [s5n6x4qu](https://wandb.ai/jlinki/seissepa-autoresearch/runs/s5n6x4qu), and all seven required local artifacts are present. Training provenance records pushed commit `c2946974a2c8db9f41259b1374e460f7b3c6e2a7`.

The score is below the verified slot-22 best of 0.211382, so the standalone evaluator and conditional repeat/seeds 43/44 were not run. The candidate's lower MSE than slot 22 does not change the ranking because the primary scores differ. Preserve slot 27 and keep slot 22 as the best valid checkpoint.

## Margin-0.1 screen at batch size 32

Hypothesis: lowering the cosine separation margin from 0.2 to 0.1 at batch size 32, with balanced reconstruction/event/station weights `1.0/0.10/0.10`, might improve retrieval by softening the pairwise separation threshold. NotebookAE, AdamW learning rate `0.0002`, zero weight decay, gradient clip `1.0`, sampling, and the 200-epoch/patience-40 budget stayed fixed. The exact screen and conditional confirmation commands were journaled and pushed before training in commit `91bd0ca3d17ed4c050dc1f1561506f84424916cc`.

Slot 28 completed 122 epochs and selected epoch 82. It scored 0.186992 (event retrieval 0.048780, station retrieval 0.325203), with canonical validation MSE 0.0756246267. Guard elapsed time was 287.9 seconds and training metrics elapsed time was 278.3 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; online W&B completed at [gxjhyom1](https://wandb.ai/jlinki/seissepa-autoresearch/runs/gxjhyom1), and all required local artifacts are present.

Slot 28 ranks below verified slot 22's score of 0.211382. Its MSE is lower than slot 22's 0.0764759242, but the scores differ, so the primary retrieval score determines the ranking. Under the predeclared rule it was not separately evaluated, and the exact seed-42 repeat and seeds 43/44 were not run. Preserve slot 28 and keep slot 22 as the best valid checkpoint.

## AdamW learning-rate-0.0001 screen at batch size 32

Hypothesis: at batch size 32, lowering AdamW learning rate from 0.0002 to 0.0001 may smooth optimization and improve retrieval for the balanced NotebookAE separation objective. Reconstruction/event/station weights `1.0/0.10/0.10`, margin 0.2, zero weight decay, gradient clip 1.0, sampling, seed 42, and the 200-epoch/patience-40 budget stayed fixed. The exact command and conditional repeat/seed commands were committed and pushed before training in commit `51bb844339f8b3739716f6cf3687db58846d1894`.

Slot 29 completed 113 epochs and selected epoch 73. It scored 0.154472 (event retrieval 0.032520, station retrieval 0.276423) with canonical validation MSE 0.1781881134. Guard elapsed time was 233.1 seconds; training metrics elapsed time was 223.7 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`, online W&B completed at [ddmfehkr](https://wandb.ai/jlinki/seissepa-autoresearch/runs/ddmfehkr), and all seven required local artifacts are present. Training provenance records commit `51bb844339f8b3739716f6cf3687db58846d1894`.

Slot 29 ranks below the verified slot-22 best (0.211382) and the prior balanced batch-32 seed-42 result (0.191057). The standalone evaluator and conditional exact seed-42 repeat/seeds 43/44 were therefore not run. Preserve the complete run and keep slot 22 as the best valid checkpoint.

## AdamW learning-rate-0.0003 screen at batch size 32

Hypothesis: at batch size 32, increasing AdamW learning rate from 0.0002 to 0.0003 may improve optimization and retrieval after the 0.0001 screen underperformed. NotebookAE, balanced reconstruction/event/station weights 1.0/0.10/0.10, margin 0.2, zero weight decay, gradient clip 1.0, sampling, and the 200-epoch/patience-40 budget stayed fixed. The exact screen and conditional confirmation commands were committed and pushed before training in commit `10d42011f3312fc71940a9383f2e43d04a559f94`.

Slot 30 completed 88 epochs and selected epoch 48. It scored 0.199187 (event retrieval 0.048780, station retrieval 0.349593) with canonical validation MSE 0.1021932648. Guard elapsed time was 205.1 seconds; training elapsed time was 195.97 seconds. The benchmark ID matched `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`; online W&B completed at [vsb52wez](https://wandb.ai/jlinki/seissepa-autoresearch/runs/vsb52wez), all seven required local artifacts are present, and provenance records training commit `10d42011f3312fc71940a9383f2e43d04a559f94`.

The score improved on the balanced batch-32 seed-42 result at slot 20 (0.191057) but remains below verified slot 22 (0.211382). The predeclared standalone evaluator, exact seed-42 repeat, and seeds 43/44 were not run. Preserve slot 30 and keep slot 22 as the best valid checkpoint.

## Current lane state and next hypothesis

Lane A has consumed 30 of 64 attempts: 24 seed-42 screens and 6 confirmations. There are 34 attempts left; at least 12 remain reserved for confirmations. The 48-screen cap allows 24 additional seed-42 screens, while preserving the future confirmation reserve limits additional screens to 22. The best valid checkpoint remains `outputs/overnight_2026-10-21/a_22_batch32_seed43/best.pt`, trained from commit `c19ef9a05f0aa589de1fb7b77d956b8f4316a128`. Its fixed configuration is NotebookAE with separation weights `reconstruction/event/station = 1.0/0.10/0.10`, margin 0.2, AdamW `lr=0.0002, weight_decay=0.0`, gradient clip 1.0, batch size 32, seed 43, maximum 200 epochs, and patience 40. It scored 0.211382 (event 0.065041, station 0.357724) with canonical validation MSE 0.0764759242 at epoch 79 after 119 epochs; its standalone evaluator matched all 101 shared scientific fields.

The balanced configuration's seed-42, seed-43, and seed-44 scores remain 0.191057, 0.211382, and 0.186992, with mean 0.196477 and sample SD 0.013067. The exact seed-42 repeat matched slot 20. Full run inventory and W&B links for slots 1–30 are above; exact commands and attempt records are in `experiments/journal.jsonl`. These are validation results after repeated selection, not untouched test results; no confidence interval is assigned to the two-task mean.

Next bounded hypothesis for a future invocation: at the batch-size-32 balanced configuration, raise AdamW learning rate from `0.0003` to `0.0004` with zero weight decay to test whether the seed-42 improvement from 0.0002 (score 0.191057) to 0.0003 (0.199187) continues. Hold the architecture, loss weights, margin, batch size, gradient clip, and budget fixed. Run the exact repeat and seeds 43/44 only if the seed-42 screen ranks ahead of the verified best and its guarded standalone evaluation matches.
