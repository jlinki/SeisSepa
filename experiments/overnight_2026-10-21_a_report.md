# Lane A overnight report

## Status

This invocation completed its declared four-attempt seed-42 screen and stopped. The lane consumed **4 of 64 attempts**: 4 seed-42 screens, 0 confirmation runs. There are 60 attempts left; keep at least 12 for confirmation work, leaving at most 44 further seed-42 screens under the 48-screen limit. No blocker occurred. No source, model, loss, configuration, evaluator, benchmark, or test files changed; the only tracked changes were the experiment journal and this report.

The lane is `research/overnight-20261021-a`. All run entries and evaluator checks are committed and pushed. Every training run passed the guarded benchmark verification, used physical GPU 0, NotebookAE, online W&B in `jlinki/seissepa-autoresearch`, and saved all required local artifacts. The frozen benchmark ID for every run is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 train and 123 validation).

Preflight: `scripts/verify_benchmark.py` passed; `pytest -q` passed all 24 tests. `scripts/leaderboard.py` ran before research. The round-one report and lane instructions were read before changes.

## Hypothesis and result

Hypothesis: independently increasing event or station separation weight may preserve the other retrieval task better than increasing both, given round one’s station gain and event loss under balanced doubled weights. The batch held architecture, reconstruction weight, margin, optimizer, batch size, seed, and schedule constant. Each run allowed 200 epochs with patience 40, batch size 128, and AdamW learning rate 0.0002. The required default baseline ran first.

| Run path | Separation weights (event / station) | Event top-1 | Station top-1 | Score | Canonical val MSE | Seed; best epoch / epochs | Guard elapsed | Source commit | W&B |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| `outputs/overnight_2026-10-21/a_01_baseline` | 0.05 / 0.05 | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 42; 1 / 41 | 300.9 s | `b0140623c71e49021ed594dccc8524f674ded1a2` | [0p7zsyx6](https://wandb.ai/jlinki/seissepa-autoresearch/runs/0p7zsyx6) |
| `outputs/overnight_2026-10-21/a_02_event010` | 0.10 / 0.05 | 0.048780 | 0.211382 | 0.130081 | 0.9858216307 | 42; 1 / 41 | 240.4 s | `d0135a6da852577a3bb0aa07abf269aade75a0b9` | [xhho1tww](https://wandb.ai/jlinki/seissepa-autoresearch/runs/xhho1tww) |
| `outputs/overnight_2026-10-21/a_03_station010` | 0.05 / 0.10 | 0.040650 | 0.211382 | 0.126016 | 0.9858217018 | 42; 1 / 41 | 262.0 s | `fb6f533c70dc263ccc82519f83578e5cfbf7a3ab` | [uspv5ol8](https://wandb.ai/jlinki/seissepa-autoresearch/runs/uspv5ol8) |
| `outputs/overnight_2026-10-21/a_04_balanced010` | 0.10 / 0.10 | 0.024390 | 0.235772 | **0.130081** | **0.9618484117** | 42; 10 / 50 | 258.3 s | `49a125d59ba35c2df1ac40d8623fd0066a9230fd` | [thjw54j9](https://wandb.ai/jlinki/seissepa-autoresearch/runs/thjw54j9) |

Run paths are under `outputs/overnight_2026-10-21/`. The exact guarded commands, full precision metrics, commits, and outcomes are in `experiments/journal.jsonl`.

The event-only run tied the balanced run on the primary score, but has higher MSE. The balanced run is therefore the best valid checkpoint in this screen under the evaluator’s tie rule. Its protected standalone evaluation at batch size 128 matched all 101 shared scientific fields exactly (zero mismatches): [evaluation JSON](../outputs/overnight_2026-10-21/a_04_balanced010/best.evaluation.json). The event-only checkpoint was also independently evaluated and matched all 101 fields with zero mismatches: [evaluation JSON](../outputs/overnight_2026-10-21/a_02_event010/best.evaluation.json).

## Best checkpoint and confirmation status

Best verified overnight seed-42 checkpoint: `outputs/overnight_2026-10-21/a_04_balanced010/best.pt`, trained from commit `49a125d59ba35c2df1ac40d8623fd0066a9230fd`.

Configuration: fixed `model=notebook`; `loss=separation`; reconstruction/event/station weights `1.0/0.10/0.10`; margin `0.2`; AdamW learning rate `0.0002`, weight decay `0.0001`; batch size 128; maximum 200 epochs; patience 40; seed 42. It stopped after 50 epochs and selected epoch 10. It matches the round-one best seed-42 score, task scores, and MSE exactly.

Same-budget confirmations are **not yet complete**. The next bounded invocation should repeat this exact config at seed 42, then run seeds 43 and 44 with the same budget. Preserve the current checkpoint pending those results; do not promote on this one seed alone.

For context only, round one tested the same 0.10/0.10 weights at 50 epochs and patience 12: seed-42 score 0.130081, seed-43 score 0.113821, seed-44 score 0.113821; mean 0.119241 and sample SD 0.009388. That is prior-budget evidence, not a substitute for the pending overnight-budget confirmations. Its mean was below the round-one default baseline of 0.121951. No confidence interval is assigned to the mean of the two retrieval tasks.

## Next hypothesis

First confirm whether the balanced 0.10/0.10 configuration reproduces and retains its score over seeds 42, 43, and 44 at the overnight budget. If the three-seed mean does not improve over the fresh overnight baseline, test one reconstruction-weight reduction with the confirmed best latent weights, keeping all other settings fixed. Validation has been used repeatedly for selection, so results remain validation evidence.
