# Overnight 2026-10-21 — lane B report

## Invocation status

This bounded invocation is complete. Lane B is on branch `research/overnight-20261021-b`; two of 64 cumulative attempt slots were consumed (slot 1 baseline, slot 2 supervised-contrastive screen). The invocation stayed within its four-attempt limit. The worktree is to be left clean and pushed.

The frozen benchmark is `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26` (1,225 traces; 1,102 training and 123 validation; 35 event and 35 station classes). Pre-research checks passed: 24 tests and `scripts/verify_benchmark.py`. The guard also verified the benchmark before each training attempt.

## Results

| Slot | Result | Event top-1 | Station top-1 | Combined score | Canonical validation MSE | Seed / best epoch | Epochs | Guard elapsed | Run |
|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 | Fresh full-budget separation baseline | 0.040650 | 0.203252 | 0.121951 | 0.9858216584 | 42 / 1 | 41 | 296.09 s | `outputs/overnight_2026-10-21/01_baseline_seed42` |
| 2 | Supervised-contrastive screen; unconfirmed | 0.032520 | 0.219512 | 0.126016 | 0.9779323249 | 42 / 5 | 45 | 293.15 s | `outputs/overnight_2026-10-21/02_supcon_t010_seed42` |

Both runs completed with online W&B and all required local artifacts. The baseline used `model=notebook`, separation weights 1.0/0.05/0.05, AdamW at 0.0002 with weight decay 0.0001, batch size 128, 200 maximum epochs, and patience 40. Its provenance commit is `18c1da48c7ab5917c15bc6098fa481ccd212b6ed`; its W&B run is [9ja0fw0w](https://wandb.ai/jlinki/seissepa-autoresearch/runs/9ja0fw0w).

The candidate kept NotebookAE and reconstruction weight 1.0, replaced the two cosine-margin losses with multi-positive supervised-contrastive terms at event/station weights 0.05 and temperature 0.1, and kept all optimizer and training settings fixed. Its source/configuration commit is `ba0ab3d5297c235a394dffca319f65fe0910c854`; its W&B run is [a8r87xmw](https://wandb.ai/jlinki/seissepa-autoresearch/runs/a8r87xmw).

Exact commands:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=separation seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_baseline_seed42 hydra.run.dir=outputs/overnight_2026-10-21/01_baseline_seed42
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/overnight.py train -- model=notebook loss=supervised_contrastive seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_b_supcon_t010_seed42 hydra.run.dir=outputs/overnight_2026-10-21/02_supcon_t010_seed42
```

## Evaluation and decision

Both checkpoints passed the guarded standalone evaluator on physical GPU 0 at batch size 128. The evaluator matched all 97 shared scientific fields with zero mismatches for each run. Reports are saved as `01_baseline_seed42/best.evaluation.json` and `02_supcon_t010_seed42/best.evaluation.json` under the run paths.

The supervised-contrastive screen improves the fresh baseline score by 0.004065 and validation MSE by 0.007889. Station retrieval rises by 0.016260, while event retrieval falls by 0.008130. The score remains below the previous round-1 seed-42 best, 0.130081. Keep the candidate as a promising screen, not a promoted configuration.

No exact seed-42 repeat or seeds 43/44 have been run for this candidate. There is therefore no three-seed mean or sample standard deviation yet. The next batch should use the unchanged configuration and full budget for an exact seed-42 repeat plus seeds 43 and 44 before promotion. Sixty-two lane slots remain, including the required minimum of twelve reserved confirmation slots.

The earlier round-1 doubled-weight candidate still has the best reported single-seed score (0.130081); its three-seed mean was 0.119241, below the round-1 baseline 0.121951. All figures here are validation results used for selection. They are not an untouched test result, and no confidence interval is assigned to the combined objective.

## Journal and source history

Both batch plans and both attempt outcomes are recorded in `experiments/journal.jsonl`. The lane B source/configuration and journal commits were pushed before training and after each attempt. The candidate source is preserved on this branch; the local checkpoint and run artifacts remain under `outputs/overnight_2026-10-21/`.
