# Overnight research, 21–22 October 2026

Owner request: run several Codex CLI GPT-6 Luna agents at maximum reasoning effort
through the night, sharing physical GPU 0. Existing W&B, Codex payload, commit and
push approvals persist. Read AGENTS.md and README.md first.

## Contract and limits

- Keep the frozen 35 × 35 benchmark and protected evaluator unchanged. Benchmark ID:
  `abafe9c36982e368359c6a5cd4005e3956d4626690f74d4516895831fb3b8a26`.
- The primary score remains `val/retrieval_accuracy`, with the existing exact tie rules.
- Use only physical GPU 0, `CUDA_VISIBLE_DEVICES=0`, `device=cuda:0`. GPUs 1–3 belong
  to other work. At most two GPU jobs total: one per researcher, including evaluation.
- Two independent branches/worktrees avoid concurrent source and journal edits.
  Never check out another branch in either worktree or edit the main checkout.
- Each researcher has at most **64 training attempts**, including failures and repeats;
  each attempt has at most **200 epochs**, fixed `train.patience=40`, and a **20-minute**
  wall-time limit. Start seed 42. At most four attempts per CLI invocation/batch.
- No new training after **22 October 2026 07:00 Europe/Oslo (05:00 UTC)**. Researcher
  reporting stops at 07:30 (05:30 UTC); final aggregation stops at 08:00 (06:00 UTC).
  The watchdog enforces these limits; do not extend or bypass them.
- Keep at least 12 of the 64 slots for exact seed-42 repeats and seeds 43/44 of the most
  promising configurations. Compare all three seeds at the same architecture and
  budget; publish all results and mean/sample SD. No lucky-seed-only promotion.
- Stop a lane on authentication, integrity, upstream-push or online logging failure.
  Preserve failures and explain the blocker; do not switch W&B offline.

## Two hypotheses

Lane A: the existing reconstruction-heavy objective prevents useful latent separation.
Tune reconstruction/event/station weights, margin, learning rate, weight decay, batch
size, gradient clipping and schedules. Prefer configuration changes and small, justified
training changes. Keep the NotebookAE architecture fixed.

Lane B: mini-batch cosine-margin separation supplies weak retrieval supervision.
Test supervised contrastive objectives, train-only auxiliary heads or balanced training
sampling, with the same NotebookAE encoder/decoder. Identity labels may supervise
latent geometry only. They must never enter model inputs or ranking. Auxiliary heads
must be included in the optimizer when trainable and unnecessary during inference.

No new external data, pretrained weights or dependencies. Do not change protected
files/tests, the notebook, this brief, or `scripts/overnight.py`. Keep inference semantics,
Git gates, W&B privacy settings, checkpoint selection and canonical metrics intact.

## Required workflow

1. Inspect the leaderboard and round-1 report. Verify the benchmark and run the tests.
   Preflight at launch: 24 tests passed; baseline seed-42 score 0.121951, best observed
   seed-42 score 0.130081; the doubled-weight candidate's three-seed mean was 0.119241.
2. The first attempt in each lane is the same fresh seed-42 baseline at the overnight
   budget: notebook model, separation weights 1/0.05/0.05, AdamW lr 0.0002, weight decay
   0.0001, batch 128, 200 epochs, patience 40. This also checks concurrent operation.
3. Before each batch of at most four attempts, append one hypothesis, exact candidate
   commands and maximum runs/epochs to the journal. Commit and push the branch before
   training. Use only the guarded entry point:

   ```bash
   uv run --frozen python scripts/overnight.py train -- model=notebook loss=separation seed=42 train.epochs=200 train.patience=40 train.batch_size=128 device=cuda:0 wandb.name=overnight_LANE_baseline hydra.run.dir=outputs/overnight_2026-10-21/01_baseline
   ```

   The supervisor supplies `SEISSEPA_OVERNIGHT_CONTROL` and `SEISSEPA_OVERNIGHT_LANE`.
   Every attempt consumes a slot before training. Use unique run directories. Do not
   call training another way, run detached training, or spawn additional agents.
4. Online W&B destination/payload remains exactly as in AGENTS.md. Keep datasets,
   checkpoints, source archives, full configuration and artifact files local. For Codex,
   read only source/configuration, instructions, metric summaries and terminal output;
   never print/attach raw waveform arrays, credentials or checkpoint tensors.
5. After every attempt, journal the exact command, status, score, separate event/station
   scores, MSE, seed, epoch, elapsed time, benchmark ID, commit and W&B URL. Commit and
   push the journal before the next attempt. Preserve weak and failed results.
6. Before changing candidate source, evaluate a promising checkpoint with:

   ```bash
   uv run --frozen python scripts/overnight.py evaluate outputs/overnight_2026-10-21/RUN/best.pt --device cuda:0 --batch-size 128
   ```

   Use its actual training/evaluation batch size. Compare every scientific report field
   exactly. If source has changed, evaluate in a separate copy restored from that run's
   `source.tar.gz` inside the lane's ignored outputs, with the same read-only frozen
   cache. From the assigned worktree pass `--source-root RESTORED_DIR` to the guard's
   `evaluate` command; never loosen evaluator checks.
7. Finish each CLI batch with a clean, pushed branch and an updated
   `experiments/overnight_2026-10-21_LANE_report.md`. State consumed slots, valid best
   checkpoint, seed evidence, findings, next bounded hypothesis and any blockers.
   The supervisor may start a fresh CLI invocation; the total 64-slot budget persists.
8. On cutoff, create final lane reports, run required checks, commit and push, then stop.
   The finalizer aggregates both lane reports on main and records branch/path/commit
   for the best valid configuration and local checkpoint. Source remains recoverable
   on each pushed research branch; do not merge unverified candidate code into main.

## Supervision and inspection

`scripts/overnight.py` runs in the persistent tmux session
`seissepa-overnight-20261021`. Local control, CLI event logs, errors, prompts, slot
reservations and status are under `outputs/autoresearch/overnight_2026-10-21/`.
Worktrees and their run artifacts are inside that directory. The supervisor runs at
most 24 CLI invocations per lane, each with `gpt-6-luna` and reasoning effort `max`.
Three consecutive invocations with no new attempts stop a lane instead of looping.
The finalizer gets one bounded CLI invocation, no new training, and at most 30 minutes.

Inspect status with:

```bash
uv run --frozen python scripts/overnight.py status outputs/autoresearch/overnight_2026-10-21
tmux attach -t seissepa-overnight-20261021
```

Creating the `STOP` file in that control directory requests a graceful stop: no new
CLI batches or training are started; existing guarded runs finish within their limits.
Validation was repeatedly used for selection; report validation evidence, not an
untouched test result. Do not attach a confidence interval to the two-task mean.
