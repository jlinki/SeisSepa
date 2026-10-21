# First autoresearch round

Owner request: use Codex CLI with GPT-6 Luna at maximum reasoning effort, GPU 0,
the current notebook model, and changes focused on training parameters and losses.
Read AGENTS.md and README.md. Their protected benchmark and evaluation contract applies.

## Hypothesis and limits

Hypothesis: stronger and better balanced event/station latent supervision, relative to
reconstruction, improves validation retrieval with the existing architecture.

- Model: `model=notebook`. Keep its architecture and learned operations fixed. The
  non-learning `encode()` inference path may be used to skip the decoder when only
  gallery latents are needed; it must return the exact same latents as `forward()`.
- GPU: physical GPU 0 only. Set `CUDA_VISIBLE_DEVICES=0` and `device=cuda:0` for
  training and standalone evaluation. Never use GPUs 1, 2 or 3 or disturb other jobs.
- One researcher writes code/Git and runs training sequentially. A read-only advisor
  may recommend candidates; it must not edit files or start training.
- Maximum **7 training runs**, each at most **50 epochs**, including failed attempts
  and repeats. Keep `train.patience=12`. Stop after **2 hours** from researcher launch;
  do not start a run when insufficient time remains. Do not silently extend any limit.
- Run 1: unchanged default notebook baseline, seed 42, 50 epochs, online W&B.
- Runs 2–4: at most three candidates, seed 42, same maximum epoch budget. Prioritize
  loss weights, margin/temperature, learning rate, weight decay and batch size.
  A simple alternative latent loss may be implemented in `src/seissepa/losses.py`
  with a new loss config if the hypothesis warrants it. Do not add dependencies.
- Reserve runs 5–7 for the best configuration: an exact seed-42 repeat, then seeds
  43 and 44 if promising. Use the same architecture and training budget. The original
  winning seed-42 run plus 43/44 supplies the three-seed mean/std. Report every result.
- No promising candidate: repeat the best seed-42 run if useful, report the outcome,
  and leave the remaining budget unused. Do not turn setup smoke checks into baselines.

## Required execution

1. Verify benchmark, run existing tests, inspect leaderboard and reports. Record the
   hypothesis and budget before each candidate batch. Never edit protected files/tests.
2. Commit and push every source/configuration change before training. Keep Git clean.
   Local run logs and artifacts belong under ignored `outputs/`. Journal/report writes
   must also be committed and pushed before the next training launch. Never force-push.
3. Use explicit unique run directories under `outputs/round1/` and descriptive W&B
   names. Online project/payload approval in AGENTS.md persists. Never upload source,
   datasets, checkpoints, complete configuration files or artifacts to W&B.
   For the Codex service, restrict context to repository instructions, source/configuration,
   experiment summaries and terminal diagnostics under the owner's separate approval.
   Never print or attach raw datasets, waveform arrays, credentials or checkpoints to
   an agent prompt. Use source scripts instead of opening the reference notebook.
4. Preserve each run's command, checkpoint, report, provenance, online URL and errors.
   Do not change evaluator code to accept older checkpoints. Re-evaluate candidates
   before editing their source; otherwise restore their saved source archive into a
   separate temporary copy and link the existing frozen cache there without modifying it.
5. Run the protected standalone evaluator on the best checkpoint before promotion.
   Compare all scientific metrics in the exact repeat, ignoring elapsed time and run
   identity. If exact repetition fails, investigate and report it rather than claiming
   reproducibility. Check the finished online run through W&B when possible.
6. Append completed/failed attempts to `experiments/journal.jsonl`. Write
   `experiments/round1_report.md` with all runs, baseline-relative deltas, event/station
   retrieval separately, canonical MSE, seed, best epoch, elapsed time, benchmark ID,
   Git commit, output path, W&B URL, repeatability, seed mean/std and next hypothesis.
   Keep the best configuration in a named Hydra config or give its exact overrides.
   Preserve unsuccessful candidates. Report validation tuning as validation evidence.
7. Run required final checks, commit and push the journal/report and code/configuration.
   Finish with a clean working tree and a concise result. If blocked by authentication,
   approvals, integrity checks or infrastructure, preserve logs and report the blocker;
   do not bypass it, alter the lock, or switch W&B offline.

## Performance correction before research resumes

The first baseline was stopped by the owner after 8 epochs because epoch evaluation
took about 160 seconds. A one-batch profile measured the complete reconstruction
diagnostics at 22.96 seconds, while warmed inference was about 0.01 seconds. Epoch
selection now extracts training-gallery latents without decoding, and computes only
the exact canonical validation MSE when an epoch beats or ties the best retrieval score.
The protected full evaluator still computes all diagnostics once for the selected checkpoint. Final reconstruction,
retrieval, centroid, k-NN, probe and silhouette values are also logged as named W&B
scalars once at run completion and remain in the local Markdown/JSON report.

The stopped baseline and a failed detached retry consume two of seven run slots. No
valid baseline has completed. Resume with a uniquely named seed-42 baseline only after
the speed fix is committed, pushed, tested and benchmark-verified.

The launcher stores CLI events, stderr, session identity, process status and final agent
messages under `outputs/autoresearch/round1/`. This brief establishes the experiment
budget; it does not claim that any research run has completed.
