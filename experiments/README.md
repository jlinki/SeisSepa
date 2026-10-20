# Research journal

Append one JSON object per completed or failed experiment to `journal.jsonl`:

```json
{"hypothesis":"Increasing event separation improves event retrieval","command":"uv run --frozen python scripts/train.py loss.event_weight=0.1","budget":{"max_runs":1,"max_epochs":50},"run":"outputs/DATE/RUN","benchmark_id":"FROM_METRICS","seed":42,"status":"completed","score":0.0,"event_accuracy":0.0,"station_accuracy":0.0,"decision":"keep or discard, with evidence"}
```

Replace every example value with the observed result. Include `git_commit` from `provenance.json` and `wandb_url` from `wandb.json`. Commit and push source/configuration changes before training, then commit and push journal entries after the experiment batch. For failures, set `status` to `failed`, use null scores, and record the exception and log path. A journal entry does not replace the saved run artifacts. Establish the default seed-42 baseline before tuning.
