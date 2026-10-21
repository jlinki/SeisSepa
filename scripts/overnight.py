"""Bounded local supervision for the explicitly authorized overnight CLI research."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BRIEF = "experiments/overnight_2026-10-21.md"


def now():
    return dt.datetime.now(dt.timezone.utc).timestamp()


def stamp(value=None):
    return dt.datetime.fromtimestamp(now() if value is None else value, dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def deadline(plan, name):
    return dt.datetime.fromisoformat(plan[name]).timestamp()


def load_control(path):
    control = Path(path).resolve()
    return control, json.loads((control / "plan.json").read_text())


def slots(control, lane):
    return sorted((control / "lanes" / lane / "slots").glob("[0-9][0-9][0-9].json"))


def descendants(pid):
    parents = {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = path.read_text().splitlines()
            parent = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
            parents.setdefault(parent, []).append(int(path.parent.name))
        except (OSError, ValueError, StopIteration):
            continue
    result, pending = [], [pid]
    while pending:
        parent = pending.pop()
        children = parents.get(parent, [])
        result.extend(children)
        pending.extend(children)
    return result


def stop_process(process):
    """Stop only this invocation and its own descendants, never other GPU jobs."""
    children = descendants(process.pid)
    for pid in reversed(children + [process.pid]):
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=40)
    except subprocess.TimeoutExpired:
        pass
    # A tool subprocess can outlive the CLI process; retain the original child list.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in children + [process.pid]:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def lane_environment(worktree, control, lane):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", SEISSEPA_OVERNIGHT_CONTROL=str(control),
               SEISSEPA_OVERNIGHT_LANE=lane, PYTHONPATH=str(worktree / "src"), UV_NO_SYNC="true")
    return env


def guarded(command, args):
    control, plan = load_control(os.environ["SEISSEPA_OVERNIGHT_CONTROL"])
    lane = os.environ["SEISSEPA_OVERNIGHT_LANE"]
    expected_root = Path(plan["lanes"][lane]["worktree"]).resolve()
    if ROOT != expected_root:
        raise RuntimeError("Guard must run from its assigned isolated worktree")
    lane_dir = control / "lanes" / lane
    check = "--check" in args
    args = [value for value in args if value not in {"--", "--check"}]
    with (lane_dir / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if command == "train":
            if (control / "STOP").exists() or now() >= deadline(plan, "training_cutoff"):
                raise RuntimeError("No new training: overnight stop/cutoff reached")
            if len(slots(control, lane)) >= plan["max_runs_per_lane"]:
                raise RuntimeError("Overnight training-attempt budget exhausted")
            if any("=" not in value or value.startswith("-") for value in args):
                raise ValueError("One training run per attempt; CLI flags/multiruns are disallowed")
            overrides = dict(value.split("=", 1) for value in args)
            if any(value.startswith(("+", "~")) for value in overrides):
                raise ValueError("Use ordinary, explicit Hydra overrides")
            if any(key.startswith("model.") or (key.startswith("hydra.") and key != "hydra.run.dir")
                   for key in overrides):
                raise ValueError("Keep the current architecture and single-run Hydra mode")
            epochs = int(overrides.get("train.epochs", "200"))
            if not 1 <= epochs <= plan["max_epochs"]:
                raise ValueError("Epoch budget exceeded")
            if overrides.get("model", "notebook") != "notebook":
                raise ValueError("This round keeps model=notebook")
            if int(overrides.get("train.patience", "40")) != 40:
                raise ValueError("Overnight patience is fixed at 40")
            if int(overrides.get("seed", "42")) not in {42, 43, 44}:
                raise ValueError("Only predeclared seeds 42, 43 and 44")
            relative = Path(overrides["hydra.run.dir"])
            output = (ROOT / relative).resolve()
            allowed = ROOT / "outputs" / "overnight_2026-10-21"
            if relative.is_absolute() or allowed not in output.parents or output.exists():
                raise ValueError("Choose an unused run directory inside the lane's overnight outputs")
            enforced = {"model": "notebook", "train.epochs": str(epochs), "train.patience": "40",
                        "seed": overrides.get("seed", "42"), "device": "cuda:0",
                        "wandb.mode": "online", "threads": "2"}
            args = [value for value in args if value.split("=", 1)[0] not in enforced]
            args += [f"{key}={value}" for key, value in enforced.items()]
            if check:
                print(json.dumps({"validated": True, "lane": lane, "overrides": args}))
                return
            index = len(slots(control, lane)) + 1
            record = {"slot": index, "lane": lane, "started_at": stamp(), "command": args,
                      "output": str(output), "status": "reserved"}
            slot = lane_dir / "slots" / f"{index:03}.json"
            write_json(slot, record)
            output.mkdir(parents=True)
            log = output / "console.log"
            subprocess.run([sys.executable, "scripts/verify_benchmark.py"], cwd=ROOT, check=True)
            child_command = [sys.executable, "scripts/train.py", *args]
            limit = min(plan["run_timeout_seconds"], deadline(plan, "worker_cutoff") - now() - 120)
        else:
            if now() >= deadline(plan, "worker_cutoff"):
                raise RuntimeError("Worker evaluation cutoff reached")
            if not args or args[0].startswith("-"):
                raise ValueError("Provide a candidate checkpoint")
            if args.count("--device") != 1 or args[args.index("--device") + 1] != "cuda:0":
                raise ValueError("Evaluation must use cuda:0")
            log = lane_dir / f"evaluation-{time.time_ns()}.log"
            child_command = [sys.executable, "scripts/evaluate.py", *args]
            limit = min(plan["run_timeout_seconds"], deadline(plan, "worker_cutoff") - now() - 30)
            record = slot = None
        print("Guard starting:", " ".join(child_command), flush=True)
        process = None
        started = time.monotonic()
        status = "failed"
        returncode = 1
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(child_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           env=lane_environment(ROOT, control, lane), start_new_session=True)
                try:
                    returncode = process.wait(timeout=max(1, limit))
                    status = "completed" if returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    stop_process(process)
                    status, returncode = "timeout", 124
        finally:
            if process is not None and process.poll() is None:
                stop_process(process)
            if record is not None:
                record.update(status=status, returncode=returncode, finished_at=stamp(),
                              elapsed_seconds=time.monotonic() - started, log=str(log))
                write_json(slot, record)
            if log.exists():
                print("\n".join(log.read_text(errors="replace").splitlines()[-35:]), flush=True)
        raise SystemExit(returncode)


def prompt_for(plan, lane, final=False):
    info = plan["lanes"][lane]
    return f"""Execute the owner's authorized overnight SeisSepa research with Codex CLI.
Read AGENTS.md, README.md, {BRIEF}, experiments/round1_report.md and your current
experiments/overnight_2026-10-21_{lane}_report.md (if present). They define the fixed
benchmark, privacy approvals, source limits, exact score, and all budgets.
You are lane {lane}: {info['hypothesis']}. Work only in {info['worktree']} on
branch {info['branch']}; do not edit another worktree or the main checkout.
This is a NEW round, so the old round-1 deadline no longer controls it. The overnight
training cutoff is {plan['training_cutoff']}; worker reporting cutoff is
{plan['worker_cutoff']}. Your cumulative cap is {plan['max_runs_per_lane']} attempts,
200 epochs each, patience 40, 20 minutes per attempt, including failures and repeats.
Count already reserved attempts in $SEISSEPA_OVERNIGHT_CONTROL/lanes/{lane}/slots.
Use ONLY `uv run --frozen python scripts/overnight.py train -- ...` for training,
and `scripts/overnight.py evaluate ... --device cuda:0 --batch-size ACTUAL` for GPU
evaluation. This enforces one job per lane. GPU 0 only. Do not bypass/edit the guard,
detach training, spawn agents, prepare data, or change the contract/evaluator/tests.
Keep NotebookAE architecture fixed; losses/parameters/sampling/schedules may change.
No dependencies, external data or pretrained weights. IDs supervise training latents
only; model inputs remain waveform/sampling_rate. Never use validation for gradients.
Keep W&B online jlinki/seissepa-autoresearch with the existing approved scalar-only
payload/privacy. Keep code, data, checkpoints/full config/artifacts local. Never print
raw data, tensors, secrets or checkpoint contents into this CLI conversation.
First ever lane attempt: run the fresh seed-42 default separation baseline at 200
epochs/patience40/batch128/lr0.0002, with a unique lane W&B name/output. Run existing
tests and benchmark verification before research. Existing best seed-42 score .130081;
previous baseline .121951, doubled-weight three-seed mean .119241. Those were 50/12
budget results; use your fresh baseline for fair overnight-budget comparisons.
Before each batch: one hypothesis, at most FOUR attempts in THIS invocation, exact
commands and budget journaled, committed and pushed. Commit/push all source/config
changes BEFORE training. Keep the working tree clean and HEAD equal to upstream.
After EACH attempt, record command, status, all score components, MSE, seed, epoch,
elapsed time, fingerprint, commit, W&B URL, and decision in experiments/journal.jsonl;
commit/push before the next launch. Preserve failed/poor runs. Stop the lane for auth,
integrity, upstream push or online logging failure; write an explicit BLOCKED file
under $SEISSEPA_OVERNIGHT_CONTROL/lanes/{lane} with the actual reason and stop.
Re-evaluate promising checkpoints BEFORE editing their source; compare all scientific
fields exactly. For promising candidates, run an exact seed42 repeat plus seeds43/44
with unchanged config/budget; report all and mean/sample SD. Reserve >=12 slots for
confirmations. Use no more than 48 seed42 screens. Prefer small, explained batches.
Finish the invocation after at most four attempts with a pushed clean tree and a clear
lane report containing consumed attempts, best valid config/checkpoint, three-seed
evidence, all runs, W&B URLs and next hypothesis. A supervisor can start another fresh
invocation within the same cumulative limits. Do not wait/sleep to consume the night.
If a STOP file exists in the control directory, make no new attempts, finish reporting.
{'FINALIZATION ONLY: no new training. Finish evaluator checks if time permits, report every result, commit/push, and stop.' if final else 'Proceed now with the next bounded batch, then stop this invocation.'}
"""


def launch_cli(worktree, prompt, output, env):
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt)
    command = ["codex", "exec", "--approve-for-me", "--add-dir", str(ROOT), "-m", "gpt-6-luna", "-c",
               'model_reasoning_effort="max"', "--json", "-o", str(output / "final.md"), "-"]
    with (output / "prompt.txt").open() as source, (output / "events.jsonl").open("w") as events, \
            (output / "stderr.log").open("w") as errors:
        process = subprocess.Popen(command, cwd=worktree, env=env, stdin=source, stdout=events,
                                   stderr=errors, start_new_session=True)
    write_json(output / "launch.json", {"pid": process.pid, "command": command, "started_at": stamp()})
    return process


def supervise(path):
    control, plan = load_control(path)
    lock = (control / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, state = {}, {"started_at": stamp(), "pid": os.getpid(), "lanes": {}}
    try:
        for lane in plan["lanes"]:
            state["lanes"][lane] = {"invocations": 0, "no_progress": 0, "done": False}
        while any(not item["done"] for item in state["lanes"].values()):
            for lane, item in state["lanes"].items():
                lane_dir = control / "lanes" / lane
                item["attempts"] = len(slots(control, lane))
                if lane in active:
                    process, launched, before, final, output = active[lane]
                    if process.poll() is None:
                        if now() >= deadline(plan, "worker_cutoff") or now() - launched > 5400:
                            stop_process(process)
                            item["stop_reason"] = "worker deadline or 90-minute CLI limit"
                        else:
                            continue
                    item["last_exit_code"] = process.poll()
                    write_json(output / "completion.json", {"finished_at": stamp(), "returncode": process.poll()})
                    item["no_progress"] = item["no_progress"] + 1 if item["attempts"] == before else 0
                    del active[lane]
                    if final or (lane_dir / "BLOCKED").exists() or item["no_progress"] >= 3:
                        item["done"] = True
                        continue
                if item["done"]:
                    continue
                if now() >= deadline(plan, "worker_cutoff") or item["invocations"] >= 24:
                    item["done"] = True
                    continue
                final = ((control / "STOP").exists() or (lane_dir / "BLOCKED").exists()
                         or item["attempts"] >= plan["max_runs_per_lane"]
                         or now() >= deadline(plan, "training_cutoff") - 1800)
                if (lane_dir / "BLOCKED").exists():
                    item.update(done=True, stop_reason=(lane_dir / "BLOCKED").read_text())
                    continue
                item["invocations"] += 1
                output = lane_dir / f"cli-{item['invocations']:02}"
                worktree = Path(plan["lanes"][lane]["worktree"])
                process = launch_cli(worktree, prompt_for(plan, lane, final), output,
                                     lane_environment(worktree, control, lane))
                active[lane] = process, now(), item["attempts"], final, output
                item.update(pid=process.pid, cli_path=str(output), finalizing=final)
                print(stamp(), lane, "launched", item["invocations"], "final" if final else "research", flush=True)
            state["updated_at"] = stamp()
            write_json(control / "status.json", state)
            if active:
                time.sleep(15)
        final_prompt = f"""Finalize the owner's overnight research; no new training or researcher agents.
Read AGENTS.md, README.md, {BRIEF}, and {control}/status.json and plan.json.
Both researcher lanes are stopped. Inspect their committed lane reports and journals,
local metrics/re-evaluations and W&B summaries using the existing approved scalar-only
payload. Worktrees and branches are listed in plan.json. Do not change protected files,
dataset/evaluator, or copy unverified candidate source into main. Keep best source/config
on its pushed research branch and preserve local checkpoints. Do not delete any result.
Write experiments/overnight_2026-10-21_report.md on main with all attempts/failures,
seed42 comparisons, event/station scores, reconstruction MSE, seed/epoch/time, benchmark
fingerprint, exact config/commit/branch/run path/W&B links, seed42/43/44 mean/sample SD,
repeatability and limitations. Distinguish unconfirmed screens from verified candidates.
If necessary record final lane outcomes/references in experiments/journal.jsonl without
duplicating existing entries or fabricating results. Inspect tests/benchmark status and
commit/push the final report/journal on main, leave main clean and equal to upstream.
You have at most 30 minutes and must stop at {plan['final_cutoff']}. No extensions.
Never print raw data, credentials or checkpoint tensors. Report actual blockers.
"""
        remaining = min(1800, deadline(plan, "final_cutoff") - now())
        if remaining > 60:
            final_dir = control / "finalizer"
            process = launch_cli(ROOT, final_prompt, final_dir, lane_environment(ROOT, control, "finalizer"))
            state["finalizer_pid"] = process.pid
            write_json(control / "status.json", state)
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop_process(process)
            state["finalizer_exit_code"] = process.poll()
        state["finished_at"] = stamp()
        write_json(control / "status.json", state)
        print(stamp(), "supervision finished", flush=True)
    finally:
        for process, *_ in active.values():
            if process.poll() is None:
                stop_process(process)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: overnight.py train -- HYDRA_OVERRIDES | evaluate CHECKPOINT OPTIONS | supervise CONTROL | status CONTROL")
    command, args = sys.argv[1], sys.argv[2:]
    if command in {"train", "evaluate"}:
        guarded(command, args)
    elif command == "supervise":
        supervise(args[0])
    elif command == "status":
        control, plan = load_control(args[0])
        print((control / "status.json").read_text() if (control / "status.json").exists() else "Not started")
        for lane in plan["lanes"]:
            print(lane, len(slots(control, lane)), "training attempts")
    else:
        raise SystemExit("Unknown overnight command")


if __name__ == "__main__":
    main()
