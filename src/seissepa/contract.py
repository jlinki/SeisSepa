"""Protected benchmark integrity; changing a frozen input/evaluator fails closed."""
import hashlib
import json

from seissepa.data import ROOT, MANIFEST, LOCK

PROTECTED = ["src/seissepa/data.py", "src/seissepa/metrics.py", "src/seissepa/evaluate.py",
             "src/seissepa/contract.py", "scripts/prepare_data.py", "scripts/evaluate.py",
             "scripts/verify_benchmark.py", "scripts/leaderboard.py",
             "AGENTS.md", "tests/test_benchmark.py"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified():
    if not LOCK.is_file():
        raise RuntimeError("Benchmark is not prepared. Run scripts/prepare_data.py once before research.")
    lock = json.loads(LOCK.read_text())
    for relative, expected in lock["files"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen benchmark changed: {relative}. Do not regenerate the lock during research.")
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    return manifest, lock


def validate_manifest(manifest):
    if len(manifest["stations"]) != len(manifest["events"]):
        raise ValueError("Frozen benchmark must be square: equal station and event counts")
    records = manifest["records"]
    pairs = {(r["station"], r["event"]) for r in records}
    expected = {(s, e) for s in manifest["stations"] for e in manifest["events"]}
    if pairs != expected or len(records) != len(expected) or len({r["trace"] for r in records}) != len(records):
        raise ValueError("Manifest must contain exactly one distinct trace per station/event cell")
    train, val = manifest["split"]["train"], manifest["split"]["validation"]
    if not train or not val or sorted(train + val) != list(range(len(records))):
        raise ValueError("Train/validation must be disjoint and partition the square")
    for key, values in (("event", manifest["events"]), ("station", manifest["stations"])):
        if {records[i][key] for i in train} != set(values):
            raise ValueError("Every validation class needs a training reference")
