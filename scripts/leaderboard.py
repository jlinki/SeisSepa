"""Compare completed runs of this frozen benchmark."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from seissepa.contract import read_verified

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=Path, default=Path("outputs"))
args = parser.parse_args()
_, lock = read_verified()
records = []
for path in args.directory.rglob("metrics.json"):
    record = json.loads(path.read_text())
    if record["benchmark_id"] == lock["benchmark_id"]:
        records.append((record["objective"]["val/retrieval_accuracy"],
                        record["reconstruction"]["validation"]["mse"], path, record))
print("retrieval  event    station  val_MSE   seed  run")
for score, mse, path, record in sorted(records, key=lambda row: (-row[0], row[1], str(row[2]))):
    objective = record["objective"]
    print(f"{score:.6f}  {objective['val/event/retrieval_top1']:.4f}   "
          f"{objective['val/station/retrieval_top1']:.4f}   {mse:.5f}  {record['seed']:4d}  {path.parent}")
