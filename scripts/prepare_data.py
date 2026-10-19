"""Owner-only, one-time preparation. Never rerun or modify during autoresearch."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from seissepa.contract import PROTECTED, sha256, validate_manifest
from seissepa.data import ROOT, FROZEN, MANIFEST, LOCK


def greedy_square(rows):
    """Largest square found by a fixed full-graph greedy search in both orientations."""
    stations = sorted({r[1] for r in rows})
    events = sorted({r[2] for r in rows})
    si, ei = {s: i for i, s in enumerate(stations)}, {e: i for i, e in enumerate(events)}
    graph = np.zeros((len(stations), len(events)), dtype=bool)
    for _, s, e, _ in rows:
        graph[si[s], ei[e]] = True
    best = None
    for orientation, matrix in (("station", graph), ("event", graph.T)):
        padded = np.pad(matrix, ((0, 0), (0, (-matrix.shape[1]) % 64)))
        bits = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little")).view(np.uint64)
        degrees = matrix.sum(1).astype(np.int64)
        ranked = np.lexsort((np.arange(len(matrix)), -degrees))
        for seed in ranked[:512]:
            if best is not None and degrees[seed] <= best["side"]:
                continue
            chosen = [int(seed)]
            used = np.zeros(len(matrix), dtype=bool)
            used[seed] = True
            common = bits[seed].copy()
            while not used.all():
                overlaps = np.bitwise_count(bits & common).sum(1).astype(np.int64)
                overlaps[used] = -1
                next_index = int(ranked[np.argmax(overlaps[ranked])])
                n_right = int(overlaps[next_index])
                if n_right < 5 or (best is not None and n_right <= best["side"]):
                    break
                chosen.append(next_index)
                used[next_index] = True
                common &= bits[next_index]
                side = min(len(chosen), n_right)
                if side >= 5 and (best is None or side > best["side"]):
                    opposite = np.flatnonzero(np.unpackbits(common.view(np.uint8), bitorder="little")[:matrix.shape[1]]).tolist()[:side]
                    left, right = sorted(chosen)[:side], opposite
                    if orientation == "event":
                        left, right = right, left
                    best = {"status": "HEURISTIC_LARGEST_SQUARE_FOUND", "geometry": "square",
                            "side": side, "area": side * side,
                            "stations": left, "events": right,
                            "method": "deterministic greedy square search in both orientations; crop larger side",
                            "station_starts": 512, "event_starts": 512}
                if n_right <= len(chosen):
                    break
    if best is None:
        raise RuntimeError("No complete square found with at least five stations and five events")
    return [stations[i] for i in best["stations"]], [events[i] for i in best["events"]], best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/Instance_events_maggt3.hdf5")
    args = parser.parse_args()
    if LOCK.exists() or MANIFEST.exists() or FROZEN.exists():
        raise RuntimeError("Frozen benchmark already exists. Preparation never overwrites or resplits it.")
    source = args.source.resolve()
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=ROOT / "data") as temporary:
        temporary = Path(temporary)
        with h5py.File(source, "r") as h5:
            columns = h5["meta/columns"][()].astype(str).tolist()
            positions = [columns.index(c) for c in ("trace_name", "station_id", "event_id", "sampling_rate")]
            rows = sorted(h5["meta/events"][()].astype(str)[:, positions].tolist())
            if len({r[0] for r in rows}) != len(rows) or len(rows) != len(h5["data"]):
                raise ValueError("Metadata must describe every source trace exactly once")
            if {r[0] for r in rows} != set(h5["data"].keys()):
                raise ValueError("Source waveform keys and metadata disagree")
            print(f"Searching full graph: {len(rows)} traces; mode=deterministic largest-found square.", flush=True)
            stations, events, selection = greedy_square(rows)
            print(f"Selected {len(stations)} × {len(events)} = {selection['area']}; {selection['status']}. Hashing source and freezing cache.", flush=True)
            pair_rows = defaultdict(list)
            for trace, station, event, sampling_rate in rows:
                pair_rows[station, event].append((trace, float(sampling_rate)))
            # Stable cell order; select the lexicographically first trace for duplicate cells.
            records = []
            for station in stations:
                for event in events:
                    trace, sampling_rate = min(pair_rows[station, event])
                    records.append({"trace": trace, "station": station, "event": event, "sampling_rate": sampling_rate})
            if len(records) != selection["area"]:
                raise ValueError("Selector returned an inconsistent square")
            shape = h5["data"][records[0]["trace"]].shape
            if len(shape) != 2 or shape[0] != 3:
                raise ValueError("Expected three-channel [C, time] waveforms")
            cache_dir = temporary / "frozen"
            cache_dir.mkdir()
            cache = np.lib.format.open_memmap(cache_dir / "waveforms.npy", mode="w+", dtype=np.float32,
                                             shape=(len(records), *shape))
            for index, record in enumerate(records):
                waveform = h5["data"][record["trace"]][()]
                if waveform.shape != shape or not np.isfinite(record["sampling_rate"]) or record["sampling_rate"] <= 0:
                    raise ValueError("Inconsistent source waveform shape or sampling rate")
                cache[index] = np.nan_to_num(waveform.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            cache.flush()
            del cache
        permutation = np.random.default_rng(42).permutation(len(records))
        cut = int(0.9 * len(records))
        # Sorted within each split gives a canonical gallery tie order, independent of batching.
        manifest = {
            "version": 1,
            "source": {"path": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                       "bytes": source.stat().st_size, "sha256": sha256(source), "trace_count": len(rows),
                       "metadata_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()},
            "selection": {**selection,
                          "min_stations": 5, "min_events": 5, "max_aspect_ratio": 1.0,
                          "duplicates": "lexicographically first trace per station/event cell"},
            "stations": stations, "events": events, "records": records,
            "waveform_shape": list(shape), "reconstruction_length": 8192,
            "split": {"seed": 42, "train_fraction": 0.9,
                      "train": sorted(permutation[:cut].tolist()), "validation": sorted(permutation[cut:].tolist())},
        }
        validate_manifest(manifest)
        MANIFEST.parent.mkdir(exist_ok=True)
        manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        benchmark_id = hashlib.sha256(manifest_bytes).hexdigest()
        # The lock is created last; interrupted preparation cannot yield a valid benchmark.
        os.rename(cache_dir, FROZEN)
        MANIFEST.write_bytes(manifest_bytes)
        protected = [*PROTECTED, "benchmark/manifest.json", "data/frozen/waveforms.npy"]
        lock = {"benchmark_id": benchmark_id, "files": {p: sha256(ROOT / p) for p in protected}}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"Frozen: {len(stations)} stations × {len(events)} events = {len(records)} traces; "
              f"train={cut}, validation={len(records) - cut}; benchmark_id={benchmark_id}", flush=True)


if __name__ == "__main__":
    main()
