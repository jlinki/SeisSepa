"""Owner-only, one-time preparation. Never rerun or modify during autoresearch."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from seissepa.contract import PROTECTED, sha256, validate_manifest
from seissepa.data import ROOT, FROZEN, MANIFEST, LOCK


def greedy_rectangle(rows):
    """Fixed deterministic full-graph search, in both orientations; no optimality claim."""
    stations = sorted({r[1] for r in rows})
    events = sorted({r[2] for r in rows})
    si, ei = {s: i for i, s in enumerate(stations)}, {e: i for i, e in enumerate(events)}
    graph = np.zeros((len(stations), len(events)), dtype=bool)
    for _, s, e, _ in rows:
        graph[si[s], ei[e]] = True
    best = None
    for orientation, matrix, starts in (("station", graph, 32), ("event", graph.T, 8)):
        padded = np.pad(matrix, ((0, 0), (0, (-matrix.shape[1]) % 64)))
        bits = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little")).view(np.uint64)
        degrees = matrix.sum(1).astype(np.int64)
        ranked = np.lexsort((np.arange(len(matrix)), -degrees))
        for seed in ranked[:starts]:
            chosen = [int(seed)]
            used = np.zeros(len(matrix), dtype=bool)
            used[seed] = True
            common = bits[seed].copy()
            while not used.all():
                overlaps = np.bitwise_count(bits & common).sum(1).astype(np.int64)
                overlaps[used] = -1
                next_index = int(ranked[np.argmax(overlaps[ranked])])
                n_right = int(overlaps[next_index])
                if n_right < 5:
                    break
                chosen.append(next_index)
                used[next_index] = True
                common &= bits[next_index]
                area = len(chosen) * n_right
                if len(chosen) >= 5 and (best is None or area > best["area"]):
                    opposite = np.flatnonzero(np.unpackbits(common.view(np.uint8), bitorder="little")[:matrix.shape[1]]).tolist()
                    left, right = sorted(chosen), opposite
                    if orientation == "event":
                        left, right = right, left
                    best = {"status": "HEURISTIC_LARGEST_FOUND", "area": area,
                            "stations": left, "events": right,
                            "method": "deterministic greedy full-graph search in both orientations",
                            "station_starts": 32, "event_starts": 8}
    if best is None:
        raise RuntimeError("No rectangle found with at least five stations and five events")
    return [stations[i] for i in best["stations"]], [events[i] for i in best["events"]], best


def exact_rectangle(rows, directory, seconds):
    stations = sorted({r[1] for r in rows})
    events = sorted({r[2] for r in rows})
    station_index = {s: i for i, s in enumerate(stations)}
    event_index = {e: i for i, e in enumerate(events)}
    pairs = sorted({(station_index[s], event_index[e]) for _, s, e, _ in rows})
    graph = directory / "graph.txt"
    with graph.open("w") as stream:
        stream.write(f"{len(stations)} {len(events)} {len(pairs)}\n")
        stream.writelines(f"{s} {e}\n" for s, e in pairs)
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Data preparation needs a C++17 compiler (c++ or g++); training does not.")
    executable = directory / "rectangle-search"
    subprocess.run([compiler, "-O3", "-std=c++17", str(ROOT / "scripts/rectangle_search.cpp"),
                    "-o", str(executable)], check=True)
    with graph.open() as stream:
        result = subprocess.run([str(executable), str(seconds)], stdin=stream, stdout=subprocess.PIPE,
                                text=True, check=True, timeout=seconds)
    selection = json.loads(result.stdout)
    if selection["status"] != "OPTIMAL":
        raise RuntimeError("A heuristic result cannot be frozen as the largest rectangle")
    selection.pop("seconds", None)  # Runtime must not make the benchmark fingerprint vary.
    selection["method"] = "exact canonical closure enumeration with safe area upper bounds"
    return sorted(stations[i] for i in selection["stations"]), sorted(events[i] for i in selection["events"]), selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/Instance_events_maggt3.hdf5")
    parser.add_argument("--search-seconds", type=float, default=3600)
    parser.add_argument("--exact", action="store_true", help="Require a proven global maximum; potentially slow")
    args = parser.parse_args()
    if args.search_seconds <= 0:
        raise ValueError("--search-seconds must be positive")
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
            print(f"Searching full graph: {len(rows)} traces; mode={'exact' if args.exact else 'deterministic largest-found'}.", flush=True)
            stations, events, selection = (exact_rectangle(rows, temporary, args.search_seconds) if args.exact
                                            else greedy_rectangle(rows))
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
                raise ValueError("Selector returned an inconsistent rectangle")
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
                          "min_stations": 5, "min_events": 5, "max_aspect_ratio": None,
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
