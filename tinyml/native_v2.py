"""Replay asynchronous packet streams and check every repair against a dense oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, ROOT, SEEDS
from .evaluate import SCORE_POINTER, Update, native_library
from .folds import CLIP_DIR, FOLDS, model_dir, read_fixture
from .protocol import TINY_EXPIRED, TINY_OK, sensor_input
from .protocol_v2 import METHODS
from .quant import Int8Model, dense_inference
from .study_v2 import HORIZON, V2

MODES = ("tap_equal", "recent_replay", "accumulator", "no_equal")
STREAM_SEED = 77237
WARMUP = 48


@dataclass(frozen=True)
class Workload:
    name: str
    period: int
    max_delay: int
    burst: int = 1
    expired: bool = False


WORKLOADS = (
    Workload("isolated_5pct", 20, 7),
    Workload("isolated_20pct", 5, 7),
    Workload("mixed_10pct", 10, 48),
    Workload("bursts_20pct", 20, 32, 4),
    Workload("expiry_stress", 10, 59, 1, True),
)


def traffic(length, workload, seed):
    rng = np.random.default_rng(seed)
    times = np.arange(length, dtype=int)
    for start in range(WARMUP, length - workload.max_delay - workload.burst, workload.period):
        # Avoid coincident ownership across contiguous packet blocks.
        offset = int(rng.integers(workload.period - workload.burst + 1))
        first = start + offset
        if first + workload.max_delay >= length:
            break
        delay = int(rng.integers(workload.burst, workload.max_delay + 1))
        if workload.expired and (start // workload.period) % 4 == 0:
            delay = min(workload.max_delay, HORIZON + 3)
        times[first:first + workload.burst] = first + delay
    return times


def _load_api(model, path):
    lib = native_library(model, HORIZON, path, extended=True)
    lib.tiny_create.restype = ct.c_void_p
    lib.tiny_destroy.argtypes = [ct.c_void_p]
    lib.tiny_destroy.restype = None
    lib.tiny_append.argtypes = [ct.c_void_p, SCORE_POINTER]
    lib.tiny_append.restype = None
    lib.tiny_repair.argtypes = [ct.c_void_p, ct.POINTER(Update), ct.c_size_t, ct.c_int]
    lib.tiny_repair.restype = ct.c_int
    lib.tiny_scores.argtypes = [ct.c_void_p, ct.c_uint8]
    lib.tiny_scores.restype = SCORE_POINTER
    return lib


def measure_trace(model, clean, arrivals, lib, modes=MODES, label=None):
    revisions = {}
    for arrival, emitted in groupby(sorted(np.flatnonzero(arrivals > np.arange(len(clean))),
                                         key=lambda i: arrivals[i]), key=lambda i: int(arrivals[i])):
        revisions[arrival] = tuple(map(int, emitted))
    provisional = np.empty_like(clean)
    verified = np.empty_like(clean)
    states = [lib.tiny_create() for _ in modes]
    if not all(states):
        for state in states:
            if state:
                lib.tiny_destroy(state)
        raise MemoryError("Could not allocate firmware state")
    before_changes = 0
    label_changes = 0
    score_checks = 0
    expected_expired = 0
    expected_ok = 0
    submitted_samples = 0
    latest_matches = 0
    last_actual_tick = -1
    provisional_correct = 0
    corrected_correct = 0
    last_actual = np.zeros(3, dtype=np.int8)
    base = dense_inference(model, clean)
    try:
        for tick, actual in enumerate(clean):
            if arrivals[tick] == tick:
                last_actual = actual
                last_actual_tick = tick
            provisional[tick] = last_actual
            verified[tick] = provisional[tick]
            arg = provisional[tick].ctypes.data_as(SCORE_POINTER)
            for state in states:
                lib.tiny_append(state, arg)
            updates = revisions.get(tick)
            if updates:
                expected = dense_inference(model, verified[:tick + 1])
                start = max(0, tick - HORIZON)
                for state in states:
                    keep = tick - start + 1
                    cached = [np.ctypeslib.as_array(lib.tiny_scores(state, age), shape=(model.classes,))
                              for age in range(keep)]
                    if not np.array_equal(cached, expected[start:tick + 1][::-1]):
                        raise AssertionError(f"Provisional stream mismatch at {tick=}")
                    score_checks += keep * model.classes
                update_age = [tick - i for i in updates]
                submit = [Update(age, (ct.c_int8 * 3)(*map(int, clean[i])))
                          for i, age in zip(updates, update_age, strict=True)]
                vector = (Update * len(submit))(*submit)
                expired = max(update_age) > HORIZON
                expected_expired += expired
                expected_ok += not expired
                submitted_samples += len(submit)
                if not expired:
                    verified[list(updates)] = clean[list(updates)]
                    corrected = dense_inference(model, verified[:tick + 1])
                    before_changes += int(np.any(expected[start:] != corrected[start:], axis=1).sum())
                    label_changes += int(np.count_nonzero(
                        expected[start:].argmax(-1) != corrected[start:].argmax(-1)
                    ))
                else:
                    corrected = expected
                if label is not None:
                    provisional_correct += int(expected[tick].argmax() == label)
                    corrected_correct += int(corrected[tick].argmax() == label)
                for state, mode in zip(states, modes, strict=True):
                    status = lib.tiny_repair(state, vector, len(vector), METHODS[mode])
                    if status != (TINY_EXPIRED if expired else TINY_OK):
                        raise AssertionError(f"Unexpected {mode} status at {tick=} {update_age=}")
                    keep = tick - start + 1
                    cached = [np.ctypeslib.as_array(lib.tiny_scores(state, age), shape=(model.classes,))
                              for age in range(keep)]
                    if not np.array_equal(cached, corrected[start:tick + 1][::-1]):
                        raise AssertionError(f"Repair mismatch in {mode} at {tick=}")
                    score_checks += keep * model.classes
                latest_matches += bool(np.array_equal(base[tick], corrected[tick]))
                newest = max(updates)
                if newest > last_actual_tick:
                    last_actual = clean[newest]
                    last_actual_tick = newest
        final = dense_inference(model, verified)
        for state in states:
            for age in range(min(HORIZON, len(clean) - 1) + 1):
                got = np.ctypeslib.as_array(lib.tiny_scores(state, age), shape=(model.classes,))
                if not np.array_equal(got, final[-age - 1]):
                    raise AssertionError("Final stream mismatch")
                score_checks += len(got)
    finally:
        for state in states:
            lib.tiny_destroy(state)
    return {
        "ticks": len(clean), "delivered_samples": submitted_samples, "accepted_transactions": expected_ok,
        "rejected_transactions": expected_expired, "score_changed_timestamps": before_changes,
        "label_changed_timestamps": label_changes, "latest_matches_clean": latest_matches,
        "provisional_correct_at_arrival": provisional_correct,
        "corrected_correct_at_arrival": corrected_correct, "score_pairs": score_checks,
        "mismatches": 0, "modes": len(modes),
    }


def _run(job):
    dataset, fold, seed, clips, results, build = job
    model = Int8Model.load(model_dir(results, dataset, fold, seed) / "int8.npz")
    lib = _load_api(model, build / dataset / f"fold{fold}" / f"seed{seed}")
    if len(clips) < 2:
        raise ValueError("Need two held-out clips to assemble a trace.")
    out = []
    for index in range(0, len(clips) - 1, 2):
        pair = clips[index:index + 2]
        clean = np.concatenate([sensor_input(model, clip) for clip in pair])
        # Use a stable trace label only when the two clips record the same activity.
        label = pair[0].label if pair[0].label == pair[1].label else None
        for order, profile in enumerate(WORKLOADS):
            delays = traffic(len(clean), profile, STREAM_SEED + fold * 1000 + index * 13 + order)
            stats = measure_trace(model, clean, delays, lib, label=label)
            out.append({"dataset": dataset, "fold": fold, "seed": seed,
                        "workload": profile.name, "clip1": pair[0].clip_id, "clip2": pair[1].clip_id,
                        "same_activity": label is not None, **stats})
    print(f"{dataset}/{fold}/{seed}: {len(out)} native streaming replays", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=V2)
    parser.add_argument("--build", type=Path, default=ROOT / "build/v2_native")
    parser.add_argument("--clips", type=Path, default=CLIP_DIR)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = []
    for dataset in DATASETS:
        for fold in FOLDS:
            clips = read_fixture(dataset, fold, args.clips)
            for seed in SEEDS:
                jobs.append((dataset, fold, seed, clips, args.results, args.build))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for local in pool.map(_run, jobs) for row in local]
    output = args.results / "streaming_native.csv.gz"
    pd.DataFrame(rows).to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{len(rows):,} workloads, {sum(row['score_pairs'] for row in rows):,} exact score comparisons")


if __name__ == "__main__":
    main()
