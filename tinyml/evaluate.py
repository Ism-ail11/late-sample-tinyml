"""Execute the paired native study and Cortex-M4 runs with per-score oracle checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .arm import run_arm
from .build import build_native
from .config import CONFIG, DATASETS, RESULTS, ROOT, SEEDS, STUDY, run_dir
from .data import DEFAULT_DATA
from .protocol import (ARRIVAL, CLIPS_DIR, HORIZON, MODES, TINY_EXPIRED, TINY_OK,
                       Case, conditions, held_out_clips, read_clips, sensor_input)
from .quant import Int8Model, dense_inference

DEFAULT_BUILD = ROOT / "build"
SCORE_POINTER = ct.POINTER(ct.c_int8)


class Update(ct.Structure):
    _fields_ = [("age", ct.c_uint8), ("sample", ct.c_int8 * 3)]


def native_library(model: Int8Model, horizon: int, output: Path, *, extended=False, cache=8):
    library = ct.CDLL(str(build_native(model, horizon, output, extended=extended, cache=cache).resolve()))
    library.tiny_native_case.argtypes = [
        SCORE_POINTER, ct.c_uint, ct.POINTER(Update), ct.c_uint, ct.c_uint,
        SCORE_POINTER, SCORE_POINTER, ct.POINTER(ct.c_uint32),
    ]
    library.tiny_native_case.restype = None
    library.tiny_state_size.argtypes = []
    library.tiny_state_size.restype = ct.c_size_t
    return library


def _fields(dataset: str, seed: int, horizon: int, case: Case) -> dict:
    return {
        "dataset": dataset, "seed": seed, "horizon": horizon,
        "case_id": case.case_id, "clip_id": case.clip.clip_id, "subject": case.clip.subject,
        "activity": case.clip.label, "start_sample": case.clip.start,
        "pattern": case.pattern, "burst_size": case.burst_size, "delay": case.delay,
        "delay_ms": 1000 * case.delay / CONFIG[dataset]["sample_hz"], "placeholder": case.placeholder,
        "status": "expired" if case.max_delay > horizon else "ok",
    }


def _native_run(model: Int8Model, dataset: str, seed: int, cases: list[Case],
                horizon: int, output: Path) -> pd.DataFrame:
    library = native_library(model, horizon, output)
    rows = []
    dense = {}
    capacity = horizon + 1
    shape = (capacity, model.classes)
    before = np.empty(shape, dtype=np.int8)
    after = np.empty(shape, dtype=np.int8)
    before_ptr = before.ctypes.data_as(SCORE_POINTER)
    after_ptr = after.ctypes.data_as(SCORE_POINTER)
    work = (ct.c_uint32 * 5)()
    macs = np.array([layer.macs for layer in model.layers])
    retained = slice(ARRIVAL, ARRIVAL - horizon - 1, -1)
    for case in cases:
        if case.clip.clip_id not in dense:
            clean = sensor_input(model, case.clip)[:ARRIVAL + 1]
            dense[case.clip.clip_id] = clean, dense_inference(model, clean)
        clean, clean_scores = dense[case.clip.clip_id]
        observed = case.provisional(clean)
        initial = dense_inference(model, observed)[retained]
        expired = case.max_delay > horizon
        final = initial if expired else clean_scores[retained]
        update_vector = [
            Update(ARRIVAL - p, (ct.c_int8 * 3)(*map(int, clean[p]))) for p in case.positions
        ]
        updates = (Update * len(update_vector))(*update_vector)
        fields = _fields(dataset, seed, horizon, case)
        fields.update({
            "state_bytes": library.tiny_state_size(),
            "revised_timestamps": int(np.any(initial != final, axis=-1).sum()),
            "changed_labels": int(np.count_nonzero(initial.argmax(axis=-1) != final.argmax(axis=-1))),
            "latest_label_revised": int(initial[0].argmax() != final[0].argmax()),
            "score_pairs_checked": 2 * final.size,
            "changed_input_samples": int(sum(np.any(observed[p] != clean[p]) for p in case.positions)),
        })
        source = observed.ctypes.data_as(SCORE_POINTER)
        for mode_id, mode in enumerate(MODES):
            library.tiny_native_case(source, len(observed), updates, len(updates),
                                     mode_id, before_ptr, after_ptr, work)
            mismatch = int(np.count_nonzero(before != initial) + np.count_nonzero(after != final))
            if mismatch or work[4] != (TINY_EXPIRED if expired else TINY_OK):
                raise AssertionError(f"Native score/status mismatch in {dataset}/{seed}/{case.case_id}/{mode}, age {horizon}")
            counts = np.asarray(work[:4], dtype=np.int64)
            rows.append({
                **fields, "mode": mode, "mismatches": mismatch,
                "conv0_sites": counts[0], "conv1_sites": counts[1],
                "conv2_sites": counts[2], "head_sites": counts[3],
                "evaluated_sites": counts.sum(), "kernel_macs": int(counts @ macs),
            })
    return pd.DataFrame(rows)


def _arm_run(model: Int8Model, dataset: str, seed: int, cases: list[Case],
             horizon: int, output: Path) -> pd.DataFrame:
    rows = run_arm(model, cases, horizon, output)
    modes = tuple(MODES)
    return pd.DataFrame([
        {
            **_fields(dataset, seed, horizon, cases[row["case_index"]]),
            "mode": modes[row["mode_id"]],
            **{key: row[key] for key in (
                "instructions", "mismatches", "stack_bytes", "state_bytes", "checksum", "revised_timestamps",
            )},
            "application_sram_bytes": row["stack_bytes"] + row["state_bytes"],
            "score_pairs_checked": 2 * (horizon + 1) * model.classes,
        }
        for row in rows
    ])


def run_one(dataset: str, seed: int, backend: str, data_dir: Path, results: Path, build: Path,
            clips_dir: Path | None):
    output = run_dir(results, dataset, seed)
    model = Int8Model.load(output / "int8.npz")
    clips = read_clips(dataset, clips_dir) if clips_dir else held_out_clips(dataset, data_dir)
    compact = backend == "renode"
    if compact:
        clips = [clip for clip in clips if int(clip.clip_id.split("-")[1]) < STUDY["renode_clips_per_class"]]
    cases = [case for clip in clips for case in conditions(clip, compact=compact)]
    # Test memory/coverage on paired conditions. Renode runs the complete primary
    # experiment, plus an explicit expiry/canary subset at the two smaller capacities.
    combined = []
    for horizon in (HORIZON, *STUDY["short_horizons"]):
        subset = cases if horizon == HORIZON else [
            case for case in cases
            if case.pattern in ("single", "burst4") and case.placeholder == "hold"
            and case.delay in (8, 24, 48)
        ]
        if compact and horizon != HORIZON:
            subset = [case for case in subset if case.clip.label in (0, 1, 2, 3)]
        path = build / backend / dataset / f"seed_{seed}" / f"h{horizon}"
        execute = _arm_run if compact else _native_run
        result = execute(model, dataset, seed, subset, horizon, path)
        combined.append(result)
        print(f"{backend} {dataset}/{seed} horizon={horizon}: {len(subset):,} cases × {len(MODES)} methods", flush=True)
    frame = pd.concat(combined, ignore_index=True)
    filename = output / f"{backend}_events.csv.gz"
    frame.to_csv(filename, index=False, compression={"method": "gzip", "mtime": 0})
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "renode"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--build", type=Path, default=Path(os.getenv("TINYML_BUILD_DIR", DEFAULT_BUILD)))
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--read-raw", action="store_true", help="Recreate replay clips from the original archives.")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    jobs = []
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        for seed in (args.seed,) if args.seed is not None else SEEDS:
            directory = run_dir(args.results, dataset, seed)
            if args.reuse and (directory / f"{args.backend}_events.csv.gz").exists():
                continue
            jobs.append((dataset, seed, args.backend, args.data_dir, args.results, args.build,
                         None if args.read_raw else args.clips_dir))
    if args.workers == 1:
        for job in jobs:
            run_one(*job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in [pool.submit(run_one, *job) for job in jobs]:
                future.result()


if __name__ == "__main__":
    main()
