"""Run paired emulated experiments with shared weights, kernels, and oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from .arm import run_arm
from .cpu import build_runner, run_cpu
from .config import CONFIG, DATASETS, RESULTS, ROOT, SEEDS
from .folds import CLIP_DIR, FOLDS, model_dir, read_fixture
from .protocol_v2 import ABLATIONS, METHODS, PRIMARY, scenarios
from .quant import Int8Model

V2 = RESULTS / "v2"
DEFAULT_BUILD = ROOT / "build" / "v2"
CHUNK = 232
HORIZON = 48


@dataclass(frozen=True)
class Cohort:
    name: str
    mode_names: tuple[str, ...]
    platform: str = "cortex-m4"
    cache: int = 8
    horizon: int = HORIZON
    scalar: bool = False
    grid: str = "primary"
    one_clip_per_subject: bool = False
    all_seeds: bool = True
    lock_kernel: bool = False


COHORTS = {
    "main": Cohort("main", PRIMARY),
    "no_cache": Cohort("no_cache", ("replay", "tap_equal", "recent_replay"), cache=0, grid="sensitivity",
                       one_clip_per_subject=True),
    "ablation": Cohort("ablation", ABLATIONS, grid="ablation", one_clip_per_subject=True),
    "zero_cache": Cohort("zero_cache", ("tap_equal", "recent_replay"), cache=0, grid="sensitivity",
                         one_clip_per_subject=True, all_seeds=False),
    "cache4": Cohort("cache4", ("recent_replay", "accumulator"), cache=4, grid="sensitivity",
                     one_clip_per_subject=True, all_seeds=False),
    "cache8": Cohort("cache8", ("recent_replay", "accumulator"), cache=8, grid="sensitivity",
                     one_clip_per_subject=True, all_seeds=False),
    "cache16": Cohort("cache16", ("recent_replay", "accumulator"), cache=16, grid="sensitivity",
                      one_clip_per_subject=True, all_seeds=False),
    "scalar": Cohort("scalar", ("tap_equal", "age_equal", "recent_replay", "accumulator"), scalar=True,
                     grid="sensitivity", one_clip_per_subject=True, all_seeds=False),
    "m33": Cohort("m33", ("tap_equal", "age_equal", "recent_replay", "accumulator"), platform="cortex-m33",
                  grid="sensitivity", one_clip_per_subject=True, all_seeds=False),
    "horizon8": Cohort("horizon8", ("recent_replay", "accumulator"), horizon=8, grid="sensitivity",
                      one_clip_per_subject=True, all_seeds=False),
    "append0": Cohort("append0", ("recent_replay",), cache=0, grid="resource",
                      one_clip_per_subject=True, all_seeds=False, lock_kernel=True),
    "append8": Cohort("append8", ("recent_replay", "accumulator"), grid="resource",
                      one_clip_per_subject=True, all_seeds=False, lock_kernel=True),
    "renode_check": Cohort("renode_check", ("replay", "tap_equal", "recent_replay", "accumulator"),
                           one_clip_per_subject=True, all_seeds=False),
    "horizon24": Cohort("horizon24", ("recent_replay", "accumulator"), horizon=24, grid="sensitivity",
                       one_clip_per_subject=True, all_seeds=False),
}
METHOD_NAMES = {v: k for k, v in METHODS.items()}


def output_file(results, name, dataset, fold, seed, chunk):
    return results / "events" / name / dataset / f"fold{fold}_seed{seed}_part{chunk:02d}.csv.gz"


def select_clips(clips, dataset, fold):
    # Stratify the hardware sensitivity checks by participant. Select a sample
    # by a predeclared hash ordering independent of firmware measurements.
    import hashlib

    key = lambda clip: hashlib.sha256(f"{dataset}/{fold}/{clip.clip_id}/43681".encode()).digest()
    cohort = {}
    for clip in sorted(clips, key=key):
        cohort.setdefault(clip.subject, clip)
    return sorted(cohort.values(), key=lambda clip: clip.clip_id)


def _execute(job):
    cohort, dataset, fold, seed, index, cases, results, build, runner = job
    model = Int8Model.load(model_dir(results, dataset, fold, seed) / "int8.npz")
    root = build / cohort.name / dataset / f"fold{fold}_seed{seed}_part{index:02d}"
    ids = tuple(METHODS[name] for name in cohort.mode_names)
    if cohort.name == "renode_check":
        backend = "renode"
        output = run_arm(model, cases, cohort.horizon, root, extended=True, mode_ids=ids,
                         cache=cohort.cache, target=cohort.platform, scalar=cohort.scalar)
    else:
        backend = "unicorn"
        output = run_cpu(model, cases, cohort.horizon, root, ids, runner,
                         cache=cohort.cache, target=cohort.platform, scalar=cohort.scalar,
                         lock_kernel=cohort.lock_kernel)
    records = []
    for row in output:
        case = cases[row["case_index"]]
        records.append({
            "cohort": cohort.name, "backend": backend, "dataset": dataset, "fold": fold, "seed": seed,
            "subject": case.clip.subject, "activity": case.clip.label, "clip_id": case.clip.clip_id,
            "case_id": case.case_id, "pattern": case.pattern, "delay": case.delay,
            "delay_ms": case.delay * 1000 / CONFIG[dataset]["sample_hz"],
            "placeholder": case.placeholder, "batch": case.burst_size,
            "mode": METHOD_NAMES[row["mode_id"]], "target": cohort.platform, "scalar": cohort.scalar,
            "horizon": cohort.horizon, "cache_frames": cohort.cache,
            "accepted": case.max_delay <= cohort.horizon,
            "score_pairs": 2 * (cohort.horizon + 1) * model.classes,
            **{key: value for key, value in row.items() if key not in ("case_index", "mode_id")},
            "method_sram": row["state_bytes"] + row["stack_bytes"],
        })
    target = output_file(results, cohort.name, dataset, fold, seed, index)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix="events-", suffix=".csv.gz", dir=target.parent, delete=False) as tmp:
        path = Path(tmp.name)
        try:
            pd.DataFrame(records).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)
    print(f"{cohort.name:<11} {dataset:<7} fold {fold} seed {seed} part {index}: {len(records):,} verified runs", flush=True)
    return target


def jobs_for(args):
    requested = ("main", "no_cache", "ablation", "zero_cache", "cache4", "cache8",
                 "cache16", "scalar", "m33", "horizon8", "horizon24", "append0", "append8", "renode_check")
    jobs = []
    manifest = []
    for group in (args.cohorts or requested):
        cohort = COHORTS[group]
        for dataset in (args.dataset,) if args.dataset else DATASETS:
            for fold in (args.fold,) if args.fold is not None else FOLDS:
                clips = read_fixture(dataset, fold, args.clips)
                if cohort.one_clip_per_subject:
                    clips = select_clips(clips, dataset, fold)
                cases = [case for clip in clips for case in scenarios(clip, cohort.grid)]
                if group == "renode_check":
                    cases = [case for case in cases
                             if case.placeholder == "hold"
                             and ((case.pattern == "single" and case.delay in (1, 3, 7, 32))
                                  or case.pattern == "burst4" and case.delay == 7)]
                selected_seeds = ((args.seed,) if args.seed is not None
                                  else SEEDS if cohort.all_seeds else SEEDS[:1])
                for seed in selected_seeds:
                    for i, start in enumerate(range(0, len(cases), CHUNK)):
                        target = output_file(args.results, cohort.name, dataset, fold, seed, i)
                        sample = cases[start:start + CHUNK]
                        manifest.append({
                            "path": target.relative_to(args.results).as_posix(), "cases": len(sample),
                            "methods": list(cohort.mode_names),
                        })
                        if not args.reuse or not target.is_file():
                            jobs.append((cohort, dataset, fold, seed, i, sample, args.results, args.build,
                                         (args.build / "cpu_runner").resolve()))
    return jobs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=V2)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--clips", type=Path, default=CLIP_DIR)
    parser.add_argument("--cohorts", nargs="+", choices=tuple(COHORTS))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse", action="store_true")
    if hasattr(os, "sched_getaffinity"):
        os.environ.setdefault("DOTNET_PROCESSOR_COUNT", "2")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    jobs, manifest = jobs_for(args)
    print(f"{len(jobs)} emulation batches to run, {len(manifest)} batches selected", flush=True)
    if any(job[0].name != "renode_check" for job in jobs):
        build_runner(args.build / "cpu_runner")
    if args.workers == 1:
        for job in jobs:
            _execute(job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(_execute, jobs):
                pass
    if not (args.cohorts or args.dataset or args.fold is not None or args.seed is not None):
        (args.results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
