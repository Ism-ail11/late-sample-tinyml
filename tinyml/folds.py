"""Subject-disjoint outer folds and class-balanced, participant-covering fixtures."""
import argparse
from pathlib import Path

import numpy as np

from .config import DATASETS, ROOT
from .data import DEFAULT_DATA, LABELS, load_segments
from .protocol import Clip

FOLDS = (0, 1, 2)
FOLD_SEED = 401839
REPLAY_SEED = 21659
CLIP_LENGTH = 120
CLIP_DIR = ROOT / "replay_clips" / "folds"


def splits(dataset, segments, fold):
    subjects = np.array(sorted({segment.subject for segment in segments}))
    order = np.random.default_rng(FOLD_SEED).permutation(subjects)
    heldout = np.array_split(order, len(FOLDS))[fold]
    other = order[~np.isin(order, heldout)]
    valid = np.random.default_rng(FOLD_SEED + fold + 1).permutation(other)[:7 if dataset == "wisdm" else 2]
    split = {
        "train": sorted(map(int, other[~np.isin(other, valid)])),
        "validation": sorted(map(int, valid)),
        "test": sorted(map(int, heldout)),
    }
    expected = set(range(len(LABELS[dataset])))
    for name, cohort in split.items():
        found = {segment.label for segment in segments if segment.subject in cohort}
        if found != expected:
            raise ValueError(f"{dataset} fold {fold} {name} missing labels: {expected - found}")
    return split


def fixture_file(dataset, fold, directory=CLIP_DIR):
    return directory / f"{dataset}_fold{fold}.npz"


def model_dir(results, dataset, fold, seed):
    return results / "models" / dataset / f"fold{fold}" / f"seed{seed}"


def make_clips(dataset, segments, fold):
    test = splits(dataset, segments, fold)["test"]
    by_label = {i: {} for i in range(len(LABELS[dataset]))}
    for index, segment in enumerate(segments):
        if segment.subject not in test:
            continue
        # Disjoint candidates, with deterministic selection inside each activity.
        for offset in range(0, len(segment.values) - CLIP_LENGTH + 1, CLIP_LENGTH):
            by_label[segment.label].setdefault(segment.subject, []).append((index, offset))
    rng = np.random.default_rng(REPLAY_SEED + fold)
    counts = {subject: 0 for subject in test}
    picked = set()
    result = []

    def add(index, offset):
        segment = segments[index]
        counts[segment.subject] += 1
        picked.add((index, offset))
        cid = f"{dataset}-f{fold}-s{segment.subject}-g{index}-t{offset}"
        result.append(Clip(cid, segment.subject, segment.label, offset,
                           segment.values[offset:offset + CLIP_LENGTH].copy()))

    for label, by_subject in by_label.items():
        if len(by_subject) < 3:
            raise ValueError(f"{dataset} fold {fold} label {label} has too few clip donors")
        shuffled = list(rng.permutation(list(by_subject)))
        for subject in sorted(shuffled, key=counts.get)[:3]:
            pool = by_subject[subject]
            add(*pool[int(rng.integers(len(pool)))])
    for subject in test:
        pool = [candidate for group in by_label.values() for candidate in group.get(subject, ())
                if candidate not in picked]
        for choice in rng.choice(len(pool), size=max(0, 2 - counts[subject]), replace=False):
            add(*pool[int(choice)])
    assert set(counts) == {clip.subject for clip in result}
    return sorted(result, key=lambda clip: (clip.label, clip.subject, clip.clip_id))


def save_fixture(path, clips):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, ids=np.array([x.clip_id for x in clips]), subjects=np.array([x.subject for x in clips]),
        labels=np.array([x.label for x in clips]), starts=np.array([x.start for x in clips]),
        values=np.stack([x.values for x in clips]),
    )


def read_fixture(dataset, fold, directory=CLIP_DIR):
    with np.load(fixture_file(dataset, fold, directory), allow_pickle=False) as x:
        return [Clip(str(cid), int(sub), int(label), int(start), values)
                for cid, sub, label, start, values in zip(
                    x["ids"], x["subjects"], x["labels"], x["starts"], x["values"], strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=CLIP_DIR)
    args = parser.parse_args()
    for dataset in DATASETS:
        segments = load_segments(dataset, args.data_dir)
        for fold in FOLDS:
            clips = make_clips(dataset, segments, fold)
            save_fixture(fixture_file(dataset, fold, args.output), clips)
            print(f"{dataset} fold {fold}: {len(clips)} clips, {len({x.subject for x in clips})} subjects", flush=True)


if __name__ == "__main__":
    main()
