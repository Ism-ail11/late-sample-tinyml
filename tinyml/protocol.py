"""Build paired scenarios from subject-disjoint, real sensor clips."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATASETS, ROOT, STUDY
from .data import DEFAULT_DATA, LABELS, SHA256, load_segments, normalize, subject_split
from .quant import Int8Model, quantize_input

MODES = {
    "replay": "Temporal replay",
    "span": "Dense span",
    "tap": "Tap mask",
    "tap_equal": "Tap mask + equality",
    "span_equal": "Dense span + equality",
    "adaptive": "Adaptive + equality",
    "serial": "Adaptive, one sample at a time",
    "adaptive_no_equal": "Adaptive, equality disabled",
}
MODE_IDS = {name: index for index, name in enumerate(MODES)}
HORIZON = STUDY["horizon"]
ARRIVAL = STUDY["arrival_tick"]
CLIPS_DIR = ROOT / "replay_clips"

TINY_OK = 0
TINY_EXPIRED = 1
TINY_DUPLICATE = 2
TINY_INVALID = 3


@dataclass(frozen=True)
class Clip:
    clip_id: str
    subject: int
    label: int
    start: int
    values: np.ndarray


@dataclass(frozen=True)
class Case:
    clip: Clip
    pattern: str
    delay: int
    max_delay: int
    placeholder: str
    positions: tuple[int, ...]

    @property
    def burst_size(self) -> int:
        return len(self.positions)

    @property
    def case_id(self) -> str:
        return f"{self.clip.clip_id}/{self.pattern}/{self.delay}/{self.placeholder}"

    def provisional(self, clean: np.ndarray) -> np.ndarray:
        observed = clean.copy()
        missing = set(self.positions)
        for position in self.positions:
            previous = position - 1
            while previous in missing:
                previous -= 1
            if self.placeholder == "hold":
                observed[position] = clean[previous] if previous >= 0 else 0
            else:
                observed[position] = 0
        return observed


def held_out_clips(dataset: str, data_dir: Path) -> list[Clip]:
    segments = load_segments(dataset, data_dir)
    held_out = set(subject_split(dataset, segments)["test"])
    samples = STUDY["clip_length"]
    quota = STUDY["host_clips_per_class"]
    rng = np.random.default_rng(STUDY["trace_seed"])
    clips = []
    for label in range(len(LABELS[dataset])):
        candidates = {}
        for index, segment in enumerate(segments):
            if segment.label != label or segment.subject not in held_out:
                continue
            for start in range(0, len(segment.values) - samples + 1, samples):
                candidates.setdefault(segment.subject, []).append((index, start))
        if not candidates:
            raise ValueError(f"No replay candidate for {dataset} class {label}")
        for subject, pool in candidates.items():
            chosen = rng.permutation(len(pool))
            candidates[subject] = [pool[index] for index in chosen]
        selected = 0
        subjects = list(rng.permutation(sorted(candidates)))
        while selected < quota:
            for subject in subjects:
                if not candidates[subject]:
                    continue
                index, start = candidates[subject].pop()
                values = segments[index].values[start:start + samples].copy()
                clip_id = f"{label:02d}-{selected:02d}-s{subject}-g{index}-t{start}"
                clips.append(Clip(clip_id, int(subject), label, start, values))
                selected += 1
                if selected == quota:
                    break
            else:
                if not any(candidates.values()) and selected < quota:
                    raise ValueError(f"Too few held-out clips for {dataset} class {label}")
    return clips


def conditions(clip: Clip, compact: bool = False) -> list[Case]:
    result = []
    for delay in STUDY["delay_grid"]:
        if not compact or delay in (1, 8, 16, 32, 48):
            result.append(Case(clip, "single", delay, delay, "hold", (ARRIVAL - delay,)))
        if not compact or delay in (8, 24, 48):
            result.append(Case(clip, "single", delay, delay, "zero", (ARRIVAL - delay,)))
    for size in STUDY["burst_sizes"]:
        for delay in STUDY["burst_delays"]:
            if size > delay + 1:
                continue
            if not compact or (size in (2, 4, 8) and delay in (8, 24, 48)):
                positions = tuple(range(ARRIVAL - delay, ARRIVAL - delay + size))
                result.append(Case(clip, f"burst{size}", delay, delay, "hold", positions))
            if not compact and delay in (16, 24, 48) and size in (4, 8):
                positions = tuple(map(int, np.linspace(ARRIVAL - delay, ARRIVAL - 1, size, dtype=int)))
                result.append(Case(clip, f"scattered{size}", delay, delay, "hold", positions))
    return result


def sensor_input(model: Int8Model, clip: Clip) -> np.ndarray:
    return quantize_input(normalize(clip.values, model.mean, model.std))


def save_clips(dataset: str, clips: list[Clip], clip_dir: Path):
    clip_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        clip_dir / f"{dataset}.npz",
        clip_id=np.array([clip.clip_id for clip in clips]),
        subject=np.array([clip.subject for clip in clips], dtype=np.int16),
        label=np.array([clip.label for clip in clips], dtype=np.int8),
        start=np.array([clip.start for clip in clips], dtype=np.int32),
        values=np.stack([clip.values for clip in clips]),
        source_sha256=SHA256[dataset], trace_seed=STUDY["trace_seed"], split_seed=STUDY["split_seed"],
    )


def read_clips(dataset: str, clip_dir: Path) -> list[Clip]:
    with np.load(clip_dir / f"{dataset}.npz", allow_pickle=False) as data:
        if (data["source_sha256"].item() != SHA256[dataset]
                or data["trace_seed"].item() != STUDY["trace_seed"]
                or data["split_seed"].item() != STUDY["split_seed"]
                or data["values"].shape != (len(LABELS[dataset]) * STUDY["host_clips_per_class"],
                                           STUDY["clip_length"], 3)):
            raise ValueError("Replay fixtures do not match study.toml. Generate them with tinyml.protocol.")
        return [
            Clip(str(cid), int(subject), int(label), int(start), values)
            for cid, subject, label, start, values in zip(
                data["clip_id"], data["subject"], data["label"], data["start"], data["values"], strict=True
            )
        ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    args = parser.parse_args()
    for dataset in DATASETS:
        clips = held_out_clips(dataset, args.data_dir)
        save_clips(dataset, clips, args.clips_dir)
        print(f"{dataset}: exported {len(clips)} held-out replay clips")


if __name__ == "__main__":
    main()
