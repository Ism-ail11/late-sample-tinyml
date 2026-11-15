"""Download original UCI archives and build subject-disjoint activity traces."""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from .config import CONFIG, DATASETS, ROOT, STUDY, WINDOW

URLS = {
    "wisdm": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
    "mhealth": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
}
SHA256 = {
    "wisdm": "6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469",
    "mhealth": "16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0",
}
WISDM_LABELS = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting",
    "E": "standing", "F": "typing", "G": "brushing teeth", "H": "eating soup",
    "I": "eating chips", "J": "eating pasta", "K": "drinking", "L": "eating a sandwich",
    "M": "kicking", "O": "catching", "P": "dribbling", "Q": "writing",
    "R": "clapping", "S": "folding clothes",
}
MHEALTH_LABELS = (
    "standing", "sitting", "lying", "walking", "stairs", "waist bends", "arm elevation",
    "knee bends", "cycling", "jogging", "running", "jumping",
)
LABELS = {"wisdm": tuple(WISDM_LABELS.values()), "mhealth": MHEALTH_LABELS}
DEFAULT_DATA = ROOT / "data"


@dataclass(frozen=True)
class Segment:
    subject: int
    label: int
    values: np.ndarray


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(dataset: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{dataset}.zip"
    if path.exists():
        if sha256(path) == SHA256[dataset]:
            return path
        raise ValueError(f"Checksum mismatch for {path}. Remove the archive to download a fresh copy.")
    with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".zip", delete=False) as transfer:
        temp_path = Path(transfer.name)
        try:
            request = Request(URLS[dataset], headers={"User-Agent": "tinyml-reproduction/1.0"})
            with urlopen(request, timeout=90) as source:
                shutil.copyfileobj(source, transfer, length=1024 * 1024)
            transfer.flush()
            if sha256(temp_path) != SHA256[dataset]:
                raise ValueError(f"The downloaded {dataset} archive failed its SHA-256 check.")
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return path


def _wisdm_subject(stream, subject: int) -> list[Segment]:
    segments = []
    rows = []
    labels = {code: index for index, code in enumerate(WISDM_LABELS)}
    previous_label = None
    previous_time = None

    def finish():
        if len(rows) >= WINDOW:
            segments.append(Segment(subject, labels[previous_label], np.asarray(rows, dtype=np.float32)))
        rows.clear()

    for line in stream:
        for record in line.strip().split(b";"):
            record = record.strip()
            if not record:
                continue
            fields = record.split(b",")
            try:
                if len(fields) != 6 or int(fields[0]) != subject:
                    raise ValueError("malformed WISDM row")
                label = fields[1].decode("ascii")
                stamp = int(fields[2])
                vector = tuple(float(value) for value in fields[3:])
                if label not in labels or stamp <= 0 or not all(np.isfinite(vector)):
                    raise ValueError("invalid WISDM value")
            except (ValueError, UnicodeError):
                finish()
                previous_time = None
                continue
            if previous_time is not None:
                gap = stamp - previous_time
                if label != previous_label or not 0 < gap <= 250_000_000:
                    finish()
            rows.append(vector)
            previous_label, previous_time = label, stamp
    finish()
    return segments


def _read_wisdm(path: Path, data_dir: Path) -> list[Segment]:
    with zipfile.ZipFile(path) as outer:
        with tempfile.TemporaryFile(dir=data_dir) as payload:
            with outer.open("wisdm-dataset.zip") as source:
                shutil.copyfileobj(source, payload, length=1024 * 1024)
            payload.seek(0)
            with zipfile.ZipFile(payload) as inner:
                result = []
                regex = re.compile(r"/raw/watch/accel/data_(\d+)_accel_watch\.txt$")
                for name in sorted(inner.namelist()):
                    match = regex.search(name)
                    if match:
                        with inner.open(name) as stream:
                            result.extend(_wisdm_subject(stream, int(match[1])))
                return result


def _read_mhealth(path: Path) -> list[Segment]:
    result = []
    with zipfile.ZipFile(path) as archive:
        for subject in range(1, 11):
            with archive.open(f"MHEALTHDATASET/mHealth_subject{subject}.log") as stream:
                data = np.loadtxt(stream, dtype=np.float32, usecols=(14, 15, 16, 23))
            labels = data[:, 3].astype(np.int16)
            boundaries = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                label = int(labels[begin])
                values = data[begin:end, :3]
                if 1 <= label <= len(MHEALTH_LABELS) and len(values) >= WINDOW and np.isfinite(values).all():
                    result.append(Segment(subject, label - 1, values.copy()))
    return result


def prepare(dataset: str, data_dir: Path = DEFAULT_DATA, rebuild: bool = False) -> Path:
    output = data_dir / f"{dataset}_segments_v1.npz"
    if output.exists() and not rebuild:
        return output
    path = fetch(dataset, data_dir)
    segments = _read_wisdm(path, data_dir) if dataset == "wisdm" else _read_mhealth(path)
    if not segments:
        raise ValueError(f"No valid {dataset} segments found.")
    offsets = np.r_[0, np.cumsum([len(segment.values) for segment in segments], dtype=np.int64)]
    with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            np.savez_compressed(
                handle, values=np.concatenate([segment.values for segment in segments]), offsets=offsets,
                subjects=np.array([segment.subject for segment in segments], dtype=np.int16),
                labels=np.array([segment.label for segment in segments], dtype=np.int8),
            )
            handle.flush()
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
    return output


def load_segments(dataset: str, data_dir: Path = DEFAULT_DATA) -> list[Segment]:
    with np.load(prepare(dataset, data_dir)) as data:
        values = data["values"]
        offsets = data["offsets"]
        return [
            Segment(int(subject), int(label), values[begin:end])
            for subject, label, begin, end in zip(data["subjects"], data["labels"], offsets[:-1], offsets[1:])
        ]


def subject_split(dataset: str, segments: list[Segment]) -> dict[str, list[int]]:
    subjects = np.array(sorted({segment.subject for segment in segments}))
    order = np.random.default_rng(STUDY["split_seed"]).permutation(subjects)
    n_test = CONFIG[dataset]["test_subjects"]
    n_valid = CONFIG[dataset]["validation_subjects"]
    split = {
        "test": sorted(map(int, order[:n_test])),
        "validation": sorted(map(int, order[n_test:n_test + n_valid])),
        "train": sorted(map(int, order[n_test + n_valid:])),
    }
    expected = set(range(len(LABELS[dataset])))
    for name, subset in split.items():
        found = {s.label for s in segments if s.subject in subset}
        if found != expected:
            raise ValueError(f"{dataset} {name} split is missing classes {sorted(expected - found)}")
    return split


def training_stats(segments: list[Segment], subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([segment.values for segment in segments if segment.subject in subjects])
    return values.mean(axis=0, dtype=np.float64), np.maximum(values.std(axis=0, dtype=np.float64), 1e-6)


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - mean) / scale, -6.0, 6.0).astype(np.float32)


def windows(segments: list[Segment], subjects: list[int], cap: int,
            mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = []
    targets = []
    for segment in segments:
        if segment.subject not in subjects:
            continue
        starts = np.arange(0, len(segment.values) - WINDOW + 1, 24)
        if len(starts) > cap:
            starts = starts[np.linspace(0, len(starts) - 1, cap, dtype=int)]
        for start in starts:
            data.append(segment.values[start:start + WINDOW])
            targets.append(segment.label)
    if not data:
        raise ValueError("Window extraction produced no inputs.")
    return normalize(np.stack(data), mean, scale), np.asarray(targets, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        path = prepare(dataset, args.data_dir, args.rebuild)
        segments = load_segments(dataset, args.data_dir)
        print(f"{dataset}: {len(segments)} segments, {sum(len(s.values) for s in segments):,} samples")
        print(f"  {subject_split(dataset, segments)}")
        print(f"  {path}")


if __name__ == "__main__":
    main()
