"""Repository paths and experiment settings."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "study.toml").open("rb") as stream:
    CONFIG = tomllib.load(stream)

STUDY = CONFIG["study"]
DATASETS = ("wisdm", "mhealth")
SEEDS = tuple(STUDY["seeds"])
WIDTHS = tuple(STUDY["widths"])
DILATIONS = tuple(STUDY["dilations"])
KERNEL = 3
RECEPTIVE_FIELD = 1 + (KERNEL - 1) * sum(DILATIONS)
WINDOW = STUDY["window"]
RESULTS = ROOT / "results"


def run_dir(results: Path, dataset: str, seed: int) -> Path:
    return results / "models" / dataset / f"seed_{seed}"
