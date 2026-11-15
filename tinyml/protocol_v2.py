"""Controlled scenarios for the fold study, including fill and burst variations."""
import numpy as np

from .protocol import ARRIVAL, Case


METHODS = {
    "replay": 0,
    "span": 1,
    "tap": 2,
    "tap_equal": 3,
    "v1_adaptive": 5,
    "age_mask": 8,
    "age_equal": 9,
    "layer_replay": 10,
    "accumulator": 11,
    "generic": 12,
    "no_delta": 13,
    "all_taps": 14,
    "no_equal": 15,
    "serial": 16,
    "recent_replay": 17,
}
PRIMARY = (
    "replay", "tap_equal", "v1_adaptive", "layer_replay", "age_equal",
    "recent_replay", "accumulator",
)
ABLATIONS = (
    "tap", "age_mask", "age_equal", "recent_replay", "accumulator",
    "generic", "no_delta", "all_taps", "no_equal", "serial",
)
AGE_SWEEP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48)


def scenarios(clip, group="primary"):
    def single(age, template="hold"):
        return Case(clip, "single", age, age, template, (ARRIVAL - age,))

    def burst(age, count, pattern):
        youngest = int(age - count + 1) if pattern == "burst" else 1
        ages = np.linspace(age, youngest, count, dtype=int)
        return Case(clip, f"{pattern}{count}", age, age, "hold", tuple(ARRIVAL - int(a) for a in ages))

    if group == "primary":
        return [*(single(age) for age in AGE_SWEEP), single(4, "zero"), single(48, "zero"),
                burst(7, 4, "burst"), burst(24, 4, "burst"), burst(48, 8, "burst"),
                burst(24, 4, "scatter"), burst(48, 8, "scatter")]
    if group == "ablation":
        return [*(single(age) for age in (1, 3, 5, 7, 24, 48)),
                single(3, "zero"), burst(7, 4, "burst"), burst(24, 4, "burst"), burst(48, 8, "burst"),
                burst(48, 8, "scatter")]
    if group == "resource":
        return [single(age) for age in range(1, 8)]
    if group == "sensitivity":
        return [*(single(age) for age in (1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32, 48)),
                single(3, "zero"), burst(7, 4, "burst"), burst(24, 8, "burst")]
    raise ValueError(f"Unknown scenario group {group!r}")
