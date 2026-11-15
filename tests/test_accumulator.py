"""Regression tests for cached accumulators, repair history, and fallback paths."""
import ctypes as ct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tinyml.config import DATASETS, SEEDS
from tinyml.evaluate import SCORE_POINTER, Update, native_library
from tinyml.folds import FOLDS, model_dir, read_fixture
from tinyml.protocol import TINY_DUPLICATE, TINY_EXPIRED, TINY_OK
from tinyml.protocol_v2 import METHODS
from tinyml.quant import Int8Model, dense_inference
from tinyml.study_v2 import V2


class CachedAccumulatorTests(unittest.TestCase):
    def check_stream(self, library, model, mode, horizon, seed):
        lib = library
        lib.tiny_create.restype = ct.c_void_p
        lib.tiny_destroy.argtypes = [ct.c_void_p]
        lib.tiny_destroy.restype = None
        lib.tiny_append.argtypes = [ct.c_void_p, SCORE_POINTER]
        lib.tiny_append.restype = None
        lib.tiny_scores.argtypes = [ct.c_void_p, ct.c_uint8]
        lib.tiny_scores.restype = SCORE_POINTER
        lib.tiny_repair.argtypes = [ct.c_void_p, ct.POINTER(Update), ct.c_size_t, ct.c_int]
        lib.tiny_repair.restype = ct.c_int

        def check_scores():
            outputs = dense_inference(model, inputs[:tick + 1])
            for age in range(min(tick, horizon) + 1):
                actual = np.ctypeslib.as_array(lib.tiny_scores(state, age), shape=(model.classes,))
                np.testing.assert_array_equal(actual, outputs[tick - age])

        rng = np.random.default_rng(seed)
        inputs = rng.integers(-128, 128, (216, 3), dtype=np.int8)
        state = lib.tiny_create()
        self.assertTrue(state)
        try:
            for tick, row in enumerate(inputs):
                lib.tiny_append(state, row.ctypes.data_as(SCORE_POINTER))
                if tick not in (0, 1, 7, 19, 38, 59, 82, 109, 140, 164, 190, 208, 215):
                    continue
                largest = min(tick, horizon)
                if tick in (59, 109, 190):
                    ages = list(range(largest + 1))
                elif tick in (1, 7, 82, 164):
                    ages = [min(largest, tick % 8)]
                else:
                    ages = sorted({0, min(largest, 2), largest})
                rng.shuffle(ages)
                replacements = rng.integers(-128, 128, (len(ages), 3), dtype=np.int8)
                entries = (Update * len(ages))(*[
                    Update(age, (ct.c_int8 * 3)(*map(int, sample)))
                    for age, sample in zip(ages, replacements, strict=True)
                ])
                self.assertEqual(lib.tiny_repair(state, entries, len(entries), mode), TINY_OK)
                inputs[tick - np.array(ages)] = replacements
                check_scores()

            dup = (Update * 2)(Update(0, (ct.c_int8 * 3)(127, 127, 127)),
                                Update(0, (ct.c_int8 * 3)(-128, -128, -128)))
            if horizon:
                self.assertEqual(lib.tiny_repair(state, dup, 2, mode), TINY_DUPLICATE)
            expired = (Update * 1)(Update(horizon + 1, (ct.c_int8 * 3)(-128, 0, 127)))
            self.assertEqual(lib.tiny_repair(state, expired, 1, mode), TINY_EXPIRED)
            check_scores()
        finally:
            lib.tiny_destroy(state)

    def test_multiple_repairs_and_wrapped_accumulators(self):
        configurations = ((0, 0), (8, 0), (4, 4), (8, 8), (24, 16), (48, 8), (63, 16))
        modes = ["accumulator", "generic", "all_taps", "no_equal", "recent_replay", "layer_replay", "serial"]
        with TemporaryDirectory() as tmp:
            for dataset in DATASETS:
                model = Int8Model.load(model_dir(V2, dataset, 0, SEEDS[0]) / "int8.npz")
                for horizon, cache in configurations:
                    path = Path(tmp) / f"{dataset}-{horizon}-{cache}"
                    lib = native_library(model, horizon, path, extended=True, cache=cache)
                    for mode in modes:
                        if horizon == 0 and mode == "serial":
                            continue
                        with self.subTest(dataset=dataset, horizon=horizon, cache=cache, mode=mode):
                            self.check_stream(lib, model, METHODS[mode], horizon,
                                              831 + list(METHODS).index(mode))

    def test_fold_fixtures_cover_every_test_subject(self):
        import json

        for dataset in DATASETS:
            outer_test = set()
            for fold in FOLDS:
                metrics = json.loads((model_dir(V2, dataset, fold, SEEDS[0]) / "metrics.json").read_text())
                groups = {name: set(group) for name, group in metrics["subjects"].items()}
                self.assertFalse(groups["train"] & groups["test"])
                self.assertFalse(groups["validation"] & groups["test"])
                self.assertFalse(groups["train"] & groups["validation"])
                self.assertFalse(outer_test & groups["test"])
                outer_test |= groups["test"]
                clips = read_fixture(dataset, fold)
                self.assertEqual(groups["test"], {clip.subject for clip in clips})
                self.assertEqual(len(clips), len({clip.clip_id for clip in clips}))


if __name__ == "__main__":
    unittest.main()
