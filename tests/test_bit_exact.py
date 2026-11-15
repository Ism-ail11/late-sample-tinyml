"""Exercise integer arithmetic, ring wrap, transactions, and future predictions."""
import ctypes as ct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tinyml.config import DATASETS, RESULTS, SEEDS, run_dir
from tinyml.evaluate import SCORE_POINTER, Update, native_library
from tinyml.protocol import MODE_IDS, TINY_DUPLICATE, TINY_EXPIRED, TINY_INVALID, TINY_OK, read_clips, CLIPS_DIR
from tinyml.quant import Int8Model, Layer, SHIFT, dense_inference


class ExactFirmwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.models = {}
        cls.libraries = {}
        for dataset in DATASETS:
            model = Int8Model.load(run_dir(RESULTS, dataset, SEEDS[0]) / "int8.npz")
            cls.models[dataset] = model
            for horizon in (8, 24, 48, 63):
                path = Path(cls.temp.name) / dataset / str(horizon)
                library = native_library(model, horizon, path)
                library.tiny_create.argtypes = []
                library.tiny_create.restype = ct.c_void_p
                library.tiny_destroy.argtypes = [ct.c_void_p]
                library.tiny_destroy.restype = None
                library.tiny_append.argtypes = [ct.c_void_p, SCORE_POINTER]
                library.tiny_append.restype = None
                library.tiny_repair.argtypes = [ct.c_void_p, ct.POINTER(Update), ct.c_size_t, ct.c_int]
                library.tiny_repair.restype = ct.c_int
                library.tiny_scores.argtypes = [ct.c_void_p, ct.c_uint8]
                library.tiny_scores.restype = SCORE_POINTER
                cls.libraries[dataset, horizon] = library

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def add(library, state, sample):
        library.tiny_append(state, sample.ctypes.data_as(SCORE_POINTER))

    def compare(self, lib, state, model, horizon, inputs):
        outputs = dense_inference(model, inputs)
        for age in range(min(horizon + 1, len(inputs))):
            actual = np.ctypeslib.as_array(lib.tiny_scores(state, age), shape=(model.classes,))
            np.testing.assert_array_equal(actual, outputs[-age - 1])

    def test_ring_wrap_repeated_corrections_and_future_predictions(self):
        rng = np.random.default_rng(78131)
        for dataset, model in self.models.items():
            for horizon in (8, 24, 48, 63):
                lib = self.libraries[dataset, horizon]
                for mode, mode_id in MODE_IDS.items():
                    with self.subTest(dataset=dataset, horizon=horizon, mode=mode):
                        stream = rng.integers(-127, 128, size=(230, 3), dtype=np.int8)
                        state = lib.tiny_create()
                        try:
                            self.assertFalse(bool(lib.tiny_scores(state, 0)))
                            for tick in range(230):
                                self.add(lib, state, stream[tick])
                                if tick not in (0, 5, 71, 141, 178, 226):
                                    continue
                                max_age = min(tick, horizon)
                                ages = list(dict.fromkeys([max_age, 0, max_age // 2]))
                                if tick == 226:
                                    ages = list(range(horizon + 1))
                                rng.shuffle(ages)
                                updated = rng.integers(-127, 128, size=(len(ages), 3), dtype=np.int8)
                                updates = (Update * len(ages))(*[
                                    Update(age, (ct.c_int8 * 3)(*map(int, sample)))
                                    for age, sample in zip(ages, updated)
                                ])
                                self.assertEqual(lib.tiny_repair(state, updates, len(updates), mode_id), TINY_OK)
                                stream[tick - np.array(ages)] = updated
                                self.compare(lib, state, model, horizon, stream[:tick + 1])
                            self.compare(lib, state, model, horizon, stream)
                        finally:
                            lib.tiny_destroy(state)

    def test_rejected_updates_are_atomic_and_identical_updates_are_noops(self):
        for dataset, model in self.models.items():
            lib = self.libraries[dataset, 8]
            values = np.arange(90, dtype=np.int8).reshape(-1, 3)
            state = lib.tiny_create()
            try:
                for sample in values:
                    self.add(lib, state, sample)
                same = (ct.c_int8 * 3)(*map(int, values[-1]))
                diff = (ct.c_int8 * 3)(-11, -24, 95)
                for mode_id in MODE_IDS.values():
                    self.assertEqual(lib.tiny_repair(state, None, 0, mode_id), TINY_OK)
                    self.assertEqual(lib.tiny_repair(state, (Update * 2)(Update(0, diff), Update(9, diff)),
                                                     2, mode_id), TINY_EXPIRED)
                    self.assertEqual(lib.tiny_repair(state, (Update * 2)(Update(0, diff), Update(0, same)),
                                                     2, mode_id), TINY_DUPLICATE)
                    self.assertEqual(lib.tiny_repair(state, None, 1, mode_id), TINY_INVALID)
                    self.assertEqual(lib.tiny_repair(state, (Update * 1)(Update(0, same)), 1, mode_id), TINY_OK)
                    self.compare(lib, state, model, 8, values)
                self.assertFalse(bool(lib.tiny_scores(state, 9)))
            finally:
                lib.tiny_destroy(state)

    def test_dense_rounding_and_saturation(self):
        identity = Layer(np.array([[[1]]], dtype=np.int8), np.zeros(1, dtype=np.int32),
                         1 << SHIFT, 1, False)
        values = np.array([[-127], [-5], [-3], [-1], [0], [1], [3], [5], [127]], dtype=np.int8)
        half = Layer(identity.weight, identity.bias, 1 << (SHIFT - 1), 1, False)
        double = Layer(identity.weight, identity.bias, 2 << SHIFT, 1, False)
        relu = Layer(identity.weight, identity.bias, 1 << SHIFT, 1, True)
        for layer, expected in [
            (half, [-64, -3, -2, -1, 0, 1, 2, 3, 64]),
            (double, [-127, -10, -6, -2, 0, 2, 6, 10, 127]),
            (relu, [0, 0, 0, 0, 0, 1, 3, 5, 127]),
        ]:
            actual = dense_inference(Int8Model((layer,), np.zeros(1), np.ones(1)), values)
            np.testing.assert_array_equal(actual.ravel(), expected)

    def test_fixture_ids_and_shapes(self):
        for dataset in DATASETS:
            clips = read_clips(dataset, CLIPS_DIR)
            self.assertEqual(len(clips), len({clip.clip_id for clip in clips}))
            self.assertTrue(all(np.isfinite(clip.values).all() for clip in clips))


if __name__ == "__main__":
    unittest.main()
