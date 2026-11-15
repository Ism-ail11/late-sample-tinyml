"""Portable INT8 format and an independent dense NumPy oracle."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHIFT = 20
INPUT_SCALE = 6.0 / 127


@dataclass(frozen=True)
class Layer:
    weight: np.ndarray
    bias: np.ndarray
    multiplier: int
    dilation: int
    relu: bool

    @property
    def macs(self) -> int:
        return int(self.weight.size)


@dataclass(frozen=True)
class Int8Model:
    layers: tuple[Layer, ...]
    mean: np.ndarray
    std: np.ndarray

    @property
    def classes(self) -> int:
        return self.layers[-1].weight.shape[0]

    def save(self, path: Path):
        arrays = {"mean": self.mean, "std": self.std}
        for index, layer in enumerate(self.layers):
            arrays[f"weight{index}"] = layer.weight
            arrays[f"bias{index}"] = layer.bias
            arrays[f"params{index}"] = np.array([layer.multiplier, layer.dilation, layer.relu])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            layers = []
            for index in range(4):
                mul, dil, relu = map(int, data[f"params{index}"])
                layers.append(Layer(data[f"weight{index}"], data[f"bias{index}"], mul, dil, bool(relu)))
            return cls(tuple(layers), data["mean"], data["std"])


def quantize_input(normalized: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(normalized / INPUT_SCALE), -127, 127).astype(np.int8)


def dense_inference(model: Int8Model, inputs: np.ndarray, intermediate: bool = False):
    """Evaluate from time zero with a layer-wise zero prefix and integer rounding."""
    activations = np.asarray(inputs, dtype=np.int8)
    if activations.ndim < 2 or activations.shape[-1] != model.layers[0].weight.shape[-1]:
        raise ValueError("Expected input with shape (..., time, channels).")
    outputs = []
    length = activations.shape[-2]
    for layer in model.layers:
        acc = np.broadcast_to(layer.bias, (*activations.shape[:-1], len(layer.bias))).copy()
        for tap, weight in enumerate(layer.weight.transpose(1, 0, 2)):
            offset = tap * layer.dilation
            if offset >= length:
                break
            act = activations[..., :length - offset, :].astype(np.int32)
            acc[..., offset:, :] += act @ weight.T.astype(np.int32)
        products = acc.astype(np.int64) * layer.multiplier
        magnitude = (np.abs(products) + (1 << (SHIFT - 1))) >> SHIFT
        rounded = np.where(products < 0, -magnitude, magnitude)
        activations = np.clip(rounded, 0 if layer.relu else -127, 127).astype(np.int8)
        outputs.append(activations)
    return tuple(outputs) if intermediate else activations


def export_layer(conv, input_scale: float, output_scale: float, relu: bool) -> Layer:
    weights = conv.weight.detach().cpu().numpy().astype(np.float64).transpose(0, 2, 1)[:, ::-1, :]
    biases = conv.bias.detach().cpu().numpy().astype(np.float64)
    weight_scale = max(float(np.abs(weights).max()) / 127, 1e-9)
    q_weights = np.clip(np.rint(weights / weight_scale), -127, 127).astype(np.int8)
    acc_scale = input_scale * weight_scale
    bias_values = np.rint(biases / acc_scale)
    bound = np.abs(bias_values) + np.abs(q_weights.astype(np.int64)).sum(axis=(1, 2)) * 127
    if np.max(bound) > np.iinfo(np.int32).max:
        raise OverflowError("Quantized convolution could overflow its INT32 accumulator.")
    mul = int(np.rint(acc_scale / output_scale * (1 << SHIFT)))
    if not 1 <= mul < 1 << 31:
        raise OverflowError("The requantization multiplier is out of range.")
    return Layer(q_weights.copy(), bias_values.astype(np.int32), mul, int(conv.dilation[0]), relu)
