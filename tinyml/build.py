"""Export model headers and build native and bare-metal C targets."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .config import ROOT
from .quant import Int8Model, SHIFT

FIRMWARE = ROOT / "firmware"


def tool(name: str, override: str | None = None) -> str:
    requested = override or name
    executable = shutil.which(requested)
    if executable is None:
        raise FileNotFoundError(f"Required executable {requested!r} was not found. See README.md.")
    return executable


def _numbers(array: np.ndarray) -> str:
    values = list(map(str, array.ravel().tolist()))
    return ",\n    ".join(", ".join(values[i:i + 16]) for i in range(0, len(values), 16))


def export_headers(model: Int8Model, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    convs = model.layers[:3]
    if (len(model.layers) != 4 or any(x.weight.shape[1] != 3 or not x.relu or not 1 <= x.dilation <= 31 for x in convs)
            or model.layers[-1].relu or model.layers[-1].weight.shape[1] != 1 or convs[0].weight.shape[-1] != 3):
        raise ValueError("The firmware requires three three-tap causal layers and a pointwise score layer.")
    for index, layer in enumerate(model.layers):
        if index and layer.weight.shape[-1] != model.layers[index - 1].weight.shape[0]:
            raise ValueError("Adjacent layers have different channel counts.")
        bound = np.abs(layer.bias.astype(np.int64)) + np.abs(layer.weight.astype(np.int64)).sum((1, 2)) * 128
        if bound.max() > np.iinfo(np.int32).max:
            raise ValueError("Layer weights and bias could overflow exact INT32 accumulation.")
    widths = [layer.weight.shape[0] for layer in convs]
    dilations = [layer.dilation for layer in convs]
    configs = ["#ifndef MODEL_CONFIG_H", "#define MODEL_CONFIG_H"]
    configs.extend(f"#define TINY_C{i} {value}" for i, value in enumerate(widths))
    configs.extend(f"#define TINY_D{i} {value}" for i, value in enumerate(dilations))
    configs += [
        f"#define TINY_CLASSES {model.classes}", f"#define TINY_RF {1 + 2 * sum(dilations)}",
        f"#define TINY_SHIFT {SHIFT}", "#endif",
    ]
    (output / "model_config.h").write_text("\n".join(configs) + "\n", encoding="ascii")
    lines = [
        "#ifndef MODEL_PARAMS_H", "#define MODEL_PARAMS_H", "#include <stdint.h>", "",
        "typedef struct {",
        "    const int8_t *weight;", "    const int32_t *bias;", "    int32_t multiplier;",
        "    uint8_t dilation;", "    uint8_t taps;", "    uint8_t relu;",
        "} TinyLayerParams;", "",
    ]
    for index, layer in enumerate(model.layers):
        lines.append(f"static const int8_t weight{index}[] = {{\n    {_numbers(layer.weight)}\n}};")
        lines.append(f"static const int32_t bias{index}[] = {{\n    {_numbers(layer.bias)}\n}};")
    lines.append("static const TinyLayerParams tiny_layers[] = {")
    for index, layer in enumerate(model.layers):
        lines.append(f"    {{weight{index}, bias{index}, {layer.multiplier}, "
                     f"{layer.dilation}, {layer.weight.shape[1]}, {int(layer.relu)}}},")
    lines += ["};", "", "#endif"]
    (output / "model_params.h").write_text("\n".join(lines) + "\n", encoding="ascii")


def build_native(model: Int8Model, horizon: int, output: Path, *, extended: bool = False,
                 cache: int = 8) -> Path:
    export_headers(model, output)
    cc = tool("cc", os.environ.get("CC"))
    library = output / "libtinyml.so"
    subprocess.run([
        cc, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        f"-DTINY_HORIZON={horizon}", "-DTINY_COUNT_CALLS",
        *(["-DTINY_EXTENDED", f"-DTINY_ACC_FRAMES={cache}"] if extended else []), "-I", str(output),
        str(FIRMWARE / "tinyml.c"), str(FIRMWARE / "native.c"), "-o", str(library),
    ], check=True)
    return library


def symbol_addresses(elf: Path) -> dict[str, int]:
    nm = tool("arm-none-eabi-nm", os.environ.get("ARM_NM"))
    output = subprocess.check_output([nm, "--defined-only", str(elf)], text=True)
    symbols = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]+) \w (\w+)", line)
        if match:
            symbols[match[2]] = int(match[1], 16)
    return symbols


def build_arm(model: Int8Model, horizon: int, output: Path, *,
              extended: bool = False, scalar: bool = False, target: str = "cortex-m4",
              cache: int = 8, lock_kernel: bool = False) -> Path:
    export_headers(model, output)
    gcc = tool("arm-none-eabi-gcc", os.environ.get("ARM_GCC"))
    flags = [
        f"-mcpu={target}", "-mthumb", "-mfloat-abi=soft", "-ffreestanding",
        "-fno-builtin", "-ffunction-sections", "-fdata-sections", "-Wall", "-Wextra", "-Werror",
        "-std=c11", "-O3", "-g", "-fstack-usage", f"-DTINY_HORIZON={horizon}", "-I", str(output),
    ]
    if extended:
        flags.extend(["-DTINY_EXTENDED", f"-DTINY_ACC_FRAMES={cache}"])
    if scalar:
        flags.append("-DTINY_SCALAR_DOT")
    if lock_kernel:
        flags.append("-DTINY_LOCK_KERNEL")
    sources = ("startup.c", "tinyml.c", "renode_bench.c")
    for name in sources:
        obj = output / Path(name).with_suffix(".o")
        subprocess.run([gcc, *flags, "-c", str(FIRMWARE / name), "-o", str(obj)], check=True)
    elf = output / "tinyml.elf"
    objects = [str(output / Path(name).with_suffix(".o")) for name in sources]
    subprocess.run([
        gcc, *flags[:3], "-nostdlib", *(str(obj) for obj in objects),
        f"-Wl,-T,{FIRMWARE / 'stm32f4.ld'}", f"-Wl,-Map={output / 'tinyml.map'}",
        "-Wl,--gc-sections", "-Wl,--build-id=none", "-lgcc", "-o", str(elf),
    ], check=True)
    return elf
