"""Run the benchmark in Renode and measure the executed Cortex-M4 instructions."""
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np

from .build import FIRMWARE, build_arm, symbol_addresses, tool
from .protocol import ARRIVAL, MODES, TINY_EXPIRED, TINY_OK, Case, sensor_input
from .quant import Int8Model, dense_inference


def _c_vector(values) -> str:
    return "{" + ", ".join(map(str, values)) + "}"


def _c_flat(values: np.ndarray) -> str:
    vector = values.ravel().tolist()
    return "{\n    " + ",\n    ".join(", ".join(map(str, vector[i:i+24]))
                                for i in range(0, len(vector), 24)) + "\n}"


def _export_cases(model: Int8Model, cases: list[Case], horizon: int, target: Path,
                  mode_ids: tuple[int, ...] | None = None) -> list[int]:
    declarations = ["/* Generated input and oracle scores. */"]
    entries = []
    changes = []
    dense = {}
    for index, case in enumerate(cases):
        if case.clip.clip_id not in dense:
            clean = sensor_input(model, case.clip)[:ARRIVAL + 1]
            dense[case.clip.clip_id] = clean, dense_inference(model, clean)
        clean, clean_scores = dense[case.clip.clip_id]
        observed = case.provisional(clean)
        provisional = dense_inference(model, observed)
        status = TINY_EXPIRED if case.max_delay > horizon else TINY_OK
        latest = slice(ARRIVAL, ARRIVAL - horizon - 1, -1)
        before = provisional[latest]
        after = (clean_scores if status == TINY_OK else provisional)[latest]
        changed = int(np.any(before != after, axis=1).sum())
        changes.append(changed)
        rows = ",\n    ".join(_c_vector(row.tolist()) for row in observed)
        declarations.append(f"static const int8_t input{index}[][TINY_INPUTS] = {{\n    {rows}\n}};")
        declarations.append(f"static const int8_t before{index}[] = {_c_flat(before)};")
        declarations.append(f"static const int8_t after{index}[] = {_c_flat(after)};")
        updates = [f"{{{ARRIVAL - p}, {_c_vector(clean[p].tolist())}}}" for p in case.positions]
        declarations.append(f"static const TinyUpdate updates{index}[] = {{{', '.join(updates)}}};")
        entries.append(f"    {{input{index}, updates{index}, before{index}, after{index}, "
                       f"{ARRIVAL + 1}, {len(updates)}, {status}}}")
    declarations += ["static const BenchCase bench_cases[] = {", ",\n".join(entries), "};"]
    if mode_ids is not None:
        modes = ", ".join(map(str, mode_ids))
        declarations += ["#define TINY_BENCH_SUBSET 1", f"static const TinyMode bench_modes[] = {{{modes}}};"]
    target.write_text("\n".join(declarations) + "\n", encoding="ascii")
    return changes


def _hooks(elf: Path, output: Path, cases: int, seconds: float, mode_count: int, extended: bool) -> str:
    addresses = symbol_addresses(elf)

    def reg(symbol: str) -> str:
        return f"0x{addresses[symbol]:08x}"

    def literal(name: str) -> str:
        return repr((output / name).as_posix()).replace('"', '\\"')

    payload_bytes = cases * mode_count * (20 if extended else 16)
    append_hooks = []
    if extended:
        for marker in ("begin", "end"):
            append_hooks.append(
                f'sysbus.cpu AddHook {reg(f"append_{marker}")} '
                f'"machine.SystemBus.WriteDoubleWord({reg(f"bench_append_{marker}_count")}, '
                'cpu.ExecutedInstructions & 0xffffffff)"'
            )
    finish = (
        "from System.IO import File; "
        f"File.WriteAllBytes({literal('renode_results.bin')}, "
        f"machine.SystemBus.ReadBytes({reg('bench_results')}, {payload_bytes})); "
        f"File.WriteAllBytes({literal('renode_state.bin')}, "
        f"machine.SystemBus.ReadBytes({reg('bench_state_bytes')}, 4))"
    )
    return "\n".join([
        'mach create "tinyml"',
        'machine LoadPlatformDescription @cortex_m4.repl',
        'sysbus LoadELF @tinyml.elf',
        f'sysbus.cpu AddHook {reg("correction_begin")} '
        f'"machine.SystemBus.WriteDoubleWord({reg("bench_begin_count")}, cpu.ExecutedInstructions & 0xffffffff)"',
        f'sysbus.cpu AddHook {reg("correction_end")} '
        f'"machine.SystemBus.WriteDoubleWord({reg("bench_end_count")}, cpu.ExecutedInstructions & 0xffffffff)"',
        *append_hooks,
        f'sysbus.cpu AddHook {reg("benchmark_finished")} "{finish}"',
        f'emulation RunFor "{seconds}"',
        "q", "",
    ])


def run_arm(model: Int8Model, cases: list[Case], horizon: int, output: Path, *,
            extended: bool = False, mode_ids: tuple[int, ...] | None = None, cache: int = 8,
            target: str = "cortex-m4", scalar: bool = False) -> list[dict]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sandbox = output / "bench_cases.h"
    if extended and mode_ids is None:
        raise ValueError("Provide the benchmark mode IDs for an extended build.")
    mode_count = len(MODES) if mode_ids is None else len(mode_ids)
    if not cases or not mode_count:
        raise ValueError("A benchmark requires at least one trial and method.")
    changes = _export_cases(model, cases, horizon, sandbox, mode_ids)
    elf = build_arm(model, horizon, output, extended=extended, cache=cache, target=target, scalar=scalar)
    binary = output / "renode_results.bin"
    state = output / "renode_state.bin"
    replay = output / "benchmark.resc"
    if not extended:
        durations = (max(1, (len(cases) + 18) // 19),)
    else:
        estimate = max(1, len(cases) / 18)
        durations = (estimate, 2 * estimate, 4 * estimate)
    layout = (FIRMWARE / "cortex_m4.repl").read_text().replace('cpuType: "cortex-m4"', f'cpuType: "{target}"')
    (output / "cortex_m4.repl").write_text(layout, encoding="utf-8")
    renode = tool("renode", os.environ.get("RENODE"))
    transcript = output / "renode.log"
    for seconds in durations:
        binary.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
        replay.write_text(_hooks(elf, output, len(cases), seconds, mode_count, extended), encoding="utf-8")
        with transcript.open("w", encoding="utf-8") as handle:
            process = subprocess.run([renode, "--disable-gui", "--console", "--plain", replay.name],
                                     cwd=output, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                     check=False, timeout=300)
        log = transcript.read_text(errors="replace")
        if process.returncode or re.search(r"(Traceback|error executing|Python error|CPU abort|HardFault)", log, re.I):
            raise RuntimeError(f"Renode logged an execution error. See {transcript}.")
        if binary.is_file() and state.is_file():
            break
    else:
        raise RuntimeError(f"Renode did not complete {output.name}. See {transcript}.")
    return decode_benchmark(binary, state, changes, tuple(range(mode_count)) if mode_ids is None else mode_ids, extended)


def decode_benchmark(binary: Path, state: Path, changes: list[int], mode_ids: tuple[int, ...],
                     extended: bool) -> list[dict]:
    mode_count = len(mode_ids)
    cases = len(changes)
    raw = binary.read_bytes()
    if len(raw) != cases * mode_count * (20 if extended else 16):
        raise RuntimeError(f"Incomplete Renode results in {binary}.")
    state_bytes, = struct.unpack("<I", state.read_bytes())
    rows = []
    for index, values in enumerate(struct.iter_unpack("<5I" if extended else "<4I", raw)):
        work, errors, stack, checksum = values[:4]
        case_index, ordinal = divmod(index, mode_count)
        mode_id = mode_ids[ordinal]
        if work <= 0 or errors or stack >= 1024 or (extended and values[4] <= 0):
            raise AssertionError(f"Invalid score, counter, or stack watermark in case {case_index}, mode {mode_id}.")
        rows.append({
            "case_index": case_index, "mode_id": mode_id, "instructions": work, "mismatches": errors,
            "stack_bytes": stack, "state_bytes": state_bytes, "checksum": checksum,
            "revised_timestamps": changes[case_index],
            **({"append_instructions": values[4]} if extended else {}),
        })
    return rows
