"""Run benchmark firmware with the Unicorn Cortex-M CPU emulator."""
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess

from .arm import _export_cases, decode_benchmark
from .build import FIRMWARE, build_arm, symbol_addresses, tool

SYMBOLS = (
    "correction_begin", "correction_end", "bench_begin_count", "bench_end_count",
    "append_begin", "append_end", "bench_append_begin_count", "bench_append_end_count",
    "benchmark_finished", "bench_results",
)


def build_runner(destination: Path):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    wheel = find_spec("unicorn")
    if not wheel or not wheel.submodule_search_locations:
        raise RuntimeError("Install unicorn==2.1.4 from requirements.txt.")
    package = Path(next(iter(wheel.submodule_search_locations)))
    libraries = sorted((package / "lib").glob("libunicorn.so*"))
    if not libraries:
        raise RuntimeError("The Unicorn benchmark requires the Linux wheel with its C library.")
    cc = tool("cc", os.getenv("CC"))
    library = libraries[0].resolve()
    subprocess.run([
        cc, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", "-I", str(package / "include"),
        str(FIRMWARE / "cpu_runner.c"), str(library), f"-Wl,-rpath,{library.parent}", "-o", str(destination),
    ], check=True)
    return destination


def run_cpu(model, cases, horizon, output, mode_ids, runner, *, cache=8, target="cortex-m4", scalar=False, lock_kernel=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    changes = _export_cases(model, cases, horizon, output / "bench_cases.h", mode_ids)
    elf = build_arm(model, horizon, output, extended=True, cache=cache, target=target, scalar=scalar,
                    lock_kernel=lock_kernel)
    addresses = symbol_addresses(elf)
    binary = output / "cpu_results.bin"
    state = output / "cpu_state.bin"
    binary.unlink(missing_ok=True)
    state.unlink(missing_ok=True)
    subprocess.run([
        str(runner), str(elf), target, *(hex(addresses[symbol]) for symbol in SYMBOLS),
        str(20 * len(cases) * len(mode_ids)), str(binary),
        hex(addresses["bench_state_bytes"]), str(state),
    ], check=True, timeout=240)
    return decode_benchmark(binary, state, changes, mode_ids, True)
