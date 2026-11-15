"""Check Renode's instruction counter with an 11-instruction Thumb program."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .build import FIRMWARE, symbol_addresses, tool

EXPECTED = 11


def check() -> dict:
    gcc = tool("arm-none-eabi-gcc", os.environ.get("ARM_GCC"))
    renode = tool("renode", os.environ.get("RENODE"))
    with tempfile.TemporaryDirectory(prefix="tinyml-counter-") as tmp:
        root = Path(tmp)
        elf = root / "counter.elf"
        subprocess.run([
            gcc, "-mcpu=cortex-m4", "-mthumb", "-nostdlib", "-Wl,--build-id=none",
            f"-Wl,-T,{FIRMWARE / 'stm32f4.ld'}", str(FIRMWARE / "counter_check.S"), "-o", str(elf),
        ], check=True)
        addresses = symbol_addresses(elf)
        (root / "cortex_m4.repl").write_bytes((FIRMWARE / "cortex_m4.repl").read_bytes())
        (root / "counter.resc").write_text("\n".join([
            'mach create "counter"',
            "machine LoadPlatformDescription @cortex_m4.repl",
            "sysbus LoadELF @counter.elf",
            f'sysbus.cpu AddHook 0x{addresses["counter_begin"]:08x} "print \'COUNT_BEGIN %d\' % cpu.ExecutedInstructions"',
            f'sysbus.cpu AddHook 0x{addresses["counter_end"]:08x} "print \'COUNT_END %d\' % cpu.ExecutedInstructions"',
            'emulation RunFor "0.02"',
            "q", "",
        ]), encoding="utf-8")
        result = subprocess.run([renode, "--disable-gui", "--console", "--plain", "counter.resc"],
                                cwd=root, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                check=True, timeout=30)
        log = result.stdout + result.stderr
        match_begin = re.findall(r"COUNT_BEGIN (\d+)", log)
        match_end = re.findall(r"COUNT_END (\d+)", log)
        if not match_begin or not match_end:
            raise RuntimeError(f"No counter markers in the Renode transcript.\n{log}")
        observed = int(match_end[0]) - int(match_begin[0])
        if observed != EXPECTED:
            raise AssertionError(f"Renode counted {observed} instructions, expected {EXPECTED}.")
    return {
        "expected": EXPECTED,
        "observed": observed,
        "compiler": subprocess.check_output([gcc, "--version"], text=True).splitlines()[0],
        "simulator": subprocess.check_output([renode, "--version"], text=True).splitlines()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
