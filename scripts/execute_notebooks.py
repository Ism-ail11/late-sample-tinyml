#!/usr/bin/env python3
"""Execute the checked-in notebooks with a local, in-process Jupyter kernel."""
import argparse
import os
from pathlib import Path

import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TYPES = {"stream", "display_data", "execute_result", "error"}


def execute(notebook):
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    try:
        for index, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code":
                continue
            message_id = client.execute(cell.source, allow_stdin=False)
            reply = client.get_shell_msg(timeout=30)["content"]
            cell.execution_count = reply["execution_count"]
            cell.outputs = []
            while True:
                message = client.get_iopub_msg(timeout=30)
                parent = message["parent_header"].get("msg_id")
                if parent is not None and parent != message_id:
                    continue
                kind = message["msg_type"]
                content = message["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind in OUTPUT_TYPES:
                    fields = {key: value for key, value in content.items() if key != "transient"}
                    cell.outputs.append(nbformat.v4.new_output(kind, **fields))
            if reply["status"] != "ok":
                context = [out.get("text", "") for out in cell.outputs if out.output_type == "stream"]
                traceback = "\n".join(reply.get("traceback", []))
                raise RuntimeError(f"Cell {index} failed.\n{''.join(context)}\n{traceback}")
    finally:
        client.stop_channels()
        manager.shutdown_kernel()
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks")
    parser.add_argument("--match", help="Execute only notebook names containing this string.")
    args = parser.parse_args()
    sources = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.match:
        sources = [path for path in sources if args.match in path.name]
    if not sources:
        parser.error("No notebooks matched.")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    os.chdir(ROOT)
    for source in sources:
        print(f"Executing {source.name}", flush=True)
        notebook = execute(nbformat.read(source, as_version=4))
        nbformat.validate(notebook)
        nbformat.write(notebook, destination / source.name)
        print(f"Saved {source.name}", flush=True)


if __name__ == "__main__":
    main()
