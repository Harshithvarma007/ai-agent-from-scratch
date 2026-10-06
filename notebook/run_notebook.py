"""Executes agent_from_scratch.ipynb top to bottom in a fresh kernel and saves the outputs.

This is the "Restart & Run All" rule made mechanical: the outputs that ship (and that the video
shows) always come from one clean, in-order run. Exits non-zero if any cell errors.

    python notebook/run_notebook.py                       # local Ollama (default model)
    AGENT_PROVIDER=gemini GEMINI_API_KEY=... python notebook/run_notebook.py
    NOTEBOOK_KERNEL=agent-notebook python notebook/run_notebook.py   # a named Jupyter kernel
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
NB = HERE / "agent_from_scratch.ipynb"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name=os.environ.get("NOTEBOOK_KERNEL", "python3"), resources={"metadata": {"path": str(HERE)}})
t0 = time.time()
try:
    client.execute()
except CellExecutionError as e:
    nbformat.write(nb, NB)
    print(f"FAILED after {time.time() - t0:.0f}s\n{e}", file=sys.stderr)
    sys.exit(1)
nbformat.write(nb, NB)
print(f"ok: {sum(c.cell_type == 'code' for c in nb.cells)} code cells ran clean in {time.time() - t0:.0f}s -> {NB.name}")
