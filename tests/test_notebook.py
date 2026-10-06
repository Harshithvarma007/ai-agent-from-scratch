"""The .ipynb is generated from notebook/build_notebook.py; fail if someone edited one and not
the other."""
import os
import runpy

import pytest
from conftest import ROOT

nbformat = pytest.importorskip("nbformat")


def test_notebook_matches_its_generator(tmp_path, monkeypatch):
    builder = runpy.run_path(os.path.join(ROOT, "notebook", "build_notebook.py"))
    fresh = builder["build"](tmp_path / "fresh.ipynb")
    committed = nbformat.read(os.path.join(ROOT, "notebook", "agent_from_scratch.ipynb"), as_version=4)
    assert [(c.id, c.cell_type, c.source) for c in fresh.cells] == \
           [(c.id, c.cell_type, c.source) for c in committed.cells], \
        "notebook out of sync: run `python notebook/build_notebook.py` then `python notebook/run_notebook.py`"


def test_notebook_code_fits_80_columns():
    builder = runpy.run_path(os.path.join(ROOT, "notebook", "build_notebook.py"))
    too_long = [(c.id, line) for c in builder["CELLS"] if c.cell_type == "code"
                for line in c.source.splitlines() if len(line) > 80]
    assert not too_long, too_long
