import os
import sys
import typing

import pytest

from agent import tools
from agent.tools import Registry, tool


def test_schema_comes_from_the_signature():
    @tool()
    def lookup(name: str, limit: int = 5, exact: bool = False,
               kind: typing.Literal["a", "b"] = "a", score: float = 0.5):
        """Look something up.

        Only the first docstring line becomes the description."""

    fn = lookup.schema["function"]
    assert fn["name"] == "lookup"
    assert fn["description"] == "Look something up."
    props = fn["parameters"]["properties"]
    assert props["name"] == {"type": "string"}
    assert props["limit"] == {"type": "integer"}
    assert props["exact"] == {"type": "boolean"}
    assert props["score"] == {"type": "number"}
    assert props["kind"] == {"type": "string", "enum": ["a", "b"]}
    assert fn["parameters"]["required"] == ["name"]
    assert lookup.dangerous is False


def test_dangerous_flag():
    assert tools.write_file.dangerous and tools.run_python.dangerous
    assert not tools.read_file.dangerous
    r = Registry(*tools.ALL_TOOLS)
    assert r.is_dangerous("run_python") and not r.is_dangerous("read_file")
    assert not r.is_dangerous("nope")


def test_registry_never_raises():
    @tool()
    def add(a: int, b: int):
        """Add two ints."""
        return a + b

    @tool()
    def boom():
        """Always fails."""
        raise RuntimeError("kaput")

    r = Registry(add, boom)
    assert [s["function"]["name"] for s in r.schemas] == ["add", "boom"]
    assert r.run("add", {"a": 2, "b": 3}) == "5"
    assert r.run("add", {"a": "2", "b": "3"}) == "5"          # small models send strings
    assert r.run("add", {"a": 2, "b": 3, "junk": 1}) == "5"    # unknown args are dropped
    assert r.run("frobnicate", {}).startswith("Error: unknown tool 'frobnicate'")
    assert "missing required argument(s): b" in r.run("add", {"a": 1})
    assert "must be a JSON object" in r.run("add", "a=1")
    assert r.run("boom", {}) == "Error: RuntimeError: kaput"


@pytest.mark.parametrize("value,annotation,expected", [
    ("3", int, 3), ("2.5", float, 2.5), (7, str, "7"), ("true", bool, True),
    ("no", bool, False), ("abc", int, "abc"), (True, bool, True),
])
def test_coerce(value, annotation, expected):
    assert tools._coerce(value, annotation) == expected


def test_prompt_lines():
    lines = Registry(*tools.ALL_TOOLS).prompt_lines()
    assert lines[0].startswith("- read_file(path): Read a text file")
    assert lines[1].startswith("- write_file(path, content):")


def test_read_and_write_inside_the_workspace(workspace):
    assert tools.write_file("notes/a.txt", "hello") == "wrote 5 chars to notes/a.txt"
    assert (workspace / "notes" / "a.txt").read_text() == "hello"
    assert tools.read_file("notes/a.txt") == "hello"


def test_read_file_truncates_big_files(workspace):
    (workspace / "big.csv").write_text("x" * 5000)
    out = tools.read_file("big.csv")
    assert out.startswith("x" * 4000)
    assert "1000 chars truncated" in out


def test_read_missing_file_is_an_error_observation(workspace):
    r = Registry(tools.read_file)
    assert r.run("read_file", {"path": "nope.csv"}) == "Error: FileNotFoundError: no such file: nope.csv"


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "data/../../outside.txt"])
def test_jail_blocks_escapes(workspace, path):
    with pytest.raises(ValueError, match="outside the workspace"):
        tools.read_file(path)
    with pytest.raises(ValueError, match="outside the workspace"):
        tools.write_file(path, "x")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
def test_jail_blocks_symlink_escapes(workspace, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    os.symlink(secret, workspace / "link.txt")
    with pytest.raises(ValueError, match="outside the workspace"):
        tools.read_file("link.txt")


def test_run_python_runs_in_the_workspace(workspace):
    (workspace / "data" / "n.txt").write_text("41")
    out = tools.run_python("print(int(open('data/n.txt').read()) + 1)")
    assert out.strip() == "42"


def test_run_python_reports_errors(workspace):
    out = tools.run_python("raise ValueError('bad')")
    assert out.startswith("Error: exited 1:") and "ValueError: bad" in out
    assert tools.run_python("x = 1") == "(no output)"


def test_run_python_does_not_inherit_secrets(workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    out = tools.run_python("import os; print(os.environ.get('OPENAI_API_KEY', 'not visible'))")
    assert out.strip() == "not visible"


def test_run_python_timeout(workspace, monkeypatch):
    monkeypatch.setattr(tools, "RUN_PYTHON_TIMEOUT", 1)
    assert tools.run_python("import time; time.sleep(5)") == "Error: run_python timed out after 1s"


def test_unsafe_exec_leaks_and_subprocess_does_not(workspace, monkeypatch):
    from agent import sandbox_ladder
    monkeypatch.setenv("OPENAI_API_KEY", sandbox_ladder.DANGEROUS_ENV_VALUE)
    code = "import os\nresult = os.environ.get('OPENAI_API_KEY', 'not visible')\nprint(result)"
    assert sandbox_ladder.unsafe_exec(code) == sandbox_ladder.DANGEROUS_ENV_VALUE
    assert tools.run_python(code).strip() == "not visible"
