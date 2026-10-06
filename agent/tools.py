"""@tool decorator -> JSON schema (from a function's own signature), a registry, and the
data-analyst tool set: `read_file` (load/preview the dataset), `write_file`, and `run_python`
(run pandas code against it).

The schema builder is the same idea as OpenAI's own "function_to_schema" cookbook pattern,
written from scratch: `inspect.signature` gives us names, type hints and defaults; the first
line of the docstring becomes the description the model reads.

This agent only ever touches a local dataset -- no calculator, no web search. Three tools is
enough to load a CSV, look at it, and compute an answer.
"""
import inspect
import os
import subprocess
import sys
import typing

# Everything the agent reads, writes or runs is jailed to this directory. Default: ./workspace,
# i.e. the repo's workspace/ when you run from the repo root (data lands in workspace/data/).
WORKSPACE = os.path.abspath(os.environ.get("AGENT_WORKSPACE", "workspace"))

_JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for_param(annotation):
    if typing.get_origin(annotation) is typing.Literal:
        return {"type": "string", "enum": list(typing.get_args(annotation))}
    return {"type": _JSON_TYPE.get(annotation, "string")}


def _coerce(value, annotation):
    """Small models hand back '3' instead of 3, or 'true' instead of true -- coerce, don't crash.
    (Ollama's own tool-calling examples do the same int() cast for the same reason.)"""
    if annotation is bool and isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if annotation in (int, float, str) and not isinstance(value, annotation):
        try:
            return annotation(value)
        except (TypeError, ValueError):
            return value
    return value


def tool(*, dangerous=False):
    """dangerous=True tools ask for y/N confirmation before running (see agent.py's confirm())."""
    def wrap(fn):
        props, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            props[name] = _schema_for_param(p.annotation)
            if p.default is inspect.Parameter.empty:
                required.append(name)
        doc = (fn.__doc__ or "").strip()
        fn.schema = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": doc.splitlines()[0] if doc else fn.__name__,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }
        fn.dangerous = dangerous
        return fn
    return wrap


class Registry:
    def __init__(self, *fns):
        self.fns = {fn.__name__: fn for fn in fns}

    @property
    def schemas(self):
        return [fn.schema for fn in self.fns.values()]

    def prompt_lines(self):
        """One line per tool, for the react-mode system prompt (no native tools= there)."""
        lines = []
        for fn in self.fns.values():
            params = ", ".join(fn.schema["function"]["parameters"]["properties"])
            lines.append(f"- {fn.__name__}({params}): {fn.schema['function']['description']}")
        return lines

    def is_dangerous(self, name):
        fn = self.fns.get(name)
        return bool(fn and getattr(fn, "dangerous", False))

    def run(self, name, args):
        """Never raises. Every failure becomes a string the model reads as its next observation."""
        if name not in self.fns:
            return f"Error: unknown tool '{name}'. Valid tools: {', '.join(self.fns)}"
        fn = self.fns[name]
        if not isinstance(args, dict):
            return f"Error: arguments for {name} must be a JSON object, got {args!r}"
        params = inspect.signature(fn).parameters
        missing = [p for p, v in params.items() if v.default is inspect.Parameter.empty and p not in args]
        if missing:
            return f"Error: {name} is missing required argument(s): {', '.join(missing)}"
        coerced = {k: _coerce(v, params[k].annotation) for k, v in args.items() if k in params}
        try:
            return str(fn(**coerced))
        except Exception as e:  # noqa: BLE001 - a tool must never crash the agent loop
            return f"Error: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ the 3 data-analyst tools

def _resolve_in_workspace(path):
    """Jail: resolve the path, then refuse anything that escapes WORKSPACE (../, symlinks, etc.)."""
    root = os.path.realpath(WORKSPACE)
    full = os.path.realpath(os.path.join(root, path))  # realpath, so a symlink can't point out
    if not (full == root or full.startswith(root + os.sep)):
        raise ValueError(f"path '{path}' is outside the workspace")
    return full


@tool()
def read_file(path: str) -> str:
    """Read a text file from the workspace directory and return its contents (large files are
    truncated -- use run_python + pandas to inspect a big CSV instead of reading it whole)."""
    full = _resolve_in_workspace(path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"no such file: {path}")
    with open(full, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > 4000:
        return content[:4000] + f"\n...[{len(content) - 4000} chars truncated -- file is large, " \
                                 "use run_python with pandas to analyze it instead of reading it whole]"
    return content


@tool(dangerous=True)
def write_file(path: str, content: str) -> str:
    """Write text to a file in the workspace directory, overwriting it if it exists."""
    full = _resolve_in_workspace(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


# set by main.py's --sandbox flag (or AGENT_SANDBOX); "docker" defers to sandbox_ladder.py
SANDBOX_MODE = os.environ.get("AGENT_SANDBOX", "subprocess")

RUN_PYTHON_TIMEOUT = int(os.environ.get("AGENT_RUN_PYTHON_TIMEOUT", "25"))  # pandas over a 138k-row CSV (read_csv + groupby/agg) needs more
                         # headroom than a one-line calculator snippet would


@tool(dangerous=True)
def run_python(code: str) -> str:
    """Run a Python snippet (pandas/numpy available) against files in the workspace and return
    its stdout -- print() whatever you want to see. Sandboxed: see sandbox_ladder.py."""
    if SANDBOX_MODE == "docker":
        from . import sandbox_ladder
        return sandbox_ladder.run_python_docker(code, WORKSPACE)
    # Level 1 of the safety ladder (chapter 9): a real subprocess, not exec() in-process, with
    # a timeout, a scrubbed environment, and a cwd jailed to the workspace. It does NOT stop
    # filesystem or network access from the code itself -- that's what the Docker rung is for.
    # NOTE: -I (isolated mode) drops PYTHONPATH and *user* site-packages but NOT a venv's own
    # site-packages, so this only sees pandas/numpy if sys.executable is a venv python that has
    # them installed directly (not `pip install --user`). If you hit ModuleNotFoundError for
    # pandas here, that's why -- see docs/RESULTS.md's gotchas.
    os.makedirs(WORKSPACE, exist_ok=True)
    env = {"PATH": "/usr/bin:/bin"}  # no inherited secrets, tokens or cwd-dependent env vars
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=WORKSPACE, env=env, timeout=RUN_PYTHON_TIMEOUT,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return f"Error: run_python timed out after {RUN_PYTHON_TIMEOUT}s"
    if proc.returncode != 0:
        return f"Error: exited {proc.returncode}: {proc.stderr[-2000:]}"
    return proc.stdout[-4000:] or "(no output)"


ALL_TOOLS = [read_file, write_file, run_python]


if __name__ == "__main__":  # python -m agent.tools
    r = Registry(*ALL_TOOLS)
    print("registered tools:", list(r.fns))
    print("read_file:  ", r.run("read_file", {"path": "data/acquirer_countries.csv"}))
    print("run_python: ", r.run("run_python", {
        "code": "import pandas as pd; df = pd.read_csv('data/payments.csv'); print(len(df))",
    }))
    print("unknown:    ", r.run("frobnicate", {}))
