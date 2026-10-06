"""Chapter 9: the code-execution safety ladder, run standalone to see the difference on screen.

Run it with `python -m agent.sandbox_ladder`: rung 0 (exec()) leaks a "secret" env var into the model's tool output;
rung 1 (our subprocess sandbox, the one tools.py's run_python ships with by default) can't see
it at all, because its environment is scrubbed to almost nothing. Rungs 2-3 are described, not
built -- they're each one escape away from "not actually a sandbox" (smolagents' own docs say
this about their AST interpreter, which is why we don't build one). Rung 4 (Docker) is real and
wired to main.py's --sandbox flag; it just needs Docker installed to run.
"""
import os
import shutil
import subprocess

from . import tools

DANGEROUS_ENV_VALUE = "sk-totally-secret-demo-key"  # stands in for a real API key


def unsafe_exec(code: str) -> str:
    """Rung 0: exec() in-process. DO NOT USE -- shown once, to justify every rung above it."""
    local_vars = {}
    exec(code, {}, local_vars)  # noqa: S102 - the whole point of this function is "don't do this"
    return str(local_vars.get("result", ""))


def run_python_docker(code, workspace):
    """Rung 4: python:3.12-slim, no network, no privileges, 512MB cap. Flags are smolagents'
    own secure-execution recipe, which we borrow rather than re-derive."""
    if not shutil.which("docker"):
        prev, tools.SANDBOX_MODE = tools.SANDBOX_MODE, "subprocess"
        try:
            return "Note: --sandbox docker requested but Docker isn't installed; falling back.\n" + tools.run_python(code)
        finally:
            tools.SANDBOX_MODE = prev
    # python:3.12-slim has no pandas; build Dockerfile.sandbox and set AGENT_DOCKER_IMAGE to give
    # the model's pandas code what it needs (see the README's "Safety" section).
    image = os.environ.get("AGENT_DOCKER_IMAGE", "python:3.12-slim")
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "--pids-limit", "100",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "nobody",
           "-v", f"{workspace}:/work:ro", "-w", "/work", image, "python", "-c", code]
    try:
        proc = subprocess.run(cmd, timeout=20, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return "Error: run_python (docker) timed out after 20s"
    return proc.stdout[-4000:] if proc.returncode == 0 else f"Error: {proc.stderr[-2000:]}"


if __name__ == "__main__":
    leak_attempt = "import os\nresult = os.environ.get('OPENAI_API_KEY', 'not visible')\nprint(result)"
    os.environ["OPENAI_API_KEY"] = DANGEROUS_ENV_VALUE

    print("rung 0 (exec, bad)       :", unsafe_exec(leak_attempt))

    tools.SANDBOX_MODE = "subprocess"
    print("rung 1 (subprocess, ok)  :", tools.run_python(leak_attempt))

    print("rung 2 (resource limits) : mention only -- resource.setrlimit in preexec_fn, POSIX-only")
    print("rung 3 (AST allowlist)   : mention only -- smolagents' own docs call this 'not a real sandbox'")

    if shutil.which("docker"):
        print("rung 4 (docker)          :", run_python_docker(leak_attempt, os.path.abspath("workspace")))
    else:
        print("rung 4 (docker)          : Docker not installed on this machine -- flag exists, path documented above")
