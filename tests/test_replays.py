"""End to end without an LLM: replay two recorded real model runs through the step files, with
the real tools running against the real dataset. Needs `python scripts/download_data.py`."""
import os
import subprocess
import sys

import pytest
from conftest import HAS_DATA, ROOT

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not HAS_DATA, reason="run python scripts/download_data.py first")]


def replay(trace, step):
    env = {**os.environ, "PYTHONPATH": ROOT, "AGENT_WORKSPACE": os.path.join(ROOT, "workspace")}
    proc = subprocess.run(
        [sys.executable, "steps/replay.py", f"steps/traces/{trace}", f"steps/{step}"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_silent_reply_before_the_fix():
    out = replay("silent_reply.json", "loop_step4.py")
    # step 4 treats "no text, no tool call" as the final answer: a blank answer
    assert "=== final ===\n\n" in out


def test_silent_reply_after_the_fix():
    out = replay("silent_reply.json", "loop_step6.py")
    assert "asking again" in out
    assert "gringotts' is based in country code **GB**" in out
    assert "[observation] ,acquirer,country_code" in out  # the real file was read


def test_repeat_guard_after_the_fix():
    out = replay("repeat_guard.json", "loop_step8.py")
    assert "[loop guard] refused to repeat run_python" in out


def test_sandbox_ladder():
    env = {**os.environ, "PYTHONPATH": ROOT}
    proc = subprocess.run([sys.executable, "-m", "agent.sandbox_ladder"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert "rung 0 (exec, bad)       : sk-totally-secret-demo-key" in proc.stdout
    assert "rung 1 (subprocess, ok)  : not visible" in proc.stdout
