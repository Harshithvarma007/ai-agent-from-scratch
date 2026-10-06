"""Replay a captured model trace through any loop step, so a rare real failure can be shown on
camera on demand. The model's replies come from the recording, in order; the tools still run
for real against the workspace.

    python steps/replay.py steps/traces/silent_reply.json steps/loop_step4.py  # before: blank answer
    python steps/replay.py steps/traces/silent_reply.json steps/loop_step6.py  # after: recovers

Run from the repo root. Traces are captured by bench/capture_traces.py from real runs of the real agent.
"""
import json
import runpy
import sys

from agent import llm

trace_path, step_path = sys.argv[1], sys.argv[2]
trace = json.load(open(trace_path))
replies = iter(trace["replies"])


def replayed_chat(*args, **kwargs):
    try:
        return next(replies)
    except StopIteration:
        print(f"\n[replay] end of recording: the real run stopped here ({trace['outcome']})")
        sys.exit(0)


llm.chat = replayed_chat
print(f"[replay] {trace_path}: {trace['model']}, recorded {trace['captured_at']}")
print(f"[replay] task: {trace['task']}\n")
sys.argv = [step_path, trace["task"]]
runpy.run_path(step_path, run_name="__main__")
