"""Capture real, naturally-occurring loop failures for Chapter 5's steps 6-8, so they can be
replayed on demand instead of hoped for live.

Runs the real agent (agent.run_agent, native mode) over golden.yaml, recording every raw model
reply, until it has seen each of:
  silent_reply    -- a reply with no text and no tool call (step 6)
  text_tool_call  -- a tool call written as plain text in `content` (step 7)
  repeat_guard    -- the same tool + args requested past REPEAT_THRESHOLD (step 8)

Each capture is saved to agent/traces/<kind>.json with the task, model and the recorded replies,
ready for agent/replay.py. Nothing is edited or staged: these are real replies, in order.

Usage: (from the repo root)
    .venv/bin/python bench/capture_traces.py [--max-runs 60] [--models qwen3.5:4b,llama3.1:8b]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "steps", "traces")

sys.path.insert(0, ROOT)
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(ROOT, "workspace"))

import yaml  # noqa: E402

from agent import llm  # noqa: E402
from agent.agent import run_agent  # noqa: E402
from agent.config import get_provider  # noqa: E402
from agent.tools import ALL_TOOLS, Registry  # noqa: E402

KINDS = ["silent_reply", "text_tool_call", "repeat_guard"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=60)
    p.add_argument("--models", default="qwen3.5:4b,qwen3:8b,llama3.1:8b")
    p.add_argument("--max-steps", type=int, default=10)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    found = {k for k in KINDS if os.path.exists(os.path.join(OUT_DIR, f"{k}.json"))}
    questions = yaml.safe_load(open(os.path.join(HERE, "golden.yaml")))
    real_chat = llm.chat
    runs = 0

    while found != set(KINDS) and runs < args.max_runs:
        for model in args.models.split(","):
            for q in questions:
                if found == set(KINDS) or runs >= args.max_runs:
                    break
                os.environ["AGENT_MODEL"] = model
                provider = get_provider("ollama")
                replies, events = [], []

                def recording_chat(*a, **k):
                    msg = real_chat(*a, **k)
                    replies.append(msg)
                    return msg

                def trace(kind, **info):
                    events.append({"kind": kind, **{k: v for k, v in info.items() if k != "result"}})

                llm.chat = recording_chat
                t0 = time.time()
                try:
                    result = run_agent(provider, q["question"], Registry(*ALL_TOOLS), mode="native",
                                       max_steps=args.max_steps, trace=trace)
                except Exception as e:  # noqa: BLE001 - keep sweeping
                    result = {"outcome": "error", "answer": str(e)}
                finally:
                    llm.chat = real_chat
                runs += 1

                hits = set()
                if any(not (m.get("content") or "").strip() and not m.get("tool_calls") for m in replies):
                    hits.add("silent_reply")
                if any(e["kind"] == "rescued_tool_call_from_text" for e in events):
                    hits.add("text_tool_call")
                if any(e["kind"] == "repeat_guard" for e in events):
                    hits.add("repeat_guard")
                print(f"[{runs}] {model} {q['id']} {result['outcome']} {time.time() - t0:.0f}s "
                      f"hits={sorted(hits) or '-'}", flush=True)

                for kind in hits - found:
                    with open(os.path.join(OUT_DIR, f"{kind}.json"), "w") as f:
                        json.dump({"kind": kind, "provider": "ollama", "model": model,
                                   "question_id": q["id"], "task": q["question"],
                                   "captured_at": time.strftime("%Y-%m-%d %H:%M"),
                                   "outcome": result["outcome"], "answer": result["answer"],
                                   "events": events, "replies": replies}, f, indent=2, default=str)
                    found.add(kind)
                    print(f"    saved traces/{kind}.json", flush=True)

    print("found:", sorted(found), "missing:", sorted(set(KINDS) - found))


if __name__ == "__main__":
    main()
