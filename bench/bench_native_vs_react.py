"""Re-measurement harness: native tool-calling vs prompt-based (ReAct) tool calling, for the
from-scratch data-analyst agent in agent/, across the locally-pulled models.

This is measurement tooling, not part of the agent: it (re-)produces the pass-rate numbers
docs/RESULTS.md cites, on this exact code. Every
single run is recorded as one line of JSONL; nothing is averaged away silently.

Usage: (from the repo root)
    python3 bench/bench_native_vs_react.py [--runs 3] [--max-steps 6] [--out bench/results/data_analyst_bench.jsonl]
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN_PATH = os.path.join(HERE, "golden.yaml")

sys.path.insert(0, ROOT)
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(ROOT, "workspace"))

import yaml  # noqa: E402

from agent import tools  # noqa: E402
from agent.agent import run_agent  # noqa: E402
from agent.config import get_provider  # noqa: E402

MODELS = ["qwen3.5:4b", "qwen3:8b", "llama3.1:8b"]
MODES = ["native", "react"]

HEDGE_WORDS = re.compile(
    r"\b(i don'?t know|cannot determine|can'?t determine|unable to|not sure|unclear|"
    r"no (?:such|data|information)|does(?:n'?t| not) exist|not (?:be )?found|not applicable|"
    r"n/a|no such merchant|couldn'?t find|could not find|not available)\b",
    re.I,
)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_golden():
    with open(GOLDEN_PATH) as f:
        return yaml.safe_load(f)


def score(item, outcome, answer):
    """Returns (verdict, confident) where verdict in {"correct", "wrong", "no_answer"} and
    confident is only meaningful when verdict == "wrong" (True = confidently wrong / fabricated,
    False = the run visibly failed -- gave up, hit max_steps, or hedged in its own words)."""
    answer = (answer or "").strip()
    gave_up = outcome != "final"
    hedged = bool(HEDGE_WORDS.search(answer))
    empty = not answer

    if item.get("type") == "na":
        # Correct behaviour: abstain / say it doesn't exist, with no invented number attached.
        # A number in the *concluding* sentence always wins over a hedge phrase found earlier in
        # the same answer -- that's exactly the "confident wrong number" shape this checks for.
        if gave_up or empty:
            return "no_answer", False
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        last = sentences[-1] if sentences else answer
        if NUM_RE.search(last):
            return "wrong", True  # stated a concrete number as the answer -- fabricated
        if hedged:
            return "correct", None
        return "wrong", True

    if gave_up or empty:
        return "no_answer", False
    if hedged:
        return "no_answer", False  # explicit hedge instead of an answer -- a visible failure

    if item.get("type") == "number":
        nums = [float(n.replace(",", "")) for n in NUM_RE.findall(answer)]
        tol = item.get("tolerance", 0) + 1e-9
        target = float(item["answer"])
        if any(abs(n - target) <= tol for n in nums):
            return "correct", None
        return "wrong", True

    # string type
    target = str(item["answer"])
    if " " in target:
        ok = target.lower() in answer.lower()
    else:
        tokens = re.findall(r"[A-Za-z0-9_]+", answer)
        ok = any(t.lower() == target.lower() for t in tokens)
    return ("correct", None) if ok else ("wrong", True)


def run_one(provider_name, model, mode, item, run_idx, max_steps):
    os.environ["AGENT_MODEL"] = model
    provider = get_provider(provider_name)
    tools.SANDBOX_MODE = "subprocess"
    registry = tools.Registry(*tools.ALL_TOOLS)

    tool_calls_log = []

    def trace(kind, **info):
        if kind == "action":
            tool_calls_log.append({"name": info["name"], "args": info["args"]})

    t0 = time.time()
    try:
        result = run_agent(
            provider, item["question"], registry, mode=mode, max_steps=max_steps,
            confirm=lambda name, args: True, trace=trace,
        )
        error = None
    except Exception as e:  # noqa: BLE001 - a benchmark run must never crash the whole sweep
        result = {"outcome": "exception", "answer": "", "messages": []}
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    verdict, confident = score(item, result["outcome"], result["answer"])
    return {
        "model": model, "mode": mode, "question_id": item["id"], "run": run_idx,
        "outcome": result["outcome"], "answer": result["answer"],
        "verdict": verdict, "confident_wrong": confident,
        "num_steps": len(tool_calls_log), "tool_calls": tool_calls_log,
        "elapsed_s": round(elapsed, 2), "error": error,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--out", default=os.path.join(HERE, "results", "data_analyst_bench.jsonl"))
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--modes", nargs="*", default=MODES)
    p.add_argument("--questions", nargs="*", default=None, help="subset of question ids, default all")
    args = p.parse_args()

    golden = load_golden()
    if args.questions:
        golden = [g for g in golden if g["id"] in args.questions]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(args.models) * len(args.modes) * len(golden) * args.runs
    done = 0
    t_start = time.time()

    with open(args.out, "a") as f:
        for model in args.models:
            for mode in args.modes:
                for item in golden:
                    for run_idx in range(1, args.runs + 1):
                        rec = run_one("ollama", model, mode, item, run_idx, args.max_steps)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        done += 1
                        elapsed = time.time() - t_start
                        print(
                            f"[{done}/{total}] {model:12} {mode:6} {item['id']:9} run{run_idx} "
                            f"-> {rec['verdict']:9} ({rec['outcome']}, {rec['elapsed_s']}s) "
                            f"total_elapsed={elapsed:.0f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
