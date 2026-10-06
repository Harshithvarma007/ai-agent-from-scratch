"""Chapter 10's built (not just researched) LangChain comparison.

Wires up langchain-experimental's create_pandas_dataframe_agent against the *same* dataset
(workspace/data/) and the *same* local Ollama model(s) as our from-scratch agent, then
runs it through the *same* golden-set questions (bench/golden.yaml), scored by the exact same
score() function bench/bench_native_vs_react.py uses, so both sides are graded identically.

Confirms two things live, by reading + exercising the installed package (not by citing docs
secondhand):
  1. create_pandas_dataframe_agent's default `agent_type` -- check with --check-only.
  2. allow_dangerous_code=True is still required -- same check.

Usage: (from the repo root, with comparisons/langchain/requirements.txt installed)
    python comparisons/langchain/compare.py --check-only
    python comparisons/langchain/compare.py --runs 3 --models qwen3.5:4b \
        --agent-types zero-shot-react-description tool-calling
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(ROOT, "workspace", "data")
OLLAMA_BASE_URL = "http://localhost:11434/v1"

sys.path.insert(0, os.path.join(ROOT, "bench"))
from bench_native_vs_react import load_golden, score  # noqa: E402


def check_allow_dangerous_code():
    """Does create_pandas_dataframe_agent still refuse to run without
    allow_dangerous_code=True? Confirmed live, not cited secondhand."""
    import pandas as pd
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model="qwen3.5:4b")
    df = pd.DataFrame({"a": [1, 2, 3]})
    try:
        create_pandas_dataframe_agent(llm, df)  # allow_dangerous_code defaults to False
        return "DID NOT RAISE -- allow_dangerous_code is no longer required (unexpected!)"
    except ValueError as e:
        return f"ValueError raised as expected: {e}"


def load_dataframes():
    import json as _json

    import pandas as pd

    payments = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
    merchant_category_codes = pd.read_csv(os.path.join(DATA_DIR, "merchant_category_codes.csv"))
    acquirer_countries = pd.read_csv(os.path.join(DATA_DIR, "acquirer_countries.csv"))
    merchant_data = pd.json_normalize(_json.load(open(os.path.join(DATA_DIR, "merchant_data.json"))))
    # Same 4 files our from-scratch agent's workspace has, so neither side has an information
    # advantage.
    return [payments, merchant_data, merchant_category_codes, acquirer_countries]


def make_agent(model, agent_type, dfs):
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model=model, temperature=0.2)
    return create_pandas_dataframe_agent(
        llm, dfs, agent_type=agent_type, allow_dangerous_code=True,
        verbose=False, max_iterations=8, early_stopping_method="force",
    )


def run_one(model, agent_type, agent, item, run_idx):
    t0 = time.time()
    error = None
    answer = ""
    outcome = "final"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = agent.invoke({"input": item["question"]})
        answer = result.get("output", "") if isinstance(result, dict) else str(result)
        # AgentExecutor's own early_stopping_method="force" returns a fixed string instead of
        # raising -- LangChain's own version of our "max_steps" outcome (a visible give-up), and
        # it must be scored as no_answer, not as a stated-but-wrong answer.
        if answer.strip().lower().startswith("agent stopped due to"):
            outcome = "max_iterations"
    except Exception as e:  # noqa: BLE001 - langchain ReAct parsing errors are common, don't crash the sweep
        error = f"{type(e).__name__}: {e}"
        outcome = "exception"
    elapsed = time.time() - t0

    verdict, confident = score(item, outcome, answer)
    return {
        "model": model, "agent_type": agent_type, "question_id": item["id"], "run": run_idx,
        "outcome": outcome, "answer": answer, "verdict": verdict, "confident_wrong": confident,
        "elapsed_s": round(elapsed, 2), "error": error,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--models", nargs="*", default=["qwen3.5:4b"])
    p.add_argument("--agent-types", nargs="*",
                    default=["zero-shot-react-description", "tool-calling"])
    p.add_argument("--questions", nargs="*", default=None)
    p.add_argument("--out", default=os.path.join(HERE, "results", "langchain_bench.jsonl"))
    p.add_argument("--check-only", action="store_true",
                    help="just run check_allow_dangerous_code() and exit")
    args = p.parse_args()

    if args.check_only:
        print(check_allow_dangerous_code())
        return

    print("allow_dangerous_code check:", check_allow_dangerous_code())

    golden = load_golden()
    if args.questions:
        golden = [g for g in golden if g["id"] in args.questions]

    dfs = load_dataframes()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total = len(args.models) * len(args.agent_types) * len(golden) * args.runs
    done = 0
    t_start = time.time()

    with open(args.out, "a") as f:
        for model in args.models:
            for agent_type in args.agent_types:
                for item in golden:
                    # Fresh agent per question: PythonAstREPLTool's locals persist across calls
                    # inside one agent instance, which could let one question's leftover
                    # variables leak into the next -- rebuilding avoids that confound.
                    for run_idx in range(1, args.runs + 1):
                        agent = make_agent(model, agent_type, dfs)
                        rec = run_one(model, agent_type, agent, item, run_idx)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        done += 1
                        elapsed = time.time() - t_start
                        print(
                            f"[{done}/{total}] {model:12} {agent_type:26} {item['id']:9} "
                            f"run{run_idx} -> {rec['verdict']:9} ({rec['outcome']}, "
                            f"{rec['elapsed_s']}s) total_elapsed={elapsed:.0f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
