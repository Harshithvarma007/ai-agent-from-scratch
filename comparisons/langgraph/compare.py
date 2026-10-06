"""Chapter 10's built (not just researched) LangGraph comparison.

Wires up langgraph.prebuilt.create_react_agent against our OWN agent/tools.py tools (the same
subprocess-sandboxed read_file/write_file/run_python from chapter 6, completely unchanged) and the
same local Ollama model(s) as comparisons/langchain/, then runs it through the same golden-set
questions (bench/golden.yaml), scored by the exact same score() function bench/
bench_native_vs_react.py and comparisons/langchain/compare.py both use, so all three sides
(from-scratch, LangChain, LangGraph) are graded identically.

Confirmed live, not cited secondhand (see README.md for the full findings and exact tracebacks):
  1. create_react_agent's default ToolNode error handling re-raises any exception that isn't its
     own ToolInvocationError -- an ordinary tool bug (e.g. a bad path arg) crashes the whole run.
     --check-only reproduces this with a single deliberately-bad tool call.
  2. handle_tool_errors=True (NOT the default) is what makes a benchmark sweep survive that.

Usage: (from the repo root, with comparisons/langgraph/requirements.txt installed)
    python comparisons/langgraph/compare.py --check-only
    python comparisons/langgraph/compare.py --runs 3 --models qwen3.5:4b
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLLAMA_BASE_URL = "http://localhost:11434/v1"
RECURSION_LIMIT = 20  # ~8 tool-call round trips (agent-node + tools-node per trip), a comparable
                      # budget to comparisons/langchain's max_iterations=8 and our own max_steps=6-10

sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, ROOT)
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(ROOT, "workspace"))

from bench_native_vs_react import load_golden, score  # noqa: E402


def check_default_tool_error_handling():
    """Does create_react_agent's default ToolNode still crash the whole run on an ordinary tool
    exception (not a validation error)? Confirmed live, not cited secondhand -- reproduced on the
    first and second attempt when writing this file (see README.md's full traceback)."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from agent import tools as agent_tools

    llm = ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model="qwen3.5:4b", temperature=0.2)
    # A plain list of tools -> create_react_agent builds a ToolNode with its own default
    # handle_tool_errors, which only catches its own ToolInvocationError (bad/missing args),
    # not exceptions your tool function itself raises.
    agent = create_react_agent(llm, tools=agent_tools.ALL_TOOLS,
                                prompt="You are a data analyst agent.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.invoke({"messages": [("user", "Read the file 'nonexistent_file_xyz.csv'.")]},
                          config={"recursion_limit": RECURSION_LIMIT})
        return "DID NOT RAISE -- default ToolNode error handling changed (unexpected!)"
    except Exception as e:  # noqa: BLE001 - this IS the check
        return f"{type(e).__name__} propagated out of agent.invoke() as expected: {e}"


def make_agent(model):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import ToolNode, create_react_agent

    from agent import tools as agent_tools

    llm = ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model=model, temperature=0.2)
    # handle_tool_errors=True is NOT the default -- see check_default_tool_error_handling() and
    # README.md. Without it this sweep would crash out (not "no_answer", a raised exception)
    # every time the model passes a bad tool arg, the same category of finding as LangChain's
    # handle_parsing_errors=False default.
    node = ToolNode(agent_tools.ALL_TOOLS, handle_tool_errors=True)
    prompt = (
        "You are a data analyst agent. The workspace contains:\n"
        "- data/payments.csv: one row per transaction\n"
        "- data/merchant_data.json: one JSON object per merchant, keyed by 'merchant'\n"
        "- data/merchant_category_codes.csv: MCC code -> description lookup\n"
        "- data/acquirer_countries.csv: acquirer -> country_code lookup\n"
        "You don't know the exact column names yet -- inspect a file (read_file, or run_python "
        "with pandas, e.g. df.columns / df.head()) before you compute an answer. Give an exact "
        "final answer (a number or short string), not a description of how you'd compute it. If "
        "the question is not answerable from this data (e.g. it asks about something that "
        "doesn't exist in these files), say so explicitly instead of guessing a number."
    )
    return create_react_agent(llm, tools=node, prompt=prompt)


def run_one(model, agent, item, run_idx):
    from langgraph.errors import GraphRecursionError

    t0 = time.time()
    error = None
    answer = ""
    outcome = "final"
    num_steps = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = agent.invoke({"messages": [("user", item["question"])]},
                                   config={"recursion_limit": RECURSION_LIMIT})
        messages = result["messages"]
        num_steps = sum(1 for m in messages if type(m).__name__ == "ToolMessage")
        answer = messages[-1].content or ""
        if not answer.strip():
            # Empty final content, no further tool call -- the same "model went silent mid-turn"
            # shape we documented for our own from-scratch agent's native mode. Not a crash,
            # not a real answer either.
            outcome = "gave_up"
    except GraphRecursionError:
        # LangGraph's own version of AgentExecutor's early_stopping_method="force" / our own
        # max_steps outcome -- a visible give-up, not a fabrication. Must be scored as no_answer.
        outcome = "max_steps"
    except Exception as e:  # noqa: BLE001 - a benchmark run must never crash the whole sweep
        error = f"{type(e).__name__}: {e}"
        outcome = "exception"
    elapsed = time.time() - t0

    verdict, confident = score(item, outcome, answer)
    return {
        "model": model, "question_id": item["id"], "run": run_idx,
        "outcome": outcome, "answer": answer, "verdict": verdict, "confident_wrong": confident,
        "num_steps": num_steps, "elapsed_s": round(elapsed, 2), "error": error,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--models", nargs="*", default=["qwen3.5:4b"])
    p.add_argument("--questions", nargs="*", default=None)
    p.add_argument("--out", default=os.path.join(HERE, "results", "langgraph_bench.jsonl"))
    p.add_argument("--check-only", action="store_true",
                    help="just run check_default_tool_error_handling() and exit")
    args = p.parse_args()

    if args.check_only:
        print(check_default_tool_error_handling())
        return

    print("default tool-error-handling check:", check_default_tool_error_handling())

    golden = load_golden()
    if args.questions:
        golden = [g for g in golden if g["id"] in args.questions]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(args.models) * len(golden) * args.runs
    done = 0
    t_start = time.time()

    with open(args.out, "a") as f:
        for model in args.models:
            # One agent per model, reused across questions and runs: unlike PythonAstREPLTool's
            # in-process REPL (whose .locals persist across calls inside one agent instance),
            # our run_python shells out to a fresh subprocess every call, so there is no
            # cross-question state-leak risk here -- confirmed by inspecting agent/tools.py,
            # not assumed.
            agent = make_agent(model)
            for item in golden:
                for run_idx in range(1, args.runs + 1):
                    rec = run_one(model, agent, item, run_idx)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done += 1
                    elapsed = time.time() - t_start
                    print(
                        f"[{done}/{total}] {model:12} {item['id']:9} run{run_idx} "
                        f"-> {rec['verdict']:9} ({rec['outcome']}, {rec['elapsed_s']}s) "
                        f"total_elapsed={elapsed:.0f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
