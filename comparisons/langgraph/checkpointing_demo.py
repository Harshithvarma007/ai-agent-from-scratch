"""Tests, live, the two claims chapter 10's "fair case" section leans on hardest: that LangGraph's
checkpointing (a) actually survives a real process restart, not just an in-memory dict, and (b)
doubles as a human-in-the-loop approval gate for free, using the SAME mechanism.

This is not a chapter script -- it's the evidence for README.md's "what would this cost to
hand-roll" table. Run it and read the printed output; don't take the README's summary on faith.

Two phases, each a genuinely separate Python process invocation of this same file:
    python3 checkpointing_demo.py --phase 1     # runs until the first tool call, then exits
                                                  # (SIGTERM-style: no graceful shutdown, no
                                                  # in-memory state carried over)
    python3 checkpointing_demo.py --phase 2     # brand-new process, opens the same sqlite file,
                                                  # proves the paused state is recoverable from
                                                  # disk, then resumes to completion
    python3 checkpointing_demo.py --both        # runs both phases in one process for convenience
                                                  # (still uses two separate SqliteSaver connection
                                                  # objects, so the "recovered from disk" claim is
                                                  # still tested, just not across a real OS process)

Mechanism: interrupt_before=["tools"] is LangGraph's older, simpler human-in-the-loop primitive
(there's a newer dynamic interrupt()+Command form for pausing inside a tool itself -- see
README.md's citation; only the interrupt_before/after form is exercised here). Every time the
graph is about to run the tools node, it stops and persists state via the checkpointer instead of
proceeding. Resuming with input=None re-enters the graph from the last checkpoint and continues.
That "stop before a side-effecting step, wait for a human, then continue from exactly there" loop
IS the human-in-the-loop pattern; the process-restart test below is really testing the same
primitive twice.
"""
import argparse
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(ROOT, "workspace"))

DB_PATH = os.path.join(HERE, "results", "checkpoint_demo.sqlite")
THREAD_ID = "checkpointing-demo-thread"
QUESTION = "How many distinct merchants have at least one transaction in payments.csv?"
PROMPT = ("You are a data analyst agent. Data lives at data/payments.csv (use that exact path). "
          "Inspect the file with a tool before answering. Give an exact final answer as plain text.")


def build_agent(saver):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import ToolNode, create_react_agent

    from agent import tools as agent_tools

    llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama",
                      model="qwen3.5:4b", temperature=0.2)
    node = ToolNode(agent_tools.ALL_TOOLS, handle_tool_errors=True)
    return create_react_agent(llm, tools=node, prompt=PROMPT, checkpointer=saver,
                               interrupt_before=["tools"])


def describe_state(agent, config):
    state = agent.get_state(config)
    pending = state.next
    msgs = state.values.get("messages", [])
    print(f"    checkpoint has {len(msgs)} message(s); next node to run: {pending or '(none -- done)'}")
    if pending and msgs and getattr(msgs[-1], "tool_calls", None):
        for tc in msgs[-1].tool_calls:
            print(f"    -> PENDING APPROVAL: {tc['name']}({tc['args']})")
    return state


def run_until_done(agent, config, first_input):
    """Loop invoke(None) past every interrupt_before pause (auto-approving each tool call) until
    the graph actually finishes. This is what a real approval loop looks like: a human (or here,
    an auto-approve stand-in) reviews describe_state()'s printed pending call between iterations."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = agent.invoke(first_input, config=config)
        steps = 1
        while agent.get_state(config).next:
            describe_state(agent, config)
            result = agent.invoke(None, config=config)
            steps += 1
    return result, steps


def phase1():
    from langgraph.checkpoint.sqlite import SqliteSaver

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    print("=== PHASE 1: start the run, pause before the FIRST tool call, then exit the process ===")
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 20}
    with SqliteSaver.from_conn_string(DB_PATH) as saver:
        agent = build_agent(saver)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.invoke({"messages": [("user", QUESTION)]}, config=config)
        print("Paused before the first tool call. State as of this exact moment:")
        describe_state(agent, config)
    print(f"Checkpoint file on disk: {DB_PATH} ({os.path.getsize(DB_PATH)} bytes)")
    print("Process exits now with no in-memory state kept anywhere.\n")


def phase2():
    from langgraph.checkpoint.sqlite import SqliteSaver

    print("=== PHASE 2: brand-new process, same thread_id, resume from the sqlite file on disk ===")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} doesn't exist -- run --phase 1 first.")
        return
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 20}
    with SqliteSaver.from_conn_string(DB_PATH) as saver:
        agent = build_agent(saver)
        print("Recovered state from disk (no invoke() has been called in this process yet):")
        describe_state(agent, config)
        print("\nResuming (auto-approving each pending tool call as we go)...")
        result, steps = run_until_done(agent, config, first_input=None)
        print(f"\nRun completed after this process issued {steps} more invoke() call(s).")
        print("FINAL ANSWER:", result["messages"][-1].content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2])
    p.add_argument("--both", action="store_true")
    args = p.parse_args()

    if args.both:
        phase1()
        phase2()
    elif args.phase == 1:
        phase1()
    elif args.phase == 2:
        phase2()
    else:
        p.error("pass --phase 1, --phase 2, or --both")


if __name__ == "__main__":
    main()
