"""Step 8 of 8: when it repeats itself. Same tool, same args, three times -> refuse, say why.

This is the finished loop: functionally the native branch of agent.py's run_agent(), minus the
prompt-based (react) branch and history trimming.
"""
import json
import os
import sys

from agent import llm
from agent.agent import DATA_CONTEXT, extract_json, truncate
from agent.config import get_provider
from agent.tools import Registry, read_file, run_python

MAX_STEPS = 10
MAX_CONSECUTIVE_ERRORS = 3
REPEAT_THRESHOLD = 3

provider = get_provider(os.environ.get("AGENT_PROVIDER", "ollama"))
registry = Registry(read_file, run_python)


def execute(name, args, seen_calls):
    key = (name, json.dumps(args, sort_keys=True, default=str))
    seen_calls[key] = seen_calls.get(key, 0) + 1
    if seen_calls[key] > REPEAT_THRESHOLD:
        print(f"[loop guard] refused to repeat {name}({args})")
        return "Error: you already called this exact tool with these exact arguments. Try something else."
    result = registry.run(name, args)
    print(f"[action] {name}({args})")
    print(f"[observation] {result[:300]}")
    return truncate(result)


def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    consecutive_errors = 0
    seen_calls = {}
    for step in range(MAX_STEPS):
        msg = llm.chat(provider, messages, tools=registry.schemas)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:300]}")

        if not calls:
            rescued = extract_json(content)
            if rescued and "tool" in rescued:
                print("[note] tool call recovered from plain text, not tool_calls")
                calls = [{"function": {"name": rescued["tool"], "arguments": rescued.get("args", {})}}]
            elif content.strip():
                return content
            else:
                print("[error] empty reply with no tool call -- asking again")
                messages.append({"role": "assistant", "content": content})
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return "Stopped: model kept replying with nothing."
                messages.append({"role": "user", "content": "Error: empty reply with no tool call. "
                                 "Call a tool, or give your final answer as plain text."})
                continue

        consecutive_errors = 0
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = c["function"].get("arguments", {})
            if isinstance(args, str):
                args = extract_json(args) or {}
            result = execute(name, args, seen_calls)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": result})

    return "Stopped: reached max_steps without a final answer."


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else (
        "What is the total eur_amount across all transactions in data/payments.csv? "
        "Round to 2 decimal places.")
    print(f"\n=== final ===\n{run_agent(task)}")
