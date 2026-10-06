"""Step 3 of 8: run the tool call. Request into the history first, then the result."""
import json
import os
import sys

from agent import llm
from agent.agent import DATA_CONTEXT
from agent.config import get_provider
from agent.tools import Registry, read_file, run_python

provider = get_provider(os.environ.get("AGENT_PROVIDER", "ollama"))
registry = Registry(read_file, run_python)
task = sys.argv[1] if len(sys.argv) > 1 else (
    "What is the total eur_amount across all transactions in data/payments.csv? "
    "Round to 2 decimal places.")

messages = [{"role": "system", "content": DATA_CONTEXT},
            {"role": "user", "content": task}]
msg = llm.chat(provider, messages, tools=registry.schemas)
content = msg.get("content") or ""
calls = msg.get("tool_calls") or []

messages.append({"role": "assistant", "content": content, "tool_calls": calls})
for c in calls:
    name = c["function"]["name"]
    args = json.loads(c["function"]["arguments"])
    result = registry.run(name, args)
    messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": result})
    print(f"[action] {name}({args})")
    print(f"[observation] {result[:500]}")
