"""Step 2 of 8: offer the tools. The model asks for one -- nothing runs yet."""
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
print("content:   ", repr(msg.get("content")))
print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=2))
