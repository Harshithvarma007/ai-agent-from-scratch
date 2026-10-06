"""Step 1 of 8: one call. This is still a chatbot."""
import os
import sys

from agent import llm
from agent.agent import DATA_CONTEXT
from agent.config import get_provider

provider = get_provider(os.environ.get("AGENT_PROVIDER", "ollama"))
task = sys.argv[1] if len(sys.argv) > 1 else (
    "What is the total eur_amount across all transactions in data/payments.csv? "
    "Round to 2 decimal places.")

messages = [{"role": "system", "content": DATA_CONTEXT},
            {"role": "user", "content": task}]
msg = llm.chat(provider, messages)
print(msg["content"])
