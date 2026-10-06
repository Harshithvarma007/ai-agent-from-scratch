"""Chapter 10's LangGraph counterpart to comparisons/langchain/minimal_example.py's "17 lines" beat.

No PythonAstREPLTool, no allow_dangerous_code -- create_react_agent ships no default tool at all.
We hand it agent/tools.py's own read_file/write_file/run_python, completely unchanged. See
README.md for the deprecation notice on create_react_agent and what that reuse implies.

Needs: pip install -r requirements.txt, and a local Ollama server with qwen3.5:4b pulled.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(os.path.dirname(__file__), "..", "..", "workspace"))

from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

from agent.tools import ALL_TOOLS  # noqa: E402 -- our own read_file/write_file/run_python, unchanged

llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen3.5:4b")

agent = create_react_agent(
    llm, tools=ALL_TOOLS,
    prompt="You are a data analyst agent. Inspect data/payments.csv with a tool before answering.",
)

result = agent.invoke({"messages": [("user", "What is the total eur_amount across all transactions?")]})
print(result["messages"][-1].content)
