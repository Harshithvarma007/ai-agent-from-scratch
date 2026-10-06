"""A data-analyst AI agent, from scratch: plain Python, free LLMs, no frameworks.

    from agent import ALL_TOOLS, Registry, get_provider, run_agent

    result = run_agent(get_provider("ollama"), "How many rows are in data/payments.csv?",
                       Registry(*ALL_TOOLS))
    print(result["outcome"], result["answer"])
"""
from .agent import run_agent
from .config import PROVIDERS, Provider, get_provider
from .llm import chat
from .tools import ALL_TOOLS, Registry, tool

__version__ = "1.0.0"

__all__ = ["ALL_TOOLS", "PROVIDERS", "Provider", "Registry", "chat", "get_provider", "run_agent",
           "tool", "__version__"]
