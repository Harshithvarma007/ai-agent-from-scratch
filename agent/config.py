"""One place to pick which LLM the agent talks to.

Every provider below speaks the same OpenAI-compatible `/chat/completions` shape, so llm.py
never has provider-specific code. Switching providers is one change: --provider on the CLI,
or the AGENT_PROVIDER env var. That one-config-switch is the whole point of chapter 3.
"""
import os
from dataclasses import dataclass


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    # Ollama-only: grammar-constrained JSON via `format=` (see llm.py). No other provider here
    # exposes this, so react+constrained mode only works against a local Ollama server.
    supports_constrained: bool = False


# Model IDs drift fast (see docs/RESULTS.md's gotchas) -- re-check these against
# `ollama list` and each provider's current free-model page before recording.
PROVIDERS = {
    "ollama": lambda: Provider(
        name="ollama",
        # OLLAMA_HOST_URL lets Docker point at the `ollama` service instead of localhost
        base_url=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434").rstrip("/") + "/v1",
        api_key="ollama",  # required by the OpenAI client shape; Ollama ignores its value
        model=os.environ.get("AGENT_MODEL", "qwen3.5:4b"),
        supports_constrained=True,
    ),
    "groq": lambda: Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        model=os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b"),
    ),
    "gemini": lambda: Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
    ),
    "openrouter": lambda: Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=os.environ.get("AGENT_MODEL", "qwen/qwen3.8-27b:free"),
    ),
}


def get_provider(name):
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose one of {list(PROVIDERS)}")
    p = PROVIDERS[name]()
    if name != "ollama" and not p.api_key:
        raise ValueError(f"provider {name!r} needs an API key: set {name.upper()}_API_KEY")
    return p


if __name__ == "__main__":
    import sys

    # `python -m agent.config ollama` vs `... groq` -- same code, one argument different.
    print(get_provider(sys.argv[1] if len(sys.argv) > 1 else "ollama"))
