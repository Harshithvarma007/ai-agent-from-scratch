import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import agent as agent_mod  # noqa: E402
from agent import tools  # noqa: E402
from agent.config import Provider  # noqa: E402

DATA_DIR = os.path.join(ROOT, "workspace", "data")
HAS_DATA = os.path.exists(os.path.join(DATA_DIR, "payments.csv"))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fresh, empty workspace the tools are jailed to."""
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    monkeypatch.setattr(tools, "WORKSPACE", str(ws))
    return ws


@pytest.fixture
def provider():
    return Provider(name="fake", base_url="http://fake.invalid/v1", api_key="x", model="fake-model")


def reply(content="", *calls):
    """An assistant message, shaped like llm.chat() returns it."""
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": args}}
                  for i, (name, args) in enumerate(calls)] or None
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


@pytest.fixture
def scripted_chat(monkeypatch):
    """Replace llm.chat with a list of canned replies; records every call it receives."""
    def install(*replies):
        it = iter(replies)
        calls = []

        def fake_chat(provider, messages, **kw):
            calls.append({"messages": [dict(m) for m in messages], **kw})
            try:
                return next(it)
            except StopIteration:
                raise AssertionError("agent asked the model more times than the test scripted") from None

        monkeypatch.setattr(agent_mod.llm, "chat", fake_chat)
        return calls
    return install
