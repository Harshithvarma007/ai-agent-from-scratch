import io
import json
import urllib.error

import pytest

from agent import llm
from agent.config import PROVIDERS, get_provider


def test_providers_share_one_shape():
    assert set(PROVIDERS) == {"ollama", "groq", "gemini", "openrouter"}


def test_ollama_needs_no_key_and_respects_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_HOST_URL", "http://ollama:11434/")
    p = get_provider("ollama")
    assert p.model == "llama3.1:8b"
    assert p.base_url == "http://ollama:11434/v1"
    assert p.supports_constrained


def test_hosted_providers_need_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_provider("groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert get_provider("groq").api_key == "gsk_test"


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("skynet")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def ok(message):
    return FakeResponse(json.dumps({"choices": [{"message": message}]}).encode())


def http_error(code, body="", headers=None):
    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(body.encode()))


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    return get_provider("groq")


def test_chat_posts_openai_shape(monkeypatch, groq):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        seen["auth"] = req.get_header("Authorization")
        return ok({"role": "assistant", "content": "hi"})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    msg = llm.chat(groq, [{"role": "user", "content": "yo"}], tools=[{"type": "function"}])
    assert msg["content"] == "hi"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["body"]["tools"] == [{"type": "function"}] and seen["body"]["temperature"] == 0.2


def test_chat_retries_rate_limits(monkeypatch, groq):
    replies = [http_error(429, headers={"retry-after": "0"}), http_error(503, headers={"retry-after": "0"}),
               ok({"role": "assistant", "content": "finally"})]

    def fake_urlopen(req, timeout):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.chat(groq, [])["content"] == "finally"


def test_chat_does_not_retry_bad_requests(monkeypatch, groq):
    def fake_urlopen(req, timeout):
        raise http_error(401, "bad key")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        llm.chat(groq, [])


def test_groq_tool_use_failed_is_recovered_as_text(monkeypatch, groq):
    body = json.dumps({"error": {"code": "tool_use_failed",
                                 "failed_generation": '{"tool": "add", "args": {}}'}})

    def fake_urlopen(req, timeout):
        raise http_error(400, body)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    msg = llm.chat(groq, [])
    assert msg == {"role": "assistant", "content": '{"tool": "add", "args": {}}', "tool_calls": None}


def test_constrained_mode_is_ollama_only(groq):
    with pytest.raises(ValueError, match="no grammar-constrained mode"):
        llm.chat(groq, [], json_schema={"type": "object"})


def test_constrained_mode_uses_ollama_native_api(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return FakeResponse(json.dumps({"message": {"content": '{"tool": "x"}'}}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    msg = llm.chat(get_provider("ollama"), [], json_schema={"type": "object"})
    assert seen["url"].endswith(":11434/api/chat") and seen["body"]["format"] == {"type": "object"}
    assert msg["content"] == '{"tool": "x"}'
