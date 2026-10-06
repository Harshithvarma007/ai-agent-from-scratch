"""One HTTP call to an OpenAI-compatible /chat/completions endpoint. No SDK.

chat() always returns a plain dict shaped like an OpenAI assistant message:
    {"role": "assistant", "content": "...", "tool_calls": [...] | None}
so agent.py never has to branch on which provider answered.
"""
import json
import time
import urllib.error
import urllib.request


def chat(provider, messages, tools=None, temperature=0.2, json_schema=None, max_retries=3):
    if json_schema is not None:
        if not provider.supports_constrained:
            raise ValueError(f"provider {provider.name!r} has no grammar-constrained mode")
        return _ollama_constrained_chat(provider, messages, json_schema, temperature)

    url = provider.base_url.rstrip("/") + "/chat/completions"
    body = {"model": provider.model, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {provider.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            payload = e.read().decode(errors="replace")
            # Groq answers a malformed tool call with 400 tool_use_failed instead of just
            # returning it -- recover the raw text so our fallback parser still gets a look.
            if e.code == 400 and "tool_use_failed" in payload:
                failed = json.loads(payload).get("error", {}).get("failed_generation", "")
                return {"role": "assistant", "content": failed, "tool_calls": None}
            # Retry only on rate-limit / server errors. Never retry a plain bad response at
            # temperature 0 -- greedy decoding just returns the same broken output again.
            if attempt < max_retries and (e.code == 429 or e.code >= 500):
                time.sleep(float(e.headers.get("retry-after", 2 ** attempt)))
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def _ollama_constrained_chat(provider, messages, json_schema, temperature):
    # `format=` grammar-constrained decoding is an Ollama-native extension, not part of the
    # OpenAI spec, so this is the one place we step outside /v1.
    host = provider.base_url.rstrip("/").removesuffix("/v1")
    body = {
        "model": provider.model, "messages": messages, "format": json_schema, "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        host + "/api/chat", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        msg = json.loads(r.read())["message"]
    return {"role": "assistant", "content": msg.get("content", ""), "tool_calls": None}


if __name__ == "__main__":  # python -m agent.llm [provider] [prompt]
    import sys

    from .config import get_provider

    provider = get_provider(sys.argv[1] if len(sys.argv) > 1 else "ollama")
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Say hi in five words."
    print(chat(provider, [{"role": "user", "content": prompt}])["content"])
