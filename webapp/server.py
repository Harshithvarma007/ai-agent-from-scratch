"""A thin web UI over the exact same agent loop as main.py. No new agent logic lives here.

    python webapp/server.py        # from the repo root, then open http://localhost:8765

It serves static/ and three endpoints:
  GET  /api/models  local Ollama models + the free-tier providers from config.PROVIDERS
  GET  /api/data    a preview of workspace/data/payments.csv
  POST /api/run     runs run_agent() and streams its trace events as Server-Sent Events
"""
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("AGENT_WORKSPACE", os.path.join(ROOT, "workspace"))
sys.path.insert(0, ROOT)  # so `agent` imports without `pip install -e .` too

from agent import agent as agent_mod  # noqa: E402
from agent import config, llm  # noqa: E402
from agent.tools import ALL_TOOLS, WORKSPACE, Registry  # noqa: E402

HOST = os.environ.get("HOST", "127.0.0.1")  # 0.0.0.0 inside Docker; keep 127.0.0.1 on a laptop
PORT = int(os.environ.get("PORT", "8765"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
DATA_DIR = os.path.join(WORKSPACE, "data")
RUN_LOCK = threading.Lock()  # one run at a time: the llm.chat wrapper below is process-global
_emit_local = threading.local()


class ClientGone(Exception):
    pass


# The agent's trace() callback only carries `content`. Wrapping llm.chat (at runtime, without
# editing agent/) additionally gives the UI each step's start time and the model's `reasoning`
# text, which Ollama returns alongside content for thinking models like qwen3.5.
_real_chat = llm.chat


def _chat_with_events(provider, messages, **kw):
    emit = getattr(_emit_local, "emit", None)
    if emit:
        emit("llm_start", {})
    t0 = time.time()
    msg = _real_chat(provider, messages, **kw)
    if emit:
        emit("llm_end", {
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "content": msg.get("content") or "",
            "has_tool_calls": bool(msg.get("tool_calls")),
            "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                           for c in (msg.get("tool_calls") or [])],
            "seconds": round(time.time() - t0, 2),
        })
    return msg


agent_mod.llm.chat = _chat_with_events


def _ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=2) as r:
            tags = json.loads(r.read())["models"]
    except Exception:  # noqa: BLE001 - Ollama not running is a normal state, not an error
        return []
    return [{"provider": "ollama", "model": m["name"],
             "size_gb": round(m.get("size", 0) / 1e9, 1),
             "params": (m.get("details") or {}).get("parameter_size", ""),
             "available": True} for m in tags]


def _free_tier_models():
    out = []
    for name, make in config.PROVIDERS.items():
        if name == "ollama":
            continue
        p = make()
        out.append({"provider": name, "model": p.model, "available": bool(p.api_key),
                    "needs": f"{name.upper()}_API_KEY"})
    return out


_data_cache = {}


def _data_summary():
    if "payments" not in _data_cache:
        import pandas as pd
        path = os.path.join(DATA_DIR, "payments.csv")
        df = pd.read_csv(path)
        preview = df.head(60).astype(object).where(df.head(60).notna(), None)
        _data_cache["payments"] = {
            "file": "data/payments.csv",
            "rows": int(len(df)),
            "size_mb": round(os.path.getsize(path) / 1e6, 1),
            "columns": [{"name": c, "dtype": str(df[c].dtype), "nulls": int(df[c].isna().sum())}
                        for c in df.columns],
            "preview": preview.values.tolist(),
            "other_files": [{"file": f"data/{f}",
                             "size_kb": round(os.path.getsize(os.path.join(DATA_DIR, f)) / 1e3, 1)}
                            for f in sorted(os.listdir(DATA_DIR)) if f != "payments.csv"],
        }
    return _data_cache["payments"]


def _json_default(o):
    return str(o)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/models":
            return self._send_json({"local": _ollama_models(), "free_tier": _free_tier_models()})
        if self.path == "/api/data":
            return self._send_json(_data_summary())
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self._send_json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send_json({"error": "body must be JSON"}, 400)
        task = (req.get("task") or "").strip()
        if not task:
            return self._send_json({"error": "empty task"}, 400)
        mode = {"native": "native", "react": "react", "constrained": "react+constrained"}.get(
            req.get("mode", "native"), "native")
        try:
            provider = config.get_provider(req.get("provider", "ollama"))
        except ValueError as e:
            return self._send_json({"error": str(e)}, 400)
        if req.get("model"):
            provider.model = req["model"]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        t_start = time.time()

        def emit(kind, data):
            payload = json.dumps({"kind": kind, "t": round(time.time() - t_start, 2), **data},
                                 default=_json_default)
            try:
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise ClientGone() from e

        def trace(kind, **info):
            emit(kind, info)

        with RUN_LOCK:
            _emit_local.emit = emit
            try:
                emit("start", {"provider": provider.name, "model": provider.model, "mode": mode})
                result = agent_mod.run_agent(provider, task, Registry(*ALL_TOOLS), mode=mode,
                                             confirm=lambda name, args: True, trace=trace)
                steps = sum(1 for m in result["messages"] if m.get("role") == "assistant")
                emit("final", {"outcome": result["outcome"], "answer": result["answer"],
                               "steps": steps})
            except ClientGone:
                pass
            except Exception as e:  # noqa: BLE001 - surface provider/network errors in the UI
                try:
                    emit("error", {"message": f"{type(e).__name__}: {e}"})
                except ClientGone:
                    pass
            finally:
                _emit_local.emit = None


def main():
    try:
        import pandas  # noqa: F401
    except ImportError:
        sys.exit("pandas isn't importable from this interpreter. run_python needs it in "
                 "sys.executable's own site-packages: start this with .venv/bin/python.")
    if not os.path.exists(os.path.join(DATA_DIR, "payments.csv")):
        sys.exit("No dataset yet: run `python scripts/download_data.py` first.")
    print(f"Data analyst agent UI on http://{HOST}:{PORT}  (workspace: {WORKSPACE})")
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
