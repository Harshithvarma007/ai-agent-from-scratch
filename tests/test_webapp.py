import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from conftest import reply

pytest.importorskip("pandas")


@pytest.fixture
def server(workspace, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST_URL", "http://127.0.0.1:9")  # nothing listens there
    from webapp import server as srv
    monkeypatch.setattr(srv, "DATA_DIR", str(workspace / "data"))
    monkeypatch.setattr(srv, "OLLAMA_HOST", "http://127.0.0.1:9")
    srv._data_cache.clear()
    (workspace / "data" / "payments.csv").write_text("merchant,eur_amount\nA,1.5\nB,\n")
    (workspace / "data" / "acquirer_countries.csv").write_text("acquirer,country_code\nx,GB\n")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_static_index(server):
    _, base = server
    status, ctype, body = get(base + "/")
    assert status == 200 and "text/html" in ctype and b"<html" in body.lower()


def test_models_without_ollama(server):
    _, base = server
    data = json.loads(get(base + "/api/models")[2])
    assert data["local"] == []
    assert {m["provider"] for m in data["free_tier"]} == {"groq", "gemini", "openrouter"}


def test_data_preview(server):
    _, base = server
    data = json.loads(get(base + "/api/data")[2])
    assert data["rows"] == 2
    assert data["columns"][1] == {"name": "eur_amount", "dtype": "float64", "nulls": 1}
    assert data["preview"][1] == ["B", None]
    assert data["other_files"][0]["file"] == "data/acquirer_countries.csv"


def test_run_streams_events(server, monkeypatch):
    srv, base = server
    answers = iter([reply("", ("read_file", {"path": "data/payments.csv"})), reply("2 rows.")])
    monkeypatch.setattr(srv, "_real_chat", lambda provider, messages, **kw: next(answers))
    status, body = post(base + "/api/run", {"task": "rows?", "provider": "ollama", "model": "m"})
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    kinds = [e["kind"] for e in events]
    assert status == 200
    assert kinds[0] == "start" and events[0]["model"] == "m"
    assert "llm_start" in kinds and "llm_end" in kinds and "action" in kinds
    assert events[-1] == {**events[-1], "kind": "final", "outcome": "final", "answer": "2 rows.", "steps": 1}


def test_run_rejects_bad_input(server):
    _, base = server
    assert post(base + "/api/run", {"task": "  "})[0] == 400
    assert post(base + "/api/run", b"not json")[0] == 400
    assert post(base + "/api/run", {"task": "x", "provider": "skynet"})[0] == 400
    assert post(base + "/api/nope", {})[0] == 404


def test_missing_static_file_is_a_clean_404(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/favicon.ico")
    assert e.value.code == 404
