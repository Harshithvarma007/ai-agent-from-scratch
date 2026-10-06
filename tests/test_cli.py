import json
import urllib.error

import pytest

from agent import __version__, main


@pytest.fixture
def fake_run(monkeypatch):
    seen = {}

    def run_agent(provider, task, registry, **kw):
        seen.update(provider=provider, task=task, **kw)
        return {"outcome": "final", "answer": "138236", "messages": []}

    monkeypatch.setattr(main, "run_agent", run_agent)
    return seen


def test_cli_runs_and_saves_the_trajectory(fake_run, tmp_path, capsys):
    code = main.main(["how many rows?", "--model", "qwen3:8b", "--yolo",
                      "--runs-dir", str(tmp_path)])
    assert code == 0
    assert fake_run["task"] == "how many rows?" and fake_run["provider"].model == "qwen3:8b"
    assert fake_run["confirm"]("run_python", {}) is True  # --yolo
    out = capsys.readouterr().out
    assert "=== final ===\n138236" in out
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["answer"] == "138236" and saved["model"] == "qwen3:8b"


def test_cli_no_save(fake_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main.main(["q", "--no-save"]) == 0
    assert not (tmp_path / "runs").exists()


def test_constrained_needs_ollama(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(SystemExit):
        main.main(["q", "--provider", "gemini", "--mode", "constrained"])


def test_missing_api_key_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        main.main(["q", "--provider", "gemini"])
    assert e.value.code == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_unreachable_server_is_a_clean_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(main, "run_agent", boom)
    assert main.main(["q", "--no-save"]) == 2
    assert "Is Ollama running?" in capsys.readouterr().err


def test_non_final_outcome_exits_1(monkeypatch):
    monkeypatch.setattr(main, "run_agent",
                        lambda *a, **k: {"outcome": "max_steps", "answer": "Stopped", "messages": []})
    assert main.main(["q", "--no-save"]) == 1


def test_version(capsys):
    with pytest.raises(SystemExit):
        main.main(["--version"])
    assert __version__ in capsys.readouterr().out
