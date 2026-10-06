<div align="center">

# AI Agent from Scratch

**A data-analyst AI agent in ~700 lines of plain Python. Free LLMs, no frameworks.**

[![CI](https://github.com/Harshithvarma007/ai-agent-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Harshithvarma007/ai-agent-from-scratch/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)
![No frameworks](https://img.shields.io/badge/frameworks-none-f5c0c0)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ba88a)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshithvarma007/ai-agent-from-scratch/blob/main/notebook/agent_from_scratch.ipynb)

https://github.com/user-attachments/assets/4e29a4e0-f0b7-4641-babb-94923a309b74

</div>

Ask a question about a CSV in plain English. The agent writes pandas code, runs it in a sandbox,
reads the result, and keeps going until it has an answer. It runs locally on
[Ollama](https://ollama.com) or on a free API tier, and the agent itself uses only the Python
standard library.

## 🎬 Full build video

<!-- FULL VIDEO PLACEHOLDER: replace with the YouTube embed when it's live, e.g.
[![Build an AI agent from scratch](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
-->

> **Coming soon.** Until then, [Build it](#-build-it-step-by-step) walks the same eight steps.

## What it does

A real run on `qwen3.5:4b`, trimmed:

```text
$ python -m agent --yolo "Which merchant had the highest total eur_amount?"
[action] read_file({'path': 'data/payments.csv'})
[action] run_python({'code': "...df.groupby('merchant')['eur_amount'].sum()..."})
[observation] Crossfit_Hanna              5076636.90
              Golfclub_Baron_Friso        2553380.62 ...
=== final ===
Crossfit_Hanna had the highest total eur_amount at 5,076,636.90 EUR.
```

<p align="center"><img src="docs/assets/webapp.png" alt="The web app showing the agent's code, output and answer" width="820"></p>

## ⚡ Quick start

```bash
git clone https://github.com/Harshithvarma007/ai-agent-from-scratch.git
cd ai-agent-from-scratch
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
python scripts/download_data.py                        # dataset, ~24 MB
ollama pull qwen3.5:4b                                 # model, ~3.4 GB
python -m agent "How many transactions are in data/payments.csv?"   # expect 138,236
```

**No GPU?** Skip Ollama and use a free tier: `export GEMINI_API_KEY=...`
([get one](https://aistudio.google.com/apikey)), then add `--provider gemini`. Groq
(`GROQ_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`) work the same way.

> Use a **venv**, not `pip install --user`: the agent runs model-written code with `python -I`,
> which only sees packages installed into the venv itself.

## 🚀 Usage

```bash
python -m agent "your question"                 # CLI (also: analyst-agent "...")
python webapp/server.py                         # web UI on http://localhost:8765
docker compose up --build                       # web UI + Ollama in Docker
make help                                       # every common task
```

| CLI flag | Default | |
|---|---|---|
| `--provider` | `ollama` | `ollama` · `gemini` · `groq` · `openrouter` |
| `--model` | per provider | e.g. `qwen3:8b`, `llama3.1:8b` |
| `--mode` | `native` | `native` (provider's `tools=` API) · `react` (JSON in the prompt) · `constrained` (react + Ollama grammar) |
| `--yolo` | off | skip the y/N prompt before running code |
| `--sandbox` | `subprocess` | or `docker` (see [Safety](#-safety)) |
| `--max-steps` | `10` | ceiling on loop turns |

All settings also work as env vars; see [`.env.example`](.env.example). From Python:

```python
from agent import ALL_TOOLS, Registry, get_provider, run_agent
result = run_agent(get_provider("ollama"), "How many merchants are there?", Registry(*ALL_TOOLS))
print(result["outcome"], result["answer"])
```

## 🛠 Build it, step by step

The agent is built in **eight steps**, run after each one. Follow along in the
[notebook](notebook/agent_from_scratch.ipynb) (one cell per idea; always *Restart & Run All*) or
with the matching scripts in [`steps/`](steps/), run from the repo root. Each step's real output
is saved in the notebook.

**Step 0. One call to a model.** Every provider here speaks the same OpenAI-compatible API, so
the whole client is one HTTP POST ([`steps/hello_llm.py`](steps/hello_llm.py)):

```python
body = {"model": model, "messages": [{"role": "user", "content": question}]}
req = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {api_key}"})
reply = json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]
```

Switching providers means changing `base_url`, the key and the model, which is all
[`agent/config.py`](agent/config.py) does.

**Step 1. Still a chatbot** ([`loop_step1.py`](steps/loop_step1.py)). Describe the data in a system
prompt and ask. The model writes the right pandas code in its reply, but **nothing runs**. An LLM
only produces text.

**Step 2. Offer tools** ([`loop_step2.py`](steps/loop_step2.py)). A tool is a function plus a JSON
schema the model reads. `@tool` builds the schema from the signature and docstring:

```python
@tool(dangerous=True)
def run_python(code: str) -> str:
    """Run a Python snippet (pandas/numpy available) against files in the workspace..."""
```

Send `tools=registry.schemas` and the reply comes back with no text, just a request:
`tool_calls: [{"function": {"name": "read_file", "arguments": "{\"path\":\"data/payments.csv\"}"}}]`.
The model *asks*; your code runs it.

**Step 3. Run the call** ([`loop_step3.py`](steps/loop_step3.py)). Append the model's request to
the history first, then each result as `role: "tool"` with its `tool_call_id`.
`registry.run()` never raises: errors come back as text the model can read and recover from.

**Step 4. The loop** ([`loop_step4.py`](steps/loop_step4.py)). Repeat until a reply has no tool
call. That's an agent: **an LLM, tools, and a loop.**

```python
while True:                                                         # condensed
    msg = llm.chat(provider, messages, tools=registry.schemas)      # think
    if not msg.get("tool_calls"):
        return msg["content"]                                       # done
    messages.append(msg)
    for c in msg["tool_calls"]:
        result = registry.run(c["function"]["name"], json.loads(c["function"]["arguments"]))  # act
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})        # observe
```

It now answers the total-revenue question correctly: **12,697,297.46**. Steps 5–8 make it survive
small models. Steps 6 and 8 replay a recorded real failure, so you can see each fix work:

**Step 5. A ceiling** ([`loop_step5.py`](steps/loop_step5.py)). `while True` becomes
`for step in range(MAX_STEPS)`, so the agent can't loop (or bill you) forever.

**Step 6. Silent replies** ([`loop_step6.py`](steps/loop_step6.py)). About 1 run in 9, a small
model replies with no text *and* no tool call, and step 4 returns that as a blank answer. The fix:
treat it as an error, say so, ask again (max 3 times).
`python steps/replay.py steps/traces/silent_reply.json steps/loop_step4.py` shows the bug; swap in
`loop_step6.py` to see it recover.

**Step 7. Tool calls written as text** ([`loop_step7.py`](steps/loop_step7.py)). Some models put
`{"tool": "read_file", "args": {...}}` in the text instead of `tool_calls`. Before treating text as
the answer, `extract_json()` checks for a tool call hiding in it.

**Step 8. Repeats** ([`loop_step8.py`](steps/loop_step8.py)). A model can send the same failing
call over and over. Count each `(tool, args)` pair and refuse the fourth, telling the model why.
`truncate()` keeps long outputs from flooding the context. **This is the finished loop**: compare
it with [`agent/agent.py`](agent/agent.py).

| Step | Became | Production adds |
|---|---|---|
| 0 | [`config.py`](agent/config.py), [`llm.py`](agent/llm.py) | 4 providers, retries on 429/5xx, Ollama constrained mode |
| 2–3 | [`tools.py`](agent/tools.py) | arg coercion, workspace jail, sandboxed `run_python`, confirm-before-run |
| 4–8 | [`agent.py`](agent/agent.py) | prompt-based (ReAct) mode, history trimming, a trace for the web app |
| — | [`main.py`](agent/main.py) | the CLI, saved trajectories, exit codes |

**Try it:** ask about merchant `'Zorblax_Fitness'`, which doesn't exist. Then compare `--mode native`
with `--mode react`, or read the same agent in 17 lines of LangChain in
[`comparisons/`](comparisons/).

## 🔒 Safety

The agent runs LLM-written code, so it has four safeguards:

- **Workspace jail:** paths can't leave `workspace/`, including via `../` or symlinks.
- **Confirmation:** `run_python` and `write_file` ask y/N unless you pass `--yolo`.
- **Subprocess sandbox (default):** the code runs in a separate process with a timeout and an
  empty environment, so your API keys aren't visible to it. It does **not** block file or
  network access.
- **Docker sandbox:** `--sandbox docker` adds no network, no capabilities and a read-only
  workspace. For pandas inside it, run
  `docker build -f Dockerfile.sandbox -t analyst-agent-sandbox .` and set
  `AGENT_DOCKER_IMAGE=analyst-agent-sandbox`.

`python -m agent.sandbox_ladder` shows the difference: `exec()` leaks a fake API key, and the
subprocess sandbox doesn't. Keep the web app on localhost.

## 📊 Results

Measured on this code with `qwen3.5:4b`, 3 runs × 11 questions per mode
([details](docs/RESULTS.md)):

- **91%** correct in native mode, **85%** in prompt-based mode.
- **0 of 18** runs, across 3 models and both modes, noticed that `Zorblax_Fitness` doesn't exist.
  All 18 answered with a number.

It *works*, and it can still be confidently wrong: this is the basic agent, with no verification.
Making it trustworthy is the sequel.

## 📁 Layout

```text
agent/          the agent (stdlib only): config, llm, tools, agent loop, CLI, sandbox
notebook/       the 8-step build (.ipynb generated by build_notebook.py)
steps/          the same steps as scripts + recorded-failure replays
webapp/         streaming web UI over run_agent()
scripts/        dataset download (checksummed)
bench/          golden questions + benchmark harness
comparisons/    the same agent in LangChain and LangGraph
tests/          75 tests, no model or network needed
```

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pandas'` inside `run_python` | install into a venv, not `--user` |
| `could not reach ollama` | `ollama serve`, then `ollama pull qwen3.5:4b` |
| agent forgets its tools mid-run | `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` |
| `needs an API key` / 404 on a free tier | export the key; free model names change, so pass `--model` |
| `no such file: data/payments.csv` | `python scripts/download_data.py`, and run from the repo root |

## Development

```bash
pip install -r requirements.txt -e ".[dev,notebook]"
pytest && ruff check .
python notebook/build_notebook.py && python notebook/run_notebook.py   # after editing notebook cells
```

CI runs lint and tests on Python 3.11–3.13, replays the recorded runs on the real dataset, and
builds the Docker image.

## Credits & license

Dataset: [DABstep](https://huggingface.co/datasets/adyen/DABstep) by Adyen (CC-BY-4.0). Fonts:
Archivo, Source Sans 3, Source Code Pro (OFL). Code: [MIT](LICENSE) © 2026 N Sai Harshith Varma.
