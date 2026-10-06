# The agent, built step by step: `agent_from_scratch.ipynb`

A data-analyst agent in plain Python, built in **eight small steps** and run after every one. No
framework, no LLM SDK, free models only. The outputs saved in the notebook come from one clean,
top-to-bottom run on `qwen3.5:4b` (local Ollama).

| Step | Cells | What it adds |
|---|---|---|
| setup | `setup-*` | the data, the model, one `chat()` call, a `show()` for the message history |
| 1 | `s1-*` | one call: still a chatbot |
| 2 | `s2-*` | the `@tool` decorator, two tools, offering their schemas |
| 3 | `s3-*` | running the tool call: request first, then result |
| 4 | `s4-*` | the loop, and its stop condition |
| 5 | `s5-*` | a hard ceiling on steps |
| 6 | `s6-*` | when the model goes silent (shown on a recorded real run) |
| 7 | `s7-*` | when the tool call arrives as text |
| 8 | `s8-*` | when it repeats itself (recorded real run), then the finished loop, live |

The finished, production version of the same agent is [`../agent/`](../agent/). The notebook's
last cell maps each cell to the file it became. The repo's main README walks the same steps in
text: [Build it, step by step](../README.md#-build-it-step-by-step).

## Run it

**Locally, with Ollama** (free, private), from the repo root:

```bash
ollama pull qwen3.5:4b
pip install -r requirements.txt -e ".[notebook]"
python scripts/download_data.py
jupyter lab notebook/agent_from_scratch.ipynb
```

**On Colab** (no install): open it with the badge in the main README, then set
`AGENT_PROVIDER=gemini` and `GEMINI_API_KEY` before running the cells (a cell with
`os.environ["AGENT_PROVIDER"] = "gemini"` works). The dataset and the two recorded traces download
themselves when the notebook isn't inside a clone.

Always **Restart & Run All**: the notebook builds `run_agent` up step by step, so running cells
out of order gives you an older step's loop.

## For maintainers

The cells are written in `build_notebook.py`; never edit the `.ipynb` by hand (a test fails if the
two drift apart).

```bash
python notebook/build_notebook.py     # regenerate the notebook (no outputs)
python notebook/run_notebook.py       # execute it top to bottom in a fresh kernel; fails on any error
```
