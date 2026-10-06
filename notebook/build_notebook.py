"""Generates agent_from_scratch.ipynb: the agent built step by step, as one notebook.

The notebook is the teaching path: one cell per idea, top to bottom, no imports from the
production package (../agent/). The package is the finished product. Every cell has a stable id
("s3-run", "s8-guard", ...) so the README and the video can point at cells, and re-generating
never breaks those references.

    python notebook/build_notebook.py            # writes agent_from_scratch.ipynb (no outputs)
    python notebook/run_notebook.py              # executes it top to bottom, saves outputs

Edit the cells HERE, not in the .ipynb.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = []


def md(cid, text):
    CELLS.append(new_markdown_cell(text.strip("\n"), id=cid))


def code(cid, text, step=None):
    c = new_code_cell(text.strip("\n"), id=cid)
    if step is not None:
        c.metadata["nous"] = {"step": step}
    CELLS.append(c)


# ---------------------------------------------------------------- 0. setup
md("title", """
# Build an AI agent from scratch

A data-analyst agent in plain Python: no framework, no LLM SDK, free models only.
We build it in **eight small steps**, and run it after every one.

Run the cells **top to bottom** (Kernel → Restart & Run All). The finished, production version
of the same agent lives in `../agent/`; the last section maps each cell to the file it became.
""")

code("setup-paths", '''
import json, os, subprocess, sys, urllib.request
from pathlib import Path

REPO_RAW = ("https://raw.githubusercontent.com/Harshithvarma007/"
            "ai-agent-from-scratch/main/")

# The data: DABstep's payments (adyen/DABstep on Hugging Face, CC-BY-4.0).
# In the repo it's in ../workspace; anywhere else (Colab) we fetch it.
WORKSPACE = Path("../workspace").resolve()
if not (WORKSPACE / "data" / "payments.csv").exists():
    WORKSPACE = Path("workspace").resolve()
    base = ("https://huggingface.co/datasets/adyen/DABstep/resolve/main/"
            "data/context/")
    (WORKSPACE / "data").mkdir(parents=True, exist_ok=True)
    for f in ["payments.csv", "merchant_data.json",
              "merchant_category_codes.csv", "acquirer_countries.csv"]:
        if not (WORKSPACE / "data" / f).exists():
            urllib.request.urlretrieve(base + f, WORKSPACE / "data" / f)

# Two recorded real runs, replayed in steps 6 and 8 (fetched if not in a clone).
TRACES = Path("../steps/traces").resolve()
if not TRACES.exists():
    TRACES = Path("traces").resolve()
    TRACES.mkdir(exist_ok=True)
    for f in ["silent_reply.json", "repeat_guard.json"]:
        if not (TRACES / f).exists():
            url = REPO_RAW + "steps/traces/" + f
            urllib.request.urlretrieve(url, TRACES / f)
print("workspace:", os.path.relpath(WORKSPACE))
print(sorted(p.name for p in (WORKSPACE / "data").iterdir()))
''', step="setup")

md("setup-provider-md", """
## Which model?

Every provider here speaks the same OpenAI-compatible `/chat/completions` API, so switching is
one variable. Local Ollama is the default; on Colab, use the Gemini free tier
(set `GEMINI_API_KEY`).
""")

code("setup-provider", '''
PROVIDER = os.environ.get("AGENT_PROVIDER", "ollama")

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama",
               "model": os.environ.get("AGENT_MODEL", "qwen3.5:4b")},
    "gemini": {"base_url":
               "https://generativelanguage.googleapis.com/v1beta/openai/",
               "api_key": os.environ.get("GEMINI_API_KEY", ""),
               "model": os.environ.get("AGENT_MODEL", "gemini-2.5-flash")},
}
provider = PROVIDERS[PROVIDER]
print(PROVIDER, "->", provider["model"])
''', step="setup")

code("setup-chat", '''
def chat(messages, tools=None):
    """One HTTP call. Returns the assistant message (content, tool_calls)."""
    body = {"model": provider["model"], "messages": messages,
            "temperature": 0.2}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        provider["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {provider['api_key']}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]

print(chat([{"role": "user", "content": "Say hi in five words."}])["content"])
''', step="setup")

code("setup-show", '''
def show(messages):
    """Print the conversation as the model sees it: this list IS the memory."""
    for m in messages:
        body = m.get("content") or ""
        if m.get("tool_calls"):
            body += " " + ", ".join(
                f"→ {c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"])
        print(f"{m['role']:>9} | {body[:110]}")
''', step="setup")

# ---------------------------------------------------------------- step 1
md("s1-md", """
## Step 1 of 8: one call

The smallest thing that talks to the model: a system prompt, the user's question, one call.
""")

code("s1-context", '''
DATA_CONTEXT = (
    "You are a data analyst agent. The workspace contains:\\n"
    "- data/payments.csv: one row per transaction\\n"
    "- data/merchant_data.json: one JSON object per merchant, keyed by "
    "'merchant'\\n"
    "- data/merchant_category_codes.csv: MCC code -> description lookup\\n"
    "- data/acquirer_countries.csv: acquirer -> country_code lookup\\n"
    "You don't know the exact column names yet -- inspect a file "
    "(read_file, or run_python with pandas, e.g. df.columns / df.head()) "
    "before you compute an answer. Give an exact final answer (a number "
    "or short string), not a description of how you'd compute it. If the "
    "question is not answerable from this data, say so explicitly instead "
    "of guessing a number.\\n"
)
TASK = ("What is the total eur_amount across all transactions in "
        "data/payments.csv? Round to 2 decimal places.")
''', step=1)

code("s1-call", '''
messages = [{"role": "system", "content": DATA_CONTEXT},
            {"role": "user", "content": TASK}]
msg = chat(messages)
print(msg["content"])
''', step=1)

md("s1-after", """
It may even write the pandas code, but nothing can run it. **That's a chatbot.**
""")

# ---------------------------------------------------------------- step 2
md("s2-md", """
## Step 2 of 8: offer the tools

A tool is a normal Python function plus a description the model reads (its *schema*).
The `@tool` decorator from Chapter 4 builds that schema from the function itself.
""")

code("s2-decorator", '''
import inspect

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

def tool(fn):
    """Read the function's own signature and docstring -> its JSON schema."""
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": JSON_TYPES.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    fn.schema = {"type": "function", "function": {
        "name": fn.__name__,
        "description": fn.__doc__.strip().splitlines()[0],
        "parameters": {"type": "object", "properties": props,
                       "required": required}}}
    return fn
''', step=2)

code("s2-tools", '''
@tool
def read_file(path: str) -> str:
    """Read a text file from the workspace (large files are truncated)."""
    full = (WORKSPACE / path).resolve()
    if not full.is_relative_to(WORKSPACE):
        raise ValueError(f"path '{path}' is outside the workspace")
    text = full.read_text(errors="replace")
    if len(text) > 4000:
        text = text[:4000] + f"\\n...[{len(text) - 4000} chars truncated]"
    return text


@tool
def run_python(code: str) -> str:
    """Run Python (pandas available) in the workspace; print what you need."""
    proc = subprocess.run([sys.executable, "-c", code], cwd=WORKSPACE,
                          capture_output=True, text=True, timeout=25)
    if proc.returncode != 0:
        return f"Error: exited {proc.returncode}: {proc.stderr[-2000:]}"
    return proc.stdout[-4000:] or "(no output)"

print(json.dumps(read_file.schema, indent=2))
''', step=2)

code("s2-registry", '''
TOOLS = {fn.__name__: fn for fn in [read_file, run_python]}
SCHEMAS = [fn.schema for fn in TOOLS.values()]

def run_tool(name, args):
    """Never raises: a failure becomes text the model reads next."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Valid tools: {', '.join(TOOLS)}"
    try:
        return str(TOOLS[name](**args))
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
''', step=2)

code("s2-call", '''
messages = [{"role": "system", "content": DATA_CONTEXT},
            {"role": "user", "content": TASK}]
msg = chat(messages, tools=SCHEMAS)
print("content:   ", repr(msg.get("content")))
print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=2))
''', step=2)

md("s2-after", """
No answer this time: the model **asks** for a tool. And nothing ran. The model asked; our code
hasn't done anything yet.
""")

# ---------------------------------------------------------------- step 3
md("s3-md", """
## Step 3 of 8: run the tool call

Two things, in this order: write the model's **request** into the history, then the **result**.
""")

code("s3-run", '''
content = msg.get("content") or ""
calls = msg.get("tool_calls") or []

# 1. the request
messages.append({"role": "assistant", "content": content,
                 "tool_calls": calls})
for c in calls:
    name = c["function"]["name"]
    args = json.loads(c["function"]["arguments"])
    result = run_tool(name, args)
    # 2. the result
    messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                     "content": result})
    print(f"[action] {name}({args})")
    print(f"[observation] {result[:300]}")
''', step=3)

code("s3-show", '''
show(messages)
''', step=3)

# ---------------------------------------------------------------- step 4
md("s4-md", """
## Step 4 of 8: loop until done

Send the results back and call again, until the model answers **without** asking for a tool.
This is the whole idea of an agent.
""")

code("s4-agent", '''
def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    while True:
        msg = chat(messages, tools=SCHEMAS)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:200]}")

        if not calls:
            return content

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"])
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})
            print(f"[action] {name}({args})")
            print(f"[observation] {result[:200]}")
''', step=4)

code("s4-run", '''
print("\\n=== final ===\\n" + run_agent(TASK))
''', step=4)

# ---------------------------------------------------------------- step 5
md("s5-md", """
## Step 5 of 8: a hard ceiling

`while True` is a promise we can't keep. Bound it, and say so honestly when we hit the limit.
""")

code("s5-agent", '''
MAX_STEPS = 10

def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    for step in range(MAX_STEPS):
        msg = chat(messages, tools=SCHEMAS)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:200]}")

        if not calls:
            return content

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"])
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})
            print(f"[action] {name}({args})")
            print(f"[observation] {result[:200]}")

    return "Stopped: reached max_steps without a final answer."
''', step=5)

# ---------------------------------------------------------------- replay helper
md("replay-md", """
## Replaying a real run

Steps 6 and 8 fix failures that happen only now and then (the silent reply is about 1 run in 9),
so we can't trigger them on demand. `replaying()` plays back the **exact replies a real model
gave** in a recorded run; the tools still run live, right now, against the real data.
""")

code("replay", '''
from contextlib import contextmanager

@contextmanager
def replaying(trace_file):
    """Swap chat() for a recorded run's replies. Tools still run for real."""
    global chat
    trace = json.loads((TRACES / trace_file).read_text())
    replies, real_chat = iter(trace["replies"]), chat

    def replayed(messages, tools=None):
        try:
            return next(replies)
        except StopIteration:
            raise RuntimeError("end of recording: the real run stopped here "
                               f"({trace['outcome']})")

    print(f"[replay] {trace_file}: {trace['model']}, "
          f"recorded {trace['captured_at']}")
    print(f"[replay] task: {trace['task']}\\n")
    chat = replayed
    try:
        yield trace["task"]
    finally:
        chat = real_chat
''')

# ---------------------------------------------------------------- step 6
md("s6-md", """
## Step 6 of 8: when the model goes silent

Our stop rule is "no tool call means done". What if there's no tool call **and no text**?
First, the bug, on a recorded real run through the step-5 loop:
""")

code("s6-before", '''
with replaying("silent_reply.json") as task:
    print("\\n=== final ===\\n" + repr(run_agent(task)))
''', step=6)

code("s6-agent", '''
MAX_CONSECUTIVE_ERRORS = 3
NUDGE = ("Error: empty reply with no tool call. "
         "Call a tool, or give your final answer as plain text.")

def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    consecutive_errors = 0
    for step in range(MAX_STEPS):
        msg = chat(messages, tools=SCHEMAS)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:200]}")

        if not calls:
            if content.strip():
                return content
            print("[error] empty reply with no tool call -- asking again")
            messages.append({"role": "assistant", "content": content})
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return "Stopped: model kept replying with nothing."
            messages.append({"role": "user", "content": NUDGE})
            continue

        consecutive_errors = 0
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"])
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})
            print(f"[action] {name}({args})")
            print(f"[observation] {result[:200]}")

    return "Stopped: reached max_steps without a final answer."
''', step=6)

code("s6-after", '''
with replaying("silent_reply.json") as task:
    print("\\n=== final ===\\n" + run_agent(task))
''', step=6)

# ---------------------------------------------------------------- step 7
md("s7-md", """
## Step 7 of 8: when the call arrives as text

Small models sometimes write the tool call **as text** instead of using `tool_calls`. Before
accepting any answer, check whether a JSON tool call is hiding in it.
""")

code("s7-extract", '''
import re

def extract_json(text):
    """The first balanced {...} in the text, or None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:          # escaped: \\" doesn't end the string
                    esc = False
                elif ch == "\\\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None

# An illustrative reply (constructed, not recorded): the call is in the text.
example = ('I will look at the file first. {"tool": "read_file", '
           '"args": {"path": "data/acquirer_countries.csv"}}')
print(extract_json(example))
''', step=7)

code("s7-agent", '''
def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    consecutive_errors = 0
    for step in range(MAX_STEPS):
        msg = chat(messages, tools=SCHEMAS)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:200]}")

        if not calls:
            rescued = extract_json(content)
            if rescued and "tool" in rescued:
                print("[note] tool call recovered from plain text, "
                      "not tool_calls")
                calls = [{"function": {"name": rescued["tool"],
                                       "arguments": rescued.get("args", {})}}]
            elif content.strip():
                return content
            else:
                print("[error] empty reply with no tool call -- asking again")
                messages.append({"role": "assistant", "content": content})
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return "Stopped: model kept replying with nothing."
                messages.append({"role": "user", "content": NUDGE})
                continue

        consecutive_errors = 0
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = c["function"].get("arguments", {})
            if isinstance(args, str):
                args = extract_json(args) or {}
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})
            print(f"[action] {name}({args})")
            print(f"[observation] {result[:200]}")

    return "Stopped: reached max_steps without a final answer."
''', step=7)

# ---------------------------------------------------------------- step 8
md("s8-md", """
## Step 8 of 8: when it repeats itself

Count every call. After three identical ones, refuse and say why. First, a recorded real run
through the step-7 loop (it was recorded on `llama3.1:8b`):
""")

code("s8-before", '''
with replaying("repeat_guard.json") as task:
    try:
        print("\\n=== final ===\\n" + run_agent(task))
    except RuntimeError as e:
        print(f"\\n[replay] {e}")
''', step=8)

code("s8-guard", '''
REPEAT_THRESHOLD = 3

def truncate(text, head=800, tail=800):
    """Keep the start and end of a long tool result."""
    text = str(text)
    if len(text) <= head + tail:
        return text
    elided = len(text) - head - tail
    return f"{text[:head]}\\n...[{elided} chars elided]...\\n{text[-tail:]}"

def execute(name, args, seen_calls):
    key = (name, json.dumps(args, sort_keys=True, default=str))
    seen_calls[key] = seen_calls.get(key, 0) + 1
    if seen_calls[key] > REPEAT_THRESHOLD:
        print(f"[loop guard] refused to repeat {name}({args})")
        return ("Error: you already called this exact tool with these exact "
                "arguments. Try something else.")
    result = run_tool(name, args)
    print(f"[action] {name}({args})")
    print(f"[observation] {result[:200]}")
    return truncate(result)
''', step=8)

code("s8-agent", '''
def run_agent(task):
    messages = [{"role": "system", "content": DATA_CONTEXT},
                {"role": "user", "content": task}]
    consecutive_errors = 0
    seen_calls = {}
    for step in range(MAX_STEPS):
        msg = chat(messages, tools=SCHEMAS)
        content = msg.get("content") or ""
        calls = msg.get("tool_calls")
        print(f"[thought] {content[:200]}")

        if not calls:
            rescued = extract_json(content)
            if rescued and "tool" in rescued:
                print("[note] tool call recovered from plain text, "
                      "not tool_calls")
                calls = [{"function": {"name": rescued["tool"],
                                       "arguments": rescued.get("args", {})}}]
            elif content.strip():
                return content
            else:
                print("[error] empty reply with no tool call -- asking again")
                messages.append({"role": "assistant", "content": content})
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return "Stopped: model kept replying with nothing."
                messages.append({"role": "user", "content": NUDGE})
                continue

        consecutive_errors = 0
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = c["function"].get("arguments", {})
            if isinstance(args, str):
                args = extract_json(args) or {}
            result = execute(name, args, seen_calls)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})

    return "Stopped: reached max_steps without a final answer."
''', step=8)

code("s8-after", '''
with replaying("repeat_guard.json") as task:
    try:
        print("\\n=== final ===\\n" + run_agent(task))
    except RuntimeError as e:
        print(f"\\n[replay] {e}")
''', step=8)

code("s8-live", '''
# The finished loop, live, on the original question.
print("\\n=== final ===\\n" + run_agent(TASK))
''', step=8)

# ---------------------------------------------------------------- notebook -> package
md("to-package", """
## From notebook to package

Every cell above became part of a file in `../agent/`. The package adds what a notebook hides:

| Notebook cell | Became | What production adds |
|---|---|---|
| `setup-provider` | `config.py` | four providers, API-key checks, one `--provider` switch |
| `setup-chat` | `llm.py` | retries on 429/5xx, Groq's `tool_use_failed` recovery, Ollama's grammar-constrained mode |
| `s2-decorator`, `s2-tools`, `s2-registry` | `tools.py` | `@tool(dangerous=...)`, the `Registry` class, argument coercion for small models, confirm-before-run for dangerous tools, a scrubbed subprocess (and the Docker rung in `sandbox_ladder.py`) |
| `s7-extract`, `s8-guard`, `s8-agent` | `agent.py` | the prompt-based (ReAct) branch, history trimming, a trace for the web app |
| (none) | `main.py` | the CLI and the coloured trace printer |

Same eight ideas, same order.
""")


def build(path: Path):
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return nb


if __name__ == "__main__":
    out = Path(__file__).with_name("agent_from_scratch.ipynb")
    nb = build(out)
    print(f"wrote {out.name}: {len(nb.cells)} cells "
          f"({sum(c.cell_type == 'code' for c in nb.cells)} code)")
