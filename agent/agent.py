"""The agent loop: think -> act -> observe -> repeat, until there's no more tool call to make.

Two teaching modes, same loop (chapter 4's native-vs-prompt comparison):
  "native"            -- the provider's tools= API; the model returns structured tool_calls.
  "react"             -- no tools=; the system prompt asks for one JSON object per turn.
  "react+constrained" -- react, but Ollama's `format=` grammar makes bad JSON and hallucinated
                         tool names physically impossible to generate (Ollama only).

Every technique that makes small models reliable lives in this file, in the smallest form that
still teaches the idea: errors fed back as observations (never crash), arg coercion (in
tools.py), the content-JSON fallback parser, a repeat-call guard, max_steps, and history
trimming.
"""
import json
import re

from . import llm

MAX_STEPS = 10
MAX_CONSECUTIVE_ERRORS = 3
REPEAT_THRESHOLD = 3          # same (tool, args) seen this many times -> stop re-running it
TRUNCATE_HEAD, TRUNCATE_TAIL = 800, 800
MAX_HISTORY_CHARS = 12_000    # rough proxy for "getting close to the context window"

CONSTRAINED_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {"thought": {"type": "string"}, "tool": {"type": "string"}, "args": {"type": "object"}},
    "required": ["thought", "tool", "args"],
}


def extract_json(text):
    """First balanced {...} in text, ignoring braces inside strings. None if there isn't one.
    This is the fallback parser: our own ~20-line version of what a package like json_repair
    does in production."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:              # this char is escaped: \" doesn't end the string
                    esc = False
                elif ch == "\\":
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


def truncate(text):
    text = str(text)
    if len(text) <= TRUNCATE_HEAD + TRUNCATE_TAIL:
        return text
    elided = len(text) - TRUNCATE_HEAD - TRUNCATE_TAIL
    return f"{text[:TRUNCATE_HEAD]}\n...[{elided} chars elided]...\n{text[-TRUNCATE_TAIL:]}"


DATA_CONTEXT = (
    "You are a data analyst agent. The workspace contains:\n"
    "- data/payments.csv: one row per transaction\n"
    "- data/merchant_data.json: one JSON object per merchant, keyed by 'merchant'\n"
    "- data/merchant_category_codes.csv: MCC code -> description lookup\n"
    "- data/acquirer_countries.csv: acquirer -> country_code lookup\n"
    "You don't know the exact column names yet -- inspect a file (read_file, or run_python with "
    "pandas, e.g. df.columns / df.head()) before you compute an answer. Give an exact final "
    "answer (a number or short string), not a description of how you'd compute it. If the "
    "question is not answerable from this data (e.g. it asks about something that doesn't exist "
    "in these files), say so explicitly instead of guessing a number.\n"
)


def _system_prompt(mode, registry):
    if mode == "native":
        return DATA_CONTEXT + "Use the tools when needed, then give a final answer."
    lines = "\n".join(registry.prompt_lines())
    return (
        DATA_CONTEXT +
        f"Available tools:\n{lines}\n\n"
        'Reply with exactly ONE JSON object and nothing else, in one of these two forms:\n'
        '{"thought": "...", "tool": "<tool name>", "args": {"<arg>": "<value>"}}\n'
        '{"thought": "...", "tool": "final_answer", "args": {"answer": "<answer for the user>"}}\n'
        "Call one tool per reply. You'll get 'Observation: <result>' back. Only use "
        "final_answer once you actually have the information you need."
    )


def trim(messages, units):
    """Keep the system prompt and the original task; drop the oldest step (as a whole unit)
    once history grows past MAX_HISTORY_CHARS, so we never orphan a tool result from its call."""
    while units and sum(len(json.dumps(m)) for m in messages) > MAX_HISTORY_CHARS:
        drop = units.pop(0)
        del messages[2:2 + drop]


def run_agent(provider, task, registry, mode="native", max_steps=MAX_STEPS, confirm=None, trace=None,
              history=None):
    """confirm(name, args) -> bool, asked before a dangerous tool runs; default allows everything.
    trace(kind, **info) -> callback for the CLI's live Thought/Action/Observation printer.
    history: extra messages spliced in right after the task (used by benchmark scripts to seed a
    prior turn and continue through this same loop instead of reimplementing it)."""
    trace = trace or (lambda *a, **k: None)
    confirm = confirm or (lambda name, args: True)
    constrained = mode == "react+constrained"
    schema = None
    if constrained:
        schema = dict(CONSTRAINED_SCHEMA_TEMPLATE)
        schema["properties"] = dict(schema["properties"])
        schema["properties"]["tool"] = {"type": "string", "enum": list(registry.fns) + ["final_answer"]}

    messages = [{"role": "system", "content": _system_prompt(mode, registry)},
                {"role": "user", "content": task}] + (history or [])
    units = []          # message counts added per step, oldest-first, so trim() drops whole steps
    seen_calls = {}      # (name, args-json) -> times executed, for the repeat guard
    consecutive_errors = 0

    for step in range(max_steps):
        trim(messages, units)
        before = len(messages)
        msg = llm.chat(provider, messages, tools=None if mode != "native" else registry.schemas,
                        json_schema=schema)
        content = msg.get("content") or ""
        trace("thought", step=step, content=content)

        if mode == "native":
            calls = msg.get("tool_calls")
            if not calls:
                rescued = extract_json(content)  # fallback: model wrote the call as plain text
                if rescued and "tool" in rescued:
                    calls = [{"function": {"name": rescued["tool"], "arguments": rescued.get("args", {})}}]
                    trace("rescued_tool_call_from_text", step=step)
                elif content.strip():
                    return _finish("final", content, messages)
                else:
                    # Empty content AND no tool call isn't a real answer, it's the model going
                    # silent mid-turn (observed sampling variance, ~1 in 9 native runs). Treat it
                    # like any other malformed turn: feed it back and ask again, capped by the
                    # same 3-strike guard as the react branch below, instead of returning a blank
                    # "final" answer.
                    messages.append({"role": "assistant", "content": content, "tool_calls": None})
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        return _finish("gave_up", "Stopped: model kept replying with nothing.", messages)
                    messages.append({"role": "user", "content": "Error: empty reply with no tool "
                                     "call. Call a tool, or give your final answer as plain text."})
                    units.append(len(messages) - before)
                    continue
            consecutive_errors = 0
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for c in calls:
                name = c["function"]["name"]
                args = c["function"].get("arguments", {})
                if isinstance(args, str):
                    args = extract_json(args) or {}
                result = _execute(name, args, registry, confirm, seen_calls, trace)
                messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": result})
        else:
            messages.append({"role": "assistant", "content": content})
            obj = extract_json(content)
            if not isinstance(obj, dict) or "tool" not in obj:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return _finish("gave_up", "Stopped: model kept replying with invalid output.", messages)
                messages.append({"role": "user", "content": 'Error: reply with exactly one JSON object '
                                 'like {"thought": "...", "tool": "...", "args": {...}}.'})
                units.append(len(messages) - before)
                continue
            consecutive_errors = 0
            name, args = obj["tool"], obj.get("args") or {}
            if name == "final_answer":
                return _finish("final", str(args.get("answer", "")), messages)
            result = _execute(name, args, registry, confirm, seen_calls, trace)
            messages.append({"role": "user", "content": f"Observation: {result}"})
        units.append(len(messages) - before)

    return _finish("max_steps", "Stopped: reached max_steps without a final answer.", messages)


def _execute(name, args, registry, confirm, seen_calls, trace):
    key = (name, json.dumps(args, sort_keys=True, default=str))
    seen_calls[key] = seen_calls.get(key, 0) + 1
    if seen_calls[key] > REPEAT_THRESHOLD:
        trace("repeat_guard", name=name, args=args)
        return "Error: you already called this exact tool with these exact arguments. Try something else."
    if registry.is_dangerous(name) and not confirm(name, args):
        result = "User denied this action."
    else:
        result = registry.run(name, args)
    trace("action", name=name, args=args, result=result)
    return truncate(result)


def _finish(outcome, answer, messages):
    return {"outcome": outcome, "answer": answer, "messages": messages}


# For a one-off run, see main.py -- the real entry point: `python -m agent "<task>"`.
