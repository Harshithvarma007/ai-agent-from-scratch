import json

import pytest
from conftest import reply

from agent import agent as agent_mod
from agent import tools
from agent.agent import extract_json, run_agent, trim, truncate
from agent.tools import Registry, tool


@tool()
def add(a: int, b: int):
    """Add two integers."""
    return a + b


@tool(dangerous=True)
def delete_everything():
    """Pretend to do something dangerous."""
    return "deleted"


REG = Registry(add, delete_everything)


# ------------------------------------------------------------------ helpers


@pytest.mark.parametrize("text,expected", [
    ('{"tool": "add", "args": {"a": 1}}', {"tool": "add", "args": {"a": 1}}),
    ('Sure! {"tool": "add", "args": {"a": 1, "b": 2}} hope that helps', {"tool": "add", "args": {"a": 1, "b": 2}}),
    ('<think>maybe {"tool": "wrong"}</think>{"tool": "right"}', {"tool": "right"}),
    ('{"thought": "a } in a string", "tool": "x"}', {"thought": "a } in a string", "tool": "x"}),
    ('{"thought": "escaped \\" quote }", "tool": "x"}', {"thought": 'escaped " quote }', "tool": "x"}),
    ('{broken {"tool": "ok"}', {"tool": "ok"}),
    ("no json here", None),
    ("", None),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_truncate():
    assert truncate("short") == "short"
    long = "a" * 800 + "b" * 500 + "c" * 800
    out = truncate(long)
    assert out.startswith("a" * 800) and out.endswith("c" * 800)
    assert "[500 chars elided]" in out


def test_trim_drops_whole_old_steps_but_keeps_system_and_task(monkeypatch):
    monkeypatch.setattr(agent_mod, "MAX_HISTORY_CHARS", 400)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    units = []
    for i in range(5):
        msgs += [{"role": "assistant", "content": f"call {i}"},
                 {"role": "tool", "content": "x" * 60}]
        units.append(2)
    trim(msgs, units)
    assert msgs[0]["content"] == "sys" and msgs[1]["content"] == "task"
    assert sum(len(json.dumps(m)) for m in msgs) <= 400
    assert msgs[2]["role"] == "assistant"  # never orphan a tool result from its call
    assert msgs[-1]["role"] == "tool" and msgs[-2]["content"] == "call 4"


# ------------------------------------------------------------------ native mode


def test_native_tool_call_then_answer(provider, scripted_chat):
    calls = scripted_chat(reply("", ("add", {"a": 2, "b": 3})), reply("The answer is 5."))
    events = []
    result = run_agent(provider, "2+3?", REG, trace=lambda k, **i: events.append((k, i)))
    assert result["outcome"] == "final" and result["answer"] == "The answer is 5."
    assert calls[0]["tools"] == REG.schemas
    tool_msg = result["messages"][-1]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_0", "content": "5"}
    assert ("action", {"name": "add", "args": {"a": 2, "b": 3}, "result": "5"}) in events


def test_arguments_as_a_json_string(provider, scripted_chat):
    scripted_chat(reply("", ("add", '{"a": 4, "b": 5}')), reply("9"))
    result = run_agent(provider, "4+5?", REG)
    assert result["messages"][-1]["content"] == "9"


def test_silent_reply_is_retried_not_returned(provider, scripted_chat):
    calls = scripted_chat(reply(""), reply("Done: 42."))
    result = run_agent(provider, "q", REG)
    assert result == {**result, "outcome": "final", "answer": "Done: 42."}
    assert "empty reply with no tool call" in calls[1]["messages"][-1]["content"]


def test_silent_reply_three_times_gives_up(provider, scripted_chat):
    scripted_chat(reply(""), reply(""), reply(""))
    result = run_agent(provider, "q", REG)
    assert result["outcome"] == "gave_up"


def test_tool_call_written_as_text_is_rescued(provider, scripted_chat):
    scripted_chat(reply('I will call {"tool": "add", "args": {"a": 1, "b": 1}}'), reply("2"))
    events = []
    result = run_agent(provider, "1+1?", REG, trace=lambda k, **i: events.append(k))
    assert "rescued_tool_call_from_text" in events
    assert result["answer"] == "2"
    assert any(m.get("role") == "tool" and m["content"] == "2" for m in result["messages"])


def test_repeat_guard(provider, scripted_chat):
    same = reply("", ("add", {"a": 1, "b": 1}))
    scripted_chat(same, same, same, same, reply("stop"))
    events = []
    result = run_agent(provider, "q", REG, trace=lambda k, **i: events.append(k))
    tool_results = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert tool_results[:3] == ["2", "2", "2"]
    assert tool_results[3].startswith("Error: you already called this exact tool")
    assert events.count("repeat_guard") == 1


def test_max_steps(provider, scripted_chat):
    scripted_chat(*[reply("", ("add", {"a": i, "b": 0})) for i in range(3)])
    result = run_agent(provider, "q", REG, max_steps=3)
    assert result["outcome"] == "max_steps"


def test_dangerous_tool_needs_confirmation(provider, scripted_chat):
    scripted_chat(reply("", ("delete_everything", {})), reply("ok"))
    asked = []
    result = run_agent(provider, "q", REG,
                       confirm=lambda name, args: asked.append(name) or False)
    assert asked == ["delete_everything"]
    assert result["messages"][-1]["content"] == "User denied this action."


def test_history_is_spliced_after_the_task(provider, scripted_chat):
    calls = scripted_chat(reply("fine"))
    run_agent(provider, "task", REG, history=[{"role": "assistant", "content": "earlier"}])
    assert [m["content"] for m in calls[0]["messages"][1:]] == ["task", "earlier"]


# ------------------------------------------------------------------ prompt (react) mode


def test_react_mode(provider, scripted_chat):
    calls = scripted_chat(
        reply('{"thought": "add them", "tool": "add", "args": {"a": 20, "b": 22}}'),
        reply('{"thought": "done", "tool": "final_answer", "args": {"answer": "42"}}'),
    )
    result = run_agent(provider, "20+22?", REG, mode="react")
    assert result["outcome"] == "final" and result["answer"] == "42"
    assert calls[0]["tools"] is None
    assert "add(a, b): Add two integers." in calls[0]["messages"][0]["content"]
    assert calls[1]["messages"][-1] == {"role": "user", "content": "Observation: 42"}


def test_react_mode_invalid_output_gives_up(provider, scripted_chat):
    calls = scripted_chat(reply("hmm"), reply("still not json"), reply("nope"))
    result = run_agent(provider, "q", REG, mode="react")
    assert result["outcome"] == "gave_up"
    assert "exactly one JSON object" in calls[1]["messages"][-1]["content"]


def test_constrained_mode_sends_a_schema_with_the_tool_enum(provider, scripted_chat):
    calls = scripted_chat(reply('{"thought": "", "tool": "final_answer", "args": {"answer": "x"}}'))
    run_agent(provider, "q", REG, mode="react+constrained")
    schema = calls[0]["json_schema"]
    assert schema["properties"]["tool"]["enum"] == ["add", "delete_everything", "final_answer"]
    # the shared template must not be mutated between runs
    assert "enum" not in agent_mod.CONSTRAINED_SCHEMA_TEMPLATE["properties"]["tool"]


def test_real_tools_end_to_end(provider, scripted_chat, workspace):
    (workspace / "data" / "payments.csv").write_text("merchant,eur_amount\nA,1.5\nB,2.5\nA,3\n")
    code = "import csv; print(sum(float(r['eur_amount']) for r in csv.DictReader(open('data/payments.csv'))))"
    scripted_chat(reply("", ("read_file", {"path": "data/payments.csv"})),
                  reply("", ("run_python", {"code": code})),
                  reply("The total eur_amount is 7.0."))
    result = run_agent(provider, "total?", Registry(*tools.ALL_TOOLS))
    observations = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert observations[0].startswith("merchant,eur_amount")
    assert observations[1].strip() == "7.0"
    assert result["answer"] == "The total eur_amount is 7.0."
