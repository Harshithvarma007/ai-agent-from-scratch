# The same agent in LangGraph (working code, not a description)

Companion to `../langchain/`. Same dataset, same local Ollama model, same golden set
(`bench/golden.yaml`), same `score()` function — so all three builds (from-scratch, LangChain,
LangGraph) are graded identically. This file covers what was built here and what running
it showed.

**Everything below was actually run on this machine, 2026-09-28** (Ollama `0.34.4`,
`qwen3.5:4b`, same M5 Max machine as `docs/RESULTS.md`), not copied from docs or estimated.

## Files

| File | What it's for |
|---|---|
| `minimal_example.py` | the whole wiring for one question, one run. Reuses `agent/tools.py`'s `read_file`/`write_file`/`run_python` **completely unchanged** — no rewrite needed. |
| `compare.py` | the full comparison: `langgraph.prebuilt.create_react_agent` against the same 11-question golden set, the same scorer `bench/` and `comparisons/langchain/` both use. |
| `checkpointing_demo.py` | a live, two-process test of state persistence (does a resumed run actually survive a process restart?) and the human-in-the-loop approval loop that uses the same primitive. Not a benchmark — read its own docstring and run it yourself. |
| `requirements.txt` | pinned `langgraph==1.2.12` and friends, confirmed installed and current as of today. |

## Run it

```bash
# from the repo root, in the same venv as the agent
pip install -r comparisons/langgraph/requirements.txt
cd comparisons/langgraph
python3 compare.py --check-only                     # confirms the default tool-error crash below
python3 minimal_example.py                           # the ~20-line version, one question
python3 compare.py --runs 3                          # the full golden-set sweep
python3 checkpointing_demo.py --phase 1               # pause before the first tool call, exit
python3 checkpointing_demo.py --phase 2               # new process, resume from disk
python3 ../bench/analyze_bench.py results/langgraph_bench.jsonl model
```

---

## 1. Is LangGraph itself in better shape than `langchain-experimental`? Checked live, not asserted.

| | `langgraph` | `langchain-experimental` (what chapter 9 is built on) |
|---|---|---|
| Archived? | **No** (`archived: false`, GitHub API, checked 2026-09-28) | **Yes**, archived 2026-05-26 |
| Stars | 42,395 | 104 |
| Last push | 2026-09-27 (yesterday) | 2026-05-22 (the day it was archived) |
| Commits, trailing 52 weeks | **784** (GitHub `stats/commit_activity`, summed) | n/a — archived, read-only |
| Commits, trailing 4 weeks | 23 | 0 |
| License | MIT | MIT |
| Latest release | `1.2.12` (2026-09, `pip index versions langgraph`) | `0.4.2` — last release before archival |

Same check for the two packages this project already cited as declining: `langchain-community`
is also archived (`archived: true`, 287 stars) — and its GitHub **description field itself** reads
`"⚠️ No longer maintained, see linked issue"`, not just a banner in the README. Main
`langchain-ai/langchain` (the package `langgraph.prebuilt`'s replacement now lives in): 147,188
stars, not archived, 2,895 commits in the trailing 52 weeks, pushed today.

**Verdict: yes, LangGraph is in a genuinely different, actively-maintained category from
`langchain-experimental` — this isn't spin, it's the same GitHub API check applied to both
packages.** The honest nuance is in section 3 below: the specific *function* this comparison
uses (`create_react_agent`) is itself mid-migration, which is a different claim from "the
package is unmaintained."

## 2. The tool-set is reused, not reimplemented — and that's itself a finding

`create_pandas_dataframe_agent` **requires** its own tool (`PythonAstREPLTool`, a bare
`exec()`/`eval()` in-process, per `comparisons/langchain/README.md`). `create_react_agent` ships
**no default tool at all** — you pass in whatever callables you want. `minimal_example.py` and
`compare.py` both pass `agent/tools.py`'s `ALL_TOOLS` list **directly, completely unchanged**:
the same workspace-jailed `read_file`/`write_file`, the same subprocess-sandboxed `run_python`
from chapter 6/9. Confirmed by passing the raw, undecorated functions straight to
`create_react_agent(tools=...)` and having it build correct OpenAI-style tool schemas from their
signatures and docstrings with zero adapter code — `langchain_core.utils.function_calling.
convert_to_openai_tool` does the same "signature + docstring → schema" trick our own `tools.py`
`@tool` decorator does by hand (chapter 4's whole point), just inside the library instead of
in ~40 lines we wrote ourselves.

**What this means for the line-count framing:** unlike LangChain's pandas agent, adopting
LangGraph does not force you to give up your own sandboxing. The "N lines of wiring" in
`minimal_example.py` sit on top of `agent/tools.py`'s 182 lines *of your own, already-audited
code*, not a replacement for it.

## 3. `create_react_agent` is deprecated — confirmed from source, not a rumor

Every run in this folder prints:

```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please
update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0
to be removed in V2.0.
```

Read directly from the installed package (`langgraph/prebuilt/chat_agent_executor.py`):

```python
@deprecated(
    "create_react_agent has been moved to `langchain.agents`. Please update your import to "
    "`from langchain.agents import create_agent`.",
    category=LangGraphDeprecatedSinceV10,
)
def create_react_agent(...):
```

**What this actually means, checked, not guessed:** `langchain.agents.create_agent` (in the main
`langchain` package, version `1.4.2` as of today) is the recommended replacement, and it **still
returns a `CompiledStateGraph`** — the same LangGraph runtime object `create_react_agent` returns.
This is a consolidation of the *entry point*, not a move away from LangGraph as the execution
engine. Confirmed directly: `type(create_agent(llm, tools=[...]))` is
`langgraph.graph.state.CompiledStateGraph`, identical to `create_react_agent`'s return type.

**What did NOT carry over cleanly, honestly reported:** `create_react_agent`'s `tools=` parameter
accepts a pre-built `ToolNode` (so you can set `handle_tool_errors=True` — see §4). `create_agent`'s
`tools=` parameter does **not** — passing a `ToolNode` raises `TypeError: 'ToolNode' object is not
iterable` from `langchain/agents/factory.py`'s own tool-list comprehension (confirmed live, this
build). `create_agent` almost certainly exposes equivalent error-handling control through its new
`middleware=` argument instead, but this pass did not chase down the exact replacement API — **that
is an open question, not a claim**, and it's exactly the kind of thing worth re-checking before
relying on it, since this ecosystem is still visibly in motion.

**In short:** "LangGraph the package is thriving; the specific prebuilt helper function
this comparison calls is being folded into the main `langchain` package's new agent API, and both
sit on the same underlying graph runtime" — not "LangGraph is being deprecated." The same
distinction as for `langchain-experimental` vs. `langchain` itself, one level
deeper into the stack.

## 4. A real, live default-error-handling crash — reproduced twice, first try

`ToolNode`'s default `handle_tool_errors` (read from `langgraph/prebuilt/tool_node.py`) only
converts its own `ToolInvocationError` (bad/missing arguments) into a message the model can read.
**Any exception your tool function itself raises is re-raised, uncaught, out of `agent.invoke()`.**
This is the same category of finding as `comparisons/langchain`'s parser-crash (finding #4): a
default that lets a very ordinary mistake — the model passing `read_file(path="payments.csv")`
instead of `"data/payments.csv"` — crash the whole run instead of feeding the error back as an
observation, which is exactly what our own `Registry.run()` (chapter 6) does for free, unconditionally.

Full traceback, captured live the first time `minimal_example.py` was actually run (the model
guessed a relative path without the `data/` prefix):

```
File ".../agent/tools.py", line 119, in read_file
    raise FileNotFoundError(f"no such file: {path}")
FileNotFoundError: no such file: payments.csv
During task with name 'tools' and id '...'
```

**Unlike LangChain's parser crash (which fired once in 3+66 runs and didn't reproduce in a later
pass), this one is common, not rare: 4 of 5 live runs of `minimal_example.py` while building this
file crashed this exact way** (`payments.csv`, then `workspace/payments.csv` — the model guesses a
different wrong relative path each time — and once correctly as `data/payments.csv`, which
succeeded). `compare.py --check-only` reproduces it on demand with a deliberately-missing file so
you don't have to rely on the model guessing wrong. **Expect it to crash more often than it doesn't.** The one-line fix —
`ToolNode(ALL_TOOLS, handle_tool_errors=True)`, used throughout `compare.py`'s actual benchmark
runs — is not the default, exactly parallel to LangChain's `handle_parsing_errors=True` not being
the default either. **Neither framework ships "errors become observations" turned on by default;
both require you to know to flip one flag** — LangGraph's default just fails far more often in
practice on this exact model/task than LangChain's did.

## 5. Golden-set results, `qwen3.5:4b`, 3 runs × 11 questions = 33 runs, complete

| build | agent config | n | correct | wrong | no_answer | confident_wrong | pass rate | avg s/run |
|---|---|---|---|---|---|---|---|---|
| **LangGraph** (`create_react_agent`, `handle_tool_errors=True`, our own tools) | this file | 33 | 29 | 3 | 1 | 3 | **88%** | 6.4 |
| From-scratch, native mode (`docs/RESULTS.md`) | — | 33 | 30 | 3 | 0 | 3 | 91% | 18.1 |
| LangChain, default `zero-shot-react-description` (`docs/RESULTS.md`) | — | 33 | 30 | 3 | 0 | 3 | 91% | 17.3 |
| LangChain, explicit `tool-calling` (`docs/RESULTS.md`) | — | 33 | 27 | 3 | 3 | 3 | 82% | 16.0 |

**All three builds land in the same 82–91% band on this golden set — no framework, and no
from-scratch loop, is the clear winner on raw correctness.** That's the AI-Agents-That-Matter
prediction (papers.md #9) holding a third time on a third independent codebase. LangGraph's per-run
latency (6.4s avg) is also the fastest of the four rows — plausibly because native tool-calling
plus a leaner dependency stack underneath means fewer round-trip tokens per step than a ReAct
text-protocol prompt, but this pass did not isolate that variable and it should be read as a
observation, not a proven cause.

**Per-question**, the two weakest were `ours_09` (67%, one confidently-wrong run — a
merchant-category-code lookup) and `ours_11` (33% — see §6). `ours_10`, the anti-join question
that broke LangChain's `tool-calling` mode 3/3 times, was **3/3 correct here** — consistent with
both other builds' pattern that this specific question rewards a ReAct-shaped or native-with-good-
prompting approach and punishes a rigid step budget.

## 6. The trap question (`ours_11`), on LangGraph — confirm, don't assume

3 runs, `qwen3.5:4b`: **1 correct, 2 confidently wrong.**

```
run 1 (wrong):    "The merchant 'Zorblax_Fitness' does not exist in the payments.csv dataset.
                   The unique merchants in the file are: ... Therefore, the total eur_amount
                   for merchant 'Zorblax_Fitness' is 0.0."

run 2 (correct):  "The merchant 'Zorblax_Fitness' does not exist in payments.csv. The dataset
                   only contains these 5 merchants: ... Therefore, there is no total eur_amount
                   for 'Zorblax_Fitness'."

run 3 (wrong):    "The total eur_amount for merchant 'Zorblax_Fitness' in payments.csv is 0.0
                   (the merchant does not appear in the dataset)."
```

**Same mechanism as the other two builds, word for word: the model correctly states the merchant
doesn't exist, then in 2 of 3 runs still reports a fabricated `0.0` as if it answered the question.**
Run 2 is a genuine, correctly-scored abstention — no invented number anywhere in the answer — so
this is *not* a total 0/3 wall the way the from-scratch grid's `ours_11` cell was (0/18 across
every model and mode tested, `docs/RESULTS.md`); it's a majority failure (2/3), the softest of
the three builds' numbers on this specific question. **Say "the same failure recurs in most
LangGraph runs too, less often than the from-scratch or LangChain builds on this 3-run sample" —
not "LangGraph fixes it" and not "identical rate."** A 3-run sample isn't enough to certify whether
33% vs. 0% vs. 17-33% (LangChain's two agent types) reflects a real difference in this exact model's
behavior under each harness's exact prompt wording, or noise — this is the same "bounded first
pass" caveat `docs/RESULTS.md` states for its own numbers.

## 7. Line counts, precisely

| What | Lines |
|---|---|
| `minimal_example.py`, real code lines (non-blank, non-comment, same method `docs/RESULTS.md` used) | **20** |
| `minimal_example.py`, `wc -l` | 27 |
| ...sitting on top of `agent/tools.py`, reused **unchanged** (your own code, not a dependency) | 182 |
| `langgraph` package installed (includes `.prebuilt`, `.checkpoint`, `.checkpoint.sqlite` — all install into the same namespace) | 41,487 |
| `langchain_core` (shared with the LangChain build) | 70,299 |
| `langchain_openai` (shared with the LangChain build) | 14,266 |
| **Total dependency lines under this build** | **126,052** |
| **Total dependency lines under the LangChain build** (`langchain_experimental` 16,627 + `langchain_core` 70,299 + `langchain_classic` 70,523 + `langchain_openai` 14,266, `docs/RESULTS.md` §2d) | **171,715** |

**LangGraph's dependency footprint is ~45,663 lines smaller than the LangChain build's for this
task** — it doesn't pull in `langchain_classic` (70,523 lines) at all, since `create_react_agent`
doesn't route through the older chains/agents machinery `create_pandas_dataframe_agent` does. Both
still share the same `langchain_core` + `langchain_openai` floor (84,565 lines) because both use
`ChatOpenAI` to reach Ollama.

The more important number for the "17 vs. 20" framing itself: LangGraph's ~20 lines reuse 182
lines of code you already wrote and audited (the sandboxing); LangChain's 17 lines replace that
same 182 lines with `PythonAstREPLTool`'s in-process `exec()`/`eval()` instead. Fewer new lines to
type is not the same claim as "less to trust" — see the research doc for the full argument.

## 8. Checkpointing and human-in-the-loop: tested, not just cited

`checkpointing_demo.py`, run for real as two separate process invocations against a file-backed
`SqliteSaver` (not `InMemorySaver` — a real file on disk, checked at 28,672 bytes after phase 1):

- **Phase 1** (one process): asks a question, `interrupt_before=["tools"]` pauses the graph before
  the first tool call executes, the process then exits with **no in-memory state kept anywhere**.
- **Phase 2** (a brand-new process, later): opens the same sqlite file, calls `agent.get_state(config)`
  **before invoking anything**, and it correctly reports the paused state — 2 messages, next node
  `('tools',)`, the exact pending tool call (`read_file({'path': 'data/payments.csv'})`) — recovered
  from disk, not from memory. `agent.invoke(None, config)` then resumes from exactly that point
  (does not re-ask the LLM the original question) and, after two more approval/resume cycles,
  finishes with a correct final answer ("5 distinct merchants").

**This is a real, verified yes to "can you resume an interrupted multi-step run" — not a docs
citation.** The same `interrupt_before` mechanism that made this resumable is also, directly, the
human-in-the-loop primitive: each pause is a point where a real human could inspect
`state.values["messages"][-1].tool_calls` and approve, edit, or reject before `run_python` (a
`dangerous=True` tool in our own taxonomy) actually executes — the durable version of what
`main.py`'s `make_confirm()` already does synchronously in the from-scratch build. See the research
doc for what it would cost to hand-roll the durable (survives-a-restart) version of that gate
ourselves, versus the one line (`checkpointer=SqliteSaver(...)`) it took here.

Not tested in this pass, cited from docs only (see the research doc): the newer dynamic
`interrupt()` + `Command(resume=...)` primitive (pausing *inside* a tool function itself, rather
than at a fixed graph edge like `interrupt_before` does) — both are real, current LangGraph
mechanisms, but only the `interrupt_before` form was exercised here.
