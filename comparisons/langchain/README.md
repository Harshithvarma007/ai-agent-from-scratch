# The same agent in LangChain: what a framework gives you (built, not described)

Read this before running a single line of code.

**`langchain-ai/langchain-experimental` on GitHub** carries the banner:

> ⚠️ No longer maintained, see linked issue

Archived 2026-05-26 (read-only). **Issue #87, "Sunsetting `langchain-experimental`"**, names
`PythonREPL` specifically — the exact tool `create_pandas_dataframe_agent` is built on — and says:

> PythonREPL predates the safer execution environments we now recommend.

Its stated recommendation is to implement this kind of functionality directly in application
code rather than depend on the package. That's this whole repo's premise, in the framework's
own words, not ours. (`langchain-ai/langchain` itself, ~147k★, is thriving — it's specifically
the `-experimental` and `-community` satellite packages being wound down. So: "LangChain's own
experimental package," not "LangChain.")

**A real CVE for the default this package requires:** `allow_dangerous_code=True` gates
`create_pandas_dataframe_agent`. Langflow hardcoded it to `True` in a shipped product —
[GHSA-3645-fxcv-hqr4](https://github.com/langflow-ai/langflow/security/advisories/GHSA-3645-fxcv-hqr4),
critical severity, remote code execution via prompt injection.

**The tool itself, read from source** (`langchain_experimental/tools/python/tool.py`,
`PythonAstREPLTool._run`):

```python
exec(ast.unparse(module), self.globals, self.locals)
...
ret = eval(module_end_str, self.globals, self.locals)
```

No subprocess, no timeout, no scrubbed environment, no confirmation gate — verbatim "rung 0" of
our own `agent/sandbox_ladder.py`, the one we label "do not use."

## Files

| File | What it's for |
|---|---|
| `minimal_example.py` | 17 lines of code: the whole wiring for one question, one run |
| `compare.py` | the full comparison: both agent types, the same 11-question golden set, the same scorer as `bench/` |
| `requirements.txt` | pinned `langchain-experimental==0.4.2` and friends |

## Run it

```bash
# from the repo root, in the same venv as the agent
pip install -r comparisons/langchain/requirements.txt
cd comparisons/langchain
python3 compare.py --check-only                 # confirms allow_dangerous_code=True is still required
python3 minimal_example.py                       # the 17-line version, one question
python3 compare.py --runs 3                      # the full golden-set sweep, both agent types
python3 ../../bench/analyze_bench.py results/langchain_bench.jsonl model agent_type
```

See [`docs/RESULTS.md`](../../docs/RESULTS.md) for the measured numbers.
