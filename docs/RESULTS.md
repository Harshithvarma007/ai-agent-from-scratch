# Measured results

Every number here comes from real runs of the code in this repo, on free local models, measured
on 2026-09-28. These are small, bounded passes (3 runs per question per cell), so read them as
"what we saw", not certified rates. Re-run them yourself with `bench/` (see the end of this
page).

**Setup:** Apple M5 Max, 36 GB · Ollama 0.34.4 · `max_steps=6` · temperature 0.2 · no seed
(Ollama's `/v1` endpoint doesn't expose one) · the 11-question golden set in
`bench/golden.yaml`, scored by `bench/bench_native_vs_react.py`.

## 1. This agent, `qwen3.5:4b`, all 11 questions

| mode | runs | correct | wrong | no answer | confidently wrong | pass rate | avg s/run |
|---|---|---|---|---|---|---|---|
| native (`tools=` API) | 33 | 30 | 3 | 0 | 3 | **91%** | 18.1 |
| react (JSON in the prompt) | 33 | 28 | 5 | 0 | 5 | **85%** | 16.0 |

An earlier version of the loop accepted "empty reply, no tool call" as a blank final answer. On
that version, react *beat* native (91% vs 82%), and native had 3 runs with no answer. Fixing that
one bug (step 6 in the build) recovered exactly those 3 runs, and the ranking flipped. The lesson:
a bug in your own loop can flip a leaderboard-style claim. Which mode wins depends on your task and
your loop, so measure it.

## 2. The trap question

`bench/golden.yaml`'s `ours_11` asks for the total `eur_amount` of merchant `Zorblax_Fitness`.
That merchant doesn't exist. The correct answer is "it isn't in the data".

| model | mode | runs | correct | confidently wrong |
|---|---|---|---|---|
| `qwen3.5:4b` | native | 3 | 0 | 3 |
| `qwen3.5:4b` | react | 3 | 0 | 3 |
| `qwen3:8b` | native | 3 | 0 | 3 |
| `qwen3:8b` | react | 3 | 0 | 3 |
| `llama3.1:8b` | native | 3 | 0 | 3 |
| `llama3.1:8b` | react | 3 | 0 | 3 |
| **total** | | **18** | **0** | **18** |

The scorer counts an answer as fabricated when its concluding sentence states a number. The
8B models simply answer `0.0`. `qwen3.5:4b` in native mode is subtler: it writes "Zorblax_Fitness
does not exist in payments.csv", lists the real merchants, and then still ends with
**"Total eur_amount for Zorblax_Fitness: 0"**. Filtering to zero rows and summing gives `0`, and
the model reports it. The agent "works", and the answer is still wrong. That's
deliberate. This repo is the *basic* agent, with no data profiling and no verification. Making it
trustworthy is the sequel.

Sampling varies. In one later session in the web app, `qwen3.5:4b` in native mode stopped at
"doesn't exist" 3 times out of 3, while prompt-based mode still answered `0`. Run it a few times.

## 3. Framework comparisons (`comparisons/`)

Same dataset, same model (`qwen3.5:4b`), same 11 questions, same scorer:

| implementation | runs | correct | pass rate | avg s/run |
|---|---|---|---|---|
| this agent, native | 33 | 30 | 91% | 18.1 |
| LangChain `create_pandas_dataframe_agent`, default (ReAct) | 33 | 30 | 91% | 17.3 |
| LangChain, explicit `tool-calling` | 33 | 27 | 82% | 16.0 |

LangGraph's numbers and its checkpointing demo are in `comparisons/langgraph/README.md`.
LangChain's trap-question result was 2 of 6 correct; the other 4 fabricated `0.0`.

**Size:** this agent's six files total 662 lines with comments. The LangChain version is 17 lines
of code, on top of roughly 170,000 lines of installed framework code (`langchain_experimental`
16,627 + `langchain_core` 70,299 + `langchain_classic` 70,523 + `langchain_openai` 14,266).

## 4. Re-measure it yourself

```bash
pip install -e ".[bench]"
ollama pull qwen3.5:4b
python bench/bench_native_vs_react.py --runs 3 --max-steps 6      # ~15-20 min on a laptop GPU
python bench/analyze_bench.py bench/results/data_analyst_bench.jsonl model mode
```

Expect your numbers to move by a few runs either way: these are 33-run cells on small, sampled
models.
