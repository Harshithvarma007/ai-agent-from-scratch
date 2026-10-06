# steps/: the build, as runnable checkpoints

The same eight steps as the notebook, one script each. Run them from the **repo root** after
`pip install -e .` and `python scripts/download_data.py`:

| File | Step |
|---|---|
| `hello_llm.py` | step 0: a complete LLM client in ~12 stdlib lines (Gemini; needs `GEMINI_API_KEY`) |
| `loop_step1.py` | one call: still a chatbot |
| `loop_step2.py` | offer the tools; the model asks, nothing runs |
| `loop_step3.py` | run the tool call |
| `loop_step4.py` | the loop |
| `loop_step5.py` | a hard ceiling (`MAX_STEPS`) |
| `loop_step6.py` | the silent-reply fix |
| `loop_step7.py` | rescuing a tool call written as text |
| `loop_step8.py` | the repeat guard: the finished loop |
| `replay.py` + `traces/` | replay a recorded real failure through any step |
| `demo/` | one tool call by hand, with `curl` |

```bash
python steps/loop_step4.py "How many distinct merchants are in data/payments.csv?"
AGENT_PROVIDER=gemini python steps/loop_step8.py
python steps/replay.py steps/traces/silent_reply.json steps/loop_step4.py   # before the fix
python steps/replay.py steps/traces/silent_reply.json steps/loop_step6.py   # after
```

The traces were captured from real runs of the real agent by `bench/capture_traces.py`.
