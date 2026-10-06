# One tool call, by hand

Two raw requests to local Ollama. There's no agent code in between; you play "your code".

```bash
# 1. Offer an `add` tool and ask. The reply has tool_calls: add(a=17, b=25). Nothing ran.
curl -s localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d @demo/tool_call_1.json | python3 -m json.tool

# 2. Same conversation, plus the model's tool call and our result ("42", role "tool").
#    Now it answers: "17 + 25 = 42".
curl -s localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d @demo/tool_call_2.json | python3 -m json.tool
```

Run these from `steps/`. Both were checked on `qwen3.5:4b` on 2026-09-28: request 1 returned an
`add(17, 25)` tool call 5/5 times, and request 2 answered "17 + 25 = 42" 3/3 times.

- `content` in reply 1 is sometimes empty and sometimes a short "I will add 17 and 25 for you."
- The reply also includes a `reasoning` field (the model's thinking).
- `tool_call_2.json` uses a fixed id (`call_1`). Ollama doesn't check that it matches reply 1's
  id, but a hosted API may.
