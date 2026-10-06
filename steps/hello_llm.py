import json, os, urllib.request

def ask(base_url, api_key, model, question):
    body = {"model": model, "messages": [{"role": "user", "content": question}]}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

print(ask("https://generativelanguage.googleapis.com/v1beta/openai/",
          os.environ["GEMINI_API_KEY"], "gemini-2.5-flash",
          "Explain what a pandas groupby does in one sentence."))
