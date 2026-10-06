# The web UI (and CLI) in a container. Pair it with Ollama via docker-compose.yml, or pass a
# free-tier API key and AGENT_PROVIDER. The container is itself a sandbox boundary for the
# model's run_python code: no host filesystem besides what you mount.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 PORT=8765 AGENT_WORKSPACE=/app/workspace AGENT_RUNS_DIR=/app/runs

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY agent ./agent
RUN pip install -r requirements.txt . && useradd --create-home --uid 10001 agent

COPY scripts ./scripts
COPY webapp ./webapp
COPY steps ./steps
RUN python scripts/download_data.py && mkdir -p /app/runs && chown -R agent:agent /app

USER agent
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/models', timeout=4)"
CMD ["python", "webapp/server.py"]
