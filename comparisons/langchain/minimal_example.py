"""Chapter 10's "17 lines" beat: this is the whole wiring for LangChain's "chat with your
dataframe" agent -- no error handling, one question, one run. Compare this file's own line count
against agent/'s ~650 lines, then read docs/RESULTS.md for what those handful of lines sit on top
of: langchain_experimental alone is tens of thousands of lines, before langchain_core,
langchain_classic and langchain_openai.

Needs: pip install -r requirements.txt, and a local Ollama server with qwen3.5:4b pulled.
"""
import os

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "data")
df = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))

llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen3.5:4b")

# allow_dangerous_code=True is not optional -- it's the confirmation checkbox this whole
# chapter is about. Without it, this next line raises ValueError.
agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True, verbose=True)

result = agent.invoke({"input": "What is the total eur_amount across all transactions?"})
print(result["output"])
