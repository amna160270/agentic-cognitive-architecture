@'

\# agentic-cognitive-architecture



A fully local, deliberative AI agent built for Assignment 1: Single-Agent Cognitive Architecture, MCP Tool Grounding \& Hierarchical Memory with Local SLMs (MAI632/MAI713, COMSATS University Islamabad, Sahiwal Campus, Fall 2026).



The agent runs entirely on local Small Language Models via Ollama, follows a Thought-Action-Observation ReAct loop, grounds itself with MCP-style tools, keeps a two-tier memory system, and learns from failed attempts through Reflexion.



\## Features



\- \*\*Local SLM engine\*\* (`core/local\_llm.py`): wraps a locally running Ollama model with configurable temperature, top-p, context window and stop tokens, and reports latency and tokens-per-second.

\- \*\*ReAct deliberation loop\*\* (`core/deliberation.py`, `core/agent.py`): builds the system prompt and parses model output into Thought, Action and Final Answer steps. The parser tolerates small formatting mistakes common in SLMs.

\- \*\*MCP-style tools\*\* (`tools/`): calculator, sandboxed Python runner and file storage, all behind a common `BaseTool` interface with Pydantic argument schemas.

\- \*\*Two-tier memory\*\* (`memory/`): in-context working memory with compaction, plus a ChromaDB-backed episodic/archival vector store.

\- \*\*Reflexion\*\* (`core/reflexion.py`): retries failed tasks using short lessons generated from earlier failures.

\- \*\*Benchmarking\*\* (`benchmark/run\_benchmark.py`): runs the agent over a task list and saves success rate and timing to JSON.



\## Project Structure



\- `core/` - agent loop, local LLM client, ReAct parser, Reflexion

\- `tools/` - calculator, Python sandbox, file storage

\- `memory/` - working memory and episodic memory

\- `benchmark/` - benchmark runner

\- `tests/` - unit tests for memory, parser, sandbox and tools



\## Requirements



\- Python 3.10+

\- Ollama installed and running, with a model pulled (for example `qwen2.5-coder:1.5b` or `phi3:mini`)

\- Python packages: ollama, chromadb, pydantic, sentence-transformers, rich, tabulate, pytest



```bash

pip install ollama chromadb pydantic sentence-transformers rich tabulate pytest

ollama pull qwen2.5-coder:1.5b

```



\## Running Tests



```bash

pytest tests/

```



\## Acknowledgements



Built for Multi-Agent Systems \& Generative AI Agents, taught by Dr. Muhammad Farhan. Based on ReAct (Yao et al., 2022) and Reflexion (Shinn et al., 2023).

'@ | Set-Content -Path README.md -Encoding utf8

