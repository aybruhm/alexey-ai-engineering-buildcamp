# AI Engineering Buildcamp

Hands-on materials and homework for the **AI Engineering Buildcamp** by Alexey Grigorev.

This repository walks through building real AI applications with LLMs: calling language models, retrieval-augmented generation (RAG), structured outputs, and streaming.

## What's inside

### Week 1 — Foundations

| Notebook | Topic |
|---|---|
| `00-llm.ipynb` | Calling LLMs with the Anthropic SDK |
| `01-rag-intro.ipynb` | Retrieval-augmented generation: indexing documents and searching |
| `02-rag-extended.ipynb` | Extending RAG: keyword search, semantic search, and filters |
| `03-structured-outputs.ipynb` | Structured outputs with Pydantic |
| `04-structured-outputs-with-rag.ipynb` | Structured outputs combined with RAG |
| `05-streaming-structured-outputs-with-rag.ipynb` | Streaming structured outputs |

### Week 1 — Homework

`week-1/homework/` contains the homework assignment: download a set of free books, convert the PDFs to Markdown with `markitdown` + Anthropic Claude, chunk the text with a sliding window, and index it with `minsearch`.

### Week 2 — Reusable RAG

| Notebook | Topic |
|---|---|
| `01-reusable-rag.ipynb` | Building a reusable RAG pipeline: documents → index → retrieval → prompt → LLM |

### Week 2 — Homework

- `week-2/homework/01-pulumi-docs-rag/` — Pulumi Docs RAG: index the Pulumi docs site and answer developer questions with a RAG system.
- `week-2/homework/02-pdf-extraction-rag/` — PDF extraction + RAG: extract a PDF textbook into structured page JSONs with Claude, then index the pages and answer questions with structured outputs.
- `week-2/homework/03-pdf-extraction-rag-with-elasticsearch/` — PDF extraction + RAG with Elasticsearch: same pipeline as `02-pdf-extraction-rag`, but full-text search is backed by Elasticsearch instead of sqlitesearch.
- `week-2/homework/04-pdf-extraction-rag-with-qdrant/` — PDF extraction + RAG with Qdrant: same pipeline as `02-pdf-extraction-rag`, but retrieval uses vector search backed by Qdrant with Jina embeddings.

### Week 3 — Agentic RAG

| Notebook | Topic |
|---|---|
| `01-agentic-rag-with-tool-calling.ipynb` | Agentic RAG with tool calling: exposing the `search` tool so the model retrieves on demand |
| `02-agentic-loop.ipynb` | The agentic loop: multi-round tool calling with the `search` and `add_entry` tools, feeding results back until a final answer |
| `03-qa-agentic-loop.ipynb` | Interactive Q&A agentic loop: a multi-turn chat that reuses the conversation history and scales `max_tokens` as it grows |
| `04-toyaikit-agent.ipynb` | Building an agent with Toyaikit: exposing index tools and running the agent loop |
| `05-toyaikit-agent-organizing-tools.ipynb` | Organizing agent tools with Toyaikit |
| `06-agentic-search-beyond-rag.ipynb` | Agentic search beyond RAG: letting the agent decide when and how to search |
| `07-pydantic-ai-agents.ipynb` | Building agents with Pydantic AI |
| `08-web-search-agent-pydanticai.ipynb` | A web-search agent built with Pydantic AI |

Shared helpers in `week-3/`: `search_tools.py` (index tools for the agent) and `invoke_llm_trace.py` (OpenTelemetry/Phoenix tracing setup).

### Week 3 — MCP

`week-3/09-mcp/` exposes the docs search tools over the Model Context Protocol (MCP):

- `main.py` — a FastMCP server that exposes the search tools over stdio.
- `run_mcp.ipynb` — an MCP client that launches the server as a subprocess and calls its tools.
- `run_mcp_with_pydanticai.ipynb` — driving the MCP server's tools from a Pydantic AI agent.
- `search_tools.py`, `utils.py`, `invoke_llm_trace.py` — the server's tools, documentation helpers, and tracing setup.

### Week 3 — Homework

- `week-3/homework/01-wikipedia-agent/` — Wikipedia agent: answer questions by searching Wikipedia with an agentic loop.

### Week 4 — Documentation Agent

`week-4/` is a standalone, installable package rather than a notebook. It answers questions about the
Pulumi documentation: the agent searches a local index, retrieves whole files when it needs them, and
returns a structured answer.

It ships its own `pyproject.toml`, `uv.lock`, `.venv` and `.gitignore`, so it is installed and run
independently of the root project.

| Module | Purpose |
|---|---|
| `pulumi_docs_agent/__init__.py` | OpenTelemetry/OpenInference tracing setup for Phoenix |
| `pulumi_docs_agent/entrypoint.py` | CLI entry point: downloads the docs, builds the index, runs the agent |
| `pulumi_docs_agent/agent.py` | Pydantic AI agent construction, blocking and streaming runs |
| `pulumi_docs_agent/helpers.py` | Streaming, tool-call, and message printing helpers |
| `pulumi_docs_agent/models.py` | `RAGResponse`, `Reference` and `Check` structured output models |
| `pulumi_docs_agent/pricing.py` | Model pricing and cost calculation, used by the test suite |
| `pulumi_docs_agent/prompts.py` | Agent instructions, self-verification checklist, and judge prompt templates |
| `pulumi_docs_agent/retrieval.py` | minsearch index and highlighter |
| `pulumi_docs_agent/tools.py` | `SearchTools`: `search`, `add_entry`, `get_file` |
| `pulumi_docs_agent/utils.py` | Downloads the Pulumi docs and reuses the local copy |

The documentation is downloaded into `week-4/docs/` on first run and reused afterwards, so only the
first run is slow.

Run it from the project directory:

```bash
cd src/alexey_ai_engineering_buildcamp/week-4
uv sync
uv run pulumi-docs-agent
```

`make run` from the repository root does the same thing.

Tracing is configured as soon as the package is imported, so `PHOENIX_API_KEY`,
`PHOENIX_COLLECTOR_ENDPOINT` and `PHOENIX_PROJECT_NAME` must be set alongside `ANTHROPIC_API_KEY`.

#### Tests

`pulumi_docs_agent/tests/` drives the real agent and grades it in two layers.

**Deterministic** — one model call per case and no judge, so this is the layer to run on every change:

| File | Purpose |
|---|---|
| `tests/conftest.py` | Module-scoped fixture that builds the index and the agent once per test module |
| `tests/test_cases.py` | Table-driven contract checks: every cited path exists in the corpus, `found_answer` agrees with the citations, tool-call budget, search-before-fetch ordering, prompt-injection safety |
| `tests/test_agent.py` | Asserts the answer is populated, that both `search` and `get_file` get called, and that answers contain code and references |
| `tests/utils.py` | Tool-call collection, token-usage capture, and cost reporting |

**LLM as a judge** — a second model grades answer quality against a rubric:

| File | Purpose |
|---|---|
| `tests/llm_judge.py` | Builds the judge agent and asserts every criterion it returns |
| `tests/test_judge.py` | Rubrics for answer quality, plus scenario tests for an ambiguous query and for adapting to the user's existing project |

Run both layers, or the cheap one on its own:

```bash
cd src/alexey_ai_engineering_buildcamp/week-4
uv run pytest                                          # both layers
uv run pytest pulumi_docs_agent/tests/test_cases.py    # deterministic only
```

`make tests` from the repository root runs everything.

These tests call the model for real, so they need a valid `ANTHROPIC_API_KEY`, they take a while, and they consume tokens — the judge layer roughly doubles the cost per case. Each test records its token usage and the session prints the total cost once the run finishes.

## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

2. Create a `.env` file from the template and add your API keys:

   ```bash
   cp .env.template .env
   # then edit .env and set ANTHROPIC_API_KEY=...
   # for the Elasticsearch homework also set ELASTICSEARCH_HOST and ELASTICSEARCH_API_KEY
   # for the Qdrant homework also set QDRANT_URL and QDRANT_API_KEY
   ```

3. Launch Jupyter and open the notebooks:

   ```bash
   uv run jupyter lab
   ```

## Project structure

```bash
.
├── pyproject.toml
├── uv.lock
├── README.md
├── Makefile
└── src/
    └── alexey_ai_engineering_buildcamp/
        ├── week-1/
        │   ├── 00-llm.ipynb
        │   ├── 01-rag-intro.ipynb
        │   ├── 02-rag-extended.ipynb
        │   ├── 03-structured-outputs.ipynb
        │   ├── 04-structured-outputs-with-rag.ipynb
        │   ├── 05-streaming-structured-outputs-with-rag.ipynb
        │   └── homework/
        │       ├── books.csv
        │       └── index.ipynb
        ├── week-2/
        │   ├── 01-reusable-rag.ipynb
        │   └── homework/
        │       ├── 01-pulumi-docs-rag/
        │       │   └── index.ipynb
        │       ├── 02-pdf-extraction-rag/
        │       │   ├── index.ipynb
        │       │   ├── extraction.py
        │       │   ├── models.py
        │       │   ├── pricing.py
        │       │   ├── prompt.py
        │       │   ├── rag.py
        │       │   ├── retrieval.py
        │       │   └── utils.py
        │       ├── 03-pdf-extraction-rag-with-elasticsearch/
        │       │   ├── index.ipynb
        │       │   ├── extraction.py
        │       │   ├── models.py
        │       │   ├── pricing.py
        │       │   ├── prompt.py
        │       │   ├── rag.py
        │       │   ├── retrieval.py
        │       │   └── utils.py
        │       └── 04-pdf-extraction-rag-with-qdrant/
        │           ├── index.ipynb
        │           ├── extraction.py
        │           ├── models.py
        │           ├── pricing.py
        │           ├── prompt.py
        │           ├── rag.py
        │           ├── retrieval.py
        │           └── utils.py
        ├── week-3/
        │   ├── 01-agentic-rag-with-tool-calling.ipynb
        │   ├── 02-agentic-loop.ipynb
        │   ├── 03-qa-agentic-loop.ipynb
        │   ├── 04-toyaikit-agent.ipynb
        │   ├── 05-toyaikit-agent-organizing-tools.ipynb
        │   ├── 06-agentic-search-beyond-rag.ipynb
        │   ├── 07-pydantic-ai-agents.ipynb
        │   ├── 08-web-search-agent-pydanticai.ipynb
        │   ├── 09-mcp/
        │   │   ├── main.py
        │   │   ├── search_tools.py
        │   │   ├── utils.py
        │   │   ├── invoke_llm_trace.py
        │   │   ├── run_mcp.ipynb
        │   │   └── run_mcp_with_pydanticai.ipynb
        │   ├── search_tools.py
        │   ├── invoke_llm_trace.py
        │   └── homework/
        │       └── 01-wikipedia-agent/
        │           ├── index.ipynb
        │           └── invoke_llm_trace.py
        └── week-4/
            ├── .env.template
            ├── .gitignore
            ├── .python-version
            ├── pyproject.toml
            ├── uv.lock
            └── pulumi_docs_agent/
                ├── __init__.py
                ├── agent.py
                ├── entrypoint.py
                ├── helpers.py
                ├── models.py
                ├── pricing.py
                ├── prompts.py
                ├── retrieval.py
                ├── tests/
                │   ├── __init__.py
                │   ├── conftest.py
                │   ├── llm_judge.py
                │   ├── test_agent.py
                │   ├── test_cases.py
                │   ├── test_judge.py
                │   └── utils.py
                ├── tools.py
                └── utils.py
```

## Requirements

- Python 3.14
- An Anthropic API key
