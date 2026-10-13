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

### Week 3 — Homework

- `week-3/homework/01-wikipedia-agent/` — Wikipedia agent: answer questions by searching Wikipedia with an agentic loop.

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
        └── week-3/
            ├── 01-agentic-rag-with-tool-calling.ipynb
            ├── 02-agentic-loop.ipynb
            ├── 03-qa-agentic-loop.ipynb
            ├── 04-toyaikit-agent.ipynb
            ├── 05-toyaikit-agent-organizing-tools.ipynb
            ├── 06-agentic-search-beyond-rag.ipynb
            ├── 07-pydantic-ai-agents.ipynb
            ├── 08-web-search-agent-pydanticai.ipynb
            ├── search_tools.py
            ├── invoke_llm_trace.py
            └── homework/
                └── 01-wikipedia-agent/
                    ├── index.ipynb
                    └── invoke_llm_trace.py
```

## Requirements

- Python 3.14
- An Anthropic API key
