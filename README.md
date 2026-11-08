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
| `pulumi_docs_agent/pricing.py` | Model pricing and cost calculation, used by the test suite and the evals |
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

#### Evaluations

`pulumi_docs_agent/evals/` measures how good the agent is, beyond the pass/fail tests. It is split into
one folder per kind of evaluation, and a single entrypoint runs any of them:

```bash
cd src/alexey_ai_engineering_buildcamp/week-4
uv run python -m pulumi_docs_agent.evals                      # list every command
uv run python -m pulumi_docs_agent.evals <group> <command> [options]
```

| Group | Command | What it does | `make` target |
|---|---|---|---|
| `scenarios` | `run` | Runs the 24 hand-designed scenarios against the agent and saves each response | `make scenarios` |
| `scenarios` | `label` | Streamlit app for labeling each response good / not good | `make label` |
| `scenarios` | `evals` | Runs the agent, judges every response and reports the percentage that are good, with the change from the previous run | `make evals` |
| `judge` | `evaluate` | Compares the LLM judge with the human labels (accuracy, precision and recall for the bad class) | `make judge` |
| `code-evaluators` | `run` | Applies the deterministic code evaluator to saved runs | `make evaluators` |
| `synthetic` | `generate` | Generates questions and reference answers from sampled documentation pages | `make questions` |
| `synthetic` | `run` | Runs the agent on the generated questions, five at a time | `make synthetic` |
| `synthetic` | `judge` | Judges each response for correctness, instruction following and trajectory | `make judge-synthetic` |
| `synthetic` | `explore` | Streamlit app for filtering and reading the judged results | `make explore` |
| `phoenix` | `traces` | Pulls the agent's traces from Phoenix into the same row format, tagged organic, scenario, synthetic or test | `make traces` |
| `phoenix` | `upload` | Creates the `pulumi-docs-scenarios` and `pulumi-docs-synthetic` datasets in Phoenix, skipping any that already exist | `make phoenix-upload` |
| `phoenix` | `experiment` | Records a Phoenix experiment for each dataset from a saved run, with the code checks and the judges as evaluators | `make phoenix-experiment` |
| `phoenix` | `annotate` | Judges traces and writes the scores onto the spans as annotations | `make annotate` |

Pass extra options with `ARGS`, for example `make scenarios ARGS="--ids edge-pricing"`. Each command also
accepts `--help`.

A typical order: run the scenarios and label the responses with `scenarios run` and `scenarios label`, check the
judge against your labels with `judge evaluate`, track the percentage of good responses with `scenarios evals`
while you change the prompt or retrieval, then widen coverage with the `synthetic` commands. Once the agent is
traced in Phoenix, `phoenix traces` brings real usage into the same pipeline and `phoenix annotate` scores it.

| Folder | Contents |
|---|---|
| `evals/scenarios/` | `scenarios.json` (scenarios grouped by documentation area and phrasing, plus edge cases), the runner, the labeling app, the `evals` pipeline, and `results/` with saved runs and the human labels |
| `evals/judge/` | The judge prompt and the script that measures its agreement with the human labels |
| `evals/code_evaluators/` | `evaluators.py`: a deterministic `evaluate(output, reference, input, metadata)` function that needs only the standard library, so it can also be pasted into a Phoenix code evaluator |
| `evals/synthetic/` | Question generation, the three judges, the results explorer, and `data/` with the generated questions and the judged runs |
| `evals/observability/` | The Phoenix integration: the trace loader, dataset upload, experiments and span annotations. `data/` holds trace dumps and is gitignored because they contain the full retrieved documents |
| `evals/common/` | Shared helpers: paths, building the agent, replaying retrieval for the judge, and parallel execution |

**Scenarios and the judge.** The scenarios cover each area of the docs with direct, vague and
wrong-terminology phrasings, plus questions the docs cannot answer and a prompt-injection attempt. The
judge sees the retrieved documents, rebuilt by replaying the saved tool calls, so it can check
grounding. It is still a work in progress: on the 24 labeled scenarios it agrees with the human labels
on about two thirds of them and catches only a quarter of the responses labeled bad, so treat the
percentage from `scenarios evals` as a trend line, not an absolute score.

**Code evaluator.** The hard checks cover an empty answer, cited paths that do not exist in the corpus,
duplicate references, `found_answer` without references, confidence out of range, a missing refusal when
one is expected, a prompt leak and the tool-call budget. Cited-source and process-narration checks only
lower the score. These checks find structural problems; they cannot judge whether an answer is correct.

**Synthetic data.** 204 questions are generated from 70 pages sampled evenly across the areas of `docs/`,
using four user personas. Each comes with a reference answer and the lines it came from, and the line
ranges are validated in code. Three separate judges grade each response: whether it contains the
reference answer, whether it follows the agent's instructions, and whether the tool-call sequence was
efficient. The generated questions and reference answers are not reviewed by a person, so check some
before treating a mismatch as an agent failure.

**Phoenix.** Monitoring in this repo uses Arize Phoenix, not the Pydantic Logfire used in the course's monitoring
week. The agent's traces are sent to the project named by `PHOENIX_PROJECT_NAME`, and the `phoenix` commands read
and write that project through `arize-phoenix-client`, using `PHOENIX_COLLECTOR_ENDPOINT` and
`PHOENIX_API_KEY`. `phoenix traces` rebuilds each one into an eval row: the
question and structured answer come from the agent span, and the tool calls and their full results come from the
tool spans, so the judges see exactly what the agent retrieved. Questions that match the scenarios, the synthetic
set or the test suite are tagged with that source, so `organic` means everything else. The loaded rows work with
the labeling app (`make label` lists trace files), the code evaluator and the judges.

`phoenix experiment` replays the agent outputs already saved on disk instead of calling the agent again, so it
costs nothing by default and the judge verdicts reuse the stored ones. Each run is recorded as
`scenarios-replay-<timestamp>` or `synthetic-replay-<timestamp>`. Add `--live-judges` to re-run them, or
`--dry-run N` to run N examples locally without recording anything. `phoenix annotate` runs the code checks and
the judges on organic traces by default and writes `code_checks`, `judge_quality`,
`judge_instruction_following` and `judge_trajectory` annotations (plus `judge_correctness` when a reference
answer is known) under the identifier `evals-v1`. Re-running updates them in place, and traces that already have
a `judge_quality` annotation are skipped unless you pass `--force`. `--dry-run` judges without writing. The
`upload`, `experiment` and `annotate` commands write to the Phoenix instance, and the client cannot delete a
dataset, so check the dataset names before uploading.

These commands call the model for real and need `ANTHROPIC_API_KEY` and the Phoenix variables. At the
default models a full `scenarios evals` run costs about $0.13; running the agent on all synthetic
questions costs about $0.80 and about ten minutes, and judging them about $0.35. The Streamlit apps
need no API key.

### Week 4 — Homework

`week-4/homework/duckdb_sql_agent/` is a SQL agent over the NYC yellow taxi dataset. It inspects the
`trips` table with `get_schema`, generates SQL, runs it with `run_sql` against DuckDB, and returns a
structured `SQLResult` containing the query, the result summary, and the row count.

The first run downloads `yellow_tripdata_2024-01.parquet` into `db/` and materializes
`db/taxi.duckdb` with a `trips` table. Both files are gitignored and reused on later runs.

| Module | Purpose |
|---|---|
| `__init__.py` | Loads `week-4/.env` and configures Phoenix tracing |
| `entrypoint.py` | CLI entry point |
| `agent.py` | Pydantic AI agent construction |
| `helpers.py` | Tool-call event logging |
| `models.py` | `SQLResult` and `SQLAgentConfig` |
| `prompts.py` | SQL instructions, self-verification checklist, and judge prompts |
| `tools.py` | `SQLTools`: `get_schema` and `run_sql` with a 50-row cap |
| `utils.py` | Parquet download and DuckDB setup |
| `tests/` | Deterministic checks and an LLM-as-a-judge suite |

Run it from the `week-4` project directory:

```bash
cd src/alexey_ai_engineering_buildcamp/week-4
uv run duckdb-sql-agent
```

The prompt asks for a question; the agent calls `get_schema` first, then `run_sql`, and prints the
structured result.

#### Tests

| File | Purpose |
|---|---|
| `tests/test_agent.py` | Exact count check, schema-before-SQL ordering, and an LLM judge for the highest-average-fare hour |
| `tests/test_cases.py` | Five table-driven data questions with expected SQL columns and result values |
| `tests/judge.py` | Judge agent plus `evaluate_agent_performance` and `assert_criteria` |
| `tests/utils.py` | Tool-call collection and numeric result parsing |

```bash
cd src/alexey_ai_engineering_buildcamp/week-4
uv run pytest homework/duckdb_sql_agent/tests -q
```

These tests call the model for real and need `ANTHROPIC_API_KEY`; the judge test adds a second model
call. The `db/` directory is generated locally and is not committed.

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
            ├── homework/
            │   └── duckdb_sql_agent/
            │       ├── __init__.py
            │       ├── agent.py
            │       ├── entrypoint.py
            │       ├── helpers.py
            │       ├── models.py
            │       ├── prompts.py
            │       ├── tools.py
            │       ├── utils.py
            │       └── tests/
            │           ├── __init__.py
            │           ├── judge.py
            │           ├── test_agent.py
            │           ├── test_cases.py
            │           └── utils.py
            └── pulumi_docs_agent/
                ├── __init__.py
                ├── agent.py
                ├── entrypoint.py
                ├── helpers.py
                ├── models.py
                ├── pricing.py
                ├── prompts.py
                ├── retrieval.py
                ├── evals/
                │   ├── __main__.py
                │   ├── common/
                │   │   ├── agent_runner.py
                │   │   ├── known_questions.py
                │   │   ├── paths.py
                │   │   ├── retrieval_replay.py
                │   │   └── utils.py
                │   ├── scenarios/
                │   │   ├── scenarios.json
                │   │   ├── run.py
                │   │   ├── run_evals.py
                │   │   ├── label.py
                │   │   └── results/
                │   ├── judge/
                │   │   ├── judge.py
                │   │   └── evaluate.py
                │   ├── code_evaluators/
                │   │   └── evaluators.py
                │   ├── synthetic/
                │   │   ├── generate.py
                │   │   ├── run.py
                │   │   ├── judges.py
                │   │   ├── run_judges.py
                │   │   ├── explore.py
                │   │   └── data/
                │   └── observability/
                │       ├── client.py
                │       ├── traces.py
                │       ├── datasets.py
                │       ├── experiments.py
                │       ├── annotate.py
                │       └── data/
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
