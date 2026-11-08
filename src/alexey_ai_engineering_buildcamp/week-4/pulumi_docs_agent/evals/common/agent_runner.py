import time

from ...agent import DocumentationAgentConfig, init_agent, run_agent_stream
from ...pricing import calculate_cost
from ...retrieval import build_index, create_highlighter
from ...tests.utils import collect_tools
from ...tools import SearchTools
from ...utils import retrieve_documentation
from .paths import DOCS_DIR


def build_agent():
    documents = retrieve_documentation(
        repo_owner="pulumi",
        repo_name="docs",
        branch_name="master",
        docs_path="content",
        save_dir=str(DOCS_DIR),
    )
    index = build_index(documents=documents)
    highlighter = create_highlighter(max_matches=3, snippet_size=50)
    file_index = {d["filename"]: d["content"] for d in documents}
    tools = SearchTools(index, highlighter, file_index)
    agent = init_agent(DocumentationAgentConfig(), tools)
    return agent, set(file_index)


def tool_summary(messages) -> list[dict]:
    return [{"name": c.name, "args": c.args} for c in collect_tools(messages)]


async def run_scenario(agent, scenario: dict, known_paths: set[str]) -> dict:
    history: list = []
    turns = scenario.get("turns") or [scenario["question"]]
    result = None
    t0 = time.perf_counter()
    input_tokens = output_tokens = 0
    all_tools: list[dict] = []

    for turn in turns:
        result = await run_agent_stream(agent, turn, history, verbose=False)
        if result is None:
            raise RuntimeError("agent produced no result")

        history = result.all_messages()
        input_tokens += result.usage.input_tokens
        output_tokens += result.usage.output_tokens
        all_tools = tool_summary(result.new_messages())  # tools of the last turn

    assert result is not None, "should have a result after all turns"

    out = result.output
    refs = [r.file_path for r in out.references]
    model_name = agent.model.model_name if agent.model else "unknown"
    try:
        cost = calculate_cost(model_name, input_tokens, output_tokens)
    except ValueError:
        cost = None

    return {
        "id": scenario["id"],
        "group": scenario["group"],
        "phrasing": scenario["phrasing"],
        "question": scenario["question"],
        "turns": turns,
        "expected_behavior": scenario["expected_behavior"],
        "expected_ref_prefixes": scenario["expected_ref_prefixes"],
        "notes": scenario.get("notes", ""),
        "answer": out.answer,
        "found_answer": out.found_answer,
        "confidence": out.confidence,
        "confidence_explanation": out.confidence_explanation,
        "answer_type": out.answer_type,
        "references": refs,
        "invented_references": sorted({r for r in refs if r not in known_paths}),
        "ref_prefix_hit": (
            any(
                r.startswith(p) for r in refs for p in scenario["expected_ref_prefixes"]
            )
            if scenario["expected_ref_prefixes"]
            else None
        ),
        "followup_questions": out.followup_questions,
        "checks": [c.model_dump() for c in out.checks],
        "tool_calls": all_tools,
        "n_tool_calls": len(all_tools),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "seconds": round(time.perf_counter() - t0, 1),
        "error": None,
    }


def error_row(scenario: dict, exc: Exception) -> dict:
    return {
        "id": scenario["id"],
        "group": scenario["group"],
        "phrasing": scenario["phrasing"],
        "question": scenario["question"],
        "expected_behavior": scenario["expected_behavior"],
        "error": f"{type(exc).__name__}: {exc}",
    }
