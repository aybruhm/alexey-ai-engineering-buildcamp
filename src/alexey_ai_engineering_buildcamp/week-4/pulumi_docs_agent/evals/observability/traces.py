import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ...pricing import calculate_cost
from ..common.known_questions import known_questions, normalize
from ..common.paths import DOCS_DIR, OBSERVABILITY_DATA_DIR
from ..common.retrieval_replay import MAX_DOC_CHARS
from ..synthetic.generate import area
from .client import get_client, project_name

AGENT_SPAN_NAME = "invoke_agent search"
SOURCES = ("organic", "scenario", "synthetic", "test")


def known_doc_paths() -> set[str]:
    return {
        p.relative_to(DOCS_DIR).as_posix()
        for pattern in ("*.md", "*.mdx")
        for p in DOCS_DIR.rglob(pattern)
    }


def parse_json(text, default=None):
    try:
        return json.loads(text)
    except TypeError, ValueError:
        return default


def user_turns(attributes: dict, fallback: str) -> list[str]:
    messages = parse_json(attributes.get("pydantic_ai.all_messages"), []) or []
    turns = []
    for message in messages:
        if message.get("role") != "user":
            continue
        for part in message.get("parts", []):
            if part.get("type") == "text" and part.get("content"):
                turns.append(part["content"])
    return turns or [fallback]


def format_retrieved(tool_calls: list[dict]) -> str:
    parts = []
    for i, call in enumerate(tool_calls, 1):
        result = call["result"]
        if call["name"] == "get_file" and len(result) > MAX_DOC_CHARS:
            result = result[:MAX_DOC_CHARS] + "\n[...truncated for the judge...]"
        parts.append(f"--- tool call {i}: {call['name']}({call['args']}) ---\n{result}")
    return "\n\n".join(parts)


def tag_source(question: str, known: dict[str, dict]) -> dict:
    match = known.get(normalize(question))
    return match if match else {"source": "organic"}


def span_to_row(
    agent_span: dict, tool_spans: list[dict], known, doc_paths: set[str]
) -> dict | None:
    attrs = agent_span["attributes"]
    output = parse_json(attrs.get("output.value"))
    if not isinstance(output, dict) or "answer" not in output:
        return None

    question = attrs.get("input.value", "")
    turns = user_turns(attrs, question)
    tool_calls = [
        {
            "name": t["attributes"].get("gen_ai.tool.name", t["name"]),
            "args": parse_json(t["attributes"].get("gen_ai.tool.call.arguments"), {}),
            "result": str(t["attributes"].get("gen_ai.tool.call.result", "")),
        }
        for t in sorted(tool_spans, key=lambda s: s["start_time"])
    ]
    refs = [r["file_path"] for r in output.get("references", [])]
    started = datetime.fromisoformat(agent_span["start_time"])
    ended = datetime.fromisoformat(agent_span["end_time"])
    tokens_in = int(attrs.get("gen_ai.aggregated_usage.input_tokens", 0))
    tokens_out = int(attrs.get("gen_ai.aggregated_usage.output_tokens", 0))
    try:
        cost = calculate_cost(attrs.get("model_name", ""), tokens_in, tokens_out)
    except ValueError:
        cost = None

    tag = tag_source(question, known)
    source = tag["source"]
    row = {
        "id": f"trace-{agent_span['context']['trace_id'][:8]}",
        "trace_id": agent_span["context"]["trace_id"],
        "span_id": agent_span["context"]["span_id"],
        "start_time": agent_span["start_time"],
        "source": source,
        "group": "organic",
        "phrasing": "organic",
        "question": question,
        "turns": turns,
        "answer": output["answer"],
        "found_answer": output.get("found_answer"),
        "confidence": output.get("confidence"),
        "confidence_explanation": output.get("confidence_explanation"),
        "answer_type": output.get("answer_type"),
        "references": refs,
        "invented_references": sorted({r for r in refs if r not in doc_paths}),
        "followup_questions": output.get("followup_questions", []),
        "checks": output.get("checks", []),
        "tool_calls": [
            {k: v for k, v in c.items() if k != "result"} for c in tool_calls
        ],
        "n_tool_calls": len(tool_calls),
        "retrieved": format_retrieved(tool_calls),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": cost,
        "seconds": round((ended - started).total_seconds(), 1),
        "expected_behavior": None,
        "expected_ref_prefixes": [],
        "error": None,
    }
    if source == "scenario":
        row["group"] = tag["group"]
        row["phrasing"] = tag["phrasing"]
        row["known_id"] = tag["id"]
        row["expected_behavior"] = tag["expected_behavior"]
        row["expected_ref_prefixes"] = tag["expected_ref_prefixes"]
        if "expected_confidence_max" in tag:
            row["expected_confidence_max"] = tag["expected_confidence_max"]
    elif source == "synthetic":
        row["group"] = area(tag["filename"])
        row["phrasing"] = tag["question_type"]
        row["known_id"] = tag["id"]
        row["reference_answer"] = tag["reference_answer"]
        row["filename"] = tag["filename"]
        row["line_number_start"] = int(tag["line_number_start"])
        row["line_number_end"] = int(tag["line_number_end"])
        row["expected_behavior"] = "answers"
        row["expected_ref_prefixes"] = [tag["filename"]]
    return row


def fetch_rows(
    since: datetime | None = None, limit: int = 10000, project: str | None = None
) -> tuple[list[dict], int]:
    client = get_client()
    project = project or project_name()
    agent_spans = client.spans.get_spans(
        project_identifier=project,
        span_kind="AGENT",
        name=AGENT_SPAN_NAME,
        start_time=since,
        limit=limit,
        timeout=120,
    )
    tool_spans = client.spans.get_spans(
        project_identifier=project,
        span_kind="TOOL",
        start_time=since,
        limit=limit * 10,
        timeout=120,
    )
    by_trace: dict[str, list[dict]] = {}
    for span in tool_spans:
        by_trace.setdefault(span["context"]["trace_id"], []).append(span)

    known = known_questions()
    doc_paths = known_doc_paths()
    rows, skipped = [], 0
    for span in sorted(agent_spans, key=lambda s: s["start_time"]):
        row = span_to_row(
            span, by_trace.get(span["context"]["trace_id"], []), known, doc_paths
        )
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    return rows, skipped


def load_rows_file(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def latest_traces_file() -> Path | None:
    files = sorted(OBSERVABILITY_DATA_DIR.glob("traces_*.jsonl"))
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="only spans after this date, e.g. 2026-10-01")
    parser.add_argument(
        "--source",
        nargs="*",
        choices=SOURCES,
        help="keep only these sources (default: all)",
    )
    parser.add_argument("--output")
    parser.add_argument("--project")
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).astimezone() if args.since else None
    rows, skipped = fetch_rows(since=since, project=args.project)
    if args.source:
        rows = [r for r in rows if r["source"] in args.source]

    OBSERVABILITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = (
        Path(args.output)
        if args.output
        else OBSERVABILITY_DATA_DIR / f"traces_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    )
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    counts = Counter(r["source"] for r in rows)
    print(f"{len(rows)} agent traces ({skipped} skipped: no structured output)")
    for source in SOURCES:
        print(f"  {source:10} {counts.get(source, 0)}")
    organic = [r for r in rows if r["source"] == "organic"]
    if organic:
        print(
            f"Organic: {organic[0]['start_time'][:10]} to {organic[-1]['start_time'][:10]}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
