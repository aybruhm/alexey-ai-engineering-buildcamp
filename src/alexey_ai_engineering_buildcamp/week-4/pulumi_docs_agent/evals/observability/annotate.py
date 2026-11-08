import argparse
import asyncio
from collections import Counter
from pathlib import Path

from ..code_evaluators.evaluators import evaluate as code_evaluate
from ..code_evaluators.evaluators import to_inputs
from ..common.utils import map_progress
from ..judge.judge import JUDGE_MODEL, create_judge, format_judge_prompt
from ..synthetic.judges import JUDGES
from ..synthetic.run_judges import CostAccumulator, judge_entry
from .client import ANNOTATION_IDENTIFIER, get_client, project_name
from .traces import SOURCES, fetch_rows, load_rows_file

QUALITY_ANNOTATION = "judge_quality"
SKIP_MARKER = QUALITY_ANNOTATION


def annotation(
    row: dict,
    name: str,
    kind: str,
    label: str,
    score: float,
    explanation: str,
    **metadata,
) -> dict:
    return {
        "name": name,
        "span_id": row["span_id"],
        "annotator_kind": kind,
        "identifier": ANNOTATION_IDENTIFIER,
        "result": {"label": label, "score": score, "explanation": explanation[:4000]},
        "metadata": {"source": row["source"], **metadata},
    }


def good_score(label: str) -> float:
    return 1.0 if label == "good" else 0.0


async def annotations_for_row(
    row: dict,
    quality_judge,
    cost: CostAccumulator,
    model: str,
) -> list[dict]:
    out = []

    checks = code_evaluate(**to_inputs(row))
    out.append(
        annotation(
            row,
            "code_checks",
            "CODE",
            checks["label"],
            checks["score"],
            checks["explanation"],
        )
    )

    result = await quality_judge.run(format_judge_prompt(row, row["retrieved"]))
    out.append(
        annotation(
            row,
            QUALITY_ANNOTATION,
            "LLM",
            result.output.label,
            good_score(result.output.label),
            result.output.reasoning,
            judge_model=model,
        )
    )

    applicable = {
        name: make(model)
        for name, (make, _) in JUDGES.items()
        if name != "judge_answer_correctness" or row.get("reference_answer")
    }
    judged = await judge_entry(dict(row), applicable, cost)
    for name in applicable:
        verdict = judged[name]
        out.append(
            annotation(
                row,
                name,
                "LLM",
                verdict["score"],
                good_score(verdict["score"]),
                verdict["reasoning"],
                judge_model=model,
                judge_error=bool(verdict.get("error")),
            )
        )
    return out


def select_rows(
    rows: list[dict],
    sources: list[str],
    since: str | None,
    limit: int | None,
) -> list[dict]:
    rows = [r for r in rows if r["source"] in sources]
    if since:
        rows = [r for r in rows if r["start_time"] >= since]
    rows.sort(key=lambda r: r["start_time"])
    return rows[-limit:] if limit else rows


def already_annotated(client, span_ids: list[str]) -> set[str]:
    if not span_ids:
        return set()
    existing = client.spans.get_span_annotations(
        span_ids=span_ids,
        project_identifier=project_name(),
        include_annotation_names=[SKIP_MARKER],
        limit=max(1000, len(span_ids) * 2),
        timeout=60,
    )
    return {a["span_id"] for a in existing}


async def main_async(args) -> None:
    client = get_client()
    if args.from_file:
        rows = load_rows_file(Path(args.from_file))
    else:
        rows, _ = fetch_rows()
    rows = select_rows(rows, args.source, args.since, args.limit)

    if not args.force:
        done = already_annotated(client, [r["span_id"] for r in rows])
        skipped = [r for r in rows if r["span_id"] in done]
        rows = [r for r in rows if r["span_id"] not in done]
        if skipped:
            print(
                f"Skipping {len(skipped)} already annotated traces (use --force to redo them)"
            )
    if not rows:
        print("Nothing to annotate.")
        return

    model = args.model
    quality_judge = create_judge(model)
    cost = CostAccumulator()

    async def process(row):
        try:
            return (
                row,
                await annotations_for_row(
                    row,
                    quality_judge,
                    cost,
                    model,
                ),
                None,
            )
        except Exception as exc:
            return row, [], str(exc)

    results = await map_progress(rows, process, max_concurrency=args.concurrency)
    results.sort(key=lambda r: r[0]["start_time"])

    annotations = [a for _, anns, _ in results for a in anns]
    failures = [(r, err) for r, _, err in results if err]
    by_name: dict[str, Counter] = {}
    for a in annotations:
        by_name.setdefault(a["name"], Counter())[a["result"]["label"]] += 1

    print(
        f"\nJudged {len(results) - len(failures)} of {len(results)} traces, {len(annotations)} annotations"
    )
    for name, counts in sorted(by_name.items()):
        print(f"  {name:30} {dict(counts)}")
    print(f"Judge cost: ${cost.total_cost:.4f}")
    for row, anns, _ in results:
        labels = " ".join(
            f"{a['name'].removeprefix('judge_')}={a['result']['label']}" for a in anns
        )
        print(f"  {row['start_time'][:10]} {row['question'][:55]!r:60} {labels}")
        for a in anns:
            if a["name"] == "code_checks" and a["result"]["label"] == "fail":
                print(f"      code_checks: {a['result']['explanation'][:160]}")
    for row, err in failures:
        print(f"  FAILED {row['id']}: {err[:120]}")

    if args.dry_run:
        print("Dry run: nothing written to Phoenix.")
        return
    client.spans.log_span_annotations(span_annotations=annotations, sync=True)
    print(
        f"Wrote {len(annotations)} annotations to Phoenix (identifier '{ANNOTATION_IDENTIFIER}')."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="*", choices=SOURCES, default=["organic"])
    parser.add_argument("--since", help="only traces from this ISO date or later")
    parser.add_argument("--limit", type=int, help="only the most recent N traces")
    parser.add_argument(
        "--from-file", help="a traces_*.jsonl file instead of fetching from Phoenix"
    )
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-judge traces that already have a judge_quality annotation",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="judge but do not write to Phoenix"
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
