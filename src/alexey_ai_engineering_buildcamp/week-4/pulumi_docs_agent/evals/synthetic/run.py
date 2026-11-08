import argparse
import asyncio
import contextlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..common.agent_runner import build_agent, error_row, run_scenario
from ..common.paths import DEFAULT_QUESTIONS, SYNTHETIC_DATA_DIR
from ..common.utils import fmt_time, map_progress
from .generate import area


def to_scenario(row: dict) -> dict:
    return {
        "id": row["id"],
        "group": area(row["filename"]),
        "phrasing": row["question_type"],
        "question": row["question"],
        "expected_behavior": "answers",
        "expected_ref_prefixes": [row["filename"]],
        "notes": "",
    }


async def run_agent_on_row(agent, known_paths: set[str], row: dict) -> dict:
    scenario = to_scenario(row)
    try:
        result = await run_scenario(agent, scenario, known_paths)
    except Exception as exc:
        result = error_row(scenario, exc)
    return {
        **result,
        "reference_answer": row["reference_answer"],
        "filename": row["filename"],
        "line_number_start": row["line_number_start"],
        "line_number_end": row["line_number_end"],
        "lines_valid": row.get("lines_valid"),
        "answer_overlap": row.get("answer_overlap"),
    }


def report(results: list[dict], elapsed: float) -> None:
    total = len(results)
    ok = [r for r in results if not r.get("error")]
    errors = total - len(ok)
    cost = sum(r.get("cost_usd") or 0 for r in ok)
    tok_in = sum(r["input_tokens"] for r in ok)
    tok_out = sum(r["output_tokens"] for r in ok)
    hit = sum(bool(r["ref_prefix_hit"]) for r in ok)
    found = sum(bool(r["found_answer"]) for r in ok)
    avg_tools = sum(r["n_tool_calls"] for r in ok) / len(ok) if ok else 0.0

    print(f"\nTotal questions: {total}, errors: {errors}")
    print(f"Cost: ${cost:.4f} ({tok_in:,} in / {tok_out:,} out)")
    print(f"Time: {fmt_time(elapsed)}")
    if ok:
        print(f"Cited the source page: {hit}/{len(ok)} ({hit / len(ok):.1%})")
        print(f"found_answer=True: {found}/{len(ok)} ({found / len(ok):.1%})")
        print(
            f"Avg tool calls: {avg_tools:.1f}; over 6: {sum(r['n_tool_calls'] > 6 for r in ok)}"
        )
    for r in results:
        if r.get("error"):
            print(f"  ERROR {r['id']}: {r['error'][:120]}")


async def main_async(args) -> None:
    df = pd.read_csv(Path(args.questions))
    df = df.where(pd.notna(df), None)
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[: args.limit]

    agent, known_paths = build_agent()

    async def process(row):
        return await run_agent_on_row(agent, known_paths, row)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        results = await map_progress(rows, process, max_concurrency=args.concurrency)
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: r["id"])

    output = (
        Path(args.output)
        if args.output
        else (
            SYNTHETIC_DATA_DIR
            / f"evals_run_{datetime.now():%Y_%m_%d_%H%M%S}_synthetic.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    report(results, elapsed)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=5)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
