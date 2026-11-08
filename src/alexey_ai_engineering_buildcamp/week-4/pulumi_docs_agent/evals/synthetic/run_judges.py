import argparse
import asyncio
import json
import time
from pathlib import Path

from ...pricing import calculate_cost
from ..common.utils import fmt_time, map_progress
from .judges import JUDGE_MODEL, JUDGES


class CostAccumulator:
    def __init__(self) -> None:
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_judge: dict[str, float] = {name: 0.0 for name in JUDGES}

    def add(self, judge: str, model: str, usage) -> None:
        cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)
        self.total_cost += cost
        self.by_judge[judge] += cost
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens


async def judge_entry(entry: dict, judges: dict, cost: CostAccumulator) -> dict:
    if entry.get("error"):
        for name in judges:
            entry[name] = {
                "reasoning": f"Agent error: {entry['error']}",
                "score": "bad",
            }
        return entry

    for name, judge in judges.items():
        formatter = JUDGES[name][1]
        try:
            result = await judge.run(formatter(entry))
            cost.add(name, judge.model.model_name, result.usage)
            entry[name] = result.output.model_dump()
        except Exception as exc:
            entry[name] = {
                "reasoning": f"Judge error: {exc}",
                "score": "bad",
                "error": True,
            }
    return entry


def report(entries: list[dict], cost: CostAccumulator, elapsed: float) -> None:
    total = len(entries)
    print(f"\nEntries judged: {total}")
    for name in JUDGES:
        good = sum(e[name]["score"] == "good" for e in entries)
        errors = sum(bool(e[name].get("error")) for e in entries)
        extra = f", {errors} judge errors" if errors else ""
        print(f"  {name:30} {good}/{total} good ({good / total:.1%}){extra}")
    all_good = sum(all(e[n]["score"] == "good" for n in JUDGES) for e in entries)
    print(f"  {'all three good':30} {all_good}/{total} ({all_good / total:.1%})")
    print(
        f"Cost: ${cost.total_cost:.4f} ({cost.input_tokens:,} in / {cost.output_tokens:,} out)"
    )
    print(f"Time: {fmt_time(elapsed)}")


async def main_async(args) -> None:
    data = Path(args.data)
    entries = json.loads(data.read_text())
    if args.limit:
        entries = entries[: args.limit]

    names = args.only or list(JUDGES)
    judges = {name: JUDGES[name][0](args.model) for name in names}
    cost = CostAccumulator()

    async def process(entry):
        return await judge_entry(entry, judges, cost)

    t0 = time.perf_counter()
    judged = await map_progress(entries, process, max_concurrency=args.concurrency)
    elapsed = time.perf_counter() - t0
    judged.sort(key=lambda e: e["id"])

    output = (
        Path(args.output)
        if args.output
        else (
            data
            if data.stem.endswith("_judged")
            else data.with_name(f"{data.stem}_judged.json")
        )
    )
    output.write_text(json.dumps(judged, indent=2, ensure_ascii=False))
    report(judged, cost, elapsed)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--only", nargs="*", choices=list(JUDGES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=5)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
