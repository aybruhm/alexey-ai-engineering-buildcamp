import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from ...pricing import calculate_cost
from ..common.agent_runner import build_agent, error_row, run_scenario
from ..common.paths import PIPELINE_RESULTS_DIR as OUT_DIR
from ..common.paths import SCENARIOS_FILE
from ..common.retrieval_replay import build_tools, replay_retrieval
from ..common.utils import fmt_time
from ..judge.judge import JUDGE_MODEL, create_judge, format_judge_prompt
MAX_TOOL_CALLS = 6


def previous_summary() -> dict | None:
    files = sorted(OUT_DIR.glob("*.json"))
    return json.loads(files[-1].read_text())["summary"] if files else None


async def run_agent_phase(scenarios: list[dict]) -> tuple[list[dict], float]:
    agent, known_paths = build_agent()
    rows = []
    t0 = time.perf_counter()
    for i, s in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] agent: {s['id']}", flush=True)
        try:
            row = await run_scenario(agent, s, known_paths)
        except Exception as exc:
            print(f"  ERROR running agent: {exc}")
            row = error_row(s, exc)
        rows.append(row)
    return rows, time.perf_counter() - t0


async def run_judge_phase(rows: list[dict], model: str, concurrency: int) -> tuple[float, float]:
    judge = create_judge(model)
    tools = build_tools()
    sem = asyncio.Semaphore(concurrency)
    cost = 0.0

    async def judge_row(row: dict) -> None:
        nonlocal cost
        if row.get("error"):
            row["judge_label"] = "bad"
            row["judge_reasoning"] = f"Agent error: {row['error']}"
            return
        prompt = format_judge_prompt(row, replay_retrieval(tools, row))
        try:
            async with sem:
                result = await judge.run(prompt)
            row["judge_label"] = result.output.label
            row["judge_reasoning"] = result.output.reasoning
            usage = result.usage
            cost += calculate_cost(
                judge.model.model_name, usage.input_tokens, usage.output_tokens
            )
        except Exception as exc:
            print(f"  ERROR judging {row['id']}: {exc}")
            row["judge_label"] = "bad"
            row["judge_reasoning"] = f"Judge error: {exc}"

    t0 = time.perf_counter()
    await asyncio.gather(*(judge_row(r) for r in rows))
    return time.perf_counter() - t0, cost


def summarize(rows, agent_cost, judge_cost, agent_time, judge_time, judge_model) -> dict:
    total = len(rows)
    good = sum(r["judge_label"] == "good" for r in rows)
    by_group: dict[str, list[int]] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], [0, 0])
        g[0] += r["judge_label"] == "good"
        g[1] += 1
    long_runs = [r["id"] for r in rows if r.get("n_tool_calls", 0) > MAX_TOOL_CALLS]
    return {
        "total": total,
        "good": good,
        "bad": total - good,
        "pct_good": round(good / total * 100, 1) if total else 0.0,
        "by_group": {k: {"good": v[0], "total": v[1]} for k, v in by_group.items()},
        "bad_ids": [r["id"] for r in rows if r["judge_label"] == "bad"],
        "long_trajectories": long_runs,
        "avg_tool_calls": round(sum(r.get("n_tool_calls", 0) for r in rows) / total, 2) if total else 0.0,
        "agent_cost_usd": round(agent_cost, 4),
        "judge_cost_usd": round(judge_cost, 4),
        "agent_seconds": round(agent_time, 1),
        "judge_seconds": round(judge_time, 1),
        "judge_model": judge_model,
    }


def report(s: dict, prev: dict | None) -> None:
    print(f"\nResults: {s['good']}/{s['total']} good ({s['pct_good']:.1f}%), {s['bad']}/{s['total']} bad")
    if prev:
        delta = s["pct_good"] - prev["pct_good"]
        print(f"Previous run: {prev['pct_good']:.1f}% good ({delta:+.1f} points, n={prev['total']})")
    print("By group:")
    for name, v in s["by_group"].items():
        print(f"  {name:28} {v['good']}/{v['total']}")
    print(f"Bad: {', '.join(s['bad_ids']) or 'none'}")
    print(f"Avg tool calls: {s['avg_tool_calls']}  Over {MAX_TOOL_CALLS}: {', '.join(s['long_trajectories']) or 'none'}")
    total_cost = s["agent_cost_usd"] + s["judge_cost_usd"]
    print(f"Cost:  agent ${s['agent_cost_usd']:.4f}, judge ${s['judge_cost_usd']:.4f}, total ${total_cost:.4f}")
    total_time = s["agent_seconds"] + s["judge_seconds"]
    print(f"Time:  agent {fmt_time(s['agent_seconds'])}, judge {fmt_time(s['judge_seconds'])}, total {fmt_time(total_time)}")


async def main_async(args) -> None:
    scenarios = json.loads(SCENARIOS_FILE.read_text())["scenarios"]
    if args.ids:
        scenarios = [s for s in scenarios if s["id"] in args.ids]
    if args.group:
        scenarios = [s for s in scenarios if s["group"] == args.group]
    if not scenarios:
        raise SystemExit("no scenarios matched")

    prev = previous_summary()
    rows, agent_time = await run_agent_phase(scenarios)
    judge_time, judge_cost = await run_judge_phase(rows, args.model, args.concurrency)
    agent_cost = sum(r.get("cost_usd") or 0 for r in rows)

    summary = summarize(rows, agent_cost, judge_cost, agent_time, judge_time, args.model)
    report(summary, prev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*")
    p.add_argument("--group")
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
