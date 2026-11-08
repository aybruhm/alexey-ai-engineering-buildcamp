import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from ..common.agent_runner import build_agent, error_row, run_scenario
from ..common.paths import SCENARIO_RESULTS_DIR, SCENARIOS_FILE


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'id':38} {'found':5} {'conf':>4} {'tools':>5} {'refs?':>5}  note")
    for r in rows:
        if r.get("error"):
            print(f"{r['id']:38} ERROR  {r['error'][:60]}")
            continue
        hit = {True: "yes", False: "NO", None: "-"}[r["ref_prefix_hit"]]
        flags = []
        if r["invented_references"]:
            flags.append("INVENTED REFS")
        if r["found_answer"] and not r["references"]:
            flags.append("found w/o refs")
        print(
            f"{r['id']:38} {r['found_answer']!s:5} {r['confidence']:>4.1f} "
            f"{r['n_tool_calls']:>5} {hit:>5}  {', '.join(flags)}"
        )
    total = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"\nTotal cost: ${total:.4f}")


async def main_async(args) -> None:
    data = json.loads(SCENARIOS_FILE.read_text())
    scenarios = data["scenarios"]
    if args.ids:
        scenarios = [s for s in scenarios if s["id"] in args.ids]
    if args.group:
        scenarios = [s for s in scenarios if s["group"] == args.group]
    if not scenarios:
        raise SystemExit("no scenarios matched")

    agent, known_paths = build_agent()
    SCENARIO_RESULTS_DIR.mkdir(exist_ok=True)
    out_path = SCENARIO_RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    rows = []
    with out_path.open("w") as f:
        for i, s in enumerate(scenarios, 1):
            print(f"[{i}/{len(scenarios)}] {s['id']}", flush=True)
            try:
                row = await run_scenario(agent, s, known_paths)
            except Exception as exc:  # keep going; one failure shouldn't lose the run
                row = error_row(s, exc)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print_summary(rows)
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*", help="only run these scenario ids")
    p.add_argument("--group", help="only run this group")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
