import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..common.labels import load_recipes
from ..common.paths import JUDGED_FILE, SYNTHETIC_JUDGED_FILE


def bad_rate(rows: list[dict]) -> str:
    bad = sum(1 for r in rows if r["judge_label"] == "bad")
    return f"{bad:>3}/{len(rows):<3} {bad / len(rows):6.1%}"


def breakdown(rows: list[dict], key: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "?"))].append(r)
    print(f"\nbad by {key}:")
    for name, group in sorted(groups.items(), key=lambda kv: kv[0]):
        print(f"  {name:28} {bad_rate(group)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise judge labels for a judged results file.")
    parser.add_argument("--data", type=Path, default=SYNTHETIC_JUDGED_FILE)
    parser.add_argument("--baseline", type=Path, default=JUDGED_FILE)
    parser.add_argument("--by", nargs="*", default=["type", "recipe_name"])
    parser.add_argument("--show-bad", action="store_true", help="print the judge reasoning for every bad")
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())
    names = {r["id"]: r["name"] for r in load_recipes()}
    for r in rows:
        if "recipe_id" in r:
            r.setdefault("recipe_name", names.get(r["recipe_id"], "?"))
    print(f"{args.data.name}: bad {bad_rate(rows)}")
    if args.baseline.exists():
        print(f"{args.baseline.name} (baseline): bad {bad_rate(json.loads(args.baseline.read_text()))}")

    for key in args.by:
        breakdown(rows, key)

    if args.show_bad:
        for r in rows:
            if r["judge_label"] == "bad":
                print(f"\n[{r.get('type')}] {r['question']}\n  {r['judge_reasoning']}")


if __name__ == "__main__":
    main()
