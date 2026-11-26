import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path

from ..common.retrieval_replay import build_tools, replay_retrieval
from .judge import JUDGE_MODEL, create_judge, format_judge_prompt


def is_holdout(row_id: str) -> bool:
    return int(hashlib.sha1(row_id.encode()).hexdigest(), 16) % 3 == 0


def load(data: Path) -> list[dict]:
    runs = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    labels = {
        row["id"]: row
        for row in json.loads(data.with_name(f"{data.stem}_results.json").read_text())
    }
    out = []
    for row in runs:
        if row["id"] not in labels or row.get("error"):
            continue
        label = labels[row["id"]]
        out.append(
            {
                **row,
                "human_label": "good" if label["label"] == "good" else "bad",
                "human_comment": label["comments"],
                "human_category": label.get("failure_category", ""),
            }
        )
    return out


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Agreement corrected for chance. Useful because most labels are 'good'."""
    n = tp + fp + fn + tn
    if not n:
        return float("nan")
    observed = (tp + tn) / n
    human_bad, human_good = tp + fn, tn + fp
    judge_bad, judge_good = tp + fp, tn + fn
    expected = (human_bad / n) * (judge_bad / n) + (human_good / n) * (judge_good / n)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["human_label"] == "bad" and r["judge_label"] == "bad" for r in rows)
    fp = sum(r["human_label"] == "good" and r["judge_label"] == "bad" for r in rows)
    fn = sum(r["human_label"] == "bad" and r["judge_label"] == "good" for r in rows)
    tn = sum(r["human_label"] == "good" and r["judge_label"] == "good" for r in rows)
    n = len(rows)
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision_bad": tp / (tp + fp) if tp + fp else float("nan"),
        "recall_bad": tp / (tp + fn) if tp + fn else float("nan"),
        "kappa": cohen_kappa(tp, fp, fn, tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def category_metrics(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Judge recall on human-labelled bad examples, split by failure category."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["human_label"] != "bad":
            continue
        category = row.get("human_category") or "(uncategorized)"
        bucket = out.setdefault(category, {"caught": 0, "total": 0})
        bucket["total"] += 1
        bucket["caught"] += row["judge_label"] == "bad"
    return dict(sorted(out.items()))


def show(title: str, m: dict) -> None:
    print(f"\n{title} (n={m['n']})")
    print(
        f"  accuracy {m['accuracy']:.0%} | kappa {m['kappa']:.2f} | "
        f"precision(bad) {m['precision_bad']:.0%} | "
        f"recall(bad) {m['recall_bad']:.0%}"
    )
    print("                judge:bad  judge:good")
    print(f"  human:bad     {m['tp']:>9}  {m['fn']:>10}")
    print(f"  human:good    {m['fp']:>9}  {m['tn']:>10}")


def show_categories(rows: list[dict]) -> None:
    categories = category_metrics(rows)
    if not categories:
        return
    print("\nHuman-labelled bad examples by category (judge recall):")
    for category, counts in categories.items():
        print(
            f"  {category:18} {counts['caught']}/{counts['total']} "
            f"({counts['caught'] / counts['total']:.0%})"
        )


def show_confidence(rows: list[dict]) -> None:
    confidences = [
        r["judge_confidence"] for r in rows if r.get("judge_confidence") is not None
    ]
    if not confidences:
        return
    average = sum(confidences) / len(confidences)
    low = [
        r["id"]
        for r in rows
        if r.get("judge_confidence") is not None and r["judge_confidence"] < 0.7
    ]
    print(
        f"\nJudge confidence: average {average:.2f}; "
        f"below 0.70: {', '.join(low) or 'none'}"
    )


async def main_async(args) -> None:
    data = Path(args.data)
    rows = load(data)
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]
    if not rows:
        raise SystemExit("no rows matched")

    judge = create_judge(args.model)
    sem = asyncio.Semaphore(args.concurrency)

    tools = build_tools()
    for r in rows:
        r["retrieved"] = replay_retrieval(tools, r)

    async def run(row):
        async with sem:
            result = await judge.run(format_judge_prompt(row, row["retrieved"]))
        return result.output

    outputs = await asyncio.gather(*(run(r) for r in rows))
    for row, output in zip(rows, outputs):
        row["judge_label"] = output.label
        row["judge_reasoning"] = output.reasoning
        row["judge_confidence"] = output.confidence
        row["split"] = "heldout" if is_holdout(row["id"]) else "dev"

    show("ALL", metrics(rows))
    show("DEV", metrics([r for r in rows if r["split"] == "dev"]))
    show("HELD-OUT", metrics([r for r in rows if r["split"] == "heldout"]))
    show_categories(rows)
    show_confidence(rows)

    print("\nDisagreements (dev only; use --show-heldout to see the rest):")
    for row in rows:
        if row["split"] == "heldout" and not args.show_heldout:
            continue
        if row["judge_label"] != row["human_label"]:
            print(
                f"\n[{row['split']}] {row['id']}  "
                f"human={row['human_label']} judge={row['judge_label']} "
                f"confidence={row['judge_confidence']:.2f}"
            )
            print(f"  human comment: {row['human_comment'] or '(none)'}")
            print(f"  judge: {row['judge_reasoning'][:600]}")

    out = Path(args.out) if args.out else data.with_name(f"{data.stem}_judge.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "split",
                "human_label",
                "judge_label",
                "judge_confidence",
                "human_comment",
                "human_category",
                "judge_reasoning",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["id"],
                    row["split"],
                    row["human_label"],
                    row["judge_label"],
                    row["judge_confidence"],
                    row["human_comment"],
                    row["human_category"],
                    row["judge_reasoning"],
                ]
            )
    print(f"\nSaved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--ids", nargs="*", help="only evaluate these scenario ids")
    parser.add_argument("--out", help="where to write the judge CSV")
    parser.add_argument("--show-heldout", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
