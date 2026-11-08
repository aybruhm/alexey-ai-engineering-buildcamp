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
    runs = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    labels = {
        x["id"]: x
        for x in json.loads(data.with_name(f"{data.stem}_results.json").read_text())
    }
    out = []
    for r in runs:
        if r["id"] in labels and not r.get("error"):
            human = "good" if labels[r["id"]]["label"] == "good" else "bad"
            out.append({**r, "human_label": human, "human_comment": labels[r["id"]]["comments"],
                        "human_category": labels[r["id"]].get("failure_category", "")})
    return out


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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def show(title: str, m: dict) -> None:
    print(f"\n{title} (n={m['n']})")
    print(f"  accuracy {m['accuracy']:.0%} | precision(bad) {m['precision_bad']:.0%} | recall(bad) {m['recall_bad']:.0%}")
    print("                judge:bad  judge:good")
    print(f"  human:bad     {m['tp']:>9}  {m['fn']:>10}")
    print(f"  human:good    {m['fp']:>9}  {m['tn']:>10}")


async def main_async(args) -> None:
    data = Path(args.data)
    rows = load(data)
    judge = create_judge(args.model)
    sem = asyncio.Semaphore(args.concurrency)

    tools = build_tools()
    for r in rows:
        r["retrieved"] = replay_retrieval(tools, r)

    async def run(row):
        async with sem:
            res = await judge.run(format_judge_prompt(row, row["retrieved"]))
        return res.output

    outs = await asyncio.gather(*(run(r) for r in rows))
    for r, o in zip(rows, outs):
        r["judge_label"], r["judge_reasoning"] = o.label, o.reasoning
        r["split"] = "heldout" if is_holdout(r["id"]) else "dev"

    show("ALL", metrics(rows))
    show("DEV", metrics([r for r in rows if r["split"] == "dev"]))
    show("HELD-OUT", metrics([r for r in rows if r["split"] == "heldout"]))

    print("\nDisagreements (dev only; use --show-heldout to see the rest):")
    for r in rows:
        if r["split"] == "heldout" and not args.show_heldout:
            continue
        if r["judge_label"] != r["human_label"]:
            print(f"\n[{r['split']}] {r['id']}  human={r['human_label']} judge={r['judge_label']}")
            print(f"  human comment: {r['human_comment'] or '(none)'}")
            print(f"  judge: {r['judge_reasoning'][:600]}")

    out = data.with_name(f"{data.stem}_judge.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "human_label", "judge_label", "human_comment", "human_category", "judge_reasoning"])
        for r in rows:
            w.writerow([r["id"], r["split"], r["human_label"], r["judge_label"], r["human_comment"], r["human_category"], r["judge_reasoning"]])
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--show-heldout", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
