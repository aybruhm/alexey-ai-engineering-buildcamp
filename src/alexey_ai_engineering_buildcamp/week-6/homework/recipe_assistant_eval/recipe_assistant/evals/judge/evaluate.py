import argparse
import json
from pathlib import Path

from ..common.labels import load_labels, run_id
from ..common.paths import JUDGED_FILE


def outcome(judge: str, human: str) -> str:
    """"bad" is the positive class: the judge's job is to catch bad responses."""
    if judge == "bad":
        return "TP" if human == "bad" else "FP"
    return "FN" if human == "bad" else "TN"


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judge labels with human labels.")
    parser.add_argument("--data", type=Path, default=JUDGED_FILE, help="judged results file")
    parser.add_argument("--show", default="", help="question substring to print in full")
    args = parser.parse_args()

    judged = json.loads(args.data.read_text())
    labels = load_labels()

    rows = []
    for r in judged:
        human = labels.get(run_id(r))
        if human is None:
            print(f"skipping unlabeled: {r['question']}")
            continue
        rows.append((r, human, outcome(r["judge_label"], human["label"])))

    counts = {k: sum(1 for *_, o in rows if o == k) for k in ("TP", "FP", "FN", "TN")}
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    total = len(rows)

    print(f"\n{'#':>2}  {'human':5} {'judge':5} {'':2}  question")
    for n, (r, human, o) in enumerate(rows, 1):
        mark = "" if o in ("TP", "TN") else o
        print(f"{n:>2}  {human['label']:5} {r['judge_label']:5} {mark:2}  {r['question'][:60]}")

    print(f"\n             human bad  human good")
    print(f"judge bad    TP = {tp:<5}  FP = {fp}")
    print(f"judge good   FN = {fn:<5}  TN = {tn}")
    print(f"\naccuracy  = (tp + tn) / total = {fmt(ratio(tp + tn, total))}  ({tp + tn}/{total})")
    print(f"precision = tp / (tp + fp)    = {fmt(ratio(tp, tp + fp))}  ({tp}/{tp + fp})")
    print(f"recall    = tp / (tp + fn)    = {fmt(ratio(tp, tp + fn))}  ({tp}/{tp + fn})")

    for r, human, o in rows:
        if args.show and args.show.lower() in r["question"].lower():
            print(f"\n--- {r['question']} ---")
            print(f"human: {human['label']} ({human.get('failure_category') or '-'}): {human.get('comments', '')}")
            print(f"judge: {r['judge_label']}: {r['judge_reasoning']}")
            print(f"outcome: {o}")


if __name__ == "__main__":
    main()
