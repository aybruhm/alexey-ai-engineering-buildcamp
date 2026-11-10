import csv

from ..common.labels import load_labels, load_results, run_id
from ..common.paths import CSV_FILE

COLUMNS = [
    "id",
    "question",
    "category",
    "type",
    "output",
    "label",
    "failure_category",
    "comments",
    "execution_time",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost",
]


def main() -> None:
    results = load_results()
    labels = load_labels()

    # utf-8-sig so Excel shows characters like "°F" correctly.
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for n, run in enumerate(results, 1):
            label = labels.get(run_id(run), {})
            writer.writerow(
                {
                    "id": n,
                    "question": run["question"],
                    "category": run["category"],
                    "type": run["type"],
                    "output": run["output"],
                    "label": label.get("label", ""),
                    "failure_category": label.get("failure_category", ""),
                    "comments": label.get("comments", ""),
                    "execution_time": run["execution_time"],
                    **run["tokens"],
                    "cost": run["cost"],
                }
            )

    print(f"Wrote {len(results)} rows to {CSV_FILE} ({len(labels)} labeled)")


if __name__ == "__main__":
    main()
