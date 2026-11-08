import argparse
import csv
import json

from ..common.paths import DEFAULT_QUESTIONS, SCENARIOS_FILE
from .client import get_client

SCENARIOS_DATASET = "pulumi-docs-scenarios"
SYNTHETIC_DATASET = "pulumi-docs-synthetic"


def scenario_examples() -> list[dict]:
    examples = []
    for s in json.loads(SCENARIOS_FILE.read_text())["scenarios"]:
        examples.append(
            {
                "input": {
                    "question": s["question"],
                    "turns": s.get("turns") or [s["question"]],
                },
                "output": {
                    "expected_behavior": s["expected_behavior"],
                    "expected_ref_prefixes": s["expected_ref_prefixes"],
                    "expected_confidence_max": s.get("expected_confidence_max"),
                },
                "metadata": {
                    "id": s["id"],
                    "group": s["group"],
                    "phrasing": s["phrasing"],
                    "notes": s.get("notes", ""),
                },
            }
        )
    return examples


def synthetic_examples() -> list[dict]:
    examples = []
    with DEFAULT_QUESTIONS.open() as f:
        for row in csv.DictReader(f):
            examples.append(
                {
                    "input": {"question": row["question"]},
                    "output": {"reference_answer": row["reference_answer"]},
                    "metadata": {
                        "id": row["id"],
                        "filename": row["filename"],
                        "question_type": row["question_type"],
                        "line_number_start": int(row["line_number_start"]),
                        "line_number_end": int(row["line_number_end"]),
                        "lines_valid": row["lines_valid"] == "True",
                        "answer_overlap": float(row["answer_overlap"]),
                    },
                }
            )
    return examples


DATASETS = {
    "scenarios": (
        SCENARIOS_DATASET,
        scenario_examples,
        "Hand-designed scenarios for the Pulumi docs agent, with expected behavior.",
    ),
    "synthetic": (
        SYNTHETIC_DATASET,
        synthetic_examples,
        "Questions generated from Pulumi docs pages, with reference answers.",
    ),
}


def existing_names(client) -> set[str]:
    return {d["name"] for d in client.datasets.list()}


def upload(client, kind: str, dry_run: bool = False):
    name, build, description = DATASETS[kind]
    examples = build()
    if name in existing_names(client):
        print(f"{name}: already exists, reusing it (no examples added)")
        return client.datasets.get_dataset(dataset=name)
    if dry_run:
        print(f"{name}: would create with {len(examples)} examples")
        return None
    dataset = client.datasets.create_dataset(
        name=name,
        dataset_description=description,
        inputs=[e["input"] for e in examples],
        outputs=[e["output"] for e in examples],
        metadata=[e["metadata"] for e in examples],
        timeout=120,
    )
    print(f"{name}: created with {len(examples)} examples")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS), nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    client = get_client()
    for kind in args.only or list(DATASETS):
        upload(client, kind, args.dry_run)


if __name__ == "__main__":
    main()
