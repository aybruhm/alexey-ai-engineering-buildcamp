import argparse
import json
from datetime import datetime
from pathlib import Path

from ..code_evaluators.evaluators import evaluate as code_evaluate
from ..common.paths import (
    PIPELINE_RESULTS_DIR,
    SCENARIO_RESULTS_DIR,
    SYNTHETIC_DATA_DIR,
)
from ..common.retrieval_replay import build_tools, replay_retrieval
from ..judge.judge import create_judge, format_judge_prompt
from ..synthetic.judges import JUDGES
from .client import get_client
from .datasets import DATASETS, upload

OUTPUT_KEYS = (
    "answer",
    "references",
    "invented_references",
    "found_answer",
    "confidence",
    "followup_questions",
    "tool_calls",
)
SYNTHETIC_JUDGES = (
    "judge_answer_correctness",
    "judge_instruction_following",
    "judge_trajectory",
)


def newest(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def load_rows(path: Path) -> dict[str, dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data["rows"] if isinstance(data, dict) else data
    return {r["id"]: r for r in rows}


def verdict(label: str, reasoning: str) -> dict:
    return {
        "label": label,
        "score": 1.0 if label == "good" else 0.0,
        "explanation": reasoning,
    }


def make_task(rows: dict[str, dict]):
    def task(example):
        row = rows[example["metadata"]["id"]]
        return {key: row.get(key) for key in OUTPUT_KEYS}

    return task


def make_code_checks(kind: str):
    def code_checks(output, expected, metadata):
        reference = dict(expected or {})
        if kind == "synthetic":
            reference = {
                "expected_behavior": "answers",
                "expected_ref_prefixes": [metadata["filename"]],
            }
        return code_evaluate(output=output, reference=reference)

    return code_checks


def make_scenario_judge(rows: dict[str, dict], stored: dict[str, dict], live: bool):
    tools = build_tools() if live else None
    judge = create_judge() if live else None

    def judge_quality(metadata):
        row_id = metadata["id"]
        if not live and row_id in stored:
            return verdict(
                "good" if stored[row_id]["judge_label"] == "good" else "bad",
                stored[row_id]["judge_reasoning"],
            )
        row = rows[row_id]
        result = judge.run_sync(format_judge_prompt(row, replay_retrieval(tools, row)))
        return verdict(
            "good" if result.output.label == "good" else "bad", result.output.reasoning
        )

    return judge_quality


def make_synthetic_judge(name: str, rows: dict[str, dict], live: bool):
    make_judge, formatter = JUDGES[name]
    judge = make_judge() if live else None

    def run(metadata):
        row = rows[metadata["id"]]
        if not live and name in row:
            return verdict(row[name]["score"], row[name]["reasoning"])
        result = judge.run_sync(formatter(row))
        return verdict(result.output.score, result.output.reasoning)

    return run


def run_kind(
    client, kind: str, run_path: Path | None, live: bool, dry_run: int | None
) -> None:
    dataset = upload(client, kind)
    rows_path = run_path or (
        newest(SCENARIO_RESULTS_DIR, "*.jsonl")
        if kind == "scenarios"
        else newest(SYNTHETIC_DATA_DIR, "*_judged.json")
    )
    if rows_path is None:
        raise SystemExit(f"no saved run found for {kind}")
    rows = load_rows(rows_path)

    if kind == "scenarios":
        pipeline = newest(PIPELINE_RESULTS_DIR, "*.json")
        stored = (
            {k: v for k, v in load_rows(pipeline).items() if "judge_label" in v}
            if pipeline
            else {}
        )
        evaluators = {
            "code_checks": make_code_checks(kind),
            "judge_quality": make_scenario_judge(rows, stored, live),
        }
    else:
        evaluators = {"code_checks": make_code_checks(kind)}
        evaluators.update(
            {n: make_synthetic_judge(n, rows, live) for n in SYNTHETIC_JUDGES}
        )

    mode = "live judges" if live else "stored judge verdicts"
    name = f"{kind}-replay-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"\n{kind}: replaying {rows_path.name} ({len(rows)} rows), {mode}")
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=make_task(rows),
        evaluators=evaluators,
        experiment_name=name,
        experiment_description=f"Replay of saved agent outputs from {rows_path.name}; {mode}.",
        experiment_metadata={"source_file": rows_path.name, "judges": mode},
        dry_run=dry_run if dry_run else False,
        timeout=120,
    )
    if not dry_run:
        print(f"Experiment recorded: {name}")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS), nargs="*")
    parser.add_argument(
        "--run", help="saved run file to replay (defaults to the newest)"
    )
    parser.add_argument(
        "--live-judges",
        action="store_true",
        help="re-run the judges instead of using stored verdicts",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        metavar="N",
        help="run N examples locally without recording anything",
    )
    args = parser.parse_args()

    client = get_client()
    kinds = args.only or list(DATASETS)
    if args.run and len(kinds) != 1:
        raise SystemExit("--run needs exactly one of --only scenarios / synthetic")
    for kind in kinds:
        run_kind(
            client,
            kind,
            Path(args.run) if args.run else None,
            args.live_judges,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
