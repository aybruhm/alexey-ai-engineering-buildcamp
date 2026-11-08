import ast
import csv
import json

from .paths import DEFAULT_QUESTIONS, SCENARIOS_FILE, TESTS_DIR


def normalize(question: str) -> str:
    return " ".join(question.split())


def known_questions() -> dict[str, dict]:
    known: dict[str, dict] = {}

    if TESTS_DIR.exists():
        for path in sorted(TESTS_DIR.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if len(node.value) > 8:
                        known.setdefault(normalize(node.value), {"source": "test"})

    if DEFAULT_QUESTIONS.exists():
        with DEFAULT_QUESTIONS.open() as f:
            for row in csv.DictReader(f):
                known[normalize(row["question"])] = {"source": "synthetic", **row}

    if SCENARIOS_FILE.exists():
        for scenario in json.loads(SCENARIOS_FILE.read_text())["scenarios"]:
            for turn in scenario.get("turns") or [scenario["question"]]:
                known[normalize(turn)] = {"source": "scenario", **scenario}

    return known
