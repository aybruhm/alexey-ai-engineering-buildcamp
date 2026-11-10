import json
from pathlib import Path

from .paths import LABELS_FILE, RECIPES_FILE, RESULTS_FILE

LABELS = ["good", "bad"]

# Why a "bad" response failed. Keep the list short so the counts mean something.
FAILURE_CATEGORIES = [
    "",
    "hallucination",  # recipe or ingredient that is not in the dataset
    "invented-advice",  # cooking tips/substitutions not supported by the recipe data
    "false-not-found",  # says it has no recipe when a matching one exists
    "incorrect-details",  # wrong times, temperatures, ingredients, tags
    "incomplete",  # misses part of the question or part of the recipe
    "wrong-scope",  # answers something it should have declined
    "other",
]


def load_results(path: Path = RESULTS_FILE) -> list[dict]:
    return json.loads(path.read_text())


def run_id(run: dict) -> str:
    """Results have no id field; the question is unique and survives reruns."""
    return run["question"]


def load_labels(path: Path = LABELS_FILE) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {item["id"]: item for item in json.loads(path.read_text())}


def save_labels(labels: dict[str, dict], path: Path = LABELS_FILE) -> None:
    path.write_text(json.dumps(list(labels.values()), indent=2, ensure_ascii=False))


def load_recipes(path: Path = RECIPES_FILE) -> list[dict]:
    return json.loads(path.read_text())
