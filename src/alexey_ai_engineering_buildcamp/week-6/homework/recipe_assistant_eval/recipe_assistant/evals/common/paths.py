from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = EVALS_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent

DATA_DIR = PACKAGE_DIR / "data"
RECIPES_FILE = DATA_DIR / "recipes.json"

SCENARIOS_DIR = EVALS_DIR / "scenarios"
SCENARIOS_FILE = SCENARIOS_DIR / "scenarios.csv"
SCENARIO_RESULTS_DIR = SCENARIOS_DIR / "results"
RESULTS_FILE = SCENARIO_RESULTS_DIR / "results.json"
LABELS_FILE = SCENARIO_RESULTS_DIR / "results_labels.json"
CSV_FILE = SCENARIO_RESULTS_DIR / "results.csv"
JUDGED_FILE = SCENARIO_RESULTS_DIR / "results_judged.json"

SYNTHETIC_DIR = EVALS_DIR / "synthetic"
SYNTHETIC_SCENARIOS_FILE = SYNTHETIC_DIR / "synthetic_scenarios.csv"
SYNTHETIC_RESULTS_DIR = SYNTHETIC_DIR / "results"
SYNTHETIC_RESULTS_FILE = SYNTHETIC_RESULTS_DIR / "results.json"
SYNTHETIC_JUDGED_FILE = SYNTHETIC_RESULTS_DIR / "results_judged.json"
