import importlib
import subprocess
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALS_DIR.parents[1]

COMMANDS = {
    "scenarios": {
        "run": (
            "scenarios.run",
            "run the scenarios against the agent and save the responses",
        ),
        "label": (
            "scenarios/label.py",
            "open the Streamlit app to label scenario runs good / bad",
        ),
        "csv": (
            "scenarios.to_csv",
            "export the results and labels to a CSV for Excel or Google Sheets",
        ),
    },
    "synthetic": {
        "generate": (
            "synthetic.generate",
            "generate 5 questions per recipe into synthetic_scenarios.csv",
        ),
        "report": (
            "synthetic.report",
            "summarise the judge's bad rate by question type and recipe",
        ),
    },
    "judge": {
        "run": (
            "judge.judge",
            "label every saved result with the LLM judge",
        ),
        "evaluate": (
            "judge.evaluate",
            "measure the judge's agreement with the human labels",
        ),
    },
}

PACKAGE = __package__ or "recipe_assistant.evals"


def usage() -> str:
    lines = [f"usage: python -m {PACKAGE} <group> <command> [args...]", ""]
    for group, commands in COMMANDS.items():
        lines.append(f"{group}")
        for name, (_, help_text) in commands.items():
            lines.append(f"  {name:10} {help_text}")
        lines.append("")
    return "\n".join(lines)


def run_streamlit(script: str, args: list[str]) -> int:
    path = EVALS_DIR / script
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(path), *args], cwd=PROJECT_DIR
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv and argv[0] in ("-h", "--help") else 2

    group, command, rest = argv[0], argv[1], argv[2:]
    if group not in COMMANDS or command not in COMMANDS[group]:
        print(f"unknown command: {group} {command}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    target = COMMANDS[group][command][0]
    if target.endswith(".py"):
        return run_streamlit(target, rest)

    module = importlib.import_module(f"{PACKAGE}.{target}")
    sys.argv = [f"{group} {command}", *rest]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
