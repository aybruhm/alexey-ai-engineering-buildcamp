import importlib
import subprocess
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
WEEK_DIR = EVALS_DIR.parents[1]

COMMANDS = {
    "scenarios": {
        "run": (
            "scenarios.run",
            "run the scenarios against the agent and save the responses",
        ),
        "label": (
            "scenarios/label.py",
            "open the Streamlit app to label scenario runs good / not good",
        ),
        "evals": (
            "scenarios.run_evals",
            "run the agent, judge it, and report the % of good responses",
        ),
    },
    "judge": {
        "evaluate": (
            "judge.evaluate",
            "measure the judge's agreement with the human labels",
        ),
    },
    "code-evaluators": {
        "run": (
            "code_evaluators.evaluators",
            "run the deterministic code evaluator on saved runs",
        ),
    },
    "synthetic": {
        "generate": (
            "synthetic.generate",
            "generate questions and reference answers from the docs",
        ),
        "run": ("synthetic.run", "run the agent on the generated questions"),
        "judge": (
            "synthetic.run_judges",
            "run the correctness, instruction and trajectory judges",
        ),
        "explore": (
            "synthetic/explore.py",
            "open the Streamlit app to explore the judged results",
        ),
    },
}

PACKAGE = __package__ or "pulumi_docs_agent.evals"


def usage() -> str:
    lines = [f"usage: python -m {PACKAGE} <group> <command> [args...]", ""]
    for group, commands in COMMANDS.items():
        lines.append(f"{group}")
        for name, (_, help_text) in commands.items():
            lines.append(f"  {name:10} {help_text}")
        lines.append("")
    lines.append("Pass --help after a command for its own options.")
    return "\n".join(lines)


def run_streamlit(script: str, args: list[str]) -> int:
    path = EVALS_DIR / script
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(path), *args], cwd=WEEK_DIR
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
