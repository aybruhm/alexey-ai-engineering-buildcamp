import argparse
import csv
import json
import time
from pathlib import Path

from ...agent import agent
from ...pricing import calculate_run_cost
from ..common.paths import RESULTS_FILE, SCENARIOS_FILE


def main():
    parser = argparse.ArgumentParser(description="Run the agent on every scenario.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_FILE)
    parser.add_argument("--output", type=Path, default=RESULTS_FILE)
    args = parser.parse_args()

    with open(args.scenarios) as f:
        scenarios = list(csv.DictReader(f))

    results = []

    for i, scenario in enumerate(scenarios):
        question = scenario["question"]
        print(f"[{i + 1}/{len(scenarios)}] {question}")

        start = time.time()
        result = agent.run_sync(question)
        elapsed = time.time() - start

        usage = result.usage
        cost = calculate_run_cost(result)

        results.append(
            {
                "question": question,
                "category": scenario["category"],
                "type": scenario["type"],
                **({"recipe_id": int(scenario["recipe_id"])} if scenario.get("recipe_id") else {}),
                "output": result.output,
                "execution_time": round(elapsed, 2),
                "tokens": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "cost": cost,
            }
        )

        print(f"  Done in {elapsed:.1f}s (${cost})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_cost = sum(r["cost"] for r in results)
    print(f"\nSaved {len(results)} results to {args.output}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
