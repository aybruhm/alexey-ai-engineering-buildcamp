import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from ...pricing import calculate_run_cost
from ..common.labels import load_recipes
from ..common.paths import JUDGED_FILE, RESULTS_FILE


class JudgeEvaluation(BaseModel):
    reasoning: str
    label: Literal["good", "bad"]


JUDGE_INSTRUCTIONS = """You are an expert evaluator assessing a recipe assistant that answers
cooking questions using a fixed recipe collection. The complete collection is below, so you
can check the response against it instead of guessing.

<RECIPES>
{recipes}
</RECIPES>

A response is "good" if:
1. It accurately answers using ONLY information from the recipe collection
2. It correctly identifies when a recipe is not available and says so
3. It correctly identifies out-of-scope questions (not about cooking/recipes) and declines

A response is "bad" if ANY of these apply:
1. It makes up recipes, ingredients, or instructions not in the collection (hallucination)
2. It provides cooking advice it invented rather than found in the recipe data
3. It answers a question it should have declined (not about recipes, or recipe not available)
4. It gives incorrect information from the recipes (wrong times, wrong ingredients, wrong
   steps, wrong tags)
5. It says it cannot help when a matching recipe exists in the collection

How to check facts:
- Verify every factual claim about a recipe (name, cuisine, times, servings, ingredients,
  steps, temperatures, tags) against <RECIPES>. A claim that matches the data is grounded,
  even when it is paraphrased, summed (e.g. prep + cook = total time) or reordered.
- Only call a fact invented or wrong if it contradicts <RECIPES> or is absent from it.
  Never mark a response bad because you "cannot verify" something: look it up.
- Check claims about the collection as a whole too ("the only X recipe", "N recipes are
  tagged Y", "no recipe for X") against the full list.

What counts as invented advice (bad, even when the response says it is not from the recipe):
- Telling the user to do something the recipe data does not support: substitutions or swaps,
  quantities, techniques or tips, adapting or modifying a recipe, make-ahead or storage
  instructions, nutrition figures or estimates. A disclaimer does not make it acceptable.

What does NOT count as invented advice on its own:
- Recommending, ranking or comparing recipes from the collection based on their data
  (quickest, lightest, most filling, best match for the request).
- Suggesting an alternative recipe from the collection when there is no exact match.
- Saying information is not in the collection and pointing the user to an outside source or
  tool for it, without supplying the missing information itself.
- Offering to look something up or asking a follow-up question.
- A brief, clearly hedged remark about why a recipe does something, when it gives the user no
  new instruction to follow. Judge whether the core answer is correct and grounded.

In your reasoning, first list the factual claims and check each against <RECIPES>, then list
any advice and classify it using the rules above, then decide the label.
"""


def build_instructions() -> str:
    recipes = json.dumps(load_recipes(), indent=1, ensure_ascii=False)
    return JUDGE_INSTRUCTIONS.format(recipes=recipes)


judge_agent = Agent(
    "anthropic:claude-haiku-5-5",
    output_type=JudgeEvaluation,
    instructions=build_instructions(),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label every agent result with the LLM judge.")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE)
    parser.add_argument("--output", type=Path, default=JUDGED_FILE)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    for i, row in enumerate(results):
        prompt = f"""Question: {row["question"]}
Agent response: {row["output"]}"""

        evaluation = judge_agent.run_sync(prompt)
        row["judge_label"] = evaluation.output.label
        row["judge_reasoning"] = evaluation.output.reasoning
        row["judge_cost"] = calculate_run_cost(evaluation)
        print(f"[{i + 1}/{len(results)}] {row['judge_label']}: {row['question']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_cost = sum(r["judge_cost"] for r in results)
    print(f"\nSaved judged results to {args.output}")
    print(f"Total judge cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
