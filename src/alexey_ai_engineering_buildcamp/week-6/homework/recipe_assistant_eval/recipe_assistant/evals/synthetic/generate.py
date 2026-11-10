import argparse
import csv
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from ...pricing import calculate_run_cost
from ..common.labels import load_recipes
from ..common.paths import SYNTHETIC_SCENARIOS_FILE

QUESTION_TYPES = ["factual", "how-to", "detail", "substitution", "comparison"]


class Question(BaseModel):
    question: str
    category: str
    type: str


class GeneratedQuestions(BaseModel):
    questions: list[Question]


GENERATOR_INSTRUCTIONS = f"""You write realistic questions that home cooks ask a recipe assistant.
Write each question the way a user would type it, without quoting the recipe text back.

For each question set:
- type: exactly one of {", ".join(QUESTION_TYPES)}
- category: a short kebab-case topic label, e.g. ingredients, method, timing, temperature,
  dietary, substitution, comparison
"""

generator_agent = Agent(
    "anthropic:claude-opus-5-5",
    output_type=GeneratedQuestions,
    instructions=GENERATOR_INSTRUCTIONS,
)


def generation_prompt(recipe: dict) -> str:
    return f"""Given this recipe, generate 5 questions a user might ask.
Include: 1 factual, 1 how-to, 1 detail, 1 substitution, 1 comparison with another dish.

Recipe: {recipe['name']}
Cuisine: {recipe['cuisine']}
Ingredients: {', '.join(recipe['ingredients'])}
Instructions: {recipe['instructions']}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 synthetic questions per recipe.")
    parser.add_argument("--output", type=Path, default=SYNTHETIC_SCENARIOS_FILE)
    args = parser.parse_args()

    rows = []
    total_cost = 0.0
    recipes = load_recipes()
    for n, recipe in enumerate(recipes, 1):
        result = generator_agent.run_sync(generation_prompt(recipe))
        total_cost += calculate_run_cost(result)
        questions = result.output.questions
        types = sorted(q.type for q in questions)
        if types != sorted(QUESTION_TYPES):
            print(f"  warning: {recipe['name']} got types {types}")
        for q in questions:
            rows.append(
                {
                    "question": q.question,
                    "category": q.category,
                    "type": q.type,
                    "recipe_id": recipe["id"],
                    "recipe_name": recipe["name"],
                }
            )
        print(f"[{n}/{len(recipes)}] {recipe['name']}: {len(questions)} questions")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} questions to {args.output}")
    print(f"Total generation cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
