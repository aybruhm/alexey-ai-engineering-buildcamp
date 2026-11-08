import argparse
import asyncio
import csv
import json
import random
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ...retrieval import Doc
from ...utils import retrieve_documentation
from ..common.paths import DOCS_DIR
from ..common.paths import SYNTHETIC_DATA_DIR as OUT_DIR

GENERATOR_MODEL = "anthropic:claude-haiku-5-5"

MIN_CHARS = 1000
MAX_PROMPT_CHARS = 20_000
MIN_QUESTIONS = 2
MAX_QUESTIONS = 4
CHARS_PER_QUESTION = 1500
SKIP_FILENAMES = ("changelog",)


class GeneratedQuestion(BaseModel):
    user_question: str = Field(description="the question we generate")
    reference_answer: str = Field(
        description="what is the correct answer based on the document we analyzed"
    )
    line_number_start: int = Field(
        description="the line of the document where the reference answer starts"
    )
    line_number_end: int = Field(
        description="the line of the document where the reference answer ends"
    )
    question_type: Literal[
        "frustrated_beginner", "hurried_developer", "context_assumer", "direct_inquirer"
    ] = Field(description="the persona style used to phrase the question")


class QuestionsResponse(BaseModel):
    questions: list[GeneratedQuestion]


INSTRUCTIONS = """
You are an expert at generating realistic user questions based on technical documentation.
Your task is to analyze the provided Pulumi documentation page and generate questions that a user might realistically ask.

We want questions from REAL users who are stuck, frustrated, or searching for specific information, not perfectly phrased textbook questions.

For each question, adopt one of these personas, and vary them across the questions:
1. frustrated_beginner: Uses non-technical terms, explains the goal they are trying to reach but are stuck. (e.g. "my stack won't let me change a value after I created it, what am I doing wrong?")
2. hurried_developer: Uses fragments, keywords or short phrases instead of full sentences. (e.g. "python pulumi secret config")
3. context_assumer: Asks assuming the assistant knows which page or step the user is on. (e.g. "it says I can use a different backend, how?")
4. direct_inquirer: Asks a straightforward, somewhat technical question in a conversational tone. (e.g. "Can I run my update on a schedule without a CI system?")

For each question you must also write the reference answer, taken directly from the document's content, and pinpoint the exact starting and ending line numbers in the numbered document where that answer is located.

Strict guidelines:
1. DO NOT end questions with phrases like "based on this text" or "according to the documentation".
2. DO NOT make the questions too formal or academic.
3. Every question must be answerable from THIS page alone. Do not ask about things the page does not cover.
4. Do not mention the page, its title, or its file name in the question.
5. The reference answer must be comprehensive, include relevant commands or code from the page, and be directly supported by the text. Do not add outside knowledge.
6. Line numbers must match the numbered lines you were given. The start line must not be after the end line.
""".strip()

PROMPT_TEMPLATE = """
Number of questions:
{number_of_questions}

Document:
{numbered_content}
""".strip()


def add_line_numbers(document: str) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(document.splitlines()))


def question_count(content: str) -> int:
    return max(MIN_QUESTIONS, min(MAX_QUESTIONS, len(content) // CHARS_PER_QUESTION))


def create_user_prompt(content: str) -> str:
    lines = content.splitlines()
    clipped = "\n".join(lines[: len(content[:MAX_PROMPT_CHARS].splitlines())])
    return PROMPT_TEMPLATE.format(
        number_of_questions=question_count(content),
        numbered_content=add_line_numbers(clipped),
    )


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", text.lower())}


def check_grounding(q: GeneratedQuestion, content: str) -> dict:
    lines = content.splitlines()
    n = len(lines)
    valid = 1 <= q.line_number_start <= q.line_number_end <= n
    overlap = 0.0
    if valid:
        cited = tokens("\n".join(lines[q.line_number_start - 1 : q.line_number_end]))
        ref = tokens(q.reference_answer)
        overlap = len(ref & cited) / len(ref) if ref else 0.0
    return {"lines_valid": valid, "answer_overlap": round(overlap, 2)}


def area(filename: str) -> str:
    parts = filename.split("/")
    if parts[0] != "docs":
        return parts[0]
    return "/".join(parts[:2]) if len(parts) > 2 else "docs"


def select_pages(documents: list[Doc], n_pages: int, seed: int) -> list[Doc]:
    candidates = [
        d
        for d in documents
        if d["filename"].startswith("docs/")
        and len(d["content"]) > MIN_CHARS
        and not any(s in d["filename"] for s in SKIP_FILENAMES)
    ]
    by_area: dict[str, list[Doc]] = defaultdict(list)
    for d in candidates:
        by_area[area(d["filename"])].append(d)

    rng = random.Random(seed)
    for pages in by_area.values():
        rng.shuffle(pages)

    chosen: list[Doc] = []
    areas = sorted(by_area)
    while len(chosen) < n_pages and any(by_area.values()):
        for a in areas:
            if by_area[a] and len(chosen) < n_pages:
                chosen.append(by_area[a].pop())
    return chosen


async def generate_for_page(agent, doc: Doc, sem: asyncio.Semaphore) -> list[dict]:
    async with sem:
        result = await agent.run(create_user_prompt(doc["content"]))
    rows = []
    for q in result.output.questions:
        rows.append(
            {
                "question": q.user_question,
                "reference_answer": q.reference_answer,
                "line_number_start": q.line_number_start,
                "line_number_end": q.line_number_end,
                "question_type": q.question_type,
                "filename": doc["filename"],
                **check_grounding(q, doc["content"]),
            }
        )
    return rows


async def main_async(args) -> None:
    documents = retrieve_documentation(
        repo_owner="pulumi",
        repo_name="docs",
        branch_name="master",
        docs_path="content",
        save_dir=str(DOCS_DIR),
    )
    pages = select_pages(documents, args.pages, args.seed)
    print(f"{len(documents)} documents, generating from {len(pages)} pages")
    by_area: dict[str, int] = defaultdict(int)
    for p in pages:
        by_area[area(p["filename"])] += 1
    for a, c in sorted(by_area.items()):
        print(f"  {a}: {c}")

    agent = Agent(
        name="data_generator",
        model=args.model,
        instructions=INSTRUCTIONS,
        output_type=QuestionsResponse,
    )
    sem = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(
        *(generate_for_page(agent, p, sem) for p in pages), return_exceptions=True
    )

    rows: list[dict] = []
    for page, outcome in zip(pages, outcomes):
        if isinstance(outcome, Exception):
            print(f"ERROR {page['filename']}: {outcome}")
        else:
            rows.extend(outcome)
    for i, row in enumerate(rows, 1):
        row["id"] = f"syn-{i:04d}"

    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / "questions_generated.csv"
    fields = ["id", *[k for k in rows[0] if k != "id"]] if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "questions_generated.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )

    bad_lines = sum(not r["lines_valid"] for r in rows)
    low_overlap = sum(r["lines_valid"] and r["answer_overlap"] < 0.4 for r in rows)
    types: dict[str, int] = defaultdict(int)
    for r in rows:
        types[r["question_type"]] += 1
    print(f"\n{len(rows)} questions from {len(pages)} pages")
    print(f"Question types: {dict(types)}")
    print(
        f"Invalid line ranges: {bad_lines}; low answer/line overlap (<0.4): {low_overlap}"
    )
    print(f"Saved: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=GENERATOR_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
