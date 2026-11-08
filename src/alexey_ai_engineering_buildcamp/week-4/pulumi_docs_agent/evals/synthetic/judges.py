from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ...prompts import INSTRUCTIONS as AGENT_INSTRUCTIONS

JUDGE_MODEL = "anthropic:claude-haiku-5-5"


class CorrectnessResult(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning about whether the agent's answer covers the reference answer."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the answer semantically contains the reference answer, 'bad' otherwise."
    )


class InstructionFollowingResult(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning about which instructions the agent followed or violated."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the agent followed its instructions overall, 'bad' if it violated an important one."
    )


class TrajectoryResult(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning about the optimality of the tool-call trajectory."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the trajectory was reasonably optimal, 'bad' if clearly wasteful."
    )
    suggestion: str = Field(
        description="Concrete suggestion for how the agent could have been more efficient, or 'none' if optimal."
    )


CORRECTNESS_INSTRUCTIONS = """
You are an expert evaluator. Your task is to decide whether an AI agent's
answer SEMANTICALLY CONTAINS the reference answer.

Rules:
- The agent's answer does NOT need to be word-for-word identical to the
  reference answer. It must convey the same key information.
- If the reference answer includes a command or code snippet, the agent's
  answer must include an equivalent one (names may differ, but the command,
  flags and logic must match).
- If the agent's answer provides additional correct information beyond the
  reference, that is fine and it should still be marked "good".
- If the agent's answer is missing the core information from the reference
  answer, or contradicts it, mark it "bad".
- If the agent says the information is not in the documentation, mark it "bad":
  the reference answer proves the documentation contains it.

Be strict: the core facts from the reference answer MUST appear in the
agent's response.
""".strip()

CORRECTNESS_PROMPT = """
User Question:
{question}

Reference Answer (ground truth):
{reference_answer}

Agent's Answer:
{agent_answer}
""".strip()

INSTRUCTION_FOLLOWING_INSTRUCTIONS = """
You are an expert evaluator. You will be given:
1. The system-prompt instructions that were given to a documentation agent.
2. A user question the agent received.
3. The agent's response and its own self-checks.

Your task is to verify whether the agent FOLLOWED its instructions.

Mark "good" if the agent followed the instructions well overall.
Mark "bad" if the agent violated any important instruction.

Only hold the agent to instructions that apply to this question. For example,
an instruction to adapt to the user's context only applies when the user gave
some context (a language, code, resource names). If the user gave none, the
agent cannot be faulted for not using it. Do not trust the agent's self-checks:
verify them against the answer yourself.

You see the agent's final answer, its references and its follow-up questions
(which are a separate output field, not part of the answer text), but not its
intermediate steps. The agent
explains each search and file retrieval in earlier steps that you cannot see.
Never mark the agent "bad" because the final answer lacks those step-by-step
explanations or a record of its iterations. Judge what you can see: whether the
answer is grounded in documentation, avoids outside knowledge, includes code
where appropriate, admits gaps honestly, and has on-topic follow-up questions.
""".strip()

INSTRUCTION_FOLLOWING_PROMPT = """
=== AGENT INSTRUCTIONS (system prompt) ===
{instructions}

=== USER QUESTION ===
{question}

=== AGENT ANSWER ===
{agent_answer}

=== AGENT REFERENCES (documents it cited) ===
{references}

=== AGENT FOLLOW-UP QUESTIONS (a separate output field) ===
{followups}

=== AGENT SELF-CHECKS ===
{self_checks}
""".strip()

TRAJECTORY_INSTRUCTIONS = """
You are an expert evaluator. You will be given:
1. A user question.
2. The sequence of tool calls (the "trajectory") the agent made.
3. The agent's final answer.
4. The documentation page the correct answer is known to come from.

The agent has two tools:
- search(query): searches the documentation index and returns matching files with snippets.
- get_file(filename): retrieves the full content of a specific documentation file.

The happy path: the agent performs one search call, then one or more get_file
calls to retrieve the relevant documents, then writes the answer. When the agent
does something more excessive than this, such as several searches, redundant
file retrievals or repeated calls, it is typically not optimal.

Evaluate the trajectory:
- Were the search queries relevant and well formulated?
- Did the agent retrieve the right files, in particular the source page?
- Were there redundant or duplicate tool calls?
- Did the agent make too many searches? Three or more searches is excessive.
  A second search is acceptable only when the first clearly missed.
- Could the agent have reached the same answer with fewer steps?

Mark "good" if the trajectory was reasonably efficient.
Mark "bad" if there are clear inefficiencies.
Provide a concrete suggestion for improvement, or "none" if optimal.
""".strip()

TRAJECTORY_PROMPT = """
User Question:
{question}

Tool Call Trajectory:
{tools}

Agent's Final Answer:
{agent_answer}

Documentation page the correct answer comes from:
{source_page}
""".strip()


def create_correctness_judge(model: str = JUDGE_MODEL) -> Agent:
    return Agent(
        name="correctness_judge",
        model=model,
        instructions=CORRECTNESS_INSTRUCTIONS,
        output_type=CorrectnessResult,
    )


def create_instruction_judge(model: str = JUDGE_MODEL) -> Agent:
    return Agent(
        name="instruction_judge",
        model=model,
        instructions=INSTRUCTION_FOLLOWING_INSTRUCTIONS,
        output_type=InstructionFollowingResult,
    )


def create_trajectory_judge(model: str = JUDGE_MODEL) -> Agent:
    return Agent(
        name="trajectory_judge",
        model=model,
        instructions=TRAJECTORY_INSTRUCTIONS,
        output_type=TrajectoryResult,
    )


def format_correctness_prompt(entry: dict[str, Any]) -> str:
    return CORRECTNESS_PROMPT.format(
        question=entry["question"],
        reference_answer=entry["reference_answer"],
        agent_answer=entry["answer"],
    )


def format_instruction_prompt(entry: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- [{'+' if c['passed'] else '-'}] {c['rule']}: {c['explanation']}"
        for c in entry.get("checks", [])
    )
    return INSTRUCTION_FOLLOWING_PROMPT.format(
        instructions=AGENT_INSTRUCTIONS,
        question=entry["question"],
        agent_answer=entry["answer"],
        references="\n".join(f"- {r}" for r in entry.get("references", [])) or "(none)",
        followups="\n".join(f"- {q}" for q in entry.get("followup_questions", []))
        or "(none)",
        self_checks=checks or "(none)",
    )


def format_trajectory_prompt(entry: dict[str, Any]) -> str:
    tools = "\n".join(
        f"{i}. {t['name']}({t['args']})" for i, t in enumerate(entry["tool_calls"], 1)
    )
    return TRAJECTORY_PROMPT.format(
        question=entry["question"],
        tools=tools or "(no tool calls)",
        agent_answer=entry["answer"],
        source_page=entry["filename"],
    )


JUDGES = {
    "judge_answer_correctness": (create_correctness_judge, format_correctness_prompt),
    "judge_instruction_following": (
        create_instruction_judge,
        format_instruction_prompt,
    ),
    "judge_trajectory": (create_trajectory_judge, format_trajectory_prompt),
}
