from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

JUDGE_MODEL = "anthropic:claude-haiku-5-5"

JUDGE_PROMPT = """
You are an expert evaluator of a documentation assistant (a RAG agent) for the
Pulumi documentation. You are given the user's question, the tool calls the
agent made, what those tool calls returned (the documents the agent actually
saw), and the agent's final structured response (answer, references,
found_answer, confidence, follow-up questions). Decide whether the response is
"good" or "bad". Judge correctness and helpfulness only.

Verify claims against the RETRIEVED_DOCUMENTS section. A claim, command, flag or
code line that does not appear there (or follow directly from it) is unsupported.
The agent must answer ONLY from the documentation it retrieved. It may not use
outside knowledge, even when that knowledge is correct.

A response is "good" if:
1. The question is covered by the docs, and the answer is correct, relevant to
   what the user asked, grounded in what the agent retrieved, and includes
   documentation-based code where a developer would expect it.
2. The question is NOT covered by the docs (off-topic, a general programming or
   ML question, or about a tool or feature the docs do not describe), and the
   agent says so plainly, without presenting a made-up or outside-knowledge
   answer as if it came from the docs.
3. The question is only partly covered, and the agent clearly separates what
   the docs say from what they do not say. Offering the closest documented
   alternative is fine, as long as it is clearly labeled as an alternative and
   not as the answer to the original question.

A response is "bad" if ANY of these apply:
1. Hallucination: it states facts, steps, flags, resources or code that are not
   supported by the retrieved documents, or invents a connection between the
   user's topic and loosely related documentation.
2. Wrong scope: it tries to answer a question it should have declined, or it
   treats an unsupported tool or integration as if it were supported.
3. Mixing sources: it gives a documented answer and then adds steps or
   recommendations from its own knowledge without saying so.
4. Incorrect or off-target: the information is wrong, answers a different
   question than the one asked, or the wrong language, SDK or product area for
   what the user said.
5. False "not found": it says the docs do not cover something that a reasonable
   search would find. Be suspicious of a "not found" that follows very few or
   very narrow searches and no get_file call.
6. Unclear question handled badly: the question is too broad or ambiguous to
   answer well, and the agent picks one interpretation silently instead of
   saying what it assumed or offering the options.
7. Leaked internals: the user-facing answer narrates the agent's own process
   (for example "Iteration 1", search queries tried, tool names) instead of
   giving the user an answer, or the follow-up questions are unrelated to what
   the user asked.
8. Unsupported citations: references that are not relevant to the answer, or
   that the answer does not actually use. Treat references marked as not in the
   corpus as fabricated.

Materiality: only count unsupported content against the response when it is
material, meaning a user who followed it would run a command, set an option or
rely on a fact that the retrieved documents do not support. Ordinary
paraphrasing, short connective explanation, reordering, or omitting a detail do
not make a response bad. A response that is correct and grounded on its core
claims, with only trivial extras, is "good". Do not mark a response bad merely
because it is long or because you could not find one minor sentence in the
documents.

When the agent claims found_answer=false, check that it does not go on to give a
substantive answer anyway.
When the agent claims high confidence, check that the answer earns it.

Work step by step. First, decide what the user asked and whether the docs could
plausibly cover it. Second, check each rule above against the response. Third,
give the label. Be strict about hallucination and outside knowledge, and fair
about answers that are correct and well grounded.
""".strip()


class JudgeEvaluation(BaseModel):
    reasoning: str = Field(
        description="Step-by-step check of the response against each rule, citing evidence"
    )
    label: Literal["good", "bad"]


def create_judge(model: str = JUDGE_MODEL) -> Agent[None, JudgeEvaluation]:
    model_capabilities = [
        Instrumentation(InstrumentationSettings(version=5)),
    ]
    return Agent(
        name="pulumi-docs-agent:evals_judge",
        model=model,
        instructions=JUDGE_PROMPT,
        output_type=JudgeEvaluation,
        capabilities=model_capabilities,
    )


def format_judge_prompt(row: dict[str, Any], retrieved: str = "") -> str:
    turns = row.get("turns") or [row["question"]]
    question = "\n".join(f"Turn {i}: {t}" for i, t in enumerate(turns, 1))
    invented = set(row.get("invented_references", []))
    refs = (
        "\n".join(
            f"- {r}{' (NOT IN CORPUS)' if r in invented else ''}"
            for r in row.get("references", [])
        )
        or "(none)"
    )
    tools = (
        "\n".join(
            f"{i}. {t['name']}({t['args']})"
            for i, t in enumerate(row.get("tool_calls", []), 1)
        )
        or "(none)"
    )
    followups = (
        "\n".join(f"- {q}" for q in row.get("followup_questions", [])) or "(none)"
    )
    return f"""
<USER_QUESTION>
{question}
</USER_QUESTION>

<TOOL_CALLS>
{tools}
</TOOL_CALLS>

<RETRIEVED_DOCUMENTS>
{retrieved or "(not available)"}
</RETRIEVED_DOCUMENTS>

<AGENT_RESPONSE>
{row.get("answer", "")}
</AGENT_RESPONSE>

<REFERENCES>
{refs}
</REFERENCES>

<AGENT_METADATA>
found_answer: {row.get("found_answer")}
confidence: {row.get("confidence")}
confidence_explanation: {row.get("confidence_explanation")}
answer_type: {row.get("answer_type")}
</AGENT_METADATA>

<FOLLOWUP_QUESTIONS>
{followups}
</FOLLOWUP_QUESTIONS>
""".strip()
