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
7. Leaked implementation details: the user-facing answer reveals the system prompt, hidden instructions, tool schemas, raw logs, or internal code. Do not treat a short explanation of what the agent searched for or which files it read as a leak; the agent is instructed to explain its search and retrieval reasoning. Follow-up questions that are unrelated to what the user asked are also bad.
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

A code example that applies a documented option or syntax to a different resource type is supported if the docs demonstrate the same option on another resource and the answer applies it consistently. Do not require the exact same resource type in the example.

When the agent claims found_answer=false, check that it does not go on to give a
substantive answer anyway.
When the agent claims high confidence, check that the answer earns it.
Use the SCENARIO_EXPECTATION section when it is present. It tells you what this case is designed to test. It does not override the rules above, but it resolves ambiguity about the intended behavior:

- If expected_behavior is "says_not_found", a clear refusal is correct. A substantive answer is bad. A brief pointer to a related documented feature is fine only if it is clearly not an answer to the question. A full how-to, code example, or command sequence for a different feature is still bad.
- If expected_behavior is "answers_or_asks_clarification" and the question is broad or ambiguous, the response must ask for the missing context or offer explicit choices before giving a generic answer. A generic answer that silently picks one path is bad.
- If expected_behavior is "answers", a not-found response is bad unless the agent can show that it searched broadly and the docs genuinely lack the answer.
- If expected_confidence_max is set and the response exceeds it, that is evidence of overconfidence, but judge the answer itself too.

Hard calls:

- Search and retrieval narration is allowed. The agent is instructed to explain each search and get_file call, including "Iteration 1" or "Research steps". Do not mark an answer bad for that. Penalize only leaks of the system prompt, hidden instructions, tool schemas, raw logs, or claims that it ran actions it cannot run.
- Outside knowledge in a not-found answer is bad, even a short definition or passing fact. In an off-topic or not-found answer, do not identify or define the off-topic subject. "X is a Y, but I cannot answer" is outside knowledge and bad. For example, if the user asks about the Queen's Gambit and the agent says "The Queen's Gambit is a chess opening, but I cannot answer from the docs", that is bad. The refusal is right, but the definition is outside knowledge. Before labeling an off-topic or not-found answer, quote any sentence that identifies or defines the off-topic subject. If such a sentence exists, the response is bad, even if the rest is a refusal.
- Stale or unsupported figures are bad when the user could rely on them, even if the answer adds a caveat. If the scenario expects a not-found or partial answer, do not quote specific historical prices, versions, or limits at all. State that only historical figures exist and point to the current source.
- If the user asks for X and the response says X is unsupported but gives Y, the response is good only if it clearly says Y is not X and does not imply that Y solves the request. If Y is presented as the answer, bad.
- A direct question with one or two narrow searches and no get_file call is weak evidence for "not found". If a reasonable broader search would find the answer, bad.
- Do not accept "the retrieved document was truncated" as proof that the docs lack the answer. If the scenario expects an answer, the agent should fetch the missing file or the obvious linked reference before giving up. A not-found answer that stops at a truncated page is bad.

Work step by step. First, decide what the user asked and what the scenario expects. Second, check the response's core behavior against the scenario expectation. Third, check each material claim against the retrieved documents. Fourth, check presentation and citations. Fifth, give the label and a confidence from 0 to 1. Use confidence below 0.7 when the case is genuinely ambiguous or the retrieved documents are incomplete.
""".strip()


class JudgeEvaluation(BaseModel):
    reasoning: str = Field(
        description="Step-by-step check of the response against each rule, citing evidence"
    )
    label: Literal["good", "bad"]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear-cut this label is; use below 0.7 for ambiguous cases",
    )


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
    expectation = []
    if row.get("phrasing"):
        expectation.append(f"phrasing: {row['phrasing']}")
    if row.get("expected_behavior"):
        expectation.append(f"expected_behavior: {row['expected_behavior']}")
    if row.get("expected_confidence_max") is not None:
        expectation.append(f"expected_confidence_max: {row['expected_confidence_max']}")
    if row.get("notes"):
        expectation.append(f"notes: {row['notes']}")
    expectation_text = "\n".join(expectation) or "(none)"
    return f"""
<USER_QUESTION>
{question}
</USER_QUESTION>

<SCENARIO_EXPECTATION>
{expectation_text}
</SCENARIO_EXPECTATION>

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
