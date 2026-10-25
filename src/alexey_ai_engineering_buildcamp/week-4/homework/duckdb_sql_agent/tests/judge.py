from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from ..models import SQLResult
from ..prompts import JUDGE_INSTRUCTIONS, build_judge_prompt
from .utils import collect_tools


class JudgeCriterion(BaseModel):
    """
    Evaluation of a single test requirement or behavioral rule.
    """

    criterion_description: str = Field(
        description="The specific requirement or rule that the agent is being evaluated against."
    )
    passed: bool = Field(
        description="Indicates whether the agent's response and actions successfully satisfied this requirement."
    )
    judgement: str = Field(
        description="A clear explanation of why the agent passed or failed, referencing specific evidence from the agent's output or tool calls."
    )


class JudgeFeedback(BaseModel):
    """
    The complete evaluation report from the judge agent, summarizing performance across all criteria.
    """

    criteria: list[JudgeCriterion] = Field(
        description="A collection of individual evaluations for each performance requirement provided in the test."
    )
    feedback: str = Field(
        description="A holistic summary of the agent's performance, providing overall context and identifying key strengths or failures."
    )


def create_judge_agent() -> Agent[None, JudgeFeedback]:
    model_capabilities = [
        Instrumentation(InstrumentationSettings(version=5)),
    ]
    agent = Agent(
        name="duckdb-sql:judge",
        model="anthropic:claude-haiku-5-5",
        instructions=JUDGE_INSTRUCTIONS,
        output_type=JudgeFeedback,
        capabilities=model_capabilities,
    )
    return agent


async def evaluate_agent_performance(
    result: AgentRunResult[SQLResult],
    criteria: list[str],
) -> JudgeFeedback:
    tool_calls = collect_tools(result.new_messages())
    output = result.output.model_dump_json()

    judge_agent = create_judge_agent()
    judge_user_prompt = build_judge_prompt(
        criteria=criteria,
        output=output,
        tool_calls=tool_calls,
    )

    judge_result = await judge_agent.run(judge_user_prompt)
    return judge_result.output


async def assert_criteria(
    result: AgentRunResult[SQLResult],
    criteria: list[str],
) -> None:
    feedback = await evaluate_agent_performance(result, criteria)

    for criterion in feedback.criteria:
        print(f"{criterion.criterion_description}: {criterion.judgement}")
        assert criterion.passed, (
            f"{criterion.criterion_description}: {criterion.judgement}"
        )
