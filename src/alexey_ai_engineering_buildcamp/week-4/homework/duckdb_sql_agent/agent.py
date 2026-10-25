from pydantic_ai import (
    Agent,
    AgentRunResult,
)
from pydantic_ai.capabilities import Instrumentation, ProcessEventStream
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.instrumented import InstrumentationSettings

from .helpers import log_events
from .models import SQLAgentConfig, SQLResult
from .tools import SQLTools


def init_agent(
    config: SQLAgentConfig,
    sql_tools: SQLTools,
):
    tools = [sql_tools.get_schema, sql_tools.run_sql]
    model_settings = AnthropicModelSettings(
        anthropic_cache=True,
        anthropic_cache_tool_definitions="1h",
    )
    model_capabilities = [
        Instrumentation(InstrumentationSettings(version=5)),
        ProcessEventStream(log_events),
    ]
    sql_agent = Agent(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools,
        output_type=SQLResult,
        model_settings=model_settings,
        capabilities=model_capabilities,
    )
    return sql_agent


async def run_agent(
    agent: Agent[object, SQLResult],
    user_prompt: str,
    message_history: list | None = None,
) -> AgentRunResult[SQLResult]:
    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        message_history=message_history,
    )
    return result
