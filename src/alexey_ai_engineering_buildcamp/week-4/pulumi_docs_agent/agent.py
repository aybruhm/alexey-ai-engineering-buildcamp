from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    AgentRunResult,
)
from pydantic_ai.capabilities import Instrumentation, ProcessEventStream
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.instrumented import InstrumentationSettings

from .helpers import log_events, print_stream_node, print_tool_calls
from .models import RAGResponse
from .prompts import INSTRUCTIONS
from .tools import SearchTools


@dataclass
class DocumentationAgentConfig:
    model = "anthropic:claude-haiku-5-5"
    name = "search"
    instructions = INSTRUCTIONS


def init_agent(
    config: DocumentationAgentConfig,
    search_tools: SearchTools,
) -> Agent:
    tools = [search_tools.search, search_tools.get_file]
    model_settings = AnthropicModelSettings(
        anthropic_cache=True,
        anthropic_cache_tool_definitions="1h",
    )
    model_capabilities = [
        Instrumentation(InstrumentationSettings(version=5)),
        ProcessEventStream(log_events),
    ]
    search_agent = Agent(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools,
        model_settings=model_settings,
        output_type=RAGResponse,
        capabilities=model_capabilities,
    )
    return search_agent  # type: ignore


async def run_agent(
    agent: Agent,
    user_prompt: str,
    message_history: list | None = None,
) -> AgentRunResult:
    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        message_history=message_history,
    )
    return result


async def run_agent_stream(
    agent: Agent,
    user_prompt: str,
    message_history: list | None = None,
) -> AgentRunResult | None:
    if message_history is None:
        message_history = []

    async with agent.iter(
        user_prompt,
        message_history=message_history,
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_user_prompt_node(node):
                print(f"User Prompt ({agent.name}) > ", node.user_prompt)
            elif Agent.is_model_request_node(node):
                await print_stream_node(node, agent_run)
            elif Agent.is_call_tools_node(node):
                await print_tool_calls(
                    node,
                    agent_run,
                    agent.name or "agent",
                )
        return agent_run.result
