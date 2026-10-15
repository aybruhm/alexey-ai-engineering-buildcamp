from collections.abc import AsyncIterable

from pydantic_ai import (
    AgentRun,
    AgentStreamEvent,
    CallToolsNode,
    ModelRequestNode,
    RunContext,
)
from pydantic_ai.messages import FunctionToolCallEvent


async def log_events(
    ctx: RunContext,
    events: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            agent_name = (
                ctx.agent.name if ctx.agent and hasattr(ctx.agent, "name") else "search"
            )
            print(f"TOOL CALL ({agent_name}): {tool_name}({args})")


async def print_stream_node(
    node: ModelRequestNode[object, str],
    agent_run: AgentRun,
):
    async with node.stream(agent_run.ctx) as stream:
        async for response in stream.stream_output(debounce_by=0.01):
            print(response)
    print()


async def print_tool_calls(
    node: CallToolsNode[object, str],
    agent_run: AgentRun,
    agent_name: str,
):
    async with node.stream(agent_run.ctx) as events:
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = event.part.tool_name
                args = event.part.args
                print(f"TOOL CALL ({agent_name}): {tool_name}({args})")


def print_messages(messages: list):
    for m in messages:
        print(m.kind)
        for p in m.parts:
            part_kind = p.part_kind
            if part_kind == "user-prompt":
                print("  USER:", p.content)
            if part_kind == "tool-call":
                print("  TOOL CALL:", p.tool_name, p.args)
            if part_kind == "tool-return":
                print("  TOOL RETURN:", p.tool_name)
            if part_kind == "text":
                print("  ASSISTANT:", p.content)
        print()
