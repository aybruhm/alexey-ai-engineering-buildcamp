from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from ..pricing import calculate_cost


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def get_model_name(agent: Agent) -> str:
    model = agent.model
    if isinstance(model, str):
        return model
    if model is not None:
        return model.model_name
    return "unknown"


def collect_tools(messages: list):
    tool_calls = []

    for m in messages:
        for p in m.parts:
            part_kind = p.part_kind
            if part_kind != "tool-call":
                continue

            if p.tool_name == "final_result":
                continue

            tool_calls.append(ToolCall(p.tool_name, p.args))

    return tool_calls


run_usages: dict[str, list[RunUsage]] = defaultdict(list)


def capture_usage(model: str, usage: RunUsage) -> None:
    run_usages[model].append(usage)


def total_cost() -> float:
    return sum(
        calculate_cost(model, usage.input_tokens, usage.output_tokens)
        for model, usages in run_usages.items()
        for usage in usages
    )


def display_total_usage() -> None:
    print()
    print(f"Total cost: ${total_cost():.6f}")
