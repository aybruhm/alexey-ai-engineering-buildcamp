import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def collect_tools(messages: list) -> list[ToolCall]:
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


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text, stripping thousands separators."""
    return [
        float(match.replace(",", ""))
        for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    ]
