from dataclasses import dataclass

from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelResponse


@dataclass(frozen=True)
class Price:
    """USD per million tokens, Anthropic first-party API rates (checked 2026-11-10)."""

    input: float
    output: float
    long_input: float | None = None
    long_output: float | None = None
    long_threshold: int | None = None


MODEL_PRICING = {
    "claude-fable-5-2": Price(8.00, 40.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(
        0.10, 0.50, long_input=0.50, long_output=2.50, long_threshold=100_000
    ),
    "claude-haiku-4-5": Price(1.00, 5.00),
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one request. The long-prompt tier is decided per request, not per run."""
    name = model_name.removeprefix("anthropic:")
    if name not in MODEL_PRICING:
        raise ValueError(f"Model {name} not found in pricing.")

    price = MODEL_PRICING[name]
    input_rate, output_rate = price.input, price.output
    if price.long_threshold is not None and input_tokens > price.long_threshold:
        input_rate = price.long_input or input_rate
        output_rate = price.long_output or output_rate

    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def calculate_run_cost(result: AgentRunResult) -> float:
    """Sum the cost of every model request in an agent run, each at its own model's rate."""
    return round(
        sum(
            calculate_cost(
                message.model_name or "",
                message.usage.input_tokens,
                message.usage.output_tokens,
            )
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
        ),
        6,
    )
