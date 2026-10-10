from anthropic.types import ParsedMessage
from pydantic import BaseModel

MODEL_PRICING = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-5-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Model {model_name} not found in pricing.")

    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def calculate_cost_response[T: BaseModel](response: ParsedMessage[T]) -> float:
    usage = response.usage
    return calculate_cost(
        model_name=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
