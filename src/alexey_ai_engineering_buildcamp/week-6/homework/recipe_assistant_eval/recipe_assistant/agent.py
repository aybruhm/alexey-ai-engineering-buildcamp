from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from .prompts import INSTRUCTIONS
from .tools import get_recipe, search_recipes

agent = Agent(
    "anthropic:claude-haiku-5-5",
    tools=[search_recipes, get_recipe],
    instructions=INSTRUCTIONS,
    capabilities=[Instrumentation(InstrumentationSettings(version=5))],
)
