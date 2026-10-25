import asyncio
import json

from .agent import init_agent, run_agent
from .models import SQLAgentConfig
from .tools import SQLTools
from .utils import init_database


async def run_user_query() -> None:
    user_prompt = input("You > ")
    # user_prompt = "What's the average trip distance for rides with 2 passengers?"
    print(f"Running agent with question: {user_prompt}...\n")

    db_path = init_database()
    tools = SQLTools(db_path=db_path)
    agent_config = SQLAgentConfig()

    agent = init_agent(agent_config, tools)
    result = await run_agent(agent, user_prompt)
    output = result.output
    print("AI > ", json.dumps(output.model_dump(), indent=4))


def main() -> None:
    asyncio.run(run_user_query())


if __name__ == "__main__":
    main()
