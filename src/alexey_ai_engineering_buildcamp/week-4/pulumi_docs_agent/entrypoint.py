import asyncio

from .agent import DocumentationAgentConfig, init_agent, run_agent_stream
from .retrieval import build_index, create_highlighter
from .tools import SearchTools
from .utils import retrieve_documentation

# from .helpers import print_messages


async def run_agent_question():
    user_prompt = input("You > ")
    # user_prompt = "How do I link a source SQS queue to a dead-letter queue in Pulumi?"
    print(f"Running agent with question: {user_prompt}...\n")

    documents = retrieve_documentation(
        repo_owner="pulumi",
        repo_name="docs",
        branch_name="master",
        docs_path="content",
        save_dir="docs",
    )
    store_index = build_index(documents=documents)
    store_highlighter = create_highlighter(
        max_matches=3,
        snippet_size=50,
    )
    file_index = {doc["filename"]: doc["content"] for doc in documents}

    tools = SearchTools(store_index, store_highlighter, file_index)
    agent_config = DocumentationAgentConfig()

    agent = init_agent(agent_config, tools)
    result = await run_agent_stream(agent, user_prompt)
    # result = await run_agent(agent, user_prompt)
    # print_messages(result.new_messages())
    output = result.output  # type: ignore
    print("AI > ", output)


def main() -> None:
    asyncio.run(run_agent_question())


if __name__ == "__main__":
    main()
