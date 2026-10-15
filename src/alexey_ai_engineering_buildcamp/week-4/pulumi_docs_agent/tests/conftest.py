import time

import pytest

from ..agent import DocumentationAgentConfig, init_agent
from ..retrieval import build_index, create_highlighter
from ..tools import SearchTools
from ..utils import retrieve_documentation
from .utils import display_total_usage


@pytest.fixture(scope="module")
def create_agent():
    t0 = time.perf_counter()
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

    t1 = time.perf_counter()
    print(f"loading agent took {t1 - t0}")
    return agent


def pytest_unconfigure(config):
    print("\n[HOOK] pytest is unconfiguring (end of run)")


def pytest_sessionfinish(session, exitstatus):
    display_total_usage()
