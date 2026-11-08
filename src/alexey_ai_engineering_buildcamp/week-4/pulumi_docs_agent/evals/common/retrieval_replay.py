import json

from ...retrieval import build_index, create_highlighter
from ...tools import SearchTools
from ...utils import retrieve_documentation
from .paths import DOCS_DIR

MAX_DOC_CHARS = 15_000


def build_tools() -> SearchTools:
    documents = retrieve_documentation(
        repo_owner="pulumi", repo_name="docs", branch_name="master",
        docs_path="content", save_dir=str(DOCS_DIR),
    )
    return SearchTools(
        build_index(documents=documents),
        create_highlighter(max_matches=3, snippet_size=50),
        {d["filename"]: d["content"] for d in documents},
    )


def replay_retrieval(tools: SearchTools, row: dict) -> str:
    """Re-run the agent's tool calls. Search and get_file are deterministic,
    so this reproduces what the agent saw without storing it in the results."""
    parts = []
    for i, call in enumerate(row.get("tool_calls", []), 1):
        args = call["args"]
        args = json.loads(args) if isinstance(args, str) else args
        if call["name"] == "search":
            body = json.dumps(tools.search(args["query"]), indent=1)
        elif call["name"] == "get_file":
            body = tools.get_file(args["filename"])
            if len(body) > MAX_DOC_CHARS:
                body = body[:MAX_DOC_CHARS] + "\n[...truncated for the judge...]"
        else:
            continue
        parts.append(f"--- tool call {i}: {call['name']}({args}) ---\n{body}")
    return "\n\n".join(parts)
