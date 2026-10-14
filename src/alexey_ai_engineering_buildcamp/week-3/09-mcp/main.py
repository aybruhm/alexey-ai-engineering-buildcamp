from pathlib import Path

from fastmcp import FastMCP
from gitsource import chunk_documents
from minsearch import AppendableIndex
from toyaikit.tools import wrap_instance_methods

from search_tools import SearchTools
from utils import retrieve_documentation

# Cache the downloaded docs next to this script so the lookup does not depend
# on the current working directory.
DOCS_DIR = Path(__file__).resolve().parent / "docs"


def init():
    # Retrieve documentation
    documents = retrieve_documentation(
        repo_owner="pulumi",
        repo_name="docs",
        branch_name="master",
        docs_path="content",
        save_dir=str(DOCS_DIR),
    )

    # Chunk documents
    document_chunks = chunk_documents(documents, size=3000, step=1500)

    # Create index store
    index = AppendableIndex(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(document_chunks)

    # Create file index
    file_index = {doc["filename"]: doc["content"] for doc in documents}
    return file_index, index


def init_tools():
    file_index, index = init()
    tools = SearchTools(index, file_index)
    return tools


def init_mcp():
    mcp = FastMCP("Demo")
    agent_tools = init_tools()
    wrap_instance_methods(mcp.tool, agent_tools)
    return mcp


if __name__ == "__main__":
    mcp = init_mcp()
    mcp.run()
