from typing import Any, TypedDict

from sqlitesearch import TextSearchIndex

import models
import utils


class Doc(TypedDict):
    filename: str
    content: str


def construct_documents() -> list[Doc]:
    documents: list[Doc] = []
    pages_dir = utils.get_extracted_pages_dir()

    for file_path in pages_dir.iterdir():
        if not file_path.is_file() or file_path.suffix != ".json":
            continue

        contents = utils.read_file_contents(file_path)
        page_response = models.PageResponse.model_validate_json(contents)

        blocks = page_response.page.blocks or []
        content = models.blocks_to_string(blocks)

        documents.append({
            "filename": file_path.name,
            "content": content,
        })

    return documents


def build_index(
    documents: list[dict[str, Any]],
    text_fields: list[str] | None = None,
) -> TextSearchIndex:
    index = TextSearchIndex(
        text_fields=text_fields or ["content"],
        db_path="db/local.sqlite",
    )
    for document in documents:
        index.add(document)
    return index
