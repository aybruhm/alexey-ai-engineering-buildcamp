import os
from typing import Any, TypedDict

from elasticsearch import Elasticsearch, helpers

import models
import utils


INDEX_NAME = "pdf-extraction-rag-documents"


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


def _create_es_client() -> Elasticsearch:
    client = Elasticsearch(
        os.getenv("ELASTICSEARCH_HOST"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        request_timeout=30,
    )
    version = client.info()["version"]["number"]
    print(f"Connected to Elasticsearch v{version}")
    return client


def _create_indices(es: Elasticsearch):
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "filename": {"type": "keyword"},
                    "content": {"type": "text"},
                }
            },
        )


def build_index(documents: list[dict[str, Any]]) -> Elasticsearch:
    es = _create_es_client()
    _create_indices(es=es)
    helpers.bulk(es, documents, index=INDEX_NAME, refresh="wait_for")
    return es


def search(query: str, es: Elasticsearch, num_results: int):
    response = es.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query": query,
                "type": "best_fields",
                "fields": ["content"],
            }
        },
        size=num_results,
    )
    for hit in response["hits"]["hits"]:
        print(f"Score: {hit['_score']}")
        print(f"Title: {hit['_source']['content']}")
