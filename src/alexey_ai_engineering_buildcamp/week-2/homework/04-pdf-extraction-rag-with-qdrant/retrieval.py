import os
from typing import Any, TypedDict

from qdrant_client import QdrantClient, models as qd_models

import models
import utils


DIMENSIONALITY = 512
COLLECTION_NAME = "pdf-extraction-rag-documents"
MODEL_HANDLE = "jinaai/jina-embeddings-v2-small-en"


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


def _create_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=443,
        https=True,
        timeout=30,
    )
    print(client.get_collections())
    return client


def _create_collection(client: QdrantClient):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qd_models.VectorParams(
            size=DIMENSIONALITY,
            distance=qd_models.Distance.COSINE,
        ),
    )


def build_index(documents: list[dict[str, Any]]) -> QdrantClient:
    points = []

    for i, doc in enumerate(documents):
        vector = qd_models.Document(
            text=doc["content"],
            model=MODEL_HANDLE,
        )
        point = qd_models.PointStruct(
            id=i,
            vector=vector,
            payload=doc,
        )
        points.append(point)

    client = _create_qdrant_client()
    _create_collection(client=client)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def search(query: str, client: QdrantClient, num_results: int):
    query_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qd_models.Document(text=query, model=MODEL_HANDLE),
        with_payload=True,
        limit=num_results,
    ).points

    results = []
    for point in query_points:
        results.append(point.payload)
    return results
