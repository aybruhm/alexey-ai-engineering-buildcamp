from anthropic import Anthropic
from qdrant_client import QdrantClient, models as qd_models
from pydantic import BaseModel

import prompt

INDEX_NAME = "pdf-extraction-rag-documents"
COLLECTION_NAME = "pdf-extraction-rag-documents"
MODEL_HANDLE = "jinaai/jina-embeddings-v2-small-en"


class RAG:
    def __init__(
        self,
        index: QdrantClient,
        llm_client: Anthropic,
        output_type: type[BaseModel],
        instructions: str = prompt.INSTRUCTIONS,
    ):
        self.index = index
        self.llm_client = llm_client
        self.output_type = output_type
        self.instructions = instructions

    def search(self, query: str, top_k: int = 5):
        query_points = self.index.query_points(
            collection_name=COLLECTION_NAME,
            query=qd_models.Document(text=query, model=MODEL_HANDLE),
            with_payload=True,
            limit=top_k,
        ).points

        results = []
        for point in query_points:
            results.append(point.payload)
        return results

    def build_prompt(self, question: str, search_results: list) -> str:
        return prompt.build_prompt(
            question=question,
            search_results=search_results,
        )

    def llm(
        self,
        user_prompt: str,
        max_tokens: int,
        model: str = "claude-opus-5",
    ):
        with self.llm_client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "assistant",
                    "content": self.instructions
                    or "You are a helpful course assistant.",
                },
                {"role": "user", "content": user_prompt},
            ],
            thinking={"type": "adaptive", "display": "summarized"},
            output_format=self.output_type,
        ) as stream:
            return stream.get_final_message()

    def invoke(self, query: str, max_tokens: int = 2056):
        search_results = self.search(query=query)
        user_prompt = self.build_prompt(
            question=query,
            search_results=search_results,
        )
        response = self.llm(
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        for block in response.content:
            if block.type == "text":
                return block.parsed_output

        return None
