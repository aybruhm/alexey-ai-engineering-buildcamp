from typing import Any, TypedDict

from minsearch import AppendableIndex, Highlighter, Tokenizer
from minsearch.tokenizer import DEFAULT_ENGLISH_STOP_WORDS


class Doc(TypedDict):
    filename: str
    content: str


def build_index(
    documents: list[dict[str, Any]],
    text_fields: list[str] | None = None,
) -> AppendableIndex:
    index = AppendableIndex(
        text_fields=text_fields or ["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index


def create_highlighter(max_matches: int = 3, snippet_size: int = 50):
    stopwords = DEFAULT_ENGLISH_STOP_WORDS | {"pulumi"}
    highlighter = Highlighter(
        highlight_fields=["content"],
        max_matches=max_matches,
        snippet_size=snippet_size,
        tokenizer=Tokenizer(stemmer="snowball", stop_words=stopwords),
    )
    return highlighter
