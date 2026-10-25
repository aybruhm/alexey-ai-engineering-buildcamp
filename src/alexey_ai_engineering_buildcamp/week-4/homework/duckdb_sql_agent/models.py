from dataclasses import dataclass

from pydantic import BaseModel

from .prompts import INSTRUCTIONS


class SQLResult(BaseModel):
    sql_query: str
    result_text: str
    row_count: int


@dataclass
class SQLAgentConfig:
    model = "anthropic:claude-haiku-5-5"
    name = "duckdb-sql"
    instructions = INSTRUCTIONS
