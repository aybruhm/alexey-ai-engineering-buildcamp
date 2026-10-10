from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer: str
    found_answer: bool
    confidence: float
    confidence_explanation: str
    answer_type: Literal[
        "how-to", "explanation", "troubleshooting", "comparison", "reference"
    ]
    followup_questions: list[str]


class PageBlock(BaseModel):
    type: str = Field(
        ...,
        description="Discriminator that identifies which kind of page block this is.",
    )


class SectionHeadingBlock(PageBlock):
    type: Literal["section_heading"] = "section_heading"
    title: str = Field(..., description="The section heading text.")


class TextBlock(PageBlock):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Explanatory prose from the textbook.")


class CodeBlock(PageBlock):
    """
    A code expression written in whatever's programming language.
    """

    type: Literal["inline", "block"] = "inline"
    code: str = Field(..., description="The code expression.")
    language: str = Field(..., description="Name of code language.")
    description: str | None = Field(
        None,
        description="Optional plain-language meaning or interpretation of the code expression.",
    )


class FigureBlock(PageBlock):
    type: Literal["figure"] = "figure"
    caption: str | None = Field(
        None, description="Figure caption or label, if present."
    )
    description: str = Field(
        ...,
        description="Conceptual description of what the figure shows and why it matters.",
    )
    figure_number: int = Field(
        ..., description="Figure number as mentioned in the book."
    )


class TableBlock(PageBlock):
    type: Literal["table"] = "table"
    caption: str | None = Field(None, description="Table caption or label, if present.")
    columns: list[str] = Field(..., description="Column headers in reading order.")
    rows: list[list[str]] = Field(..., description="Table rows aligned with columns.")


PageBlockUnion = Annotated[
    SectionHeadingBlock | TextBlock | CodeBlock | FigureBlock | TableBlock,
    Field(discriminator="type"),
]


class Page(BaseModel):
    page_number: int = Field(..., description="Printed page number in the textbook.")
    header: str | None = Field(None, description="Running page header text, if any.")
    blocks: list[PageBlockUnion] | None = Field(
        ..., description="Ordered list of extracted page blocks."
    )

    def display(self):
        print(self.page_number)
        print(self.header)

        for block in self.blocks or []:
            if block.type == "text":
                print(block.text)

            elif block.type == "inline":
                print(f">>>{block.code}")

            elif block.type == "block":
                print(f"```{block.language}\n{block.code}\n```")

            elif block.type == "figure":
                print(block.caption)
                print(block.description)
                print("Fig.", block.figure_number)

            else:
                print(block)

            print()


class PageResponse(BaseModel):
    page: Page
    cost: float


def block_to_string(block: PageBlockUnion) -> str:
    if block.type == "section_heading":
        return block.title

    if block.type == "text":
        return block.text

    if block.type == "inline":
        return f">>>{block.code}"

    if block.type == "block":
        return f"```{block.language}\n{block.code}\n```"

    if block.type == "figure":
        lines = [
            block.caption or "",
            block.description or "",
            f"Fig. {block.figure_number}",
        ]
        return "\n".join(lines)

    if block.type == "table":
        lines = []
        if block.caption:
            lines.append(block.caption)
        lines.append(" | ".join(block.columns))
        lines.extend(" | ".join(row) for row in block.rows)
        return "\n".join(lines)

    return str(block)


def blocks_to_string(blocks: list[PageBlockUnion]) -> str:
    return "\n".join(block_to_string(block) for block in blocks)
