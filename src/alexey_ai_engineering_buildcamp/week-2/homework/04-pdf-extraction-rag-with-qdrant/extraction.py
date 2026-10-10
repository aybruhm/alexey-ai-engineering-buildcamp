import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from anthropic import Anthropic
from pymupdf import Page as PDFPage
from tqdm.auto import tqdm

import utils
from models import Page, PageResponse
from pricing import calculate_cost_response

SYSTEM_PROMPT = """
You are extracting a textbook page into structured page blocks.

Extract text and code expressions verbatim.

The `type` field is a discriminator and MUST be exactly one of these values:
- "section_heading" for section/chapter/subsection titles.
- "text" for explanatory prose.
- "inline" for a short inline code expression.
- "block" for a full, fenced code block.
- "figure" for figures, diagrams, plots, and images.
- "table" for tables.
Never emit any other value for `type` (for example, never use "code").

Extraction rules:
1) Preserve reading order. `Page.blocks` must match the order a human reads the page.
2) Do NOT include OCR or layout details (no coordinates, fonts, line breaks, or scan artifacts).
3) Prefer fewer, larger TextBlocks over many tiny ones. Group adjacent paragraphs when they belong together.
4) Choose the correct block type for every element:
     - SectionHeadingBlock (type="section_heading"): section/chapter/subsection titles only.
         Never include body text.
     - TextBlock (type="text"): explanatory prose from the textbook.
     - CodeBlock (type="inline" or type="block"): code expressions. Use type="inline" for
         short inline expressions, type="block" for multi-line code. Put the code text in
         the `code` field and the language name in `language`. Use `description` to explain
         what the expression means when it helps the reader.
     - FigureBlock (type="figure"): figures, diagrams, plots, and images. Set `figure_number`
         to the number shown in the book and copy the caption if present.
     - TableBlock (type="table"): tables with `columns` in reading order and `rows` aligned
         to columns. Include units in column names when the source shows them.
5) Some inline equations should be treated as block-level expressions when there is
     little surrounding text.
6) FigureBlock.description should explain what the figure conveys conceptually
     (graphs, curves, relationships), not how it looks on the page.
7) Store the running page header (if any) in `Page.header`, and the printed page
     number in `Page.page_number`.
8) Important: don't skip any text. If something is not possible to recognize,
     include a placeholder.
9) If uncertain, make a best-faith concise extraction; do not invent content.
""".strip()


def convert_page_to_image_b64(page: PDFPage, image_format: str = "png") -> str:
    matrix = pymupdf.Matrix(1.5, 1.5)
    pixmap = page.get_pixmap(matrix=matrix)
    png_bytes = pixmap.tobytes(output=image_format)
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return image_b64


def _page_file(filename: str, path: str = "output/") -> Path:
    if not filename.endswith(".json"):
        filename += ".json"

    output_dir = Path(utils.get_working_directory(path=path))
    return output_dir / filename


def page_exists(filename: str, path: str = "output/") -> bool:
    return _page_file(filename, path).exists()


def save_page_as_json(
    result: str,
    filename: str,
    path: str = "output/",
) -> None:
    page_file = _page_file(filename, path)
    page_file.parent.mkdir(parents=True, exist_ok=True)
    page_file.write_text(result, encoding="utf-8")


def extract_page_information(
    client: Anthropic,
    image_b64: str,
    max_tokens: int = 1024,
) -> PageResponse:
    response = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=max_tokens,
        messages=[
            {"role": "assistant", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
        output_format=Page,
    )
    cost = calculate_cost_response(response=response)
    return PageResponse(
        page=response.parsed_output,
        cost=cost,
    )


def _map_progress(pool: ThreadPoolExecutor, seq: list[int], func) -> list:
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(func, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results


def extract_pdf(
    client: Anthropic,
    pdf_document: pymupdf.Document,
    max_tokens: int = 20789,
    max_workers: int = 4,
) -> None:
    """Extract every page of the PDF into the knowledge base (output/)."""

    def process_document(page_number: int) -> bool:
        try:
            filename = f"page_{page_number + 1}.json"
            if page_exists(filename):
                print(f"{filename} already processed")
                return True

            image_b64 = convert_page_to_image_b64(page=pdf_document[page_number])
            page_response = extract_page_information(
                client=client,
                image_b64=image_b64,
                max_tokens=max_tokens,
            )
            save_page_as_json(
                result=page_response.model_dump_json(indent=4),
                filename=filename,
            )
            return True
        except Exception as e:  # noqa: BLE001
            print(f"Error occurred for page #{page_number}: {e}")
            return False

    pages = list(range(len(pdf_document)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        _map_progress(pool, pages, process_document)
