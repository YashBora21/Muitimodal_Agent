from typing import Any

import fitz

from ..extraction.image import vision
from ..extraction.pdf import pdf_page, relevant_pdf_pages
from ..extraction.youtube import youtube_text
from ..models import Asset


def run_tool(
    name: str,
    args: dict[str, Any],
    asset_by_id: dict[str, Asset],
) -> str:
    if name != "read_asset":
        return f"Unknown tool: {name}"

    asset_id = str(args.get("asset_id") or "")
    asset = asset_by_id.get(asset_id)
    if asset is None:
        valid_ids = ", ".join(asset_by_id) or "(none)"
        return f"No such asset: {asset_id!r}. Valid asset ids: {valid_ids}."

    if asset["kind"] == "youtube" and not asset.get("content"):
        asset["content"] = youtube_text(asset["name"])

    content = asset.get("content") or "(empty)"
    query = str(args.get("query") or "").strip()
    page_number = args.get("page_number")
    visual = args.get("visual") is True

    if asset["kind"] == "pdf" and page_number is not None:
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            return "page_number must be a positive integer."

        page_content = pdf_page(content, page_number)
        if page_content is None:
            return f"Page {page_number} is not available in {asset_id}."
        if not visual:
            return page_content

        pdf_data = asset.get("data")
        if not pdf_data:
            return f"No visual PDF data is available for {asset_id!r}."

        with fitz.open(stream=pdf_data, filetype="pdf") as document:
            if page_number > len(document):
                return f"Page must be between 1 and {len(document)}."
            pixmap = document[page_number - 1].get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False,
            )
        return vision(pixmap.tobytes("jpeg"), "image/jpeg")

    if visual:
        return "visual=true requires a PDF page_number."
    if asset["kind"] == "pdf" and query:
        return relevant_pdf_pages(content, query)
    return content
