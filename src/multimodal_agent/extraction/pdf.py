import re

import fitz

from ..config import TESSDATA
from ..models import Asset
from ..registry import new_asset
from ..retrieval import hybrid_search
from .youtube import YOUTUBE_RE, youtube_id


def ocr_page(page: fitz.Page) -> str:
    if not TESSDATA.is_dir():
        raise RuntimeError(
            f"Tesseract language data was not found at {TESSDATA}. "
            "Set TESSDATA_PREFIX to its tessdata folder."
        )
    textpage = page.get_textpage_ocr(
        dpi=300,
        full=True,
        language="eng",
        tessdata=str(TESSDATA),
    )
    return page.get_text(textpage=textpage).strip()


def read_pdf(data: bytes, name: str, existing: list[Asset]) -> list[Asset]:
    pages: list[str] = []
    urls: list[str] = []

    with fitz.open(stream=data, filetype="pdf") as document:
        for page_number, page in enumerate(document, 1):
            text = page.get_text().strip() or ocr_page(page)
            pages.append(f"--- Page {page_number} ---\n{text}")

            visible_urls = [
                url.rstrip(".,;!?")
                for url in YOUTUBE_RE.findall(text)
                if youtube_id(url)
            ]
            urls.extend(visible_urls)

            for link in page.get_links():
                url = link.get("uri", "").rstrip(".,;!?")
                if youtube_id(url) and url not in visible_urls:
                    urls.append(url)

    unique_urls = list(dict.fromkeys(urls))
    content = "\n\n".join(pages)

    if urls:
        content += (
            f"\n\nTotal YouTube links: {len(urls)}\n"
            f"Unique YouTube URLs: {len(unique_urls)}\n"
            + "\n".join(
                f"{index}. {url}"
                for index, url in enumerate(unique_urls, 1)
            )
        )

    assets = [new_asset(existing, "pdf", name, content, data=data)]
    for url in unique_urls:
        assets.append(new_asset(existing + assets, "youtube", url, ""))
    return assets


def relevant_pdf_pages(content: str, query: str) -> str:
    return hybrid_search(content, query)


def pdf_page(content: str, page_number: int) -> str | None:
    marker = re.compile(r"(?m)^--- Page (\d+) ---$")
    matches = list(marker.finditer(content))
    for index, match in enumerate(matches):
        if int(match.group(1)) != page_number:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        return content[match.start():end].strip()
    return None
