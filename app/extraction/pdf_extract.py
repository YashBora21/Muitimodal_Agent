import re

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+)[^\s)>\]\"']*"
)

# a page with fewer than this many extractable chars is treated as scanned/image-only
MIN_CHARS_PER_PAGE = 20


def extract_pdf_text(path: str) -> tuple[str, list[str]]:
    """Returns (full_text, youtube_urls_found). Falls back to OCR per-page when
    a page has no text layer (i.e. it's a scanned image)."""
    doc = fitz.open(path)
    pages: list[str] = []

    for page in doc:
        text = page.get_text().strip()
        if len(text) < MIN_CHARS_PER_PAGE:
            text = _ocr_page(page)
        pages.append(text)

    full_text = "\n\n".join(p for p in pages if p).strip()
    urls = list(dict.fromkeys(YOUTUBE_URL_RE.findall(full_text)))  # dedupe, keep order
    return full_text, urls


def _ocr_page(page) -> str:
    # render at 2x for better OCR accuracy on scanned text
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img).strip()
