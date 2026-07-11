import os
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

# Use Windows path locally, Linux (Docker) uses system tesseract
if os.name == "nt":  # Windows only
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_image_text(path: str) -> tuple[str, float]:
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = ImageEnhance.Contrast(img).enhance(2.5)

    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT
    )

    confidences = [
        int(c)
        for c in data["conf"]
        if c not in ("-1", -1)
    ]

    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config="--oem 3 --psm 6"
    )

    return text.strip(), round(avg_conf, 1)