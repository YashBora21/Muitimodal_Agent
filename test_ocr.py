from app.extraction.image_ocr import extract_image_text

text, conf = extract_image_text("tests/fixtures/code.jpg")

print("=" * 50)
print("OCR TEXT")
print("=" * 50)
print(text)
print("=" * 50)
print("Confidence:", conf)