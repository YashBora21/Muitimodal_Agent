import base64
import logging

from ..config import VISION_MODEL, client


logger = logging.getLogger(__name__)


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def vision(data: bytes, mime: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe all visible text, then briefly describe "
                        "the image. Use plain text, keep the wording natural, "
                        "and do not return JSON."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(data, mime)},
                },
            ],
        }
    ]
    try:
        response = client().chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=900,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as error:
        logger.warning("Image processing failed: %s", error)
        raise RuntimeError(
            "Image analysis is temporarily unavailable. Please retry shortly."
        ) from error
