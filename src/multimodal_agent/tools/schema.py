from typing import Any


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_asset",
            "description": (
                "Read an uploaded PDF, image, audio transcript, or YouTube "
                "transcript. PDFs support query search, exact page retrieval, "
                "and optional visual page inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": (
                            "The asset identifier, such as pdf-1, image-2, "
                            "audio-1, or youtube-1."
                        ),
                    },
                    "query": {
                        "type": ["string", "null"],
                        "description": (
                            "Question or keywords used to retrieve relevant "
                            "content from PDFs and long transcripts."
                        ),
                    },
                    "page_number": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "For PDFs, return this exact page.",
                    },
                    "visual": {
                        "type": ["boolean", "null"],
                        "description": (
                            "Visually inspect page_number for charts, images, "
                            "diagrams, screenshots, or layout."
                        ),
                        "default": False,
                    },
                },
                "required": ["asset_id"],
            },
        },
    }
]
