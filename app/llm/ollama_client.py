import requests

from app.config import settings
from app.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self):
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
