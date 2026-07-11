from functools import lru_cache

from app.config import settings
from app.llm.base import LLMClient


@lru_cache
def get_llm_client() -> LLMClient:
    """
    Cached so we build one client per process, not one per request.
    Add a new provider by writing a class + one elif branch here.
    """
    if settings.llm_provider == "groq":
        from app.llm.groq_client import GroqClient
        return GroqClient()

    if settings.llm_provider == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient()

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
