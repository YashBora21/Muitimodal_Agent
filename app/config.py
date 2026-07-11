import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # "groq" for deployment, "ollama" for local dev without burning API calls
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

    def validate(self):
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Set it in your .env file."
            )


settings = Settings()
