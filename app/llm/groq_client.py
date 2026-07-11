from groq import Groq

from app.config import settings
from app.llm.base import LLMClient

# Groq pricing per million tokens (as of 2025)
GROQ_PRICING = {
    "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.08}
}


class GroqClient(LLMClient):
    def __init__(self):
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def name(self) -> str:
        return f"groq:{self._model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            **kwargs,
        )

        # Track usage
        usage = response.usage
        if usage:
            self.total_input_tokens  += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens

            pricing = GROQ_PRICING.get(self._model, {"input": 0.05, "output": 0.08})
            cost = (
                usage.prompt_tokens     / 1_000_000 * pricing["input"] +
                usage.completion_tokens / 1_000_000 * pricing["output"]
            )
            self.total_cost_usd += cost

        return response.choices[0].message.content

    def get_usage_summary(self) -> dict:
        return {
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens":  self.total_input_tokens + self.total_output_tokens,
            "cost_usd":      round(self.total_cost_usd, 6),
        }

    def reset_usage(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost_usd      = 0.0