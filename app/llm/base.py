from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Every provider (Groq, Ollama, whatever comes next) implements this.
    The rest of the app never imports a provider directly - only this
    interface - so swapping providers is a one-line config change.
    """

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """
        Send a single system+user turn, return the raw text response.
        json_mode=True asks the provider to constrain output to valid JSON
        (used by the planner - see app/planner/planner.py).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in the plan trace shown to the user."""
        raise NotImplementedError
