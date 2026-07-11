from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    tool: str
    status: str          # "ok" | "failed" | "skipped"
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentState:
    query: str
    file_paths: dict[str, str] = field(default_factory=dict)   # {"pdf": "/tmp/x.pdf", ...}

    pdf_text: str | None = None
    image_text: str | None = None
    audio_text: str | None = None
    youtube_urls: list[str] = field(default_factory=list)
    youtube_transcript: str | None = None

    trace: list[TraceStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    result: dict[str, Any] | None = None   # final tool output shown to user

    def combined_text(self) -> str:
        parts = [t for t in (self.pdf_text, self.image_text, self.audio_text,
                              self.youtube_transcript) if t]
        return "\n\n".join(parts)

    def log(self, tool: str, status: str, message: str = "", duration_ms: float = 0.0):
        self.trace.append(TraceStep(tool, status, message, duration_ms))
