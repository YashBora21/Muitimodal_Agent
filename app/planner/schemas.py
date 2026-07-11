from pydantic import BaseModel


class PlanStep(BaseModel):
    tool: str


class Plan(BaseModel):
    needs_clarification: bool
    question: str | None = None
    intent: str
    steps: list[PlanStep] = []
