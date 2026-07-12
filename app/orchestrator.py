from app.planner.planner import build_plan
from app.state import AgentState
from app.tools.registry import TOOLS


def run_agent(state: AgentState) -> AgentState:
    plan = build_plan(state)
    state.log("planner", "ok", f"intent={plan.intent}")

    if plan.needs_clarification:
        state.result = {"type": "clarification_needed", "content": plan.question}
        return state

    for step in plan.steps:
        tool = TOOLS.get(step.tool)

        if tool is None:
            state.warnings.append(f"planner requested unknown tool '{step.tool}'")
            state.log(step.tool, "skipped", "unknown tool")
            continue

        tool(state)

        last = state.trace[-1]

        if last.status == "failed" and step.tool in {
            "pdf_extract",
            "image_ocr",
            "audio_transcribe",
            "youtube_transcript",
        }:
            state.result = {
                "type": "partial_failure",
                "content": f"Couldn't complete the task — {step.tool} failed: {last.message}",
            }
            return state

    return state