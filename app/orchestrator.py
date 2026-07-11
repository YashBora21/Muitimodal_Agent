from app.planner.planner import build_plan
from app.state import AgentState
from app.tools.registry import TOOLS


def run_agent(state: AgentState) -> AgentState:
    plan = build_plan(state)
    print(plan)
    state.log("planner", "ok", f"intent={plan.intent}")

    if plan.needs_clarification:
        state.result = {"type": "clarification_needed", "content": plan.question}
        return state

    for step in plan.steps:
        print("=" * 50)
        print("EXECUTING:", step.tool)
        print("=" * 50)

        tool = TOOLS.get(step.tool)

        if tool is None:
            state.warnings.append(f"planner requested unknown tool '{step.tool}'")
            state.log(step.tool, "skipped", "unknown tool")
            continue

        tool(state)

        last = state.trace[-1]
        print(f"DEBUG ORCH: tool={last.tool} status={last.status} msg={last.message}")

        if last.status == "failed" and step.tool in {
            "pdf_extract",
            "image_ocr",
            "audio_transcribe",
            "youtube_transcript",
        }:
            state.result = {
                "type": "partial_failure",
                "content": f"{step.tool} failed",
            }
            return state

    return state  # ← THIS WAS MISSING