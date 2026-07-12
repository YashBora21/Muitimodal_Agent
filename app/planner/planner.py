import json

from app.llm.factory import get_llm_client
from app.planner.schemas import Plan, PlanStep
from app.state import AgentState

AVAILABLE_TOOLS = """
- pdf_extract: extract text from an uploaded PDF, also detects any YouTube URLs inside it
- image_ocr: extract text from an uploaded image
- audio_transcribe: transcribe an uploaded audio file to text
- youtube_transcript: fetch the transcript of a YouTube video, given a URL already found in state
- summarize: produce a 1-line summary + 3 bullets + 5-sentence summary of the combined extracted text
- sentiment: label + confidence + one-line justification for the combined extracted text
- code_explain: explain what a code snippet does, flag bugs, note time complexity (input is combined text)
- compare: compare two or more extracted sources and answer a comparison question
- conversational: answer a general question directly, no extraction needed
"""

SYSTEM_PROMPT = f"""You are the planning module of an agentic assistant.
Given the user's query and which file types they attached, decide:
1. whether you have enough information to act, or need to ask a clarifying question
2. the user's intent
3. the minimal ordered sequence of tools needed

Available tools:
{AVAILABLE_TOOLS}

Rules:
- If the query is empty OR too vague to determine the task
  (e.g. a file with no instruction), set needs_clarification=true
  and write a short specific question. This takes priority over all other rules.
- Only include extraction tools (pdf_extract, image_ocr, audio_transcribe) for file types that were actually attached.
- If the query references a URL inside an attached file (e.g. "summarize the video in this PDF"), plan pdf_extract
  BEFORE youtube_transcript - the URL is only known after extraction.
- If multiple file types are attached, you MUST include an extraction tool for EVERY
  attached file type. For example, if both "pdf" and "audio" are attached, include
  BOTH pdf_extract AND audio_transcribe before any answer tool.
- If the query asks to compare, relate, find similarities, or asks whether files discuss the same topic,
  ALWAYS use "compare" as the final step after all extractions.
- If there are NO attached files and the query is a direct question or statement,
  ALWAYS use only the "conversational" tool. Never ask for clarification in this case.
- After extraction steps, always include at least one answer step
  (summarize, code_explain, compare, sentiment, or conversational)
  that uses the extracted content to answer the user's query.
- If the query says "explain" or "what does this do" and an image is attached,
  plan image_ocr followed by code_explain.

Respond ONLY with JSON matching this shape, no other text:
{{"needs_clarification": bool, "question": string|null, "intent": string, "steps": [{{"tool": string}}, ...]}}
"""


def _llm_plan_with_forced_extraction(
    extraction_steps: list[PlanStep],
    system_prompt: str,
    user_prompt: str,
    fallback_answer: str = "conversational",
) -> Plan:
    """Call LLM for answer steps, prepend forced extraction steps."""
    client = get_llm_client()
    raw = client.complete(system=system_prompt, user=user_prompt, json_mode=True)
    try:
        llm_plan = Plan(**json.loads(raw))
        extraction_tool_names = {s.tool for s in extraction_steps}
        answer_steps = [
            s for s in llm_plan.steps
            if s.tool not in {"pdf_extract", "image_ocr", "audio_transcribe"}
        ]
        if not answer_steps:
            answer_steps = [PlanStep(tool=fallback_answer)]
        return Plan(
            needs_clarification=False,
            question=None,
            intent=llm_plan.intent,
            steps=extraction_steps + answer_steps,
        )
    except Exception:
        return Plan(
            needs_clarification=False,
            question=None,
            intent="extract_and_answer",
            steps=extraction_steps + [PlanStep(tool=fallback_answer)],
        )


def build_plan(state: AgentState) -> Plan:
    q = (state.query or "").strip().lower()

    has_pdf   = "pdf"   in state.file_paths
    has_image = "image" in state.file_paths
    has_audio = "audio" in state.file_paths

    attached = ", ".join(state.file_paths.keys()) or "none"
    user_prompt = (
        f"User query: {state.query or '(empty)'}\n"
        f"Attached file types: {attached}"
    )

    # ------------------------------------------
    # 1. Files uploaded but no instruction
    # ------------------------------------------
    if state.file_paths and not q:
        return Plan(
            needs_clarification=True,
            question="What would you like me to do with the uploaded file(s)?",
            intent="clarification",
            steps=[],
        )

    # ------------------------------------------
    # 2. Pure conversation — no files
    # ------------------------------------------
    if not state.file_paths:
        return Plan(
            needs_clarification=False,
            question=None,
            intent="conversation",
            steps=[PlanStep(tool="conversational")],
        )

    # ------------------------------------------
    # 3. Image with code-related query
    # ------------------------------------------
    code_keywords = [
        "explain", "code", "bug", "complexity", "what does",
        "function", "error", "syntax", "debug",
    ]
    if has_image and any(x in q for x in code_keywords):
        return Plan(
            needs_clarification=False,
            question=None,
            intent="code_explain",
            steps=[
                PlanStep(tool="image_ocr"),
                PlanStep(tool="code_explain"),
            ],
        )

    # ------------------------------------------
    # 4. PDF contains YouTube URL
    # ------------------------------------------
    if has_pdf and any(w in q for w in ["youtube", "video", "transcript", "yt"]):
        return Plan(
            needs_clarification=False,
            question=None,
            intent="youtube_summary",
            steps=[
                PlanStep(tool="pdf_extract"),
                PlanStep(tool="youtube_transcript"),
                PlanStep(tool="summarize"),
            ],
        )

    # ------------------------------------------
    # 5. Compare intent — multiple files
    # ------------------------------------------
    compare_words = [
        "compare", "difference", "similar", "same", "relation",
        "related", "match", "topic", "discuss", "both", "agree", "overlap",
    ]
    if sum([has_pdf, has_audio, has_image]) >= 2 and any(w in q for w in compare_words):
        steps = []
        if has_pdf:   steps.append(PlanStep(tool="pdf_extract"))
        if has_audio: steps.append(PlanStep(tool="audio_transcribe"))
        if has_image: steps.append(PlanStep(tool="image_ocr"))
        steps.append(PlanStep(tool="compare"))
        return Plan(
            needs_clarification=False,
            question=None,
            intent="compare",
            steps=steps,
        )

    # ------------------------------------------
    # 6. Image attached — always OCR first
    # ------------------------------------------
    if has_image and not any(w in q for w in code_keywords):
        return _llm_plan_with_forced_extraction(
            extraction_steps=[PlanStep(tool="image_ocr")],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_answer="conversational",
        )

    # ------------------------------------------
    # 7. PDF attached — always extract first
    # ------------------------------------------
    if has_pdf and not any(w in q for w in ["youtube", "video", "transcript", "yt"]):
        return _llm_plan_with_forced_extraction(
            extraction_steps=[PlanStep(tool="pdf_extract")],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_answer="conversational",
        )

    # ------------------------------------------
    # 8. Audio attached — always transcribe first
    # ------------------------------------------
    if has_audio:
        return _llm_plan_with_forced_extraction(
            extraction_steps=[PlanStep(tool="audio_transcribe")],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_answer="summarize",
        )

    # ------------------------------------------
    # 9. LLM fallback
    # ------------------------------------------
    client = get_llm_client()
    raw = client.complete(system=SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    try:
        return Plan(**json.loads(raw))
    except Exception:
        return Plan(
            needs_clarification=True,
            question="I couldn't understand the request. Could you clarify?",
            intent="unknown",
            steps=[],
        )