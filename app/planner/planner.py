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
3. the minimal ordered sequence of tools neededs

Available tools:
{AVAILABLE_TOOLS}

Rules:
- If the query is empty OR too vague to determine the task 
  (e.g. a file with no instruction), set needs_clarification=true 
  and write a short specific question. This takes priority over all other rules
- Only include extraction tools (pdf_extract, image_ocr, audio_transcribe) for file types that were actually attached.
- If the query references a URL inside an attached file (e.g. "summarize the video in this PDF"), plan pdf_extract
  BEFORE youtube_transcript - the URL is only known after extraction.
- If multiple file types are attached, you MUST include an extraction tool for EVERY 
  attached file type. For example, if both "pdf" and "audio" are attached, include 
  BOTH pdf_extract AND audio_transcribe before any answer tool.
- If the query asks to compare, relate, or find similarities between files, always 
  use "compare" as the final step after all extractions.
- If the query asks whether two files are related, discuss the same topic, 
  are similar, or any cross-file question with 2+ files attached, 
  ALWAYS use "compare" as the final step.
  
- If there are NO attached files and the query is a direct question or statement, 
  ALWAYS use only the "conversational" tool. Never ask for clarification in this case
- If the query is a general conversational question with no files, use only the "conversational" tool.
- After extraction steps, always include at least one answer step 
  (summarize, code_explain, compare, sentiment, or conversational) 
  that uses the extracted content to answer the user's query.
- If the query says "explain" or "what does this do" and an image is attached, 
  plan image_ocr followed by code_explain.
Respond ONLY with JSON matching this shape, no other text:
{{"needs_clarification": bool, "question": string|null, "intent": string, "steps": [{{"tool": string}}, ...]}}
"""

def build_plan(state: AgentState) -> Plan:

    q = (state.query or "").strip().lower()

    has_pdf = "pdf" in state.file_paths
    has_image = "image" in state.file_paths
    has_audio = "audio" in state.file_paths

    attached = ", ".join(state.file_paths.keys()) or "none"

    user_prompt = (
        f"User query: {state.query or '(empty)'}\n"
        f"Attached file types: {attached}"
    )

    # ------------------------------------------
    # 1. Uploaded files but no instruction
    # ------------------------------------------

    if state.file_paths and not q:
        return Plan(
            needs_clarification=True,
            question="What would you like me to do with the uploaded file(s)?",
            intent="clarification",
            steps=[],
        )

    # ------------------------------------------
    # 2. Pure conversation
    # ------------------------------------------

    if not state.file_paths:
        return Plan(
            needs_clarification=False,
            question=None,
            intent="conversation",
            steps=[
                PlanStep(tool="conversational")
            ],
        )

    # ------------------------------------------
    # 3. Image explanation
    # ------------------------------------------

    if has_image and any(
        x in q for x in [
            "explain",
            "code",
            "bug",
            "complexity",
            "what does",
        ]
    ):
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
    # 4. PDF contains YouTube
    # ------------------------------------------

    if has_pdf and (
        "youtube" in q
        or "video" in q
        or "transcript" in q
    ):
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
    # 5. Compare intent
    # ------------------------------------------

# ------------------------------------------
# 5. Compare intent
# ------------------------------------------

    compare_words = [
        "compare", "difference", "similar", "same",
        "relation", "related", "match", "topic",
        "discuss", "both", "agree", "overlap",
    ]

    

    if sum([has_pdf, has_audio, has_image]) >= 2 and any(
        w in q for w in compare_words
    ):
        steps = []

        if has_pdf:
            steps.append(PlanStep(tool="pdf_extract"))

        if has_audio:
            steps.append(PlanStep(tool="audio_transcribe"))

        if has_image:
            steps.append(PlanStep(tool="image_ocr"))

        steps.append(PlanStep(tool="compare"))
        # top of build_plan, after computing has_pdf etc:
        print(f"DEBUG: has_pdf={has_pdf} has_audio={has_audio} q={q!r}")
        print(f"DEBUG: file_paths keys={list(state.file_paths.keys())}")

        return Plan(
            needs_clarification=False,
            question=None,
            intent="compare",
            steps=steps,
        )
    # Add as section 6.5, before the LLM fallback:

    # ------------------------------------------
    # 6.5 PDF attached with any query → extract first
    # ------------------------------------------
    if has_pdf and not any(
            w in q for w in ["youtube", "video", "transcript"]
        ):
            client = get_llm_client()
            raw = client.complete(system=SYSTEM_PROMPT, user=user_prompt, json_mode=True)
            try:
                llm_plan = Plan(**json.loads(raw))
                answer_steps = [s for s in llm_plan.steps if s.tool not in {
                    "pdf_extract", "image_ocr", "audio_transcribe"
                }]
                return Plan(
                    needs_clarification=False,
                    question=None,
                    intent=llm_plan.intent,
                    steps=[PlanStep(tool="pdf_extract")] + answer_steps,
                )
            except Exception:
                # Fallback — just extract + summarize
                return Plan(
                    needs_clarification=False,
                    question=None,  
                    intent="extract_and_answer",
                    steps=[
                        PlanStep(tool="pdf_extract"),
                        PlanStep(tool="conversational"),
                    ],
                )
    # ------------------------------------------
    # 6. Let the LLM decide
    # ------------------------------------------

    client = get_llm_client()

    raw = client.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        json_mode=True,
    )

    try:
        return Plan(**json.loads(raw))

    except Exception:
        return Plan(
            needs_clarification=True,
            question="I couldn't understand the request. Could you clarify?",
            intent="unknown",
            steps=[],
        )