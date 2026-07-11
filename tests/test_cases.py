"""
Runs the assignment's Section 8 test cases against a live server.

Usage:
    uvicorn app.main:app &          # start the server first
    python3 -m tests.make_fixtures  # generate PDF/image fixtures once
    python3 -m tests.test_cases     # run this

Exits non-zero if any test fails, so it's CI-friendly too.
"""
import sys

import requests

BASE_URL = "http://localhost:8000"
FIXTURES = "tests/fixtures"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def run(query: str = "", files: dict | None = None) -> dict:
    data = {"query": query}
    file_handles = {}
    try:
        for label, path in (files or {}).items():
            file_handles[label] = open(path, "rb")
        resp = requests.post(f"{BASE_URL}/run", data=data, files=file_handles, timeout=120)
        resp.raise_for_status()
        return resp.json()
    finally:
        for fh in file_handles.values():
            fh.close()


def trace_tools(response: dict) -> list[str]:
    return [step["tool"] for step in response["trace"]]


def test_case_2_pdf_query():
    print("\nTest Case 2 - PDF + natural language query")
    resp = run("What are the action items?", {"pdf": f"{FIXTURES}/meeting_notes.pdf"})
    tools = trace_tools(resp)
    check("pdf_extract ran", "pdf_extract" in tools)
    check("did not hallucinate a youtube step", "youtube_transcript" not in tools)
    check("got a result", resp["result"] is not None, str(resp["result"]))
    if resp["result"]:
        content = str(resp["result"].get("content", "")).lower()
        check("mentions dashboard/budget action items", "dashboard" in content or "budget" in content)


def test_case_3_image_code():
    print("\nTest Case 3 - Image with code, 'Explain'")
    resp = run("Explain this code", {"image": f"{FIXTURES}/code.jpg"})
    tools = trace_tools(resp)
    check("image_ocr ran", "image_ocr" in tools)
    check("code_explain ran", "code_explain" in tools)
    check("OCR extracted something", bool(resp["extracted"]["image_text"]))


def test_case_4_pdf_youtube_chain():
    print("\nTest Case 4 - PDF containing a YouTube URL, chained fetch + summary")
    resp = run(
        "Hit the YT URL in this PDF and give me a summary of it",
        {"pdf": f"{FIXTURES}/pdf_with_youtube_link.pdf"},
    )
    tools = trace_tools(resp)
    check("pdf_extract ran before youtube_transcript",
          "pdf_extract" in tools and "youtube_transcript" in tools
          and tools.index("pdf_extract") < tools.index("youtube_transcript"))
    check("url was actually detected", len(resp["extracted"]["youtube_urls"]) > 0)
    # transcript fetch may legitimately fail (disabled/unavailable) - that's graceful
    # degradation, not a bug, so just check we didn't crash
    check("no crash, got a result", resp["result"] is not None)


def test_case_5_multi_file_compare():
    print("\nTest Case 5 - audio + PDF, comparative question")
    import os
    audio_path = f"{FIXTURES}/test_audio.mp3"
    if not os.path.exists(audio_path):
        print("  SKIP  no audio fixture at tests/fixtures/test_audio.mp3 - add one to run this")
        return
    resp = run(
        "Do the audio and the document discuss the same topic?",
        {"pdf": f"{FIXTURES}/meeting_notes.pdf", "audio": audio_path},
    )
    tools = trace_tools(resp)
    check("pdf_extract ran", "pdf_extract" in tools)
    check("audio_transcribe ran", "audio_transcribe" in tools)
    check("compare ran", "compare" in tools)


def test_clarification_on_ambiguous_input():
    print("\nEdge case - PDF with no query should trigger a clarifying question")
    resp = run("", {"pdf": f"{FIXTURES}/meeting_notes.pdf"})
    check("marked as needing clarification",
          resp["result"] is not None and resp["result"].get("type") == "clarification_needed",
          str(resp["result"]))


def test_conversational_no_files():
    print("\nEdge case - plain text question, no files")
    resp = run("What is the capital of France?")
    tools = trace_tools(resp)
    check("used conversational tool", "conversational" in tools)
    check("no extraction tools ran", not any(t in tools for t in
          ("pdf_extract", "image_ocr", "audio_transcribe")))


if __name__ == "__main__":
    test_case_2_pdf_query()
    test_case_3_image_code()
    test_case_4_pdf_youtube_chain()
    test_case_5_multi_file_compare()
    test_clarification_on_ambiguous_input()
    test_conversational_no_files()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
