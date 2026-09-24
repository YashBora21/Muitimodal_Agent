import io
import wave
from types import SimpleNamespace

import fitz

from multimodal_agent import graph as graph_module
from multimodal_agent.extraction import audio, image, pdf, youtube
from multimodal_agent.tools import asset_reader
from multimodal_agent.tools.schema import TOOLS


def test_root_app_uses_packaged_application():
    import app as root_app
    from multimodal_agent.app import app as package_app

    assert root_app.app is package_app


def test_youtube_url_variants():
    assert youtube.youtube_id("https://youtu.be/abc123?t=2") == "abc123"
    assert youtube.youtube_id("https://youtube.com/watch?v=abc123") == "abc123"
    assert youtube.youtube_id("https://youtube.com/shorts/abc123") == "abc123"


def test_wav_duration():
    data = io.BytesIO()
    with wave.open(data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\0\0" * 8000)
    assert audio.audio_duration(data.getvalue(), ".wav") == 1.0


def test_vision_uses_one_plain_text_request(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            assert "response_format" not in kwargs
            prompt = kwargs["messages"][0]["content"][0]["text"]
            assert "do not return JSON" in prompt
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="TOTAL: $12\nA store receipt.")
                )]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(image, "client", lambda: fake_client)

    assert image.vision(b"image", "image/png") == "TOTAL: $12\nA store receipt."


def test_tools_are_only_sent_when_assets_exist(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=[])
                )]
            )

    monkeypatch.setattr(
        graph_module,
        "client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
    )
    graph_module.run_agent("What is an RNN?", [], [])

    asset = {
        "id": "image-3",
        "kind": "image",
        "name": "screen.png",
        "content": "A code screenshot.",
    }
    graph_module.run_agent("Explain the image", [asset], [], ["image-3"])

    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[1]["tool_choice"] == "auto"
    assert any(
        "Grounding evidence from image-3" in message["content"]
        for message in calls[1]["messages"]
    )
    assert len(calls[1]["tools"]) == 1
    assert calls[1]["tools"][0]["function"]["name"] == "read_asset"
    assert "[current upload]" in calls[1]["messages"][0]["content"]


def test_tool_schema_has_pdf_text_and_visual_modes():
    assert len(TOOLS) == 1
    properties = TOOLS[0]["function"]["parameters"]["properties"]
    assert set(properties) == {"asset_id", "query", "scope", "page_number", "visual"}
    assert properties["query"]["type"] == ["string", "null"]
    assert properties["scope"]["enum"] == ["auto", "focused", "whole", None]
    assert properties["page_number"]["type"] == ["integer", "null"]
    assert properties["visual"]["type"] == ["boolean", "null"]


def test_read_asset_accepts_null_optional_arguments():
    asset = {
        "id": "youtube-8",
        "kind": "youtube",
        "name": "video",
        "content": "transcript",
    }

    result = asset_reader.run_tool(
        "read_asset",
        {
            "asset_id": "youtube-8",
            "query": None,
            "page_number": None,
            "visual": False,
        },
        {"youtube-8": asset},
    )

    assert result == "transcript"


def test_pdf_page_can_be_visually_inspected(monkeypatch):
    seen = []
    monkeypatch.setattr(
        asset_reader,
        "vision",
        lambda data, mime: seen.append((data, mime)) or "A bar chart.",
    )
    document = fitz.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    asset = {
        "id": "pdf-1",
        "kind": "pdf",
        "name": "charts.pdf",
        "content": "--- Page 1 ---",
        "data": data,
    }

    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "page_number": 1, "visual": True},
        {"pdf-1": asset},
    )

    assert result == "A bar chart."
    assert seen[0][1] == "image/jpeg"


def test_scanned_pdf_ocr_reads_page_58(monkeypatch):
    pages_read = []

    def fake_ocr(page):
        page_number = page.number + 1
        pages_read.append(page_number)
        return f"The answer is on page {page_number}."

    monkeypatch.setattr(pdf, "ocr_page", fake_ocr)
    document = fitz.open()
    for _ in range(60):
        document.new_page()
    data = document.tobytes()
    document.close()

    assets = pdf.read_pdf(data, "scanned.pdf", [])

    assert pages_read == list(range(1, 61))
    assert "The answer is on page 58." in assets[0]["content"]
    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "page_number": 58},
        {"pdf-1": assets[0]},
    )
    assert "--- Page 58 ---" in result
    assert "The answer is on page 58." in result
    assert "--- Page 59 ---" not in result


def test_repeated_tool_call_reuses_result(monkeypatch):
    api_calls = []
    tool_runs = []

    class Completions:
        def create(self, **kwargs):
            api_calls.append(kwargs)
            if len(api_calls) == 3:
                message = SimpleNamespace(content="done", tool_calls=[])
            else:
                call_id = f"call-{len(api_calls)}"
                tool_call = SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name="read_asset",
                        arguments='{"asset_id": "image-1"}',
                    ),
                    model_dump=lambda: {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_asset",
                            "arguments": '{"asset_id": "image-1"}',
                        },
                    },
                )
                message = SimpleNamespace(content="", tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(
        graph_module,
        "client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    monkeypatch.setattr(
        graph_module,
        "run_tool",
        lambda name, args, assets: tool_runs.append((name, args)) or "content",
    )
    asset = {
        "id": "image-1",
        "kind": "image",
        "name": "one.png",
        "content": "content",
    }

    result = graph_module.run_agent("Explain it", [asset], [])

    assert result["answer"] == "done"
    assert len(tool_runs) == 1
    assert any("[cached]" in log for log in result["logs"])


def test_pdf_registers_youtube_links_without_fetching_them(monkeypatch):
    monkeypatch.setattr(
        youtube,
        "youtube_text",
        lambda url: (_ for _ in ()).throw(AssertionError("eager download")),
    )
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Video: https://youtu.be/abc123")
    data = document.tobytes()
    document.close()

    assets = pdf.read_pdf(data, "links.pdf", [])

    assert [asset["kind"] for asset in assets] == ["pdf", "youtube"]
    assert "Total YouTube links: 1" in assets[0]["content"]
    assert assets[1]["content"] == ""


def test_run_tool_fetches_only_the_requested_youtube(monkeypatch):
    fetched = []
    monkeypatch.setattr(
        asset_reader,
        "youtube_text",
        lambda url: fetched.append(url) or "transcript",
    )
    assets = {
        "youtube-1": {
            "id": "youtube-1",
            "kind": "youtube",
            "name": "https://youtu.be/one",
            "content": "",
        },
        "youtube-2": {
            "id": "youtube-2",
            "kind": "youtube",
            "name": "https://youtu.be/two",
            "content": "",
        },
    }

    assert asset_reader.run_tool(
        "read_asset",
        {"asset_id": "youtube-2"},
        assets,
    ) == "transcript"
    assert fetched == ["https://youtu.be/two"]



def test_database_is_required_for_authenticated_routes(monkeypatch):
    from fastapi import HTTPException
    from multimodal_agent import app as app_module

    monkeypatch.setattr(app_module.database, "enabled", lambda: False)

    try:
        app_module.current_user("Bearer token")
    except HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("database-free authentication was allowed")


def test_chunking_keeps_relevant_tail_content(monkeypatch):
    from multimodal_agent import retrieval

    monkeypatch.setattr(retrieval, "CHUNK_SIZE", 120)
    monkeypatch.setattr(retrieval, "CHUNK_OVERLAP", 20)
    content = ("irrelevant introduction " * 30) + "Section 302 IPC punishment is life imprisonment."

    parts = retrieval.chunks(content)

    assert "life imprisonment" in parts[-1]
    assert "".join(part for part in parts if part) != ""

def test_agent_passes_complete_tool_result(monkeypatch):
    api_calls = []
    complete_result = "\n".join(
        f"--- Page {page} ---\n" + "grounded evidence " * 120
        for page in range(1, 7)
    )

    class Completions:
        def create(self, **kwargs):
            api_calls.append(kwargs)
            if len(api_calls) == 2:
                assert kwargs["messages"][-1]["content"] == complete_result
                message = SimpleNamespace(content="grounded answer", tool_calls=[])
            else:
                tool_call = SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="read_asset",
                        arguments='{"asset_id": "pdf-1", "query": "evidence"}',
                    ),
                    model_dump=lambda: {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_asset",
                            "arguments": '{"asset_id": "pdf-1", "query": "evidence"}',
                        },
                    },
                )
                message = SimpleNamespace(content="", tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(
        graph_module,
        "client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    monkeypatch.setattr(graph_module, "run_tool", lambda *args: complete_result)
    asset = {
        "id": "pdf-1",
        "kind": "pdf",
        "name": "evidence.pdf",
        "content": complete_result,
    }

    result = graph_module.run_agent("Find evidence", [asset], [], ["pdf-1"])

    assert result["answer"] == "grounded answer"
    assert api_calls[0]["tool_choice"] == "auto"
    assert api_calls[1]["tool_choice"] == "auto"
    assert any("Grounding | pdf-1 preloaded" in log for log in result["logs"])
    assert any("Retrieval | pdf-1 | hybrid" in log for log in result["logs"])



def test_pdf_discovered_youtube_assets_are_not_preloaded(monkeypatch):
    discovered_assets = [
        {"id": "pdf-1", "kind": "pdf", "name": "links.pdf", "content": "PDF"},
        {"id": "youtube-1", "kind": "youtube", "name": "https://youtu.be/one", "content": ""},
        {"id": "youtube-2", "kind": "youtube", "name": "https://youtu.be/two", "content": ""},
    ]
    monkeypatch.setattr(graph_module, "ingest_upload", lambda upload, existing: discovered_assets)

    assets, current_asset_ids = graph_module.ingest_assets(
        "What is this PDF about?",
        [{"name": "links.pdf", "data": b"pdf"}],
        [],
        True,
    )

    assert len(assets) == 3
    assert current_asset_ids == ["pdf-1"]
    assert assets[1]["content"] == assets[2]["content"] == ""



def test_rag_threshold_is_based_on_configured_content_size(monkeypatch):
    from multimodal_agent import retrieval

    monkeypatch.setattr(retrieval, "RAG_MIN_CHARS", 100)

    assert not retrieval.needs_hybrid_retrieval("x" * 100)
    assert retrieval.needs_hybrid_retrieval("x" * 101)


def test_database_asset_uses_persistent_hybrid_search(monkeypatch):
    from multimodal_agent import database

    content = "\n".join(
        f"--- Page {page} ---\n" + "evidence " * 250
        for page in range(1, 7)
    )
    asset = {
        "db_id": "00000000-0000-0000-0000-000000000001",
        "id": "pdf-1",
        "kind": "pdf",
        "name": "large.pdf",
        "content": content,
    }
    monkeypatch.setattr(database, "enabled", lambda: True)
    monkeypatch.setattr(database, "hybrid_search", lambda *args: "database evidence")

    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "query": "evidence"},
        {"pdf-1": asset},
    )

    assert result == "database evidence"


def test_thread_management_routes_are_registered():
    from multimodal_agent.app import app

    routes = {
        (route.path, method)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }

    assert ("/threads", "POST") in routes
    assert ("/threads", "GET") in routes
    assert ("/threads/{thread_id}", "GET") in routes
    assert ("/threads/{thread_id}", "DELETE") in routes
    assert ("/chat", "POST") in routes

def test_health_route_is_registered():
    from multimodal_agent.app import app

    routes = {
        (route.path, method)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }

    assert ("/health", "GET") in routes


def test_database_clean_text_removes_postgres_unsafe_characters():
    from multimodal_agent.database import clean_text

    value = "hello\x00world\x01\n\t!" + chr(0xD800)
    assert clean_text(value) == "helloworld\n\t!?"


def test_database_chat_persists_assets_before_agent(monkeypatch):
    import asyncio
    import uuid
    from multimodal_agent import app as app_module

    order = []
    added = {"id": "pdf-1", "kind": "pdf", "name": "file.pdf", "content": "content"}
    monkeypatch.setattr(app_module.database, "enabled", lambda: True)
    monkeypatch.setattr(app_module.database, "ensure_thread", lambda *args: None)
    monkeypatch.setattr(app_module.database, "load_thread", lambda *args: ([], []))
    monkeypatch.setattr(
        app_module,
        "ingest_assets",
        lambda *args: ([added], ["pdf-1"]),
    )

    def save_assets(*args):
        order.append("save_assets")
        added["db_id"] = "asset-uuid"

    def run_agent(*args):
        order.append("run_agent")
        assert added["db_id"] == "asset-uuid"
        return {"answer": "done", "decision": "answer", "logs": []}

    monkeypatch.setattr(app_module.database, "save_assets", save_assets)
    monkeypatch.setattr(app_module, "run_agent", run_agent)
    monkeypatch.setattr(app_module.database, "save_messages", lambda *args: order.append("save_messages"))

    result = asyncio.run(
        app_module.chat(
            message="question",
            thread_id=str(uuid.uuid4()),
            files=[],
            user={"id": "user-1"},
        )
    )

    assert result["answer"] == "done"
    assert order == ["save_assets", "run_agent", "save_messages"]


def test_database_asset_does_not_fall_back_to_memory_search(monkeypatch):
    from multimodal_agent import database

    content = "evidence " * 2000
    asset = {
        "db_id": "asset-uuid",
        "id": "pdf-1",
        "kind": "pdf",
        "name": "large.pdf",
        "content": "\n".join(
            f"--- Page {page} ---\n{content}" for page in range(1, 7)
        ),
    }
    monkeypatch.setattr(database, "enabled", lambda: True)
    monkeypatch.setattr(database, "hybrid_search", lambda *args: None)
    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "query": "evidence"},
        {"pdf-1": asset},
    )

    assert result == "No content matched that query."


def test_lazy_youtube_is_persisted_before_database_retrieval(monkeypatch):
    from multimodal_agent import database

    events = []
    asset = {
        "db_id": "asset-uuid",
        "id": "youtube-1",
        "kind": "youtube",
        "name": "https://youtu.be/example",
        "content": "",
    }
    monkeypatch.setattr(asset_reader, "youtube_text", lambda url: "transcript " * 1000)
    monkeypatch.setattr(database, "enabled", lambda: True)
    monkeypatch.setattr(database, "update_asset_content", lambda value: events.append("persist"))
    monkeypatch.setattr(database, "hybrid_search", lambda *args: events.append("search") or "database evidence")

    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "youtube-1", "query": "specific lyric"},
        {"youtube-1": asset},
    )

    assert result == "database evidence"
    assert events == ["persist", "search"]



def test_whole_document_request_uses_configured_hybrid_overview(monkeypatch):
    from multimodal_agent import database

    calls = []
    asset = {
        "db_id": "asset-uuid",
        "id": "pdf-1",
        "kind": "pdf",
        "name": "large.pdf",
        "content": "content " * 2000,
    }
    monkeypatch.setattr(
        database,
        "hybrid_search",
        lambda asset_id, query, top_k: calls.append((asset_id, query, top_k)) or "overview evidence",
    )

    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "query": "Summarize the whole document", "scope": "whole"},
        {"pdf-1": asset},
    )

    assert result == "overview evidence"
    assert calls[0][0] == "asset-uuid"
    assert "Summarize the whole document" in calls[0][1]
    assert asset_reader.WHOLE_DOCUMENT_QUERY in calls[0][1]
    assert calls[0][2] == asset_reader.RETRIEVAL_TOP_K


def test_long_whole_asset_with_null_query_never_returns_full_content(monkeypatch):
    from multimodal_agent import database

    calls = []
    asset = {
        "db_id": "asset-uuid",
        "id": "pdf-1",
        "kind": "pdf",
        "name": "large.pdf",
        "content": "private source content " * 1000,
    }
    monkeypatch.setattr(
        database,
        "hybrid_search",
        lambda asset_id, query, top_k: calls.append((query, top_k)) or "bounded evidence",
    )

    result = asset_reader.run_tool(
        "read_asset",
        {"asset_id": "pdf-1", "query": None, "scope": "whole"},
        {"pdf-1": asset},
    )

    assert result == "bounded evidence"
    assert calls == [(asset_reader.WHOLE_DOCUMENT_QUERY, asset_reader.RETRIEVAL_TOP_K)]
    assert result != asset["content"]
