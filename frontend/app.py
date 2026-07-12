"""
Streamlit UI for the Multimodal Agentic Assistant.
Run with: streamlit run frontend/app.py
"""

import os
import streamlit as st
import requests

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0f1117; color: #e2e8f0; }
.app-header {
    background: linear-gradient(90deg, #1a1f2e 0%, #141824 100%);
    border-bottom: 1px solid #2d3748;
    padding: 1rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 0.75rem;
}
.app-header h1 { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; margin: 0; letter-spacing: -0.02em; }
.app-header .badge {
    background: #7c3aed22; border: 1px solid #7c3aed55; color: #a78bfa;
    font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.5rem;
    border-radius: 999px; letter-spacing: 0.05em; text-transform: uppercase;
}
section[data-testid="stSidebar"] { background: #141824 !important; border-right: 1px solid #1e2535 !important; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-label {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #4a5568; margin-bottom: 0.5rem; margin-top: 1.25rem;
}
.file-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 999px;
    padding: 0.25rem 0.75rem; font-size: 0.75rem; color: #94a3b8; margin: 0.2rem;
}
.chat-bubble {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px;
    padding: 1rem 1.25rem; margin-bottom: 0.75rem; font-size: 0.9rem; line-height: 1.6;
}
.chat-bubble.user { background: #1e1b4b; border-color: #3730a3; margin-left: 2rem; }
.chat-bubble.assistant { background: #0f2027; border-color: #134e4a; margin-right: 2rem; }
.chat-bubble .role { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.chat-bubble.user .role { color: #818cf8; }
.chat-bubble.assistant .role { color: #34d399; }
.trace-step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.75rem; border-radius: 8px; margin-bottom: 0.35rem;
    font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
    background: #141824; border: 1px solid #1e2535;
}
.trace-step.ok { border-left: 3px solid #10b981; }
.trace-step.failed { border-left: 3px solid #ef4444; }
.trace-step.skipped { border-left: 3px solid #f59e0b; }
.trace-step .tool-name { color: #e2e8f0; font-weight: 500; min-width: 140px; }
.trace-step .status-ok { color: #10b981; }
.trace-step .status-failed { color: #ef4444; }
.trace-step .status-skipped { color: #f59e0b; }
.trace-step .duration { color: #4a5568; margin-left: auto; }
.trace-step .msg { color: #64748b; }
.answer-box {
    background: #0d1f1a; border: 1px solid #134e4a; border-radius: 10px;
    padding: 1.25rem; font-size: 0.9rem; line-height: 1.7; color: #d1fae5; white-space: pre-wrap;
}
.answer-type-badge {
    display: inline-block; background: #064e3b; color: #34d399;
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 0.2rem 0.6rem; border-radius: 999px; margin-bottom: 0.75rem;
}
.extracted-box {
    background: #141824; border: 1px solid #2d3748; border-radius: 10px; padding: 1rem;
    font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8;
    line-height: 1.6; max-height: 300px; overflow-y: auto; white-space: pre-wrap;
}
.extracted-label {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #4a5568; margin-bottom: 0.4rem;
}
.warn-box {
    background: #1c1407; border: 1px solid #78350f; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.8rem; color: #fbbf24; margin-bottom: 0.5rem;
}
.cost-bar {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
    padding: 0.75rem 1.25rem; display: flex; align-items: center;
    gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;
}
.cost-item { display: flex; flex-direction: column; gap: 0.1rem; }
.cost-label { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #4a5568; }
.cost-value { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
.cost-value.green { color: #34d399; }
.context-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #1e1b4b; border: 1px solid #3730a3; border-radius: 6px;
    padding: 0.3rem 0.7rem; font-size: 0.75rem; color: #818cf8; margin-bottom: 0.75rem;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9); color: white; border: none;
    border-radius: 8px; font-weight: 600; font-size: 0.9rem; padding: 0.6rem 1.5rem;
    width: 100%; transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }
.stTabs [data-baseweb="tab-list"] {
    background: #141824; border-radius: 8px; padding: 0.25rem; gap: 0.25rem; border: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #1e2535 !important; color: #e2e8f0 !important; }
.stTextArea textarea {
    background: #141824 !important; border: 1px solid #2d3748 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
}
.stTextArea textarea:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 2px #7c3aed22 !important; }
hr { border-color: #1e2535 !important; }
.stSpinner > div { border-top-color: #7c3aed !important; }
.empty-state { text-align: center; padding: 3rem 1rem; color: #4a5568; }
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-state p { font-size: 0.85rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span style="font-size:1.4rem">🤖</span>
    <h1>Agentic Assistant</h1>
    <span class="badge">Multimodal</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">Backend</div>', unsafe_allow_html=True)
    backend_url = st.text_input("API URL", value=BACKEND, label_visibility="collapsed")

    try:
        health = requests.get(f"{backend_url}/health", timeout=3).json()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.78rem;color:#34d399;
                    background:#0d1f1a;border:1px solid #134e4a;border-radius:6px;padding:0.4rem 0.7rem;">
            <span>●</span> Connected · <span style="color:#64748b">{health.get('llm_provider','?')}</span>
        </div>""", unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.78rem;color:#ef4444;
                    background:#1c0a0a;border:1px solid #7f1d1d;border-radius:6px;padding:0.4rem 0.7rem;">
            <span>●</span> Backend offline
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Upload Files</div>', unsafe_allow_html=True)
    pdf_file   = st.file_uploader("PDF",   type=["pdf"],              label_visibility="collapsed", key="pdf")
    image_file = st.file_uploader("Image", type=["png","jpg","jpeg"], label_visibility="collapsed", key="img")
    audio_file = st.file_uploader("Audio", type=["mp3","wav","m4a"],  label_visibility="collapsed", key="aud")

    attached = []
    if pdf_file:   attached.append(("📄", pdf_file.name))
    if image_file: attached.append(("🖼️", image_file.name))
    if audio_file: attached.append(("🎵", audio_file.name))

    if attached:
        pills_html = "".join(f'<span class="file-pill">{icon} {name}</span>' for icon, name in attached)
        st.markdown(f'<div style="margin-top:0.5rem">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-8:])):
            q = item["query"][:40] + ("…" if len(item["query"]) > 40 else "")
            if st.button(q, key=f"hist_{i}", use_container_width=True):
                st.session_state.last_response = item["response"]
    else:
        st.markdown('<p style="font-size:0.75rem;color:#4a5568">No history yet.</p>', unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_response = None
            st.rerun()


def _get_context_from_history() -> str:
    """Pull extracted text from the most recent response for follow-up queries."""
    if not st.session_state.history:
        return ""
    last = st.session_state.history[-1]
    extracted = last["response"].get("extracted", {})
    parts = [
        v for k, v in extracted.items()
        if isinstance(v, str) and v.strip()
    ]
    if not parts:
        return ""
    return "\n\n[Context from previous upload]:\n" + "\n\n".join(parts)[:3000]


# ── Main workspace ───────────────────────────────────────────
with st.container():

    # ── Chat history ─────────────────────────────────────────
    if st.session_state.history:
        st.markdown('<p class="sidebar-label">Conversation</p>', unsafe_allow_html=True)
        for item in st.session_state.history[-6:]:
            st.markdown(f"""
            <div class="chat-bubble user">
                <div class="role">You</div>
                {item['query']}
            </div>""", unsafe_allow_html=True)

            result  = item["response"].get("result") or {}
            content = result.get("content", "—")
            preview = str(content)[:300] + ("…" if len(str(content)) > 300 else "")
            st.markdown(f"""
            <div class="chat-bubble assistant">
                <div class="role">Assistant</div>
                {preview}
            </div>""", unsafe_allow_html=True)
        st.markdown("---")

    # ── Prompt input ──────────────────────────────────────────
    st.markdown('<p class="sidebar-label">Your prompt</p>', unsafe_allow_html=True)

    # Show context badge when no files attached but history has extracted content
    no_files = not any([pdf_file, image_file, audio_file])
    has_prev_context = bool(st.session_state.history and _get_context_from_history())

    if no_files and has_prev_context:
        st.markdown(
            '<div class="context-badge">🔗 Using context from previous upload</div>',
            unsafe_allow_html=True
        )

    query = st.text_area(
        "Query",
        placeholder="Ask a question, request a summary, explain code… or just chat.",
        height=100,
        label_visibility="collapsed",
    )

    send = st.button("⚡ Run", use_container_width=False)

    # ── Run ───────────────────────────────────────────────────
    if send:
        if not query.strip() and not any([pdf_file, image_file, audio_file]):
            st.warning("Enter a prompt or upload a file.")
        else:
            with st.spinner("Running agent…"):
                files = {}
                if pdf_file:   files["pdf"]   = (pdf_file.name,   pdf_file,   "application/pdf")
                if image_file: files["image"] = (image_file.name, image_file, image_file.type)
                if audio_file: files["audio"] = (audio_file.name, audio_file, audio_file.type)

                # Append previous extracted context for follow-up queries
                final_query = query
                if no_files and has_prev_context:
                    final_query = query + "\n\n" + _get_context_from_history()

                try:
                    resp = requests.post(
                        f"{backend_url}/run",
                        data={"query": final_query},
                        files=files,
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.last_response = data
                    # Store original query (not with appended context) for display
                    st.session_state.history.append({"query": query, "response": data})
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend. Is `uvicorn app.main:app` running?")
                except requests.exceptions.Timeout:
                    st.error("⏱ Request timed out (300s). Try a shorter audio file.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Results panel ─────────────────────────────────────────
    resp_data = st.session_state.last_response

    if resp_data:
        st.markdown("---")

        # ── Cost estimator ────────────────────────────────────
        usage = resp_data.get("usage", {})
        if usage and usage.get("total_tokens"):
            st.markdown(f"""
            <div class="cost-bar">
                <div class="cost-item">
                    <span class="cost-label">📥 Input tokens</span>
                    <span class="cost-value">{usage['input_tokens']:,}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">📤 Output tokens</span>
                    <span class="cost-value">{usage['output_tokens']:,}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">🔢 Total tokens</span>
                    <span class="cost-value">{usage['total_tokens']:,}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">💰 Est. cost</span>
                    <span class="cost-value green">${usage['cost_usd']:.5f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["⚡ Execution Trace", "📄 Extracted Text", "✅ Final Answer"])

        with tab1:
            trace    = resp_data.get("trace", [])
            warnings = resp_data.get("warnings", [])

            if warnings:
                for w in warnings:
                    st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

            if trace:
                for step in trace:
                    tool   = step.get("tool", "?")
                    status = step.get("status", "?")
                    msg    = step.get("message", "")
                    dur    = step.get("duration_ms", 0)
                    icon   = {"ok": "✓", "failed": "✗", "skipped": "⊘"}.get(status, "?")
                    st.markdown(f"""
                    <div class="trace-step {status}">
                        <span class="status-{status}">{icon}</span>
                        <span class="tool-name">{tool}</span>
                        <span class="msg">{msg[:80]}</span>
                        <span class="duration">{dur:.0f}ms</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-state"><div class="icon">🔍</div><p>No trace yet.</p></div>', unsafe_allow_html=True)

        with tab2:
            extracted = resp_data.get("extracted", {})
            has_any   = False

            for label, icon, key in [
                ("PDF Text",           "📄", "pdf_text"),
                ("Image OCR",          "🖼️", "image_text"),
                ("Audio Transcript",   "🎵", "audio_text"),
                ("YouTube Transcript", "▶️", "youtube_transcript"),
            ]:
                val = extracted.get(key)
                if val:
                    has_any = True
                    st.markdown(f'<div class="extracted-label">{icon} {label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="extracted-box">{val[:2000]}{"…" if len(val) > 2000 else ""}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

            urls = extracted.get("youtube_urls", [])
            if urls:
                has_any = True
                st.markdown('<div class="extracted-label">🔗 YouTube URLs found</div>', unsafe_allow_html=True)
                for url in urls:
                    st.markdown(f"- {url}")

            if not has_any:
                st.markdown('<div class="empty-state"><div class="icon">📭</div><p>No files were extracted in this run.</p></div>', unsafe_allow_html=True)

        with tab3:
            result = resp_data.get("result")
            if result:
                rtype   = result.get("type", "response")
                content = result.get("content", "")
                type_labels = {
                    "summary":              "📋 Summary",
                    "code_explanation":     "💻 Code Explanation",
                    "comparison":           "⚖️ Comparison",
                    "sentiment":            "💬 Sentiment",
                    "conversation":         "🗣 Conversation",
                    "clarification_needed": "❓ Clarification Needed",
                    "partial_failure":      "⚠️ Partial Failure",
                }
                badge = type_labels.get(rtype, rtype)
                st.markdown(f'<span class="answer-type-badge">{badge}</span>', unsafe_allow_html=True)

                if rtype == "clarification_needed":
                    st.info(content)
                elif rtype == "partial_failure":
                    st.warning(content)
                else:
                    st.markdown(f'<div class="answer-box">{content}</div>', unsafe_allow_html=True)
                    with st.expander("📋 Raw text"):
                        st.code(content, language=None)
            else:
                st.markdown('<div class="empty-state"><div class="icon">💬</div><p>Run a query to see the answer here.</p></div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state" style="margin-top:3rem">
            <div class="icon">✨</div>
            <p style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:0.5rem">
                Ready when you are
            </p>
            <p>Upload a PDF, image, or audio file —<br>
            or just type a question and hit <strong>Run</strong>.</p>
        </div>
        """, unsafe_allow_html=True)