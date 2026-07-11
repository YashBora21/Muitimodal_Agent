# 🤖 Multimodal Agentic Assistant

A deployed, agentic application that accepts multiple input types simultaneously (Text, Images, PDFs, Audio), extracts content, understands the user's goal, and autonomously performs the correct task — including complex, multi-step queries that require chaining several tools in a single request.

---

## 🌐 Live Deployment

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | `https://agentic-frontend.onrender.com` |
| **Backend (FastAPI)** | `https://agentic-backend.onrender.com` |
| **API Docs** | `https://agentic-backend.onrender.com/docs` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Streamlit Frontend)                      │
│   File Upload (PDF/Image/Audio) + Text Query + Chat UI      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST /run
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                      /app/main.py                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                              │
│                  /app/orchestrator.py                        │
│         Runs plan steps → executes tools → returns state    │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────┐     ┌──────────────────────────────────┐
│      Planner      │     │           Tool Registry           │
│ /app/planner/     │     │        /app/tools/registry.py    │
│                   │     │                                   │
│ Keyword rules +   │     │  Extraction Tools:               │
│ LLM fallback      │     │  ├── pdf_extract   (PyMuPDF+OCR) │
│                   │     │  ├── image_ocr     (Tesseract)   │
│ Returns:          │     │  ├── audio_transcribe (Whisper)  │
│ - intent          │     │  └── youtube_transcript (yt-dlp) │
│ - steps[]         │     │                                   │
│ - needs_clarif.   │     │  Answer Tools:                   │
└───────────────────┘     │  ├── summarize                   │
                          │  ├── sentiment                   │
                          │  ├── code_explain                │
                          │  ├── compare                     │
                          │  └── conversational              │
                          └──────────────┬───────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────┐
                          │        LLM Client         │
                          │    /app/llm/              │
                          │                           │
                          │  GroqClient (default)     │
                          │  OllamaClient (local dev) │
                          └──────────────────────────┘
```

---

## ✨ Features

### Input Types Supported
- **Text** — plain natural language queries
- **Image** (JPG/PNG) — OCR via Tesseract
- **PDF** — text extraction via PyMuPDF + OCR fallback for scanned pages
- **Audio** (MP3/WAV/M4A) — Speech-to-Text via faster-whisper
- **Multiple inputs simultaneously** — e.g., PDF + audio + text query in one request

### Tasks Handled
| Task | Description |
|------|-------------|
| Image/PDF Extraction | Returns cleaned text + OCR confidence score |
| YouTube Transcript | Detects URL in any input → fetches transcript via yt-dlp |
| Conversational Answering | Friendly response for general questions |
| Summarization | 1-line summary + 3 bullets + 5-sentence summary |
| Sentiment Analysis | Label + confidence + one-line justification |
| Code Explanation | Language detection + explanation + bug detection + time complexity |
| Audio Transcription | Convert audio → text → summarize |
| Cross-Input Reasoning | Compare/combine content from multiple inputs |

### Bonus Features
- 💰 **Cost Estimator** — token count + estimated API cost shown after every run
- 🔍 **Tool Call Visualization** — real-time execution trace with status and timing per step

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Groq API key ([get one free](https://console.groq.com))
- Tesseract OCR installed (Windows only for local dev)

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/YashBora21/agentic-assistant.git
cd agentic-assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Start backend
uvicorn app.main:app --reload

# 6. Start frontend (new terminal)
streamlit run frontend/app.py
```

Visit `http://localhost:8501`

### Docker (Recommended)

```bash
# Build and run both backend + frontend
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider: "groq" for cloud, "ollama" for local
LLM_PROVIDER=groq

# Groq (recommended)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Ollama (local dev only)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud

# Whisper model size: tiny, base, small, medium
WHISPER_MODEL_SIZE=base

# Max file upload size in MB
MAX_UPLOAD_MB=25
```

---

## 📁 Project Structure

```
agentic-assistant/
├── app/
│   ├── main.py               # FastAPI entrypoint
│   ├── orchestrator.py       # Runs plan steps
│   ├── state.py              # AgentState dataclass
│   ├── config.py             # Settings from env vars
│   ├── extraction/
│   │   ├── pdf_extract.py    # PyMuPDF + OCR fallback
│   │   ├── image_ocr.py      # Tesseract OCR
│   │   ├── audio_transcribe.py # faster-whisper
│   │   └── youtube_fetch.py  # yt-dlp transcript
│   ├── llm/
│   │   ├── base.py           # LLMClient interface
│   │   ├── factory.py        # Provider selection
│   │   ├── groq_client.py    # Groq + token tracking
│   │   └── ollama_client.py  # Local Ollama
│   ├── planner/
│   │   ├── planner.py        # Intent detection + LLM fallback
│   │   └── schemas.py        # Plan / PlanStep models
│   └── tools/
│       └── registry.py       # All tool implementations
├── frontend/
│   └── app.py                # Streamlit UI
├── tests/
│   ├── test_cases.py         # Section 8 test suite
│   ├── make_fixtures.py      # Generate test fixtures
│   └── fixtures/             # Test PDFs, images, audio
├── Dockerfile                # Backend container
├── Dockerfile.frontend       # Frontend container
├── docker-compose.yml        # Orchestrates both services
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## 🧪 Test Cases

Generate fixtures first:
```bash
python -m tests.make_fixtures
```

Run all tests (requires backend running):
```bash
uvicorn app.main:app &
python -m tests.test_cases
```

### Test Results
| Test | Description | Status |
|------|-------------|--------|
| TC1 | Audio transcription + summary | ✅ |
| TC2 | PDF + natural language query | ✅ |
| TC3 | Image with code, explain | ✅ |
| TC4 | PDF → YouTube URL → transcript → summary | ✅ |
| TC5 | Audio + PDF comparative analysis | ✅ |
| Edge | PDF with no query → clarification | ✅ |
| Edge | Plain text question, no files | ✅ |

---

## 🔌 API Reference

### `POST /run`
Main endpoint. Accepts multipart form data.

**Parameters:**
| Field | Type | Description |
|-------|------|-------------|
| `query` | string | User's text query |
| `pdf` | file | PDF document (optional) |
| `image` | file | Image file (optional) |
| `audio` | file | Audio file (optional) |

**Response:**
```json
{
  "result": {
    "type": "summary",
    "content": "..."
  },
  "extracted": {
    "pdf_text": "...",
    "image_text": "...",
    "audio_text": "...",
    "youtube_urls": [],
    "youtube_transcript": "..."
  },
  "trace": [
    {"tool": "pdf_extract", "status": "ok", "message": "extracted 264 chars", "duration_ms": 120}
  ],
  "warnings": [],
  "usage": {
    "input_tokens": 847,
    "output_tokens": 213,
    "total_tokens": 1060,
    "cost_usd": 0.00005
  }
}
```

### `GET /health`
Returns backend status and active LLM provider.

---

## 🐳 Deployment on Render

### Backend
1. New Web Service → Docker → select repo
2. Dockerfile: `./Dockerfile`
3. Port: `8000`
4. Add environment variables from `.env`

### Frontend
1. New Web Service → Docker → select repo
2. Dockerfile: `./Dockerfile.frontend`
3. Port: `8501`
4. Set `BACKEND_URL=https://your-backend.onrender.com`

---

## 🎯 Design Decisions

**Hybrid Planner** — keyword rules handle common cases (compare, code explain, YouTube chain) deterministically, with LLM fallback for complex/ambiguous queries. This reduces LLM calls and improves reliability.

**Fail-fast orchestration** — if a required extraction step fails (e.g., audio_transcribe), the orchestrator stops immediately and returns a `partial_failure` result rather than feeding empty content to downstream tools.

**CPU-first Whisper** — faster-whisper runs with `device="cpu"` and `compute_type="int8"` for maximum compatibility across deployment environments without GPU.

**yt-dlp over youtube-transcript-api** — yt-dlp handles YouTube's bot detection more reliably and doesn't require authentication for most videos.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Backend API framework |
| `streamlit` | Frontend UI |
| `groq` | LLM inference (cloud) |
| `faster-whisper` | Audio transcription |
| `pytesseract` | OCR for images |
| `pymupdf` | PDF text extraction |
| `yt-dlp` | YouTube transcript fetching |
| `pillow` | Image preprocessing |

---

## 👤 Author

**Yash Bora**
- GitHub: [YashBora21](https://github.com/YashBora21)
- Portfolio: [yash-bora-portfolio.vercel.app](https://yash-bora-portfolio.vercel.app)