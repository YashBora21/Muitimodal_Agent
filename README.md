# Context-aware Multimodal Agent

A ChatGPT-style multimodal assistant built with FastAPI, LangGraph, Groq, Neon PostgreSQL, and pgvector. Users can register, sign in, keep separate conversation threads, upload files, and ask grounded questions about PDFs, images, audio, and YouTube videos.

## Features

- JWT registration and login with Argon2 password hashing
- Persistent users, conversations, messages, and assets in Neon PostgreSQL
- PDF text extraction with OCR fallback for scanned pages
- Image understanding through a Groq vision model
- Audio transcription through Groq Whisper
- YouTube caption extraction with audio-transcription fallback
- Hybrid RAG using PostgreSQL full-text search and pgvector
- Chat history sidebar and file uploads
- Reusable Psycopg connection pool
- Request timing middleware
- Public `/health` readiness endpoint

## Architecture

```text
Browser UI
    |
    v
FastAPI
    |-- JWT authentication
    |-- LangGraph agent
    |-- Groq chat, vision, and transcription APIs
    |
    v
Neon PostgreSQL
    |-- users
    |-- threads
    |-- messages
    |-- assets
    `-- asset_chunks (full-text search + pgvector)
```

FastAPI serves both the UI and API from the same origin. CORS is therefore not required in the current deployment.

## Project Structure

```text
GENAI_Intern/
|-- app.py                         # Uvicorn compatibility entry point
|-- database/
|   `-- schema.sql                 # PostgreSQL and pgvector schema
|-- src/multimodal_agent/
|   |-- app.py                     # FastAPI routes, middleware, lifespan
|   |-- auth.py                    # Password hashing and JWT helpers
|   |-- config.py                  # Environment-based configuration
|   |-- database.py                # PostgreSQL access and connection pool
|   |-- graph.py                   # LangGraph workflow and model loop
|   |-- ingest.py                  # Upload routing
|   |-- retrieval.py               # Chunking and hybrid retrieval
|   |-- extraction/                # PDF, image, audio, YouTube extraction
|   |-- tools/                     # Agent tool schemas and execution
|   |-- static/                    # Browser JavaScript and CSS
|   `-- templates/                 # HTML UI
|-- tests/test_agent.py
|-- requirements.txt
`-- requirements-dev.txt
```

## Prerequisites

- Python 3.11 or newer
- A Groq API key
- A Neon PostgreSQL project with pgvector support
- FFmpeg/ffprobe for YouTube audio fallback and non-WAV duration detection
- Tesseract OCR language data for scanned PDFs

Text-based PDFs do not require Tesseract. YouTube videos with accessible captions do not require FFmpeg.

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For tests:

```powershell
python -m pip install -r requirements-dev.txt
```

### 3. Configure environment variables

Create `.env` in the repository root:

```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
GROQ_TRANSCRIBE_MODEL=whisper-large-v3-turbo

DATABASE_URL=your-neon-pooled-connection-string
DATABASE_URL_UNPOOLED=your-neon-direct-connection-string

JWT_SECRET=replace-with-a-long-random-secret
JWT_EXPIRE_MINUTES=30
DB_POOL_SIZE=5
MAX_FILE_MB=25

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=6
RAG_MIN_CHARS=10000
RAG_MIN_PDF_PAGES=5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

Generate a JWT secret with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Required variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Chat, vision, and transcription API access |
| `DATABASE_URL` | Neon pooled connection used by normal application queries |
| `DATABASE_URL_UNPOOLED` | Neon direct connection used for schema initialization |
| `JWT_SECRET` | Signs and verifies access tokens |

The remaining variables have defaults and are optional.

## Neon Database

Use the pooled Neon connection string for `DATABASE_URL`. Its hostname normally contains `-pooler`. Use the direct connection string for `DATABASE_URL_UNPOOLED`.

The application automatically executes `database/schema.sql` during startup. The schema creates:

- `users`: account credentials and profile information
- `threads`: conversations owned by a user
- `messages`: ordered by precise creation time
- `assets`: extracted files and transcripts
- `asset_chunks`: keyword and vector index for large assets

`asset_chunks.embedding` uses `VECTOR(384)`. PostgreSQL full-text search and vector similarity are combined for hybrid retrieval.

## Run Locally

```powershell
python -m uvicorn app:app --reload --port 8000
```

Open:

- UI: <http://127.0.0.1:8000>
- API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

A healthy response is:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

The health endpoint returns HTTP `503` when the database is not configured or unavailable.

## API

| Method | Path | Authentication | Purpose |
|---|---|---:|---|
| `GET` | `/` | No | Serve the browser UI |
| `GET` | `/health` | No | Check application and database readiness |
| `POST` | `/auth/register` | No | Create an account and return a JWT |
| `POST` | `/auth/login` | No | Verify credentials and return a JWT |
| `GET` | `/auth/me` | Yes | Return the current authenticated user |
| `POST` | `/threads` | Yes | Create a conversation |
| `GET` | `/threads` | Yes | List the user's conversations |
| `GET` | `/threads/{thread_id}` | Yes | Load one conversation, messages, and assets |
| `DELETE` | `/threads/{thread_id}` | Yes | Delete an owned conversation |
| `POST` | `/chat` | Yes | Send a message and optional files |

`POST /chat` expects `multipart/form-data`:

- `message`: optional when files are present
- `thread_id`: conversation UUID
- `files`: zero or more uploads

Protected requests send:

```http
Authorization: Bearer <access-token>
```

## Authentication Flow

```text
Register or login
    -> backend verifies password
    -> backend signs JWT containing the user ID
    -> frontend stores the JWT in localStorage
    -> frontend sends Authorization: Bearer <token>
    -> FastAPI validates the JWT
    -> database queries are filtered by the authenticated user ID
```

Access tokens expire after `JWT_EXPIRE_MINUTES`. The current implementation has no refresh-token or server-side token-revocation table; logout removes the token from the browser.

## RAG Behavior

Small assets are read in full. Large PDFs, audio transcripts, and YouTube transcripts are split into chunks and stored in `asset_chunks`.

For a question about a large asset:

1. The question is embedded.
2. PostgreSQL runs full-text keyword search.
3. pgvector runs semantic similarity search.
4. Reciprocal-rank fusion combines both result lists.
5. Only the best chunks are sent to Groq as grounding evidence.

This avoids sending complete large transcripts to the model and reduces token-limit failures.

## Middleware

The HTTP middleware records every request's method, path, response status, and duration. It also adds an `X-Process-Time` response header.

Example:

```text
INFO:multimodal_agent.app:GET /threads -> 200 (0.0234s)
INFO:multimodal_agent.app:POST /chat -> 200 (2.3451s)
```

## Tests

```powershell
python -m pytest -q
```

Current suite: 20 tests.

# Deploy on Render

The simplest deployment uses one Render Web Service because FastAPI serves both the frontend and backend.

## 1. Push the repository

Push the project to GitHub, GitLab, or Bitbucket. Do not commit `.env`, API keys, JWT secrets, Neon credentials, or YouTube cookies.

## 2. Create a Web Service

In the Render dashboard:

1. Select **New > Web Service**.
2. Connect the repository.
3. Select the production branch.
4. Choose the **Python** runtime.
5. Leave **Root Directory** empty when this repository is the project root.

## 3. Configure build and start commands

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Do not use `--reload` in production. Render provides the `PORT` environment variable and requires the service to bind to `0.0.0.0`.

## 4. Add environment variables

Add these in **Environment > Environment Variables**:

```text
PYTHON_VERSION=3.11.11
GROQ_API_KEY=<secret>
DATABASE_URL=<Neon pooled URL>
DATABASE_URL_UNPOOLED=<Neon direct URL>
JWT_SECRET=<long random secret>
JWT_EXPIRE_MINUTES=30
DB_POOL_SIZE=5
MAX_FILE_MB=25
```

Optional model and RAG settings can use the defaults listed in the local `.env` example.

Never add secrets directly to `README.md`, `render.yaml`, or source code.

## 5. Configure the health check

In the service's **Settings > Health Checks**, set:

```text
/health
```

Render will send `GET /health` periodically. A `200` response means the FastAPI process can query Neon. A `503` response prevents a broken instance from receiving traffic and can trigger a restart.

Render requires HTTP health checks to respond within five seconds, so keep this endpoint lightweight.

## 6. Deploy

Select **Create Web Service** or **Manual Deploy > Deploy latest commit**. After deployment, verify:

```text
https://<your-service>.onrender.com/health
https://<your-service>.onrender.com/docs
https://<your-service>.onrender.com/
```

## Render Native Runtime Limitations

The core application, authentication, text PDFs, image analysis, direct audio transcription, and caption-based YouTube extraction can run on Render's native Python runtime.

Two fallback paths require OS-level programs:

- Scanned-PDF OCR requires Tesseract and its `eng` language data.
- YouTube audio fallback requires FFmpeg.

Render recommends Docker when an application requires OS packages not included in its native runtime. For full OCR and YouTube fallback support in production, deploy this repository as a Docker service with Tesseract and FFmpeg installed in the image. Do not set the Windows `TESSDATA_PREFIX` value on Linux.

Render free web services can spin down after inactivity, so the first request after an idle period can be slower. The local SentenceTransformer embedding model also increases memory usage; monitor memory and move to a larger instance if the process is terminated for exceeding its limit.

Official references:

- [Deploying on Render](https://render.com/docs/deploys)
- [Render health checks](https://render.com/docs/health-checks)
- [Render Python versions](https://render.com/docs/python-version)
- [Docker on Render](https://render.com/docs/docker)

## Security Notes

- Keep `.env` and cookie files out of Git.
- Use a strong random `JWT_SECRET`.
- Use HTTPS in production; Render provides HTTPS for its service domain.
- Keep `DATABASE_URL` pooled and reserve `DATABASE_URL_UNPOOLED` for schema operations.
- Uploaded raw files are currently stored in PostgreSQL. Monitor database size and move large binaries to object storage when storage growth becomes significant.
- The JWT is stored in browser `localStorage`; avoid introducing unsafe JavaScript or unsanitized HTML.
