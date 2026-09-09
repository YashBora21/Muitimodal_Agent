# Context-aware Multimodal Agent

A FastAPI and LangGraph application for questions about text, images, PDFs,
audio, and YouTube videos.

## Project structure

~~~text
GENAI_INtern/
|-- src/
|   +-- multimodal_agent/
|       |-- extraction/
|       |   |-- audio.py
|       |   |-- image.py
|       |   |-- pdf.py
|       |   +-- youtube.py
|       |-- tools/
|       |   |-- asset_reader.py
|       |   +-- schema.py
|       |-- static/
|       |   |-- app.js
|       |   +-- styles.css
|       |-- templates/
|       |   +-- index.html
|       |-- agent.py
|       |-- app.py
|       |-- config.py
|       |-- graph.py
|       |-- ingest.py
|       |-- models.py
|       |-- prompts.py
|       +-- registry.py
|-- tests/
|   +-- test_agent.py
|-- pyproject.toml
+-- requirements.txt
~~~

## Module responsibilities

- extraction/: image, audio, PDF/OCR, and YouTube processing.
- tools/: the single model-tool schema and its executor.
- ingest.py: routes uploaded files to the correct extractor.
- registry.py: asset IDs, indexes, and safe content shortening.
- graph.py: model loop and LangGraph conversation flow.
- agent.py: small public compatibility entry point.
- app.py: FastAPI routes.
- templates/ and static/: browser interface.

## Install

Python 3.11+, Tesseract OCR, and FFmpeg/ffprobe are required.

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

Add GROQ_API_KEY to .env.

## Run

~~~powershell
python -m uvicorn multimodal_agent.app:app --app-dir src --reload
~~~

Open http://127.0.0.1:8000.

## Test

~~~powershell
pip install -r requirements-dev.txt
pytest -q
~~~
