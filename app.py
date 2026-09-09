"""Compatibility entry point for the original Uvicorn command."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_agent.app import app

__all__ = ["app"]
