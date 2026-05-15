from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "gpt-4o-mini"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("DEEPSEEK_TIMEOUT_SECONDS") or "90")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
