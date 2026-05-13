from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.getenv("FORECAST_DB_PATH", DATA_DIR / "forecast_agent.db"))
MAX_MAP_ROUNDS = int(os.getenv("FORECAST_MAX_MAP_ROUNDS", "3"))
STOP_DELTA_THRESHOLD = float(os.getenv("FORECAST_STOP_DELTA_THRESHOLD", "0.06"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("FORECAST_SEARCH_RESULTS_PER_QUERY", "3"))
SEARCH_PROVIDER = os.getenv("FORECAST_SEARCH_PROVIDER", "synthetic").lower()
CONTENT_PROVIDER = os.getenv("FORECAST_CONTENT_PROVIDER", SEARCH_PROVIDER).lower()
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_SEARCH_URL = os.getenv("EXA_SEARCH_URL", "https://api.exa.ai/search")
EXA_CONTENTS_URL = os.getenv("EXA_CONTENTS_URL", "https://api.exa.ai/contents")
EXA_SEARCH_TYPE = os.getenv("EXA_SEARCH_TYPE", "auto")
EXA_TIMEOUT_SECONDS = float(os.getenv("EXA_TIMEOUT_SECONDS", "20"))
EXA_TEXT_MAX_CHARACTERS = int(os.getenv("EXA_TEXT_MAX_CHARACTERS", "4000"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30"))
