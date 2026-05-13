from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.db.database import db
from app.routers.api import router as api_router


app = FastAPI(title="AI 预测智能体", version="0.1.0")
app.include_router(api_router)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    db.init_db()
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")
