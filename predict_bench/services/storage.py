from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from predict_bench.config import RUNS_DIR


class JsonStorage:
    def __init__(
        self,
        runs_dir: Path = RUNS_DIR,
        run_id: str | None = None,
    ) -> None:
        self.runs_dir = runs_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_raw(self, payload: Any, domain: str = "sources") -> Path:
        return self._save(domain, "raw", payload)

    def save_events(self, payload: Any, domain: str = "sources") -> Path:
        return self._save(domain, "events", payload)

    def save_benchmark(self, payload: Any, domain: str = "sports") -> Path:
        return self._save(domain, "benchmark", payload)

    def _save(self, domain: str, kind: str, payload: Any) -> Path:
        directory = self.runs_dir / self.run_id / domain
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{kind}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path