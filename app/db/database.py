from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.core.config import DB_PATH


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_db()

    @contextmanager
    def connect(self) -> Any:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    external_id TEXT,
                    category TEXT NOT NULL,
                    domain TEXT,
                    source TEXT,
                    source_url TEXT,
                    question_text TEXT NOT NULL,
                    resolution_date TEXT NOT NULL,
                    resolution_rule TEXT,
                    event_type TEXT,
                    event_status TEXT,
                    as_of_date TEXT,
                    timezone TEXT NOT NULL,
                    candidate_options TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    original_payload TEXT
                );

                CREATE TABLE IF NOT EXISTS forecast_runs (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    run_status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    progress_stage TEXT NOT NULL,
                    error TEXT,
                    trace_summary TEXT NOT NULL,
                    latest_probabilities TEXT,
                    latest_evidence_summary TEXT,
                    FOREIGN KEY(question_id) REFERENCES questions(id)
                );

                CREATE TABLE IF NOT EXISTS forecast_results (
                    run_id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    question TEXT NOT NULL,
                    direct_answer TEXT NOT NULL,
                    confidence_level TEXT NOT NULL,
                    confidence_rationale TEXT NOT NULL,
                    evidence_basis TEXT NOT NULL,
                    candidate_probabilities TEXT NOT NULL,
                    counterfactual_fragility TEXT NOT NULL,
                    conflict_summary TEXT NOT NULL,
                    evidence_items TEXT,
                    round_snapshots TEXT,
                    monitoring_items TEXT,
                    report_quality_notes TEXT,
                    sub_agent_results TEXT,
                    markdown_report TEXT,
                    FOREIGN KEY(question_id) REFERENCES questions(id),
                    FOREIGN KEY(run_id) REFERENCES forecast_runs(id)
                );

                CREATE TABLE IF NOT EXISTS resolution_records (
                    question_id TEXT PRIMARY KEY,
                    selected_run_id TEXT NOT NULL,
                    resolved_answer TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    scoring_metrics TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES questions(id),
                    FOREIGN KEY(selected_run_id) REFERENCES forecast_runs(id)
                );
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        self._ensure_columns(
            conn,
            "questions",
            {
                "external_id": "TEXT",
                "domain": "TEXT",
                "source": "TEXT",
                "source_url": "TEXT",
                "resolution_rule": "TEXT",
                "event_type": "TEXT",
                "event_status": "TEXT",
                "as_of_date": "TEXT",
                "original_payload": "TEXT",
            },
        )
        self._ensure_columns(
            conn,
            "forecast_results",
            {
                "evidence_items": "TEXT",
                "round_snapshots": "TEXT",
                "monitoring_items": "TEXT",
                "report_quality_notes": "TEXT",
                "sub_agent_results": "TEXT",
                "markdown_report": "TEXT",
            },
        )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, column_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")

    def create_question(
        self,
        category: str,
        question_text: str,
        resolution_date: str,
        timezone_name: str,
        candidate_options: list[str],
    ) -> str:
        question_id = str(uuid.uuid4())
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO questions (id, category, question_text, resolution_date, timezone, candidate_options, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    category,
                    question_text,
                    resolution_date,
                    timezone_name,
                    json.dumps(candidate_options, ensure_ascii=False),
                    "draft",
                    utcnow_iso(),
                ),
            )
        return question_id

    def import_benchmark_event(
        self,
        event_payload: dict[str, Any],
        candidate_options: list[str],
        cutoff_iso: str,
    ) -> tuple[str, str]:
        external_id = event_payload["id"]
        with self._lock, self.connect() as conn:
            existing = conn.execute("SELECT id, status FROM questions WHERE external_id = ?", (external_id,)).fetchone()
            if existing:
                return existing["id"], existing["status"]

            question_id = str(uuid.uuid4())
            status = "resolved" if event_payload["event_status"].lower() == "resolved" else "draft"
            conn.execute(
                """
                INSERT INTO questions (
                    id, external_id, category, domain, source, source_url, question_text, resolution_date,
                    resolution_rule, event_type, event_status, as_of_date, timezone, candidate_options,
                    status, created_at, original_payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    external_id,
                    event_payload.get("domain") or event_payload.get("category") or "benchmark",
                    event_payload["domain"],
                    event_payload["source"],
                    event_payload["source_url"],
                    event_payload["question"],
                    event_payload["resolution_date"],
                    event_payload["resolution_rule"],
                    event_payload["event_type"],
                    event_payload["event_status"],
                    cutoff_iso,
                    event_payload.get("timezone") or "UTC",
                    json.dumps(candidate_options, ensure_ascii=False),
                    status,
                    event_payload.get("created_at") or utcnow_iso(),
                    json.dumps(event_payload, ensure_ascii=False),
                ),
            )
        return question_id, status

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
        return self._row_to_question(row) if row else None

    def list_questions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM questions ORDER BY created_at DESC").fetchall()
        return [self._row_to_question(row) for row in rows]

    def update_question_status(self, question_id: str, status: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("UPDATE questions SET status = ? WHERE id = ?", (status, question_id))

    def create_run(self, question_id: str) -> str:
        run_id = str(uuid.uuid4())
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO forecast_runs (id, question_id, run_status, started_at, finished_at, progress_stage, error, trace_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, question_id, "running", utcnow_iso(), None, "queued", None, "[]"),
            )
        return run_id

    def append_run_trace(
        self,
        run_id: str,
        progress_stage: str,
        trace_entry: dict[str, Any],
        latest_probabilities: list[dict[str, Any]] | None = None,
        latest_evidence_summary: str | None = None,
    ) -> None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT trace_summary FROM forecast_runs WHERE id = ?", (run_id,)).fetchone()
            trace = json.loads(row["trace_summary"]) if row and row["trace_summary"] else []
            trace.append(trace_entry)
            conn.execute(
                """
                UPDATE forecast_runs
                SET progress_stage = ?, trace_summary = ?, latest_probabilities = ?, latest_evidence_summary = ?
                WHERE id = ?
                """,
                (
                    progress_stage,
                    json.dumps(trace, ensure_ascii=False),
                    json.dumps(latest_probabilities, ensure_ascii=False) if latest_probabilities is not None else None,
                    latest_evidence_summary,
                    run_id,
                ),
            )

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                UPDATE forecast_runs
                SET run_status = ?, finished_at = ?, error = ?, progress_stage = ?
                WHERE id = ?
                """,
                (status, utcnow_iso(), error, "finished" if status == "completed" else "failed", run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM forecast_runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs_for_question(self, question_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forecast_runs WHERE question_id = ? ORDER BY started_at DESC",
                (question_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def save_result(self, run_id: str, question_id: str, payload: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_results (
                    run_id, question_id, prediction_date, question, direct_answer, confidence_level,
                    confidence_rationale, evidence_basis, candidate_probabilities, counterfactual_fragility,
                    conflict_summary, evidence_items, round_snapshots, monitoring_items, report_quality_notes,
                    sub_agent_results, markdown_report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    question_id,
                    payload["prediction_date"],
                    payload["question"],
                    payload["direct_answer"],
                    payload["confidence_level"],
                    payload["confidence_rationale"],
                    payload["evidence_basis"],
                    json.dumps(payload["candidate_probabilities"], ensure_ascii=False),
                    payload["counterfactual_fragility"],
                    payload["conflict_summary"],
                    json.dumps(payload.get("evidence_items", []), ensure_ascii=False),
                    json.dumps(payload.get("round_snapshots", []), ensure_ascii=False),
                    json.dumps(payload.get("monitoring_items", []), ensure_ascii=False),
                    json.dumps(payload.get("report_quality_notes", {}), ensure_ascii=False),
                    json.dumps(payload.get("sub_agent_results", []), ensure_ascii=False),
                    payload.get("markdown_report", ""),
                ),
            )

    def get_latest_result(self, question_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT r.* FROM forecast_results r
                JOIN forecast_runs fr ON fr.id = r.run_id
                WHERE r.question_id = ? AND fr.run_status = 'completed'
                ORDER BY fr.finished_at DESC
                LIMIT 1
                """,
                (question_id,),
            ).fetchone()
        return self._row_to_result(row) if row else None

    def get_result_by_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM forecast_results WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_result(row) if row else None

    def save_resolution(
        self,
        question_id: str,
        selected_run_id: str,
        resolved_answer: str,
        scoring_metrics: dict[str, Any],
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO resolution_records
                (question_id, selected_run_id, resolved_answer, resolved_at, scoring_metrics)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    selected_run_id,
                    resolved_answer,
                    utcnow_iso(),
                    json.dumps(scoring_metrics, ensure_ascii=False),
                ),
            )

    def get_benchmark_resolution_answer(self, question_id: str) -> str | None:
        question = self.get_question(question_id)
        if not question:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT original_payload FROM questions WHERE id = ?", (question_id,)).fetchone()
        if not row or not row["original_payload"]:
            return None
        payload = json.loads(row["original_payload"])
        if str(payload.get("event_status", "")).lower() != "resolved":
            return None
        resolved_answer = payload.get("resolved_answer")
        if resolved_answer in question["candidate_options"]:
            return resolved_answer
        return None

    def get_resolution(self, question_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM resolution_records WHERE question_id = ?", (question_id,)).fetchone()
        if not row:
            return None
        return {
            "question_id": row["question_id"],
            "selected_run_id": row["selected_run_id"],
            "resolved_answer": row["resolved_answer"],
            "resolved_at": row["resolved_at"],
            "scoring_metrics": json.loads(row["scoring_metrics"]),
        }

    def _row_to_question(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "external_id": row["external_id"],
            "category": row["category"],
            "domain": row["domain"] or row["category"],
            "source": row["source"] or "",
            "source_url": row["source_url"] or "",
            "question_text": row["question_text"],
            "resolution_date": row["resolution_date"],
            "resolution_rule": row["resolution_rule"] or "",
            "event_type": row["event_type"] or "",
            "event_status": row["event_status"] or row["status"],
            "as_of_date": row["as_of_date"],
            "timezone": row["timezone"],
            "candidate_options": json.loads(row["candidate_options"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def _row_to_run(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["id"],
            "question_id": row["question_id"],
            "run_status": row["run_status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "progress_stage": row["progress_stage"],
            "error": row["error"],
            "trace_summary": json.loads(row["trace_summary"]) if row["trace_summary"] else [],
            "latest_probabilities": json.loads(row["latest_probabilities"]) if row["latest_probabilities"] else [],
            "latest_evidence_summary": row["latest_evidence_summary"] or "",
        }

    def _row_to_result(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "question_id": row["question_id"],
            "prediction_date": row["prediction_date"],
            "question": row["question"],
            "direct_answer": row["direct_answer"],
            "confidence_level": row["confidence_level"],
            "confidence_rationale": row["confidence_rationale"],
            "evidence_basis": row["evidence_basis"],
            "candidate_probabilities": json.loads(row["candidate_probabilities"]),
            "counterfactual_fragility": row["counterfactual_fragility"],
            "conflict_summary": row["conflict_summary"],
            "evidence_items": json.loads(row["evidence_items"]) if row["evidence_items"] else [],
            "round_snapshots": json.loads(row["round_snapshots"]) if row["round_snapshots"] else [],
            "monitoring_items": json.loads(row["monitoring_items"]) if row["monitoring_items"] else [],
            "report_quality_notes": json.loads(row["report_quality_notes"]) if row["report_quality_notes"] else {},
            "sub_agent_results": json.loads(row["sub_agent_results"]) if row["sub_agent_results"] else [],
            "markdown_report": row["markdown_report"] or "",
        }


db = Database(str(DB_PATH))
