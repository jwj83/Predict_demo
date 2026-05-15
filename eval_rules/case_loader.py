from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .schemas import EvaluationCase
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from schemas import EvaluationCase


def load_cases_jsonl(path: str | Path, domain: str | None = None, limit: int | None = None) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            case = case_from_dict(payload)
            if domain and case.domain != domain:
                continue
            cases.append(case)
            if limit and len(cases) >= limit:
                break
    return cases


def case_from_dict(payload: dict[str, Any]) -> EvaluationCase:
    return EvaluationCase(
        case_id=str(payload["case_id"]),
        domain=str(payload.get("domain", "generic")),
        question=str(payload.get("question", "")),
        resolved_answer=payload.get("resolved_answer"),
        candidate_probabilities=list(payload.get("candidate_probabilities", [])),
        scoring_metrics={key: float(value) for key, value in payload.get("scoring_metrics", {}).items()},
        evidence_items=list(payload.get("evidence_items", [])),
        sub_agent_results=list(payload.get("sub_agent_results", [])),
        round_snapshots=list(payload.get("round_snapshots", [])),
        markdown_report=str(payload.get("markdown_report", "")),
    )


def load_default_sample_cases(domain: str | None = None) -> list[EvaluationCase]:
    sample_path = Path(__file__).resolve().parent / "data" / "sample_cases.jsonl"
    return load_cases_jsonl(sample_path, domain=domain)
