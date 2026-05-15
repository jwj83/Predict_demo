from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str, prefix: str = "pb") -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


class RawNewsItem(BaseModel):
    id: str
    domain: str
    candidate_domains: list[str] = Field(default_factory=list)
    source: str
    feed_name: str
    title: str
    link: str
    description: str = ""
    published_at: str | None = None
    fetched_at: str = Field(default_factory=utc_now_iso)


class FutureEvent(BaseModel):
    id: str
    domain: str
    secondary_domains: list[str] = Field(default_factory=list)
    source: str
    source_url: str
    raw_item_id: str
    event_type: str
    event_summary: str
    entities: list[str] = Field(default_factory=list)
    deadline: date | None = None
    event_status: str = "unresolved"
    resolution_date: date | None = None
    resolved_answer: str | None = None
    answer_evidence: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class BenchmarkItem(BaseModel):
    id: str
    domain: str
    source: str
    source_url: str
    question: str
    options: list[str]
    resolution_date: date
    resolution_rule: str
    event_type: str
    event_status: str = "unresolved"
    resolved_answer: str | None = None
    answer_evidence: str | None = None
    quality_score: float | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("options")
    @classmethod
    def clean_options(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("options must be unique")
        return cleaned


class ValidationIssue(BaseModel):
    item_id: str
    reason: str


class AgentRunResult(BaseModel):
    raw_path: str
    events_path: str
    benchmark_path: str
    raw_count: int
    event_count: int
    item_count: int
    rejected: list[ValidationIssue] = Field(default_factory=list)
    items: list[BenchmarkItem] = Field(default_factory=list)


class SourceAgentRunResult(BaseModel):
    raw_path: str
    events_path: str
    benchmark_paths: dict[str, str] = Field(default_factory=dict)
    raw_count: int
    event_count: int
    item_count: int
    rejected: list[ValidationIssue] = Field(default_factory=list)
    items: list[BenchmarkItem] = Field(default_factory=list)
