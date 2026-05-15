from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CandidateProbability(BaseModel):
    option: str
    probability: float = Field(ge=0.0, le=1.0)


class BenchmarkEventInput(BaseModel):
    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    resolution_date: date
    resolution_rule: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    event_status: str = Field(min_length=1)
    resolved_answer: str | None = None
    answer_evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime | None = None
    as_of_date: datetime | None = None

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least two non-empty options are required.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Options must be unique.")
        return cleaned


class CreateQuestionRequest(BaseModel):
    category: str = Field(min_length=1)
    question: str = Field(min_length=10)
    resolution_date: date
    timezone: str = Field(min_length=1)
    candidate_options: list[str] = Field(min_length=2)

    @field_validator("candidate_options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(cleaned) < 2:
            raise ValueError("至少需要两个非空候选项。")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("候选项不能重复。")
        return cleaned


class CreateQuestionResponse(BaseModel):
    question_id: str


class NativeForecastRequest(CreateQuestionRequest):
    wait_timeout_seconds: int = Field(default=300, ge=1, le=1800)


class ImportBenchmarkEventResponse(BaseModel):
    question_id: str
    external_id: str
    status: str


class BenchmarkGenerateRequest(BaseModel):
    domain: Literal["all", "economy", "finance", "international", "politics", "sports", "weather"] = "all"
    limit: int = Field(default=10, ge=1)
    max_items_per_feed: int = Field(default=10, ge=1)


class BenchmarkGenerateResponse(BaseModel):
    run_id: str
    total: int
    items: list[BenchmarkEventInput]
    output_paths: dict[str, str]
    domain_counts: dict[str, int]
    status_counts: dict[str, int]


class ForecastRunResponse(BaseModel):
    run_id: str
    status: str


class TraceEntry(BaseModel):
    round_index: int
    role: str
    thought_summary: str
    action: str
    observation_summary: str


class RunStatusResponse(BaseModel):
    run_id: str
    question_id: str
    run_status: str
    progress_stage: str
    started_at: str
    finished_at: str | None
    error: str | None
    trace_summary: list[TraceEntry]
    latest_probabilities: list[CandidateProbability]
    latest_evidence_summary: str


class EvidenceItem(BaseModel):
    claim: str
    supports_option: str
    stance: Literal["support", "oppose", "neutral"]
    strength: float = Field(ge=0.0, le=1.0)
    source_url: str
    source_title: str = ""
    source_excerpt_summary: str
    published_at: str
    cutoff_compliant: bool
    recency_score: float = Field(ge=0.0, le=1.0)
    evidence_role: Literal["direct", "supporting", "opposing", "context"] = "supporting"
    rationale: str = ""


class ReActStep(BaseModel):
    t: int
    thought: str
    action: dict[str, Any]
    observation: str


class LocalConclusion(BaseModel):
    favored_option: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_findings: list[str] = []
    conflicts: list[str] = []
    information_gaps: list[str] = []


class SubAgentResult(BaseModel):
    agent_id: str
    mission: str
    trajectory: list[ReActStep]
    evidence_items: list[EvidenceItem]
    local_conclusion: LocalConclusion


class RoundSnapshot(BaseModel):
    round_index: int
    probabilities: list[CandidateProbability]
    conflict_summary: str
    evidence_count: int
    stop_reason: str | None = None


class ReportQualityNotes(BaseModel):
    evidence_detail: str
    probability_rigor: str
    counterfactual_completeness: str
    monitoring_plan: str


class ForecastResultResponse(BaseModel):
    prediction_date: datetime
    question: str
    direct_answer: str
    confidence_level: Literal["low", "medium", "high"]
    confidence_rationale: str
    evidence_basis: str
    candidate_probabilities: list[CandidateProbability]
    counterfactual_fragility: Literal["low", "medium", "high"]
    conflict_summary: str
    evidence_items: list[EvidenceItem] = []
    round_snapshots: list[RoundSnapshot] = []
    monitoring_items: list[str] = []
    report_quality_notes: ReportQualityNotes | dict[str, Any] = Field(default_factory=dict)
    sub_agent_results: list[SubAgentResult] = []
    markdown_report: str = ""


class ForecastApiEvidence(BaseModel):
    claim: str
    role: str
    stance: str
    supports_option: str | None = None
    strength: float | None = None
    source_url: str
    source_title: str | None = None
    published_at: datetime | None = None
    cutoff_compliant: bool = True
    why_important: str


class ForecastApiQualityAssessment(BaseModel):
    evidence_granularity: str
    probability_rigor: str
    counterfactual_completeness: str
    actionable_monitoring: str


class ForecastApiReportResponse(BaseModel):
    prediction_date: date
    question: str
    direct_answer: str
    confidence_level: Literal["low", "medium", "high"]
    candidate_probability_table: list[CandidateProbability]
    confidence_basis: str
    evidence_details: list[ForecastApiEvidence]
    counterfactual_fragility: str
    monitoring_items: list[str]
    report_quality_assessment: ForecastApiQualityAssessment
    markdown_report: str


class NativeForecastResponse(BaseModel):
    question_id: str
    run_id: str
    status: Literal["completed"]
    report: ForecastApiReportResponse


class ResolutionRequest(BaseModel):
    resolved_answer: str = Field(min_length=1)
    run_id: str | None = None


class EvaluationResponse(BaseModel):
    question_id: str
    resolved_answer: str
    resolved_at: datetime
    selected_run_id: str
    scoring_metrics: dict


class RuleSearchRequest(BaseModel):
    domains: list[Literal["all", "economy", "finance", "international", "politics", "sports", "weather"]] = Field(default_factory=lambda: ["all"])
    case_ids: list[str] = Field(default_factory=list)
    cases_per_domain: int | None = Field(default=None, ge=1)
    iterations: int = Field(default=2, ge=1)
    candidates_per_round: int = Field(default=3, ge=1)


class RuleSearchCaseSummary(BaseModel):
    case_id: str
    domain: str
    question: str
    scoring_metrics: dict


class RuleSearchCaseListResponse(BaseModel):
    cases: list[RuleSearchCaseSummary]
    domain_counts: dict[str, int]


class RuleSearchDomainResult(BaseModel):
    case_count: int
    domain: str
    best_rule_set_id: str
    validation_score: float
    correlation_with_resolved_probability: float
    correlation_with_brier: float
    correlation_with_accuracy: float
    report_path: str


class RuleSearchSkippedDomain(BaseModel):
    domain: str
    reason: str


class RuleSearchResponse(BaseModel):
    mode: str
    total_case_count: int
    results: list[RuleSearchDomainResult]
    skipped: list[RuleSearchSkippedDomain] = Field(default_factory=list)


class RuleSearchJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress_events: list[dict[str, Any]]
    best_by_round: list[dict[str, Any]]
    result: RuleSearchResponse | None = None
    error: str | None = None


class QuestionListItem(BaseModel):
    id: str
    category: str
    question_text: str
    resolution_date: date
    timezone: str
    candidate_options: list[str]
    status: str
    created_at: datetime
