from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringDimension:
    name: str
    description: str
    score_5: str
    score_3: str
    score_1: str
    weight: float = 1.0


@dataclass
class ScoringRuleSet:
    rule_set_id: str
    domain: str
    description: str
    dimensions: list[ScoringDimension]
    parent_id: str | None = None


@dataclass
class EvaluationCase:
    case_id: str
    domain: str
    question: str
    resolved_answer: str | None
    candidate_probabilities: list[dict[str, Any]]
    scoring_metrics: dict[str, float]
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    sub_agent_results: list[dict[str, Any]] = field(default_factory=list)
    round_snapshots: list[dict[str, Any]] = field(default_factory=list)
    markdown_report: str = ""


@dataclass
class TrajectoryFeatures:
    evidence_count: int
    real_url_ratio: float
    has_direct_evidence: bool
    has_opposing_evidence: bool
    cutoff_violation_count: int
    react_step_complete_rate: float
    sub_agent_count: int
    has_monitoring: bool
    has_probability_table: bool
    source_diversity: int


@dataclass
class CaseScore:
    case_id: str
    rule_set_id: str
    trajectory_quality_score: float
    dimension_scores: list[dict[str, Any]]
    target_metrics: dict[str, float]


@dataclass
class CandidateValidation:
    rule_set_id: str
    validation_score: float
    correlation_with_resolved_probability: float
    correlation_with_brier: float
    correlation_with_accuracy: float
    case_count: int


@dataclass
class RuleSearchRound:
    round_index: int
    candidates: list[ScoringRuleSet]
    validations: list[CandidateValidation]
    best_rule_set_id: str
    feedback: str


@dataclass
class RuleSearchResult:
    domain: str
    seed_rule_set: ScoringRuleSet
    best_rule_set: ScoringRuleSet
    rounds: list[RuleSearchRound]
    case_scores: list[CaseScore]
    validation_summary: CandidateValidation
