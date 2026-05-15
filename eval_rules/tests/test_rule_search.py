from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from case_loader import load_default_sample_cases
from feature_extractor import extract_features
from rule_search import RuleSearchRunner
from seed_rules import generic_v0_rule_set
from trajectory_scorer import score_case


def test_seed_rule_has_generic_dimensions() -> None:
    rule_set = generic_v0_rule_set()
    assert rule_set.rule_set_id == "generic_v0"
    assert len(rule_set.dimensions) == 6
    assert {dimension.name for dimension in rule_set.dimensions}


def test_feature_extractor_reads_react_trace_and_sources() -> None:
    case = load_default_sample_cases()[0]
    features = extract_features(case)
    assert features.evidence_count >= 3
    assert features.real_url_ratio > 0
    assert features.react_step_complete_rate == 1.0
    assert features.has_direct_evidence


def test_trajectory_scorer_outputs_dimension_scores() -> None:
    case = load_default_sample_cases()[0]
    rule_set = generic_v0_rule_set(domain="politics_governance")
    score = score_case(case, rule_set)
    assert score.case_id == case.case_id
    assert 1.0 <= score.trajectory_quality_score <= 5.0
    assert len(score.dimension_scores) == len(rule_set.dimensions)
    assert "resolved_option_probability" in score.target_metrics


def test_rule_search_runs_two_rounds() -> None:
    cases = load_default_sample_cases()
    result = RuleSearchRunner().run(cases, domain="politics_governance", iterations=2, candidates_per_round=2)
    assert result.seed_rule_set.rule_set_id == "generic_v0"
    assert result.best_rule_set.rule_set_id.startswith("round_")
    assert len(result.rounds) == 2
    assert result.validation_summary.case_count == len(cases)
    assert result.case_scores
