from __future__ import annotations

import json
from typing import Any

try:
    from .schemas import CaseScore, EvaluationCase, ScoringDimension, ScoringRuleSet
    from .feature_extractor import extract_features
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from schemas import CaseScore, EvaluationCase, ScoringDimension, ScoringRuleSet
    from feature_extractor import extract_features


def score_case(case: EvaluationCase, rule_set: ScoringRuleSet, use_llm_scorer: bool = False) -> CaseScore:
    if use_llm_scorer:
        try:
            return _score_case_with_llm(case, rule_set)
        except Exception:
            pass
    return _score_case_with_features(case, rule_set)


def _score_case_with_features(case: EvaluationCase, rule_set: ScoringRuleSet) -> CaseScore:
    features = extract_features(case)
    dimension_scores = []
    weighted_total = 0.0
    total_weight = 0.0
    for dimension in rule_set.dimensions:
        score, rationale = _score_dimension(dimension, features)
        dimension_scores.append(
            {
                "dimension": dimension.name,
                "score": score,
                "rationale": rationale,
                "weight": dimension.weight,
            }
        )
        weighted_total += score * dimension.weight
        total_weight += dimension.weight
    quality_score = weighted_total / total_weight if total_weight else 0.0
    return CaseScore(
        case_id=case.case_id,
        rule_set_id=rule_set.rule_set_id,
        trajectory_quality_score=round(quality_score, 4),
        dimension_scores=dimension_scores,
        target_metrics=case.scoring_metrics,
    )


def _score_case_with_llm(case: EvaluationCase, rule_set: ScoringRuleSet) -> CaseScore:
    from predict_bench.services.llm import OpenAICompatibleLLMClient

    client = OpenAICompatibleLLMClient()
    system = (
        "You are a forecast-agent trajectory evaluator. Score process quality only. "
        "Do not infer from resolved answers, Brier score, accuracy, or any outcome labels. "
        "Return JSON only."
    )
    user = (
        "Scoring rule set:\n"
        f"{json.dumps(_rule_set_to_dict(rule_set), ensure_ascii=False)}\n\n"
        "Prediction case visible to the agent evaluator:\n"
        f"{json.dumps(_case_visible_payload(case), ensure_ascii=False)}\n\n"
        "Score every dimension using only 1, 3, or 5. Each rationale must cite observable trajectory, "
        "evidence, or report behavior. Return shape: "
        "{\"dimension_scores\": [{\"dimension\": str, \"score\": 1|3|5, \"rationale\": str}], "
        "\"trajectory_quality_score\": number, \"failure_modes\": [str]}."
    )
    parsed = client.generate_json(system=system, user=user)
    raw_scores = parsed.get("dimension_scores", [])
    by_dimension = {str(item.get("dimension")): item for item in raw_scores if isinstance(item, dict)}
    dimension_scores = []
    weighted_total = 0.0
    total_weight = 0.0
    for dimension in rule_set.dimensions:
        raw = by_dimension.get(dimension.name, {})
        score = _coerce_llm_score(raw.get("score", 1))
        rationale = str(raw.get("rationale") or "LLM scorer did not provide a rationale for this dimension.")
        dimension_scores.append(
            {
                "dimension": dimension.name,
                "score": score,
                "rationale": rationale,
                "weight": dimension.weight,
            }
        )
        weighted_total += score * dimension.weight
        total_weight += dimension.weight
    quality_score = weighted_total / total_weight if total_weight else 0.0
    return CaseScore(
        case_id=case.case_id,
        rule_set_id=rule_set.rule_set_id,
        trajectory_quality_score=round(quality_score, 4),
        dimension_scores=dimension_scores,
        target_metrics=case.scoring_metrics,
    )


def _coerce_llm_score(value: Any) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return 1
    if numeric >= 5:
        return 5
    if numeric >= 3:
        return 3
    return 1


def _rule_set_to_dict(rule_set: ScoringRuleSet) -> dict[str, Any]:
    return {
        "rule_set_id": rule_set.rule_set_id,
        "domain": rule_set.domain,
        "description": rule_set.description,
        "dimensions": [dimension.__dict__ for dimension in rule_set.dimensions],
    }


def _case_visible_payload(case: EvaluationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "question": case.question,
        "candidate_probabilities": case.candidate_probabilities,
        "evidence_items": _truncate_jsonable(case.evidence_items, 12000),
        "sub_agent_results": _truncate_jsonable(case.sub_agent_results, 12000),
        "round_snapshots": _truncate_jsonable(case.round_snapshots, 6000),
        "markdown_report": case.markdown_report[:8000],
    }


def _truncate_jsonable(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return json.loads(text[:max_chars].rsplit(",", 1)[0] + "]") if text.startswith("[") else text[:max_chars]


def _score_dimension(dimension: ScoringDimension, features) -> tuple[int, str]:
    name = dimension.name.lower()
    if "question" in name or "resolv" in name or "裁定" in name:
        if features.has_probability_table and features.react_step_complete_rate >= 0.95:
            return 5, "The trajectory has a complete ReAct record and explicit probability output."
        if features.react_step_complete_rate >= 0.7:
            return 3, "The trajectory is mostly complete but settlement/probability detail is partial."
        return 1, "The trajectory lacks enough structure to verify problem understanding."
    if "retrieval" in name or "检索" in name:
        if features.source_diversity >= 3 and features.evidence_count >= 4:
            return 5, "Multiple independent sources and enough evidence items were retrieved."
        if features.source_diversity >= 1 and features.evidence_count >= 2:
            return 3, "Retrieval found relevant evidence but diversity or depth is limited."
        return 1, "Retrieval is sparse or low-diversity."
    if "source" in name or "credibility" in name or "来源" in name:
        if features.real_url_ratio >= 0.9 and features.has_direct_evidence:
            return 5, "Evidence mostly uses real URLs and includes direct evidence."
        if features.real_url_ratio >= 0.5:
            return 3, "Some real sources are present but direct/authoritative coverage is incomplete."
        return 1, "Sources are mostly synthetic, missing, or unverifiable."
    if "reasoning" in name or "推理" in name:
        if features.has_opposing_evidence and features.has_direct_evidence and features.react_step_complete_rate >= 0.95:
            return 5, "The trajectory combines direct evidence, opposing evidence, and complete ReAct reasoning."
        if features.react_step_complete_rate >= 0.8 and features.evidence_count >= 3:
            return 3, "Reasoning is auditable but conflict coverage is incomplete."
        return 1, "Reasoning is weakly connected to evidence or incomplete."
    if "probability" in name or "概率" in name:
        if features.has_probability_table and features.evidence_count >= 4 and features.cutoff_violation_count == 0:
            return 5, "The probability output is supported by enough cutoff-compliant evidence."
        if features.has_probability_table:
            return 3, "Probability output exists but evidence support is limited."
        return 1, "No clear probability output was found."
    if "tool" in name or "execution" in name or "工具" in name:
        if features.sub_agent_count >= 4 and features.react_step_complete_rate >= 0.95:
            return 5, "Expected sub-agents and tool actions are complete."
        if features.sub_agent_count >= 2 and features.react_step_complete_rate >= 0.7:
            return 3, "Some sub-agent/tool structure is present but incomplete."
        return 1, "Tool use is missing or poorly recorded."
    if features.evidence_count >= 4 and features.real_url_ratio >= 0.5:
        return 3, "Generic dimension receives a medium score from evidence depth and source quality."
    return 1, "Generic dimension lacks enough matching evidence."
