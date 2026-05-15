from __future__ import annotations

import json
import os
from typing import Any

try:
    from .schemas import ScoringDimension, ScoringRuleSet
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from schemas import ScoringDimension, ScoringRuleSet


class RuleCandidateGenerator:
    def generate(
        self,
        domain: str,
        current_best: ScoringRuleSet,
        feedback: str,
        round_index: int,
        candidates_per_round: int,
    ) -> list[ScoringRuleSet]:
        if _llm_enabled():
            try:
                return self._generate_with_llm(domain, current_best, feedback, round_index, candidates_per_round)
            except Exception:
                pass
        return self._generate_mock(domain, current_best, feedback, round_index, candidates_per_round)

    def _generate_with_llm(
        self,
        domain: str,
        current_best: ScoringRuleSet,
        feedback: str,
        round_index: int,
        candidates_per_round: int,
    ) -> list[ScoringRuleSet]:
        from predict_bench.services.llm import OpenAICompatibleLLMClient

        client = OpenAICompatibleLLMClient()
        system = (
            "You design forecast-agent trajectory scoring rubrics. Return JSON only. "
            "Each candidate must have exactly 20 observable dimensions with 5/3/1 scoring criteria."
        )
        user = (
            f"Domain: {domain}\n"
            f"Round: {round_index}\n"
            f"Current best rule set: {json.dumps(_rule_set_to_dict(current_best), ensure_ascii=False)}\n"
            f"Feedback from validation: {feedback}\n\n"
            f"Generate {candidates_per_round} improved candidate rule sets. "
            "Adapt the generic rules to the domain while keeping them scoreable from ReAct trajectories, "
            "evidence items, URLs, cutoff compliance, probability outputs, and reports. "
            "Each candidate must contain exactly 20 dimensions. Do not use resolved answers, Brier score, "
            "accuracy, or other outcome labels as scoring inputs.\n\n"
            "Return shape: {\"candidates\": [{\"rule_set_id\": str, \"description\": str, "
            "\"dimensions\": [{\"name\": str, \"description\": str, \"score_5\": str, \"score_3\": str, "
            "\"score_1\": str, \"weight\": number}]}]}"
        )
        parsed = client.generate_json(system=system, user=user)
        candidates = []
        for index, raw in enumerate(parsed.get("candidates", [])[:candidates_per_round]):
            candidates.append(_rule_set_from_dict(raw, domain, current_best.rule_set_id, round_index, index))
        return candidates or self._generate_mock(domain, current_best, feedback, round_index, candidates_per_round)

    def _generate_mock(
        self,
        domain: str,
        current_best: ScoringRuleSet,
        feedback: str,
        round_index: int,
        candidates_per_round: int,
    ) -> list[ScoringRuleSet]:
        del feedback
        candidates = []
        for idx in range(candidates_per_round):
            dimensions = [_adapt_dimension(dimension, domain, idx) for dimension in current_best.dimensions]
            if domain in {"politics", "politics_governance", "governance"} and idx == 0:
                existing_names = {dimension.name for dimension in dimensions}
                dimensions.extend(dimension for dimension in _politics_dimensions() if dimension.name not in existing_names)
            dimensions = _ensure_twenty_dimensions(dimensions, domain)
            candidates.append(
                ScoringRuleSet(
                    rule_set_id=f"round_{round_index}_candidate_{idx + 1}",
                    domain=domain,
                    description=f"Mock domain-adapted candidate {idx + 1} for {domain}.",
                    dimensions=dimensions,
                    parent_id=current_best.rule_set_id,
                )
            )
        return candidates


def _llm_enabled() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _adapt_dimension(dimension: ScoringDimension, domain: str, idx: int) -> ScoringDimension:
    weight_boost = 0.25 if idx == 0 and ("source" in dimension.name.lower() or "retrieval" in dimension.name.lower()) else 0.0
    return ScoringDimension(
        name=dimension.name,
        description=f"{dimension.description} Domain adaptation target: {domain}.",
        score_5=dimension.score_5,
        score_3=dimension.score_3,
        score_1=dimension.score_1,
        weight=round(dimension.weight + weight_boost, 2),
    )


def _politics_dimensions() -> list[ScoringDimension]:
    return [
        ScoringDimension(
            name="Official institution sourcing",
            description="Whether the trajectory traces claims to election bodies, courts, agencies, legislation, or official schedules.",
            score_5="Uses primary institutions or direct legal/procedural records.",
            score_3="Uses credible media summaries but limited primary institutional evidence.",
            score_1="Relies on commentary or campaign claims without institutional verification.",
            weight=1.4,
        ),
        ScoringDimension(
            name="Statement versus execution distinction",
            description="Whether the agent separates rhetoric, pledges, media narratives, legal acts, and executed government decisions.",
            score_5="Clearly distinguishes public statements from binding actions or official execution.",
            score_3="Mentions the distinction but inconsistently applies it.",
            score_1="Treats rhetoric or headlines as resolved political action.",
            weight=1.2,
        ),
        ScoringDimension(
            name="Procedural and timing feasibility",
            description="Whether the trajectory checks deadlines, legal thresholds, vote counts, procedural calendars, and remaining feasible actions.",
            score_5="Uses concrete procedural constraints and timing feasibility in probability reasoning.",
            score_3="Mentions timing or procedure but lacks quantitative/procedural grounding.",
            score_1="Ignores legal/procedural feasibility.",
            weight=1.3,
        ),
    ]


def _ensure_twenty_dimensions(dimensions: list[ScoringDimension], domain: str) -> list[ScoringDimension]:
    expanded = list(dimensions)
    existing_names = {dimension.name for dimension in expanded}
    for dimension in _generic_expansion_dimensions(domain):
        if len(expanded) >= 20:
            break
        if dimension.name not in existing_names:
            expanded.append(dimension)
            existing_names.add(dimension.name)
    return expanded[:20]


def _generic_expansion_dimensions(domain: str) -> list[ScoringDimension]:
    names = [
        "Settlement rule extraction",
        "Entity and option grounding",
        "Temporal boundary compliance",
        "Official source prioritization",
        "Independent source diversity",
        "Direct evidence identification",
        "Supporting evidence coverage",
        "Opposing evidence coverage",
        "Conflict resolution",
        "Evidence recency",
        "Cutoff violation avoidance",
        "Quantitative anchor use",
        "Probability calibration rationale",
        "Residual uncertainty allocation",
        "Counterfactual fragility",
        "Monitoring actionability",
        "Tool call necessity",
        "ReAct trajectory completeness",
        "Report auditability",
        "Domain-specific procedural reasoning",
    ]
    return [
        ScoringDimension(
            name=name,
            description=f"Evaluate {name.lower()} for {domain} forecast trajectories using only trajectory, evidence, and report artifacts.",
            score_5=f"Excellent {name.lower()} with explicit, source-backed, cutoff-compliant support.",
            score_3=f"Partial {name.lower()} with some gaps or weak traceability.",
            score_1=f"Missing or unreliable {name.lower()}.",
            weight=1.0,
        )
        for name in names
    ]


def _rule_set_to_dict(rule_set: ScoringRuleSet) -> dict[str, Any]:
    return {
        "rule_set_id": rule_set.rule_set_id,
        "domain": rule_set.domain,
        "description": rule_set.description,
        "dimensions": [dimension.__dict__ for dimension in rule_set.dimensions],
    }


def _rule_set_from_dict(raw: dict[str, Any], domain: str, parent_id: str, round_index: int, index: int) -> ScoringRuleSet:
    dimensions = []
    for item in raw.get("dimensions", []):
        dimensions.append(
            ScoringDimension(
                name=str(item.get("name", "Unnamed dimension")),
                description=str(item.get("description", "")),
                score_5=str(item.get("score_5", "Excellent")),
                score_3=str(item.get("score_3", "Medium")),
                score_1=str(item.get("score_1", "Poor")),
                weight=float(item.get("weight", 1.0)),
            )
        )
    dimensions = _ensure_twenty_dimensions(dimensions, domain)
    return ScoringRuleSet(
        rule_set_id=str(raw.get("rule_set_id") or f"round_{round_index}_candidate_{index + 1}"),
        domain=domain,
        description=str(raw.get("description", "")),
        dimensions=dimensions,
        parent_id=parent_id,
    )
