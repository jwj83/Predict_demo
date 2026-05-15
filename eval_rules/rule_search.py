from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

try:
    from .metrics import validation_score
    from .rule_generator import RuleCandidateGenerator
    from .schemas import CandidateValidation, EvaluationCase, RuleSearchResult, RuleSearchRound, ScoringRuleSet
    from .seed_rules import generic_v0_rule_set
    from .trajectory_scorer import score_case
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from metrics import validation_score
    from rule_generator import RuleCandidateGenerator
    from schemas import CandidateValidation, EvaluationCase, RuleSearchResult, RuleSearchRound, ScoringRuleSet
    from seed_rules import generic_v0_rule_set
    from trajectory_scorer import score_case


class RuleSearchRunner:
    def __init__(self, generator: RuleCandidateGenerator | None = None, use_llm_scorer: bool = False) -> None:
        self.generator = generator or RuleCandidateGenerator()
        self.use_llm_scorer = use_llm_scorer

    def run(
        self,
        cases: list[EvaluationCase],
        domain: str = "generic",
        iterations: int = 2,
        candidates_per_round: int = 3,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RuleSearchResult:
        if not cases:
            raise ValueError("Rule search requires at least one evaluation case.")
        seed = generic_v0_rule_set(domain=domain)
        current_best = seed
        rounds: list[RuleSearchRound] = []
        all_case_scores = []
        feedback = "No prior feedback. Start from the generic forecast trajectory rubric."

        for round_index in range(1, iterations + 1):
            if progress_callback:
                progress_callback(
                    {
                        "type": "round_started",
                        "domain": domain,
                        "round_index": round_index,
                        "current_best_rule_set_id": current_best.rule_set_id,
                    }
                )
            candidates = self.generator.generate(domain, current_best, feedback, round_index, candidates_per_round)
            validations: list[CandidateValidation] = []
            candidate_scores = {}
            for candidate_index, candidate in enumerate(candidates, start=1):
                scores = [score_case(case, candidate, use_llm_scorer=self.use_llm_scorer) for case in cases]
                candidate_scores[candidate.rule_set_id] = scores
                validation = _validate_candidate(candidate, scores)
                validations.append(validation)
                if progress_callback:
                    progress_callback(
                        {
                            "type": "candidate_scored",
                            "domain": domain,
                            "round_index": round_index,
                            "candidate_index": candidate_index,
                            "rule_set_id": candidate.rule_set_id,
                            "validation_score": validation.validation_score,
                            "correlation_with_resolved_probability": validation.correlation_with_resolved_probability,
                            "correlation_with_brier": validation.correlation_with_brier,
                            "correlation_with_accuracy": validation.correlation_with_accuracy,
                            "case_count": validation.case_count,
                        }
                    )

            best_validation = max(validations, key=lambda item: item.validation_score)
            current_best = next(candidate for candidate in candidates if candidate.rule_set_id == best_validation.rule_set_id)
            all_case_scores = candidate_scores[current_best.rule_set_id]
            feedback = _feedback_from_validation(best_validation, current_best)
            if progress_callback:
                progress_callback(
                    {
                        "type": "round_completed",
                        "domain": domain,
                        "round_index": round_index,
                        "best_rule_set_id": current_best.rule_set_id,
                        "validation_score": best_validation.validation_score,
                        "feedback": feedback,
                    }
                )
            rounds.append(
                RuleSearchRound(
                    round_index=round_index,
                    candidates=candidates,
                    validations=validations,
                    best_rule_set_id=current_best.rule_set_id,
                    feedback=feedback,
                )
            )

        final_validation = _validate_candidate(current_best, all_case_scores)
        return RuleSearchResult(
            domain=domain,
            seed_rule_set=seed,
            best_rule_set=current_best,
            rounds=rounds,
            case_scores=all_case_scores,
            validation_summary=final_validation,
        )


def _validate_candidate(rule_set: ScoringRuleSet, case_scores) -> CandidateValidation:
    quality_scores = [score.trajectory_quality_score for score in case_scores]
    resolved_probs = [score.target_metrics.get("resolved_option_probability", 0.0) for score in case_scores]
    brier_scores = [score.target_metrics.get("brier_score", 1.0) for score in case_scores]
    accuracies = [score.target_metrics.get("accuracy", 0.0) for score in case_scores]
    score, corr_resolved, corr_brier, corr_accuracy = validation_score(quality_scores, resolved_probs, brier_scores, accuracies)
    return CandidateValidation(
        rule_set_id=rule_set.rule_set_id,
        validation_score=round(score, 4),
        correlation_with_resolved_probability=round(corr_resolved, 4),
        correlation_with_brier=round(corr_brier, 4),
        correlation_with_accuracy=round(corr_accuracy, 4),
        case_count=len(case_scores),
    )


def _feedback_from_validation(validation: CandidateValidation, rule_set: ScoringRuleSet) -> str:
    return (
        f"{rule_set.rule_set_id} is currently best. "
        f"Validation score={validation.validation_score}, "
        f"corr(resolved probability)={validation.correlation_with_resolved_probability}, "
        f"corr(-Brier)={validation.correlation_with_brier}, "
        f"corr(accuracy)={validation.correlation_with_accuracy}. "
        "Improve dimensions that better distinguish credible evidence, cutoff compliance, and probability calibration."
    )


def result_to_dict(result: RuleSearchResult) -> dict:
    return asdict(result)
