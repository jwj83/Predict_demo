from __future__ import annotations


def compute_brier_score(probabilities: list[dict], resolved_answer: str) -> float:
    score = 0.0
    for item in probabilities:
        outcome = 1.0 if item["option"] == resolved_answer else 0.0
        score += (item["probability"] - outcome) ** 2
    return round(score, 6)


def compute_accuracy(probabilities: list[dict], resolved_answer: str) -> float:
    winner = max(probabilities, key=lambda item: item["probability"])["option"]
    return 1.0 if winner == resolved_answer else 0.0


def compute_confidence_gap(probabilities: list[dict], resolved_answer: str) -> float:
    resolved_probability = next(
        (item["probability"] for item in probabilities if item["option"] == resolved_answer),
        0.0,
    )
    return round(resolved_probability, 6)
