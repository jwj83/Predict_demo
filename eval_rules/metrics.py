from __future__ import annotations

import math


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def validation_score(
    quality_scores: list[float],
    resolved_probs: list[float],
    brier_scores: list[float],
    accuracies: list[float],
) -> tuple[float, float, float, float]:
    corr_resolved = pearson(quality_scores, resolved_probs)
    corr_brier = pearson(quality_scores, [-score for score in brier_scores])
    corr_accuracy = pearson(quality_scores, accuracies)
    score = (0.6 * corr_brier) + (0.3 * corr_resolved) + (0.1 * corr_accuracy)
    return score, corr_resolved, corr_brier, corr_accuracy
