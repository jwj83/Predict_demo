from __future__ import annotations

try:
    from .schemas import ScoringDimension, ScoringRuleSet
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from schemas import ScoringDimension, ScoringRuleSet


def generic_v0_rule_set(domain: str = "generic") -> ScoringRuleSet:
    return ScoringRuleSet(
        rule_set_id="generic_v0",
        domain=domain,
        description=(
            "Generic forecast trajectory rubric. It evaluates whether a ReAct forecasting agent "
            "understands a resolvable event, retrieves reliable evidence, reasons faithfully, "
            "updates probabilities, and uses tools efficiently."
        ),
        dimensions=[
            ScoringDimension(
                name="Question understanding and resolvability",
                description="Whether the agent identifies event subject, time, place, conditions, and settlement criteria.",
                score_5="Clearly extracts event subject, deadline, options, settlement rule, ambiguity, and edge cases.",
                score_3="Understands the broad event but misses some boundary or settlement details.",
                score_1="Treats an open-ended or non-resolvable prompt as directly forecastable.",
                weight=1.5,
            ),
            ScoringDimension(
                name="Information retrieval quality",
                description="Whether retrieval covers multiple relevant, independent, fresh, and documented sources.",
                score_5="Uses multiple independent, relevant sources and records search/fetch paths.",
                score_3="Uses relevant sources but lacks diversity, freshness, or trace detail.",
                score_1="Uses sparse, irrelevant, duplicate, or undocumented retrieval.",
                weight=2.0,
            ),
            ScoringDimension(
                name="Source credibility and news quality",
                description="Whether evidence comes from authoritative, transparent, accurate, and independent sources.",
                score_5="Prioritizes primary/official/high-quality sources and distinguishes source independence.",
                score_3="Uses generally credible secondary sources but limited primary verification.",
                score_1="Relies on low-quality, unclear, promotional, or duplicated sources.",
                weight=2.0,
            ),
            ScoringDimension(
                name="Reasoning chain quality",
                description="Whether the agent links evidence to forecast-relevant variables and uncertainty.",
                score_5="Separates facts, assumptions, counter-evidence, causal links, and uncertainty clearly.",
                score_3="Provides plausible reasoning but weak conflict handling or causal explanation.",
                score_1="Stacks news snippets without auditable evidence-to-forecast rationale.",
                weight=2.0,
            ),
            ScoringDimension(
                name="Probability update quality",
                description="Whether probabilities are updated from evidence with calibrated magnitude and residual uncertainty.",
                score_5="Shows evidence-driven probability movement, reasonable magnitude, and remaining uncertainty.",
                score_3="Probability is connected to evidence but update magnitude is underspecified.",
                score_1="Probability appears arbitrary or jumps to extremes from weak evidence.",
                weight=1.5,
            ),
            ScoringDimension(
                name="Tool use and execution efficiency",
                description="Whether the agent uses necessary tools without wasteful or irrelevant steps.",
                score_5="Uses appropriate search/fetch/reasoning steps with complete ReAct trajectories and controlled cost.",
                score_3="Uses the right tools but has some redundant or shallow steps.",
                score_1="Misses necessary tools or performs mostly irrelevant tool calls.",
                weight=1.0,
            ),
        ],
    )
