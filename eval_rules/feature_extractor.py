from __future__ import annotations

from urllib.parse import urlparse

try:
    from .schemas import EvaluationCase, TrajectoryFeatures
except ImportError:  # pragma: no cover - supports direct script/test execution.
    from schemas import EvaluationCase, TrajectoryFeatures


def extract_features(case: EvaluationCase) -> TrajectoryFeatures:
    evidence_items = case.evidence_items
    urls = [str(item.get("source_url", "")) for item in evidence_items if item.get("source_url")]
    real_urls = [url for url in urls if _is_real_url(url)]
    source_domains = {_source_domain(url) for url in real_urls}
    all_steps = [
        step
        for agent in case.sub_agent_results
        for step in agent.get("trajectory", [])
        if isinstance(step, dict)
    ]
    complete_steps = [
        step for step in all_steps if step.get("thought") and step.get("action") and step.get("observation")
    ]
    complete_rate = len(complete_steps) / len(all_steps) if all_steps else 0.0
    markdown = case.markdown_report.lower()
    return TrajectoryFeatures(
        evidence_count=len(evidence_items),
        real_url_ratio=len(real_urls) / len(urls) if urls else 0.0,
        has_direct_evidence=any(item.get("evidence_role") == "direct" for item in evidence_items),
        has_opposing_evidence=any(item.get("stance") == "oppose" or item.get("evidence_role") == "opposing" for item in evidence_items),
        cutoff_violation_count=sum(1 for item in evidence_items if item.get("cutoff_compliant") is False),
        react_step_complete_rate=complete_rate,
        sub_agent_count=len(case.sub_agent_results),
        has_monitoring="monitor" in markdown or "监控" in case.markdown_report,
        has_probability_table="probability" in markdown or "概率" in case.markdown_report,
        source_diversity=len(source_domains),
    )


def _is_real_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc and parsed.netloc != "example.com"


def _source_domain(url: str) -> str:
    return urlparse(url).netloc.lower()
