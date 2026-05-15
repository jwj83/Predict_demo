from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import DATA_DIR
from app.db.database import db
from eval_rules.rule_search import RuleSearchRunner, result_to_dict
from eval_rules.schemas import EvaluationCase
from predict_bench.agents.question_agent import SourceFirstQuestionAgent
from predict_bench.domains import get_source_configs
from predict_bench.services.storage import JsonStorage

_RULE_SEARCH_JOBS: dict[str, dict[str, Any]] = {}
_RULE_SEARCH_LOCK = threading.Lock()


def generate_benchmark(domain: str, limit: int, max_items_per_feed: int) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sources = get_source_configs(domain=domain)
    domain_filter = None if domain == "all" else domain
    storage = JsonStorage(run_id=run_id)
    result = SourceFirstQuestionAgent(sources=sources, storage=storage).run(
        limit=limit,
        domain_filter=domain_filter,
        max_items_per_feed=max_items_per_feed,
    )
    items = [item.model_dump(mode="json") for item in result.items]
    run_dir = Path(result.raw_path).parents[1]
    benchmark_all_path = run_dir / "benchmark_all.json"
    benchmark_all_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    output_paths = {
        "benchmark_all": str(benchmark_all_path),
        "raw": result.raw_path,
        "events": result.events_path,
        **{f"benchmark_{key}": value for key, value in result.benchmark_paths.items()},
    }
    return {
        "run_id": run_id,
        "total": len(items),
        "items": items,
        "output_paths": output_paths,
        "domain_counts": dict(Counter(item["domain"] for item in items)),
        "status_counts": dict(Counter(item["event_status"] for item in items)),
    }


def list_rule_search_cases(domains: list[str] | None = None) -> dict[str, Any]:
    cases = db.list_rule_search_case_summaries(domains)
    return {
        "cases": cases,
        "domain_counts": dict(Counter(item["domain"] for item in cases)),
    }


def run_rule_search(
    domains: list[str],
    iterations: int,
    candidates_per_round: int,
    case_ids: list[str] | None = None,
    cases_per_domain: int | None = None,
) -> dict[str, Any]:
    available_domains = db.list_rule_search_domains()
    selected_domains = available_domains if "all" in domains else domains
    if not selected_domains:
        raise ValueError("No completed resolved benchmark predictions are available for rule search.")

    results = []
    skipped = []
    total_case_count = 0
    for domain in selected_domains:
        case_payloads = db.list_rule_search_cases(
            domain=domain,
            case_ids=case_ids or None,
            limit=cases_per_domain,
        )
        if not case_payloads:
            skipped.append({"domain": domain, "reason": "No eligible resolved completed samples."})
            continue
        result = _run_rule_search_for_domain(domain, case_payloads, iterations, candidates_per_round)
        total_case_count += result["case_count"]
        results.append(result)
    if not results:
        raise ValueError("No selected domains have eligible resolved completed benchmark predictions.")
    return {
        "mode": "per_domain",
        "total_case_count": total_case_count,
        "results": results,
        "skipped": skipped,
    }


def start_rule_search_job(
    domains: list[str],
    iterations: int,
    candidates_per_round: int,
    case_ids: list[str] | None = None,
    cases_per_domain: int | None = None,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    with _RULE_SEARCH_LOCK:
        _RULE_SEARCH_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": "Rule search queued.",
            "progress_events": [],
            "best_by_round": [],
            "result": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_run_rule_search_job,
        args=(job_id, domains, iterations, candidates_per_round, case_ids or [], cases_per_domain),
        daemon=True,
    )
    thread.start()
    return get_rule_search_job(job_id)


def get_rule_search_job(job_id: str) -> dict[str, Any]:
    with _RULE_SEARCH_LOCK:
        job = _RULE_SEARCH_JOBS.get(job_id)
        if not job:
            raise ValueError("Rule search job not found.")
        return json.loads(json.dumps(job, ensure_ascii=False))


def _run_rule_search_job(
    job_id: str,
    domains: list[str],
    iterations: int,
    candidates_per_round: int,
    case_ids: list[str],
    cases_per_domain: int | None,
) -> None:
    def progress(event: dict[str, Any]) -> None:
        with _RULE_SEARCH_LOCK:
            job = _RULE_SEARCH_JOBS[job_id]
            job["progress_events"].append(event)
            if event["type"] == "candidate_scored":
                job["message"] = (
                    f"{event['domain']} round {event['round_index']} candidate "
                    f"{event['candidate_index']} scored {event['validation_score']}."
                )
            if event["type"] == "round_completed":
                job["best_by_round"].append(event)
                job["message"] = (
                    f"{event['domain']} round {event['round_index']} best rule "
                    f"{event['best_rule_set_id']} scored {event['validation_score']}."
                )

    try:
        result = _run_rule_search_with_progress(
            domains=domains,
            iterations=iterations,
            candidates_per_round=candidates_per_round,
            case_ids=case_ids,
            cases_per_domain=cases_per_domain,
            progress_callback=progress,
        )
        with _RULE_SEARCH_LOCK:
            _RULE_SEARCH_JOBS[job_id]["status"] = "completed"
            _RULE_SEARCH_JOBS[job_id]["message"] = "Rule search completed."
            _RULE_SEARCH_JOBS[job_id]["result"] = result
    except Exception as exc:
        with _RULE_SEARCH_LOCK:
            _RULE_SEARCH_JOBS[job_id]["status"] = "failed"
            _RULE_SEARCH_JOBS[job_id]["message"] = "Rule search failed."
            _RULE_SEARCH_JOBS[job_id]["error"] = str(exc)


def _run_rule_search_with_progress(
    domains: list[str],
    iterations: int,
    candidates_per_round: int,
    case_ids: list[str] | None = None,
    cases_per_domain: int | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    available_domains = db.list_rule_search_domains()
    selected_domains = available_domains if "all" in domains else domains
    if not selected_domains:
        raise ValueError("No completed resolved benchmark predictions are available for rule search.")

    results = []
    skipped = []
    total_case_count = 0
    for domain in selected_domains:
        case_payloads = db.list_rule_search_cases(
            domain=domain,
            case_ids=case_ids or None,
            limit=cases_per_domain,
        )
        if not case_payloads:
            skipped.append({"domain": domain, "reason": "No eligible resolved completed samples."})
            continue
        result = _run_rule_search_for_domain(
            domain,
            case_payloads,
            iterations,
            candidates_per_round,
            progress_callback=progress_callback,
        )
        total_case_count += result["case_count"]
        results.append(result)
    if not results:
        raise ValueError("No selected domains have eligible resolved completed benchmark predictions.")
    return {
        "mode": "per_domain",
        "total_case_count": total_case_count,
        "results": results,
        "skipped": skipped,
    }


def _run_rule_search_for_domain(
    domain: str,
    case_payloads: list[dict[str, Any]],
    iterations: int,
    candidates_per_round: int,
    progress_callback=None,
) -> dict[str, Any]:
    cases = [
        EvaluationCase(
            case_id=str(payload["case_id"]),
            domain=str(payload.get("domain") or "generic"),
            question=str(payload.get("question") or ""),
            resolved_answer=payload.get("resolved_answer"),
            candidate_probabilities=list(payload.get("candidate_probabilities") or []),
            scoring_metrics={
                key: float(value)
                for key, value in (payload.get("scoring_metrics") or {}).items()
                if isinstance(value, int | float)
            },
            evidence_items=list(payload.get("evidence_items") or []),
            sub_agent_results=list(payload.get("sub_agent_results") or []),
            round_snapshots=list(payload.get("round_snapshots") or []),
            markdown_report=str(payload.get("markdown_report") or ""),
        )
        for payload in case_payloads
    ]
    result = RuleSearchRunner(use_llm_scorer=True).run(
        cases=cases,
        domain=domain,
        iterations=iterations,
        candidates_per_round=candidates_per_round,
        progress_callback=progress_callback,
    )
    output_dir = DATA_DIR / "eval_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"rule_search_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    validation = result.validation_summary
    return {
        "case_count": validation.case_count,
        "domain": domain,
        "best_rule_set_id": result.best_rule_set.rule_set_id,
        "validation_score": validation.validation_score,
        "correlation_with_resolved_probability": validation.correlation_with_resolved_probability,
        "correlation_with_brier": validation.correlation_with_brier,
        "correlation_with_accuracy": validation.correlation_with_accuracy,
        "report_path": str(report_path),
    }
