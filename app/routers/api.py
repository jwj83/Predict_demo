from __future__ import annotations

import time as clock
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException

from app.db.database import db
from app.schemas import (
    BenchmarkEventInput,
    BenchmarkGenerateRequest,
    BenchmarkGenerateResponse,
    CreateQuestionRequest,
    CreateQuestionResponse,
    EvaluationResponse,
    ForecastResultResponse,
    ForecastApiEvidence,
    ForecastApiQualityAssessment,
    ForecastApiReportResponse,
    ForecastRunResponse,
    ImportBenchmarkEventResponse,
    NativeForecastRequest,
    NativeForecastResponse,
    QuestionDetailResponse,
    QuestionHistoryResponse,
    QuestionListItem,
    RuleSearchCaseListResponse,
    RuleSearchJobResponse,
    RuleSearchRequest,
    RuleSearchResponse,
    ResolutionRequest,
    RunStatusResponse,
)
from app.services.evaluation import compute_accuracy, compute_brier_score, compute_confidence_gap
from app.services.benchmarking import (
    generate_benchmark,
    get_rule_search_job,
    list_rule_search_cases,
    run_rule_search,
    start_rule_search_job,
)
from app.services.forecasting import forecast_service


router = APIRouter(prefix="/api")


def _prediction_cutoff_iso(payload: BenchmarkEventInput) -> str:
    if payload.as_of_date:
        return payload.as_of_date.isoformat()
    cutoff = datetime.combine(payload.resolution_date, time.min, tzinfo=timezone.utc)
    return cutoff.isoformat()


def _question_cutoff_iso(payload: CreateQuestionRequest) -> str:
    try:
        hour_text, minute_text = payload.resolution_time.split(":", 1)
        local_time = time(hour=int(hour_text), minute=int(minute_text))
        local_datetime = datetime.combine(payload.resolution_date, local_time, tzinfo=ZoneInfo(payload.timezone))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="揭晓时间或时区格式无效。") from exc
    return local_datetime.astimezone(timezone.utc).isoformat()


@router.get("/questions", response_model=list[QuestionListItem])
def list_questions() -> list[QuestionListItem]:
    return db.list_questions()


@router.get("/questions/history", response_model=QuestionHistoryResponse)
def list_question_history(page: int = 1, page_size: int = 10) -> QuestionHistoryResponse:
    return QuestionHistoryResponse(**db.list_questions_page(page=page, page_size=page_size))


@router.post("/questions", response_model=CreateQuestionResponse)
def create_question(payload: CreateQuestionRequest) -> CreateQuestionResponse:
    question_id = db.create_question(
        category=payload.category,
        question_text=payload.question,
        resolution_date=payload.resolution_date.isoformat(),
        timezone_name=payload.timezone,
        candidate_options=payload.candidate_options,
        as_of_date=_question_cutoff_iso(payload),
    )
    return CreateQuestionResponse(question_id=question_id)


@router.post("/forecast", response_model=NativeForecastResponse)
def forecast_and_return_report(payload: NativeForecastRequest) -> NativeForecastResponse:
    question_id = db.create_question(
        category=payload.category,
        question_text=payload.question,
        resolution_date=payload.resolution_date.isoformat(),
        timezone_name=payload.timezone,
        candidate_options=payload.candidate_options,
        as_of_date=_question_cutoff_iso(payload),
    )
    run_id = forecast_service.start_forecast(question_id)
    deadline = clock.monotonic() + payload.wait_timeout_seconds
    while clock.monotonic() < deadline:
        run = db.get_run(run_id)
        if run and run["run_status"] == "completed":
            result = db.get_latest_result(question_id)
            if not result:
                raise HTTPException(status_code=500, detail="预测完成但未找到报告结果。")
            return NativeForecastResponse(
                question_id=question_id,
                run_id=run_id,
                status="completed",
                report=_build_api_report(result),
            )
        if run and run["run_status"] == "failed":
            raise HTTPException(status_code=500, detail=run.get("error") or "预测失败。")
        clock.sleep(0.5)
    raise HTTPException(status_code=504, detail=f"预测未在 {payload.wait_timeout_seconds} 秒内完成，请稍后通过 run_id 查询。")


@router.post("/benchmark-events", response_model=ImportBenchmarkEventResponse)
def import_benchmark_event(payload: BenchmarkEventInput) -> ImportBenchmarkEventResponse:
    event_payload = payload.model_dump(mode="json")
    question_id, status = db.import_benchmark_event(
        event_payload=event_payload,
        candidate_options=payload.options,
        cutoff_iso=_prediction_cutoff_iso(payload),
    )
    return ImportBenchmarkEventResponse(question_id=question_id, external_id=payload.id, status=status)


@router.post("/benchmark/generate", response_model=BenchmarkGenerateResponse)
def generate_benchmark_events(payload: BenchmarkGenerateRequest) -> BenchmarkGenerateResponse:
    try:
        result = generate_benchmark(
            domain=payload.domain,
            limit=payload.limit,
            max_items_per_feed=payload.max_items_per_feed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BenchmarkGenerateResponse(**result)


@router.post("/questions/{question_id}/forecast", response_model=ForecastRunResponse)
def start_forecast(question_id: str) -> ForecastRunResponse:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="问题不存在。")
    run_id = forecast_service.start_forecast(question_id)
    return ForecastRunResponse(run_id=run_id, status="running")


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str) -> RunStatusResponse:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行任务不存在。")
    return RunStatusResponse(**run)


@router.get("/questions/{question_id}/result", response_model=ForecastResultResponse)
def get_result(question_id: str) -> ForecastResultResponse:
    result = db.get_latest_result(question_id)
    if not result:
        raise HTTPException(status_code=404, detail="暂无预测结果。")
    return ForecastResultResponse(
        prediction_date=datetime.fromisoformat(result["prediction_date"]),
        question=result["question"],
        direct_answer=result["direct_answer"],
        confidence_level=result["confidence_level"],
        confidence_rationale=result["confidence_rationale"],
        evidence_basis=result["evidence_basis"],
        candidate_probabilities=result["candidate_probabilities"],
        counterfactual_fragility=result["counterfactual_fragility"],
        conflict_summary=result["conflict_summary"],
        evidence_items=result["evidence_items"],
        round_snapshots=result["round_snapshots"],
        monitoring_items=result["monitoring_items"],
        report_quality_notes=result["report_quality_notes"],
        sub_agent_results=result["sub_agent_results"],
        markdown_report=result["markdown_report"],
    )


@router.get("/questions/{question_id}/detail", response_model=QuestionDetailResponse)
def get_question_detail(question_id: str) -> QuestionDetailResponse:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="问题不存在。")
    runs = db.list_runs_for_question(question_id)
    latest_run = RunStatusResponse(**runs[0]) if runs else None
    latest_result = None
    raw_result = db.get_latest_result(question_id)
    if raw_result:
        latest_result = ForecastResultResponse(
            prediction_date=datetime.fromisoformat(raw_result["prediction_date"]),
            question=raw_result["question"],
            direct_answer=raw_result["direct_answer"],
            confidence_level=raw_result["confidence_level"],
            confidence_rationale=raw_result["confidence_rationale"],
            evidence_basis=raw_result["evidence_basis"],
            candidate_probabilities=raw_result["candidate_probabilities"],
            counterfactual_fragility=raw_result["counterfactual_fragility"],
            conflict_summary=raw_result["conflict_summary"],
            evidence_items=raw_result["evidence_items"],
            round_snapshots=raw_result["round_snapshots"],
            monitoring_items=raw_result["monitoring_items"],
            report_quality_notes=raw_result["report_quality_notes"],
            sub_agent_results=raw_result["sub_agent_results"],
            markdown_report=raw_result["markdown_report"],
        )
    evaluation = None
    resolution = db.get_resolution(question_id)
    if resolution:
        evaluation = EvaluationResponse(
            question_id=question_id,
            resolved_answer=resolution["resolved_answer"],
            resolved_at=datetime.fromisoformat(resolution["resolved_at"]),
            selected_run_id=resolution["selected_run_id"],
            scoring_metrics=resolution["scoring_metrics"],
        )
    return QuestionDetailResponse(
        question=QuestionListItem(**question),
        latest_run=latest_run,
        latest_result=latest_result,
        evaluation=evaluation,
    )


@router.get("/questions/{question_id}/api-report", response_model=ForecastApiReportResponse)
def get_api_report(question_id: str) -> ForecastApiReportResponse:
    result = db.get_latest_result(question_id)
    if not result:
        raise HTTPException(status_code=404, detail="暂无预测结果。")
    return _build_api_report(result)


def _build_api_report(result: dict) -> ForecastApiReportResponse:
    quality = result["report_quality_notes"] or {}
    evidence_items = [
        ForecastApiEvidence(
            claim=item.get("claim", ""),
            role=item.get("evidence_role", "supporting"),
            stance=item.get("stance", "neutral"),
            supports_option=item.get("supports_option"),
            strength=item.get("strength"),
            source_url=item.get("source_url", ""),
            source_title=item.get("source_title"),
            published_at=datetime.fromisoformat(item["published_at"]) if item.get("published_at") else None,
            cutoff_compliant=bool(item.get("cutoff_compliant", True)),
            why_important=item.get("rationale") or item.get("source_excerpt_summary") or item.get("claim", ""),
        )
        for item in result["evidence_items"][:6]
    ]
    return ForecastApiReportResponse(
        prediction_date=datetime.fromisoformat(result["prediction_date"]).date(),
        question=result["question"],
        direct_answer=result["direct_answer"],
        confidence_level=result["confidence_level"],
        candidate_probability_table=result["candidate_probabilities"],
        confidence_basis=result["evidence_basis"],
        evidence_details=evidence_items,
        counterfactual_fragility=f"{result['counterfactual_fragility']}: {result['conflict_summary']}",
        monitoring_items=result["monitoring_items"],
        report_quality_assessment=ForecastApiQualityAssessment(
            evidence_granularity=quality.get("evidence_detail", ""),
            probability_rigor=quality.get("probability_rigor", ""),
            counterfactual_completeness=quality.get("counterfactual_completeness", ""),
            actionable_monitoring=quality.get("monitoring_plan", ""),
        ),
        markdown_report=result["markdown_report"],
    )


@router.post("/questions/{question_id}/resolve", response_model=EvaluationResponse)
def resolve_question(question_id: str, payload: ResolutionRequest) -> EvaluationResponse:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="问题不存在。")
    if payload.resolved_answer not in question["candidate_options"]:
        raise HTTPException(status_code=400, detail="真实答案必须属于候选项集合。")

    selected_run_id = payload.run_id
    if selected_run_id:
        result = db.get_result_by_run(selected_run_id)
        if not result or result["question_id"] != question_id:
            raise HTTPException(status_code=404, detail="指定的运行结果不存在。")
    else:
        result = db.get_latest_result(question_id)
        if not result:
            raise HTTPException(status_code=404, detail="暂无可结算的预测结果。")
        selected_run_id = result["run_id"]

    probabilities = result["candidate_probabilities"]
    metrics = {
        "accuracy": compute_accuracy(probabilities, payload.resolved_answer),
        "brier_score": compute_brier_score(probabilities, payload.resolved_answer),
        "resolved_option_probability": compute_confidence_gap(probabilities, payload.resolved_answer),
    }
    db.save_resolution(question_id, selected_run_id, payload.resolved_answer, metrics)
    db.update_question_status(question_id, "resolved")
    resolution = db.get_resolution(question_id)
    return EvaluationResponse(
        question_id=question_id,
        resolved_answer=resolution["resolved_answer"],
        resolved_at=datetime.fromisoformat(resolution["resolved_at"]),
        selected_run_id=resolution["selected_run_id"],
        scoring_metrics=resolution["scoring_metrics"],
    )


@router.get("/questions/{question_id}/evaluation", response_model=EvaluationResponse)
def get_evaluation(question_id: str) -> EvaluationResponse:
    resolution = db.get_resolution(question_id)
    if not resolution:
        raise HTTPException(status_code=404, detail="尚未结算。")
    return EvaluationResponse(
        question_id=question_id,
        resolved_answer=resolution["resolved_answer"],
        resolved_at=datetime.fromisoformat(resolution["resolved_at"]),
        selected_run_id=resolution["selected_run_id"],
        scoring_metrics=resolution["scoring_metrics"],
    )


@router.get("/evaluation/rule-search/cases", response_model=RuleSearchCaseListResponse)
def list_scoring_rule_cases(domains: str = "all") -> RuleSearchCaseListResponse:
    selected_domains = [item.strip() for item in domains.split(",") if item.strip()]
    result = list_rule_search_cases(selected_domains)
    return RuleSearchCaseListResponse(**result)


@router.post("/evaluation/rule-search", response_model=RuleSearchResponse)
def search_scoring_rules(payload: RuleSearchRequest) -> RuleSearchResponse:
    try:
        result = run_rule_search(
            domains=payload.domains,
            iterations=payload.iterations,
            candidates_per_round=payload.candidates_per_round,
            case_ids=payload.case_ids,
            cases_per_domain=payload.cases_per_domain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RuleSearchResponse(**result)


@router.post("/evaluation/rule-search/jobs", response_model=RuleSearchJobResponse)
def start_scoring_rule_search_job(payload: RuleSearchRequest) -> RuleSearchJobResponse:
    try:
        result = start_rule_search_job(
            domains=payload.domains,
            iterations=payload.iterations,
            candidates_per_round=payload.candidates_per_round,
            case_ids=payload.case_ids,
            cases_per_domain=payload.cases_per_domain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RuleSearchJobResponse(**result)


@router.get("/evaluation/rule-search/jobs/{job_id}", response_model=RuleSearchJobResponse)
def get_scoring_rule_search_job(job_id: str) -> RuleSearchJobResponse:
    try:
        result = get_rule_search_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RuleSearchJobResponse(**result)
