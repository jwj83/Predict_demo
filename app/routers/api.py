from __future__ import annotations

from datetime import datetime, time, timezone

from fastapi import APIRouter, HTTPException

from app.db.database import db
from app.schemas import (
    BenchmarkEventInput,
    CreateQuestionRequest,
    CreateQuestionResponse,
    EvaluationResponse,
    ForecastResultResponse,
    ForecastRunResponse,
    ImportBenchmarkEventResponse,
    QuestionListItem,
    ResolutionRequest,
    RunStatusResponse,
)
from app.services.evaluation import compute_accuracy, compute_brier_score, compute_confidence_gap
from app.services.forecasting import forecast_service


router = APIRouter(prefix="/api")


def _prediction_cutoff_iso(payload: BenchmarkEventInput) -> str:
    if payload.as_of_date:
        return payload.as_of_date.isoformat()
    cutoff = datetime.combine(payload.resolution_date, time.min, tzinfo=timezone.utc)
    return cutoff.isoformat()


@router.get("/questions", response_model=list[QuestionListItem])
def list_questions() -> list[QuestionListItem]:
    return db.list_questions()


@router.post("/questions", response_model=CreateQuestionResponse)
def create_question(payload: CreateQuestionRequest) -> CreateQuestionResponse:
    question_id = db.create_question(
        category=payload.category,
        question_text=payload.question,
        resolution_date=payload.resolution_date.isoformat(),
        timezone_name=payload.timezone,
        candidate_options=payload.candidate_options,
    )
    return CreateQuestionResponse(question_id=question_id)


@router.post("/benchmark-events", response_model=ImportBenchmarkEventResponse)
def import_benchmark_event(payload: BenchmarkEventInput) -> ImportBenchmarkEventResponse:
    event_payload = payload.model_dump(mode="json")
    question_id, status = db.import_benchmark_event(
        event_payload=event_payload,
        candidate_options=payload.options,
        cutoff_iso=_prediction_cutoff_iso(payload),
    )
    return ImportBenchmarkEventResponse(question_id=question_id, external_id=payload.id, status=status)


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
