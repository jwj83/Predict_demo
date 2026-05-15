from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


def build_client(tmp_path: Path) -> TestClient:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ["FORECAST_DB_PATH"] = str(tmp_path / "test.db")
    for module_name in list(sys.modules.keys()):
        if module_name == "app" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    app_main = importlib.import_module("app.main")
    return TestClient(app_main.app)


def create_question(client: TestClient) -> str:
    response = client.post(
        "/api/questions",
        json={
            "category": "经济与金融",
            "question": "截至 2026 年 3 月 31 日美东时间收盘，哪家公司将成为全球市值最大的公司？",
            "resolution_date": "2026-03-31",
            "timezone": "America/New_York",
            "candidate_options": ["英伟达（NVIDIA）", "苹果（Apple）", "微软（Microsoft）"],
        },
    )
    assert response.status_code == 200
    return response.json()["question_id"]


def wait_for_run(client: TestClient, run_id: str) -> dict:
    deadline = time.time() + 5
    last_payload = None
    while time.time() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["run_status"] in {"completed", "failed"}:
            return last_payload
        time.sleep(0.1)
    raise AssertionError(f"run did not finish in time: {last_payload}")


def benchmark_payload(event_status: str = "resolved") -> dict:
    payload = {
        "id": f"bm_test_{event_status}",
        "domain": "sports",
        "source": "sohu_nba",
        "source_url": "https://www.sohu.com/a/295061110_458722",
        "question": "洛斯是否赢得了2019年NBA名人赛MVP？",
        "options": ["是", "否"],
        "resolution_date": "2026-05-10",
        "resolution_rule": "以NBA官方公布的名人赛MVP得主为准。",
        "event_type": "award",
        "event_status": event_status,
        "confidence": 0.9,
        "created_at": "2026-05-13T07:13:57.101586+00:00",
    }
    if event_status == "resolved":
        payload["resolved_answer"] = "是"
        payload["answer_evidence"] = "DO_NOT_LEAK_ANSWER_EVIDENCE"
    return payload


def test_create_question_requires_multiple_options(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    response = client.post(
        "/api/questions",
        json={
            "category": "测试",
            "question": "这是一个长度足够但候选项不足的问题文本。",
            "resolution_date": "2026-05-08",
            "timezone": "Asia/Shanghai",
            "candidate_options": ["只有一个"],
        },
    )
    assert response.status_code == 422


def test_forecast_flow_and_result_shape(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    question_id = create_question(client)

    run_response = client.post(f"/api/questions/{question_id}/forecast")
    assert run_response.status_code == 200
    run_id = run_response.json()["run_id"]

    run_payload = wait_for_run(client, run_id)
    assert run_payload["run_status"] == "completed"
    assert run_payload["latest_probabilities"]
    total_probability = round(sum(item["probability"] for item in run_payload["latest_probabilities"]), 4)
    assert total_probability == 1.0

    result_response = client.get(f"/api/questions/{question_id}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    winner = max(result_payload["candidate_probabilities"], key=lambda item: item["probability"])["option"]
    assert result_payload["direct_answer"] == winner
    assert result_payload["confidence_level"] in {"low", "medium", "high"}
    assert result_payload["conflict_summary"]


def test_native_forecast_api_creates_prediction_and_returns_report(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/api/forecast",
        json={
            "category": "sports",
            "question": "Will Team A win the 2026 demo championship?",
            "resolution_date": "2026-05-20",
            "timezone": "Asia/Shanghai",
            "candidate_options": ["yes", "no"],
            "wait_timeout_seconds": 20,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["question_id"]
    assert payload["run_id"]
    assert payload["status"] == "completed"
    report = payload["report"]
    assert report["direct_answer"] in {"yes", "no"}
    assert report["candidate_probability_table"]
    assert report["confidence_basis"]
    assert report["evidence_details"]
    assert report["report_quality_assessment"]["probability_rigor"]


def test_resolution_and_evaluation(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    question_id = create_question(client)
    run_id = client.post(f"/api/questions/{question_id}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    result = client.get(f"/api/questions/{question_id}/result").json()
    resolved_answer = result["candidate_probabilities"][0]["option"]
    resolve_response = client.post(
        f"/api/questions/{question_id}/resolve",
        json={"resolved_answer": resolved_answer},
    )
    assert resolve_response.status_code == 200
    payload = resolve_response.json()
    assert "brier_score" in payload["scoring_metrics"]
    assert payload["selected_run_id"] == run_id

    evaluation_response = client.get(f"/api/questions/{question_id}/evaluation")
    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["resolved_answer"] == resolved_answer


def test_benchmark_event_import_preserves_metadata_and_auto_scores(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    response = client.post("/api/benchmark-events", json=benchmark_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["external_id"] == "bm_test_resolved"

    from app.db.database import db

    question = db.get_question(payload["question_id"])
    assert question["external_id"] == "bm_test_resolved"
    assert question["domain"] == "sports"
    assert question["source_url"] == "https://www.sohu.com/a/295061110_458722"
    assert question["resolution_rule"] == "以NBA官方公布的名人赛MVP得主为准。"
    assert question["as_of_date"] == "2026-05-10T00:00:00+00:00"

    run_id = client.post(f"/api/questions/{payload['question_id']}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    evaluation_response = client.get(f"/api/questions/{payload['question_id']}/evaluation")
    assert evaluation_response.status_code == 200
    metrics = evaluation_response.json()["scoring_metrics"]
    assert "accuracy" in metrics
    assert "brier_score" in metrics


def test_unresolved_benchmark_event_uses_same_forecast_without_result_scoring(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload("open")).json()
    run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
    run_payload = wait_for_run(client, run_id)
    assert run_payload["run_status"] == "completed"

    result_response = client.get(f"/api/questions/{created['question_id']}/result")
    assert result_response.status_code == 200
    evaluation_response = client.get(f"/api/questions/{created['question_id']}/evaluation")
    assert evaluation_response.status_code == 404


def test_benchmark_generate_endpoint_returns_items_and_paths(tmp_path: Path, monkeypatch) -> None:
    client = build_client(tmp_path)

    import app.routers.api as api_module

    def fake_generate_benchmark(domain: str, limit: int, max_items_per_feed: int) -> dict:
        assert domain == "all"
        assert limit == 10
        assert max_items_per_feed == 5
        return {
            "run_id": "test_run",
            "total": 1,
            "items": [benchmark_payload()],
            "output_paths": {
                "benchmark_all": str(tmp_path / "benchmark_all.json"),
                "raw": str(tmp_path / "sources" / "raw.json"),
                "events": str(tmp_path / "sources" / "events.json"),
            },
            "domain_counts": {"sports": 1},
            "status_counts": {"resolved": 1},
        }

    monkeypatch.setattr(api_module, "generate_benchmark", fake_generate_benchmark)

    response = client.post(
        "/api/benchmark/generate",
        json={"domain": "all", "limit": 10, "max_items_per_feed": 5},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["run_id"] == "test_run"
    assert data["total"] == 1
    assert data["items"][0]["id"] == "bm_test_resolved"
    assert data["domain_counts"] == {"sports": 1}
    assert data["status_counts"] == {"resolved": 1}
    assert data["output_paths"]["benchmark_all"].endswith("benchmark_all.json")


def test_rule_search_endpoint_uses_completed_resolved_benchmark_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = build_client(tmp_path)

    import app.services.benchmarking as benchmarking_module

    monkeypatch.setattr(benchmarking_module, "DATA_DIR", tmp_path)

    imported = client.post("/api/benchmark-events", json=benchmark_payload())
    assert imported.status_code == 200
    question_id = imported.json()["question_id"]
    run = client.post(f"/api/questions/{question_id}/forecast")
    wait_for_run(client, run.json()["run_id"])

    response = client.post(
        "/api/evaluation/rule-search",
        json={"domains": ["all"], "iterations": 1, "candidates_per_round": 1},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["mode"] == "per_domain"
    assert data["total_case_count"] == 1
    assert data["results"][0]["domain"] == "sports"
    assert data["results"][0]["case_count"] == 1
    assert data["results"][0]["best_rule_set_id"]
    assert data["results"][0]["report_path"].endswith(".json")
    assert Path(data["results"][0]["report_path"]).exists()


def test_rule_search_case_list_filters_by_domain(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    imported = client.post("/api/benchmark-events", json=benchmark_payload())
    question_id = imported.json()["question_id"]
    run = client.post(f"/api/questions/{question_id}/forecast")
    wait_for_run(client, run.json()["run_id"])

    response = client.get("/api/evaluation/rule-search/cases?domains=sports")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["domain_counts"] == {"sports": 1}
    assert data["cases"][0]["domain"] == "sports"


def test_rule_search_job_reports_candidate_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = build_client(tmp_path)

    imported = client.post("/api/benchmark-events", json=benchmark_payload())
    question_id = imported.json()["question_id"]
    run = client.post(f"/api/questions/{question_id}/forecast")
    wait_for_run(client, run.json()["run_id"])

    started = client.post(
        "/api/evaluation/rule-search/jobs",
        json={"domains": ["sports"], "iterations": 1, "candidates_per_round": 1},
    )

    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    deadline = time.time() + 5
    final_payload = None
    while time.time() < deadline:
        response = client.get(f"/api/evaluation/rule-search/jobs/{job_id}")
        assert response.status_code == 200
        final_payload = response.json()
        if final_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert final_payload["status"] == "completed"
    assert final_payload["progress_events"]
    assert final_payload["best_by_round"]
    assert final_payload["result"]["results"][0]["domain"] == "sports"


def test_rule_search_endpoint_returns_clear_error_without_cases(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    response = client.post(
        "/api/evaluation/rule-search",
        json={"domains": ["all"], "iterations": 1, "candidates_per_round": 1},
    )

    assert response.status_code == 400
    assert "No completed resolved benchmark predictions" in response.json()["detail"]


def test_forecast_report_contains_required_structured_fields(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload()).json()
    run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    result = client.get(f"/api/questions/{created['question_id']}/result").json()
    assert result["direct_answer"]
    assert result["confidence_level"] in {"low", "medium", "high"}
    assert result["candidate_probabilities"]
    assert round(sum(item["probability"] for item in result["candidate_probabilities"]), 4) == 1.0
    assert result["evidence_basis"]
    assert result["counterfactual_fragility"] in {"low", "medium", "high"}
    assert result["conflict_summary"]
    assert result["evidence_items"]
    assert result["round_snapshots"]
    assert result["monitoring_items"]
    assert result["report_quality_notes"]["evidence_detail"]
    assert 4 <= len(result["evidence_items"]) <= 6
    assert result["markdown_report"]
    assert "Forecast Report" in result["markdown_report"]


def test_api_report_returns_formal_prediction_template(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload()).json()
    run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    response = client.get(f"/api/questions/{created['question_id']}/api-report")

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["prediction_date"]
    assert report["question"]
    assert report["direct_answer"]
    assert report["confidence_level"] in {"low", "medium", "high"}
    assert report["candidate_probability_table"]
    assert report["confidence_basis"]
    assert report["evidence_details"]
    assert report["counterfactual_fragility"]
    assert report["monitoring_items"]
    assert report["report_quality_assessment"]["evidence_granularity"]
    assert report["markdown_report"]


def test_sub_agents_record_react_trajectories_and_local_conclusions(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload()).json()
    run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    result = client.get(f"/api/questions/{created['question_id']}/result").json()
    sub_agent_results = result["sub_agent_results"]
    options = {item["option"] for item in result["candidate_probabilities"]}
    assert {item["agent_id"] for item in sub_agent_results} == {
        "official_sources_agent",
        "supporting_evidence_agent",
        "opposing_evidence_agent",
        "resolution_rule_agent",
    }
    for sub_agent in sub_agent_results:
        assert len(sub_agent["trajectory"]) == 3
        assert [step["t"] for step in sub_agent["trajectory"]] == [0, 1, 2]
        assert [step["action"]["type"] for step in sub_agent["trajectory"]] == [
            "websearch",
            "webfetch",
            "reason_over_evidence",
        ]
        assert all(step["thought"] for step in sub_agent["trajectory"])
        assert all(step["observation"] for step in sub_agent["trajectory"])
        assert sub_agent["local_conclusion"]["favored_option"] in options
        assert 0.0 <= sub_agent["local_conclusion"]["confidence"] <= 1.0
        assert sub_agent["evidence_items"]


def test_cutoff_filters_future_search_results(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload()).json()

    from app.services.forecasting import forecast_service

    class MockSearchProvider:
        def search(self, query: str, limit: int | None = None, published_before: str | None = None) -> list[dict]:
            del query, limit, published_before
            return [
                {
                    "title": "Allowed source",
                    "url": "https://example.com/allowed",
                    "snippet": "Published before cutoff.",
                    "source_type": "mock",
                    "published_at": "2026-05-09T12:00:00+00:00",
                },
                {
                    "title": "Future source",
                    "url": "https://example.com/future",
                    "snippet": "Published after cutoff.",
                    "source_type": "mock",
                    "published_at": "2026-05-11T12:00:00+00:00",
                },
            ]

    forecast_service.researcher.search_provider = MockSearchProvider()
    run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
    wait_for_run(client, run_id)

    result = client.get(f"/api/questions/{created['question_id']}/result").json()
    assert result["evidence_items"]
    assert {item["source_url"] for item in result["evidence_items"]} == {"https://example.com/allowed"}
    assert all(item["cutoff_compliant"] for item in result["evidence_items"])
    assert all(item["published_at"] <= "2026-05-10T00:00:00+00:00" for item in result["evidence_items"])


def test_resolved_answer_evidence_is_not_in_agent_prompts(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    created = client.post("/api/benchmark-events", json=benchmark_payload()).json()

    from app.services.forecasting import forecast_service

    prompts: list[str] = []
    original_generate = forecast_service.llm.generate_structured

    def record_prompt(role: str, prompt: str, schema: dict) -> dict:
        prompts.append(prompt)
        return original_generate(role, prompt, schema)

    forecast_service.llm.generate_structured = record_prompt
    try:
        run_id = client.post(f"/api/questions/{created['question_id']}/forecast").json()["run_id"]
        wait_for_run(client, run_id)
    finally:
        forecast_service.llm.generate_structured = original_generate

    prompt_blob = "\n".join(prompts)
    assert "DO_NOT_LEAK_ANSWER_EVIDENCE" not in prompt_blob
    assert "answer_evidence" not in prompt_blob
    assert "resolved_answer" not in prompt_blob


def test_exa_search_provider_normalizes_results_and_cutoff_payload() -> None:
    from app.services.search import ExaSearchProvider

    requests: list[dict] = []

    class MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "title": "Official result",
                        "url": "https://example.com/official",
                        "highlights": ["Candidate A is confirmed."],
                        "publishedDate": "2026-05-09T12:00:00Z",
                    }
                ]
            }

    def mock_post(url: str, headers: dict, json: dict, timeout: float) -> MockResponse:
        requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return MockResponse()

    import app.services.search as search_module

    original_post = search_module.httpx.post
    search_module.httpx.post = mock_post
    try:
        provider = ExaSearchProvider(api_key="test-key")
        results = provider.search("who won", published_before="2026-05-10T00:00:00+00:00")
    finally:
        search_module.httpx.post = original_post

    assert requests[0]["headers"]["x-api-key"] == "test-key"
    assert requests[0]["json"]["endPublishedDate"] == "2026-05-10T00:00:00+00:00"
    assert requests[0]["json"]["text"]["maxCharacters"] == 600
    assert results == [
        {
            "title": "Official result",
            "url": "https://example.com/official",
            "snippet": "Candidate A is confirmed.",
            "source_type": "exa_web",
            "published_at": "2026-05-09T12:00:00+00:00",
            "raw": {
                "title": "Official result",
                "url": "https://example.com/official",
                "highlights": ["Candidate A is confirmed."],
                "publishedDate": "2026-05-09T12:00:00Z",
            },
        }
    ]


def test_exa_page_reader_fetches_contents_without_network() -> None:
    from app.services.page_reader import ExaPageReader

    requests: list[dict] = []

    class MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "title": "Official page",
                        "url": "https://example.com/page",
                        "text": "Long clean text from the page.",
                        "highlights": ["Relevant sentence."],
                        "publishedDate": "2026-05-09T12:00:00Z",
                    }
                ]
            }

    def mock_post(url: str, headers: dict, json: dict, timeout: float) -> MockResponse:
        requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return MockResponse()

    import app.services.page_reader as reader_module

    original_post = reader_module.httpx.post
    reader_module.httpx.post = mock_post
    try:
        reader = ExaPageReader(api_key="test-key")
        result = reader.fetch("https://example.com/page")
    finally:
        reader_module.httpx.post = original_post

    assert requests[0]["json"]["urls"] == ["https://example.com/page"]
    assert requests[0]["json"]["text"]["maxCharacters"] > 0
    assert result["title"] == "Official page"
    assert result["content_summary"] == "Relevant sentence."
    assert result["published_at"] == "2026-05-09T12:00:00Z"


def test_front_page_and_questions_listing(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    create_question(client)

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "Forecast Agent v1" in index_response.text

    list_response = client.get("/api/questions")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
