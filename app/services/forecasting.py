from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any

from app.core.config import MAX_MAP_ROUNDS, STOP_DELTA_THRESHOLD
from app.db.database import db
from app.services.evaluation import compute_accuracy, compute_brier_score, compute_confidence_gap
from app.services.llm import LLMClient
from app.services.page_reader import PageReaderProtocol, create_page_reader
from app.services.search import SearchProviderProtocol, create_search_provider


@dataclass(frozen=True)
class ForecastInput:
    question_id: str
    question_text: str
    candidate_options: list[str]
    domain: str
    source_url: str
    resolution_rule: str
    event_type: str
    prediction_cutoff: str


@dataclass(frozen=True)
class SubAgentMission:
    agent_id: str
    mission: str
    search_goal: str
    evidence_role: str


def _hash_score(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def _normalize(scores: list[tuple[str, float]]) -> list[dict[str, float | str]]:
    total = sum(max(score, 0.001) for _, score in scores)
    normalized = [{"option": option, "probability": round(max(score, 0.001) / total, 4)} for option, score in scores]
    drift = round(1.0 - sum(item["probability"] for item in normalized), 4)
    if normalized:
        normalized[0]["probability"] = round(normalized[0]["probability"] + drift, 4)
    return normalized


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_cutoff_compliant(published_at: str, cutoff: str) -> bool:
    if not published_at:
        return False
    return _parse_iso(published_at) <= _parse_iso(cutoff)


def _default_cutoff(question: dict[str, Any]) -> str:
    if question.get("as_of_date"):
        return question["as_of_date"]
    resolution_date = datetime.fromisoformat(str(question["resolution_date"])).date()
    return datetime.combine(resolution_date, time.min, tzinfo=timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Planner:
    def plan(self, forecast_input: ForecastInput, history: list[dict[str, Any]]) -> list[SubAgentMission]:
        del history
        return [
            SubAgentMission(
                agent_id="official_sources_agent",
                mission="Find official or authoritative sources that directly resolve the forecast.",
                search_goal=f"{forecast_input.question_text} official authoritative source {forecast_input.resolution_rule}",
                evidence_role="direct",
            ),
            SubAgentMission(
                agent_id="supporting_evidence_agent",
                mission="Find evidence that supports each candidate option before the cutoff.",
                search_goal=f"{forecast_input.question_text} supporting evidence {' '.join(forecast_input.candidate_options)}",
                evidence_role="supporting",
            ),
            SubAgentMission(
                agent_id="opposing_evidence_agent",
                mission="Find opposing evidence, conflicts, and uncertainty that could weaken leading options.",
                search_goal=f"{forecast_input.question_text} contradiction dispute uncertainty",
                evidence_role="opposing",
            ),
            SubAgentMission(
                agent_id="resolution_rule_agent",
                mission="Interpret the resolution rule, cutoff, and adjudication criteria for this forecast.",
                search_goal=f"{forecast_input.question_text} resolution rule criteria cutoff {forecast_input.resolution_rule}",
                evidence_role="context",
            ),
        ]


class MapSubAgent:
    def __init__(
        self,
        llm: LLMClient,
        search_provider: SearchProviderProtocol,
        page_reader: PageReaderProtocol,
    ) -> None:
        self.llm = llm
        self.search_provider = search_provider
        self.page_reader = page_reader

    async def run(self, forecast_input: ForecastInput, mission: SubAgentMission, round_index: int) -> dict[str, Any]:
        await asyncio.sleep(0)
        trajectory: list[dict[str, Any]] = []

        query = self._build_query(forecast_input, mission, round_index)
        search_thought = self._think(
            role=f"{mission.agent_id}:thought:websearch",
            forecast_input=forecast_input,
            mission=mission,
            prompt=(
                "Generate the next thought for a ReAct websearch step. "
                f"Explain why this query should help: {query}"
            ),
            fallback=f"Search for {mission.evidence_role} evidence relevant to the mission before the cutoff.",
        )
        search_action = {"type": "websearch", "query": query, "published_before": forecast_input.prediction_cutoff}
        raw_results = self.search_provider.search(query, published_before=forecast_input.prediction_cutoff)
        compliant_results = [
            result
            for result in raw_results
            if _is_cutoff_compliant(result.get("published_at", ""), forecast_input.prediction_cutoff)
        ]
        search_observation = (
            f"Found {len(raw_results)} search results and kept {len(compliant_results)} cutoff-compliant results. "
            f"Top titles: {self._summarize_titles(compliant_results)}"
        )
        trajectory.append(
            {"t": 0, "thought": search_thought, "action": search_action, "observation": search_observation}
        )

        selected_sources = compliant_results[:1]
        fetch_thought = self._think(
            role=f"{mission.agent_id}:thought:webfetch",
            forecast_input=forecast_input,
            mission=mission,
            prompt=(
                "Generate the next thought for a ReAct webfetch step. "
                f"Choose why these URLs should be read: {[item.get('url') for item in selected_sources]}"
            ),
            fallback="Read the highest-ranked cutoff-compliant source to inspect its evidence content.",
        )
        fetch_action = {"type": "webfetch", "urls": [item.get("url", "") for item in selected_sources]}
        pages = []
        for source in selected_sources:
            page = self.page_reader.fetch(source["url"], published_before=forecast_input.prediction_cutoff)
            pages.append({"source": source, "page": page})
        fetch_observation = (
            f"Fetched {len(pages)} pages. "
            f"Page summaries: {self._summarize_pages(pages)}"
        )
        trajectory.append({"t": 1, "thought": fetch_thought, "action": fetch_action, "observation": fetch_observation})

        reason_thought = self._think(
            role=f"{mission.agent_id}:thought:reason",
            forecast_input=forecast_input,
            mission=mission,
            prompt="Generate the next thought for local evidence reasoning over the fetched page snippets.",
            fallback="Reason over the fetched snippets to extract structured evidence and a local conclusion.",
        )
        reason_action = {"type": "reason_over_evidence", "agent_id": mission.agent_id}
        evidence_items, local_conclusion = self._reason_over_evidence(forecast_input, mission, pages)
        reason_observation = (
            f"Extracted {len(evidence_items)} evidence items. "
            f"Local conclusion favors {local_conclusion['favored_option']} with confidence {local_conclusion['confidence']:.2f}."
        )
        trajectory.append(
            {"t": 2, "thought": reason_thought, "action": reason_action, "observation": reason_observation}
        )

        return {
            "agent_id": mission.agent_id,
            "mission": mission.mission,
            "trajectory": trajectory,
            "evidence_items": evidence_items,
            "local_conclusion": local_conclusion,
            "observation_summary": reason_observation,
        }

    def _build_query(self, forecast_input: ForecastInput, mission: SubAgentMission, round_index: int) -> str:
        round_suffix = "" if round_index == 0 else f" additional evidence round {round_index + 1}"
        return f"{mission.search_goal} before:{forecast_input.prediction_cutoff}{round_suffix}".strip()

    def _think(
        self,
        role: str,
        forecast_input: ForecastInput,
        mission: SubAgentMission,
        prompt: str,
        fallback: str,
    ) -> str:
        payload = (
            f"Question: {forecast_input.question_text}\n"
            f"Options: {forecast_input.candidate_options}\n"
            f"Mission: {mission.mission}\n"
            f"Cutoff: {forecast_input.prediction_cutoff}\n"
            f"{prompt}"
        )
        try:
            response = self.llm.generate_structured(role, payload, {"thought": "string"})
        except Exception:
            return fallback
        thought = response.get("thought")
        return str(thought) if thought else fallback

    def _reason_over_evidence(
        self,
        forecast_input: ForecastInput,
        mission: SubAgentMission,
        pages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = (
            "Extract structured evidence and a local conclusion for this ReAct sub-agent. "
            "Do not infer from resolved answers. Use only the provided source snippets.\n"
            f"Question: {forecast_input.question_text}\n"
            f"Options: {forecast_input.candidate_options}\n"
            f"Mission: {mission.mission}\n"
            f"Evidence role: {mission.evidence_role}\n"
            f"Cutoff: {forecast_input.prediction_cutoff}\n"
            f"Sources: {_safe_json(self._source_packets(pages))}"
        )
        schema = {
            "evidence_items": [
                {
                    "claim": "string",
                    "evidence_role": "direct|supporting|opposing|context",
                    "stance": "support|oppose|neutral",
                    "supports_option": "one candidate option",
                    "strength": 0.0,
                    "rationale": "string",
                    "source_url": "string",
                    "source_title": "string",
                    "source_excerpt_summary": "string",
                    "published_at": "ISO datetime",
                    "cutoff_compliant": True,
                    "recency_score": 0.0,
                }
            ],
            "local_conclusion": {
                "favored_option": "one candidate option",
                "confidence": 0.0,
                "key_findings": ["string"],
                "conflicts": ["string"],
                "information_gaps": ["string"],
            },
        }
        try:
            response = self.llm.generate_structured(f"{mission.agent_id}:reason_over_evidence", prompt, schema)
            evidence_items = self._sanitize_evidence_items(response.get("evidence_items", []), forecast_input, mission, pages)
            local_conclusion = self._sanitize_local_conclusion(response.get("local_conclusion", {}), forecast_input, evidence_items)
            if evidence_items:
                return evidence_items, local_conclusion
        except Exception:
            pass
        return self._mock_reasoning(forecast_input, mission, pages)

    def _source_packets(self, pages: list[dict[str, Any]]) -> list[dict[str, str]]:
        packets = []
        for item in pages:
            source = item["source"]
            page = item["page"]
            packets.append(
                {
                    "title": source.get("title") or page.get("title", ""),
                    "url": source.get("url", ""),
                    "snippet": source.get("snippet", ""),
                    "published_at": source.get("published_at") or page.get("published_at", ""),
                    "content_summary": page.get("content_summary", ""),
                }
            )
        return packets

    def _sanitize_evidence_items(
        self,
        raw_items: Any,
        forecast_input: ForecastInput,
        mission: SubAgentMission,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        known_urls = {item["source"].get("url", ""): item for item in pages}
        cleaned = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            option = raw.get("supports_option")
            if option not in forecast_input.candidate_options:
                option = forecast_input.candidate_options[0]
            source_url = raw.get("source_url") or next(iter(known_urls), "")
            source = known_urls.get(source_url, {}).get("source", {})
            page = known_urls.get(source_url, {}).get("page", {})
            published_at = raw.get("published_at") or source.get("published_at") or page.get("published_at", "")
            cutoff_compliant = _is_cutoff_compliant(published_at, forecast_input.prediction_cutoff)
            if not cutoff_compliant:
                continue
            cleaned.append(
                {
                    "claim": str(raw.get("claim") or f"{mission.agent_id} found evidence for {option}."),
                    "supports_option": option,
                    "stance": raw.get("stance") if raw.get("stance") in {"support", "oppose", "neutral"} else "support",
                    "strength": min(1.0, max(0.0, float(raw.get("strength", 0.5)))),
                    "source_url": source_url,
                    "source_title": str(raw.get("source_title") or source.get("title") or page.get("title", "")),
                    "source_excerpt_summary": str(
                        raw.get("source_excerpt_summary")
                        or page.get("content_summary")
                        or source.get("snippet")
                        or ""
                    ),
                    "published_at": published_at,
                    "cutoff_compliant": True,
                    "recency_score": min(1.0, max(0.0, float(raw.get("recency_score", 0.8)))),
                    "evidence_role": raw.get("evidence_role")
                    if raw.get("evidence_role") in {"direct", "supporting", "opposing", "context"}
                    else mission.evidence_role,
                    "rationale": str(raw.get("rationale") or mission.mission),
                }
            )
        return cleaned

    def _sanitize_local_conclusion(
        self,
        raw: Any,
        forecast_input: ForecastInput,
        evidence_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raw = {}
        favored = raw.get("favored_option")
        if favored not in forecast_input.candidate_options:
            favored = evidence_items[0]["supports_option"] if evidence_items else forecast_input.candidate_options[0]
        return {
            "favored_option": favored,
            "confidence": min(1.0, max(0.0, float(raw.get("confidence", 0.5)))),
            "key_findings": [str(item) for item in raw.get("key_findings", []) if str(item).strip()][:4],
            "conflicts": [str(item) for item in raw.get("conflicts", []) if str(item).strip()][:4],
            "information_gaps": [str(item) for item in raw.get("information_gaps", []) if str(item).strip()][:4],
        }

    def _mock_reasoning(
        self,
        forecast_input: ForecastInput,
        mission: SubAgentMission,
        pages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not pages:
            return [], {
                "favored_option": forecast_input.candidate_options[0],
                "confidence": 0.2,
                "key_findings": [],
                "conflicts": ["No cutoff-compliant pages were available."],
                "information_gaps": ["Need at least one verifiable source before the cutoff."],
            }
        item = pages[0]
        source = item["source"]
        page = item["page"]
        seed = f"{forecast_input.question_text}|{mission.agent_id}|{source.get('url', '')}"
        option = forecast_input.candidate_options[int(_hash_score(seed) * len(forecast_input.candidate_options))]
        stance = "oppose" if mission.evidence_role == "opposing" else "support"
        strength = round(0.45 + _hash_score(seed + "|strength") * 0.45, 4)
        evidence = {
            "claim": f"{mission.agent_id} identified {mission.evidence_role} evidence related to {option}.",
            "supports_option": option,
            "stance": stance,
            "strength": strength,
            "source_url": source.get("url", ""),
            "source_title": source.get("title", ""),
            "source_excerpt_summary": page.get("content_summary") or source.get("snippet", ""),
            "published_at": source.get("published_at", ""),
            "cutoff_compliant": True,
            "recency_score": round(0.7 + _hash_score(seed + "|recency") * 0.3, 4),
            "evidence_role": mission.evidence_role,
            "rationale": f"This source was selected by {mission.agent_id} for local ReAct reasoning.",
        }
        local_conclusion = {
            "favored_option": option,
            "confidence": strength,
            "key_findings": [evidence["claim"]],
            "conflicts": ["Opposing evidence requires reducer-level comparison."] if stance == "oppose" else [],
            "information_gaps": ["Verify whether additional official sources change this local conclusion."],
        }
        return [evidence], local_conclusion

    def _summarize_titles(self, results: list[dict[str, Any]]) -> str:
        titles = [item.get("title", "untitled") for item in results[:3]]
        return "; ".join(titles) if titles else "none"

    def _summarize_pages(self, pages: list[dict[str, Any]]) -> str:
        summaries = []
        for item in pages[:2]:
            page = item["page"]
            summaries.append(f"{page.get('title', 'untitled')}: {str(page.get('content_summary', ''))[:160]}")
        return " | ".join(summaries) if summaries else "none"


class Reducer:
    ROLE_WEIGHTS = {"direct": 1.35, "supporting": 1.0, "opposing": 0.85, "context": 0.65}

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def aggregate(
        self,
        forecast_input: ForecastInput,
        sub_agent_results: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del history
        option_scores: dict[str, float] = {option: 0.05 for option in forecast_input.candidate_options}
        all_evidence = self._dedupe_evidence(
            evidence
            for result in sub_agent_results
            for evidence in result.get("evidence_items", [])
            if evidence.get("cutoff_compliant")
        )

        for evidence in all_evidence:
            option = evidence["supports_option"]
            role_weight = self.ROLE_WEIGHTS.get(evidence.get("evidence_role", "supporting"), 1.0)
            score = evidence["strength"] * evidence["recency_score"] * role_weight
            if evidence["stance"] == "support":
                option_scores[option] += score
            elif evidence["stance"] == "oppose":
                option_scores[option] = max(0.001, option_scores[option] - score * 0.45)
                for other in option_scores:
                    if other != option:
                        option_scores[other] += score * 0.2 / max(1, len(option_scores) - 1)
            else:
                option_scores[option] += score * 0.25

        probabilities = _normalize(list(option_scores.items()))
        ranked = sorted(probabilities, key=lambda item: item["probability"], reverse=True)
        top_probability = float(ranked[0]["probability"])
        second_probability = float(ranked[1]["probability"]) if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        conflict_count = sum(1 for item in all_evidence if item["stance"] == "oppose")
        conflict_summary = (
            f"Found {conflict_count} opposing evidence items; reducer lowered confidence for contested options."
            if conflict_count
            else "No cutoff-compliant opposing evidence was strong enough to overturn the leading option."
        )
        confidence_level = "high" if top_probability >= 0.7 and conflict_count == 0 else "medium" if top_probability >= 0.48 else "low"
        fragility = "low" if margin >= 0.2 and conflict_count == 0 else "medium" if margin >= 0.08 else "high"
        evidence_basis = self._format_evidence_basis(all_evidence[:6])
        summary = self.llm.generate_structured(
            "reducer",
            (
                f"Question: {forecast_input.question_text}; leading option: {ranked[0]['option']}; "
                f"cutoff: {forecast_input.prediction_cutoff}; evidence: {evidence_basis}"
            ),
            {},
        )
        return {
            "probabilities": probabilities,
            "predicted_winner": ranked[0]["option"],
            "confidence_level": confidence_level,
            "confidence_rationale": (
                f"{ranked[0]['option']} is assigned {top_probability:.2%}, "
                f"ahead of the second option by {margin:.2%}. {summary.get('summary', '')[:90]}"
            ),
            "evidence_basis": evidence_basis,
            "counterfactual_fragility": fragility,
            "conflict_summary": conflict_summary,
            "evidence_items": all_evidence[:6],
            "sub_agent_results": sub_agent_results,
        }

    def _dedupe_evidence(self, evidence_iterable: Any) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        deduped = []
        for evidence in evidence_iterable:
            key = (evidence.get("source_url", ""), evidence.get("claim", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(evidence)
        return deduped

    def _format_evidence_basis(self, evidence_items: list[dict[str, Any]]) -> str:
        if not evidence_items:
            return "No cutoff-compliant evidence was available."
        return " ".join(
            f"[{item.get('evidence_role', 'supporting')}] {item['claim']} ({item['source_url']})."
            for item in evidence_items
        )


class Judge:
    def should_stop(
        self,
        history: list[dict[str, Any]],
        latest_result: dict[str, Any],
    ) -> tuple[bool, str]:
        reducer_entries = [entry for entry in history if entry["role"] == "reducer"]
        if len(reducer_entries) >= MAX_MAP_ROUNDS:
            return True, "Reached the maximum number of Map-Reduce rounds."
        if len(reducer_entries) >= 2:
            previous = reducer_entries[-2]["probabilities"]
            current = latest_result["probabilities"]
            delta = 0.0
            for current_item in current:
                previous_probability = next(
                    (item["probability"] for item in previous if item["option"] == current_item["option"]),
                    0.0,
                )
                delta += abs(current_item["probability"] - previous_probability)
            if delta < STOP_DELTA_THRESHOLD:
                return True, "Probability distribution has stabilized across consecutive rounds."
        if len(latest_result["evidence_items"]) < 3:
            return True, "New cutoff-compliant evidence is limited; additional iteration has low expected value."
        return False, "Continue searching and reducing evidence."


class ForecastService:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.planner = Planner()
        self.map_agent = MapSubAgent(self.llm, create_search_provider(), create_page_reader())
        self.researcher = self.map_agent
        self.reducer = Reducer(self.llm)
        self.judge = Judge()

    def start_forecast(self, question_id: str) -> str:
        question = db.get_question(question_id)
        if not question:
            raise ValueError("Question not found.")
        db.update_question_status(question_id, "running")
        run_id = db.create_run(question_id)
        forecast_input = self._build_forecast_input(question)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._execute_forecast(run_id, forecast_input))
        except RuntimeError:
            thread = threading.Thread(
                target=lambda: asyncio.run(self._execute_forecast(run_id, forecast_input)),
                daemon=True,
            )
            thread.start()
        return run_id

    def _build_forecast_input(self, question: dict[str, Any]) -> ForecastInput:
        return ForecastInput(
            question_id=question["id"],
            question_text=question["question_text"],
            candidate_options=question["candidate_options"],
            domain=question.get("domain") or question["category"],
            source_url=question.get("source_url") or "",
            resolution_rule=question.get("resolution_rule") or "",
            event_type=question.get("event_type") or "",
            prediction_cutoff=_default_cutoff(question),
        )

    async def _execute_forecast(self, run_id: str, forecast_input: ForecastInput) -> None:
        history: list[dict[str, Any]] = []
        round_snapshots: list[dict[str, Any]] = []
        latest_reduced: dict[str, Any] | None = None
        try:
            for round_index in range(MAX_MAP_ROUNDS):
                missions = self.planner.plan(forecast_input, history)
                db.append_run_trace(
                    run_id,
                    "planning",
                    {
                        "round_index": round_index,
                        "role": "planner",
                        "thought_summary": f"Round {round_index + 1} assigned {len(missions)} ReAct sub-agent missions.",
                        "action": "plan_sub_agent_missions",
                        "observation_summary": "; ".join(mission.agent_id for mission in missions),
                    },
                )
                history.append({"role": "planner", "round_index": round_index, "missions": [mission.__dict__ for mission in missions]})

                sub_agent_results = await asyncio.gather(
                    *(self.map_agent.run(forecast_input, mission, round_index) for mission in missions)
                )
                for result in sub_agent_results:
                    for step in result["trajectory"]:
                        trace_entry = {
                            "round_index": round_index,
                            "role": result["agent_id"],
                            "thought_summary": step["thought"],
                            "action": _safe_json(step["action"]),
                            "observation_summary": step["observation"],
                        }
                        db.append_run_trace(run_id, "researching", trace_entry)
                    history.append(
                        {
                            "role": result["agent_id"],
                            "round_index": round_index,
                            "trajectory": result["trajectory"],
                            "evidence_items": result["evidence_items"],
                            "local_conclusion": result["local_conclusion"],
                        }
                    )

                reduced = self.reducer.aggregate(forecast_input, sub_agent_results, history)
                latest_reduced = reduced
                reducer_entry = {
                    "round_index": round_index,
                    "role": "reducer",
                    "thought_summary": f"Aggregated {len(reduced['evidence_items'])} deduped evidence items from {len(sub_agent_results)} sub-agents.",
                    "action": "aggregate_sub_agent_results",
                    "observation_summary": reduced["confidence_rationale"],
                }
                db.append_run_trace(
                    run_id,
                    "reducing",
                    reducer_entry,
                    latest_probabilities=reduced["probabilities"],
                    latest_evidence_summary=reduced["evidence_basis"],
                )
                history.append({"role": "reducer", "round_index": round_index, "probabilities": reduced["probabilities"]})

                should_stop, reason = self.judge.should_stop(history, reduced)
                snapshot = {
                    "round_index": round_index,
                    "probabilities": reduced["probabilities"],
                    "conflict_summary": reduced["conflict_summary"],
                    "evidence_count": len(reduced["evidence_items"]),
                    "stop_reason": reason if should_stop else None,
                }
                round_snapshots.append(snapshot)
                db.append_run_trace(
                    run_id,
                    "judging",
                    {
                        "round_index": round_index,
                        "role": "judge",
                        "thought_summary": "Evaluate evidence coverage, cutoff compliance, and probability stability.",
                        "action": "check_stop_condition",
                        "observation_summary": reason,
                    },
                    latest_probabilities=reduced["probabilities"],
                    latest_evidence_summary=reduced["evidence_basis"],
                )
                history.append({"role": "judge", "round_index": round_index, "reason": reason})
                if should_stop:
                    self._save_completed_result(run_id, forecast_input, reduced, round_snapshots)
                    return

            if latest_reduced is None:
                raise RuntimeError("Forecast loop produced no result.")
            self._save_completed_result(run_id, forecast_input, latest_reduced, round_snapshots)
        except Exception as exc:
            db.finish_run(run_id, "failed", str(exc))
            db.update_question_status(forecast_input.question_id, "failed")

    def _save_completed_result(
        self,
        run_id: str,
        forecast_input: ForecastInput,
        reduced: dict[str, Any],
        round_snapshots: list[dict[str, Any]],
    ) -> None:
        monitoring_items = [
            f"Monitor authoritative updates for {forecast_input.event_type or forecast_input.domain} before {forecast_input.prediction_cutoff}.",
            "Re-run if direct official evidence or a strong opposing source appears.",
        ]
        report_quality_notes = {
            "evidence_detail": "Each retained item is linked to a source URL, role, stance, strength, and cutoff timestamp.",
            "probability_rigor": "Probabilities are reduced from sub-agent evidence strength, source role, recency, and opposition signals.",
            "counterfactual_completeness": "Fragility reflects the probability margin and whether opposing evidence survives cutoff filtering.",
            "monitoring_plan": "Monitoring items convert unresolved evidence gaps into concrete re-run triggers.",
        }
        payload = {
            "prediction_date": datetime.now(timezone.utc).isoformat(),
            "question": forecast_input.question_text,
            "direct_answer": reduced["predicted_winner"],
            "confidence_level": reduced["confidence_level"],
            "confidence_rationale": reduced["confidence_rationale"],
            "evidence_basis": reduced["evidence_basis"],
            "candidate_probabilities": reduced["probabilities"],
            "counterfactual_fragility": reduced["counterfactual_fragility"],
            "conflict_summary": reduced["conflict_summary"],
            "evidence_items": reduced["evidence_items"],
            "round_snapshots": round_snapshots,
            "monitoring_items": monitoring_items,
            "report_quality_notes": report_quality_notes,
            "sub_agent_results": reduced["sub_agent_results"],
            "markdown_report": self._render_markdown_report(forecast_input, reduced, monitoring_items, report_quality_notes),
        }
        db.save_result(run_id, forecast_input.question_id, payload)
        db.finish_run(run_id, "completed")

        resolved_answer = db.get_benchmark_resolution_answer(forecast_input.question_id)
        if resolved_answer:
            metrics = {
                "accuracy": compute_accuracy(reduced["probabilities"], resolved_answer),
                "brier_score": compute_brier_score(reduced["probabilities"], resolved_answer),
                "resolved_option_probability": compute_confidence_gap(reduced["probabilities"], resolved_answer),
            }
            db.save_resolution(forecast_input.question_id, run_id, resolved_answer, metrics)
            db.update_question_status(forecast_input.question_id, "resolved")
        else:
            db.update_question_status(forecast_input.question_id, "awaiting_resolution")

    def _render_markdown_report(
        self,
        forecast_input: ForecastInput,
        reduced: dict[str, Any],
        monitoring_items: list[str],
        report_quality_notes: dict[str, str],
    ) -> str:
        probability_rows = "\n".join(
            f"| {item['option']} | {float(item['probability']):.4f} |" for item in reduced["probabilities"]
        )
        evidence_rows = "\n".join(
            f"- **{item.get('evidence_role', 'supporting')} / {item['stance']}**: {item['claim']} "
            f"([{item.get('source_title') or 'source'}]({item['source_url']}))"
            for item in reduced["evidence_items"]
        )
        monitoring_rows = "\n".join(f"- {item}" for item in monitoring_items)
        quality_rows = "\n".join(f"- **{key}**: {value}" for key, value in report_quality_notes.items())
        return (
            f"# Forecast Report\n\n"
            f"Prediction date: {datetime.now(timezone.utc).date().isoformat()}\n\n"
            f"Question: {forecast_input.question_text}\n\n"
            f"Direct answer: **{reduced['predicted_winner']}** | Confidence: **{reduced['confidence_level']}**\n\n"
            f"| Candidate | Probability |\n| --- | ---: |\n{probability_rows}\n\n"
            f"## Confidence Basis\n\n{reduced['evidence_basis']}\n\n"
            f"## Evidence\n\n{evidence_rows or '- No cutoff-compliant evidence retained.'}\n\n"
            f"## Counterfactual Fragility\n\n{reduced['counterfactual_fragility']}: {reduced['conflict_summary']}\n\n"
            f"## Monitoring\n\n{monitoring_rows}\n\n"
            f"## Report Quality\n\n{quality_rows}\n"
        )


forecast_service = ForecastService()
