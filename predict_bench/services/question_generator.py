from __future__ import annotations

import json
from datetime import date
from typing import Any

from predict_bench.domains import DomainConfig
from predict_bench.models import BenchmarkItem, FutureEvent, stable_id
from predict_bench.services.llm import OpenAICompatibleLLMClient


class QuestionGenerator:
    def __init__(
        self,
        llm: OpenAICompatibleLLMClient | None = None,
        batch_size: int = 3,
        config: DomainConfig | None = None,
    ) -> None:
        self.llm = llm or OpenAICompatibleLLMClient()
        self.batch_size = batch_size
        self.config = config

    def generate(self, events: list[FutureEvent], limit: int) -> list[BenchmarkItem]:
        items: list[BenchmarkItem] = []
        events_by_id = {event.id: event for event in events}
        for start in range(0, len(events), self.batch_size):
            batch = events[start : start + self.batch_size]
            response = self.llm.generate_json(
                system=(
                    "你是 Polymarket 风格 benchmark 问题生成器。只返回 JSON，不要 Markdown。"
                    "问题必须简单、答案有限、可结算。"
                ),
                user=self._build_prompt(batch, remaining=limit - len(items)),
            )
            for raw_item in response.get("items", []):
                if not isinstance(raw_item, dict):
                    continue
                item = self._coerce_item(raw_item, events_by_id)
                if item is not None:
                    items.append(item)
                if len(items) >= limit:
                    return items
        return items

    def _build_prompt(self, events: list[FutureEvent], remaining: int) -> str:
        payload = [event.model_dump(mode="json") for event in events]
        today = date.today().isoformat()
        domain = self.config.domain if self.config else events[0].domain if events else "sports"
        guidance = self.config.resolution_guidance if self.config else "以官方或主流公开来源为准。"
        return (
            f"今天日期是 {today}。\n"
            f"领域是 {domain}。请从事件候选中生成最多 {remaining} 条 Polymarket 风格预测 benchmark item。\n"
            "优先生成是/否问题；如果事件天然是多选冠军/赢家问题，可以生成有限多选。\n"
            "resolved 事件要改写成事前口吻问题，不能在 question 中泄露答案，并且必须输出 resolved_answer；unresolved 事件输出 resolved_answer:null。\n"
            "不要生成复杂分析题、主观题。\n"
            "question 中每个词都必须有明确、无歧义的含义。\n"
            "禁止：包含无法客观验证的形容词或副词（如：低廉、温和、重大、显著、显著增长、历史低位）。\n"
            "可以用：具体数值、具体日期、具体地点、可量化的市场数据。\n"
            f"resolution_rule 必须写清楚官方/公开来源如何结算。结算来源提示：{guidance}\n"
            "返回 JSON 格式：\n"
            '{"items":[{"event_id":"string","question":"string","options":["是","否"],"resolution_date":"YYYY-MM-DD","resolution_rule":"string","event_status":"resolved|unresolved","resolved_answer":"string|null","answer_evidence":"string|null"}]}\n'
            f"事件候选：\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _coerce_item(self, raw_item: dict[str, Any], events_by_id: dict[str, FutureEvent]) -> BenchmarkItem | None:
        event_id = str(raw_item.get("event_id", ""))
        event = events_by_id.get(event_id)
        if event is None:
            return None
        question = str(raw_item.get("question", "")).strip()
        if not question:
            return None
        resolution_date = raw_item.get("resolution_date") or event.resolution_date or event.deadline
        if not resolution_date:
            return None
        event_status = raw_item.get("event_status") or event.event_status
        if event_status not in {"resolved", "unresolved"}:
            event_status = event.event_status
        return BenchmarkItem.model_validate(
            {
                "id": stable_id(event.id, question, prefix="bm"),
                "domain": event.domain,
                "source": event.source,
                "source_url": event.source_url,
                "question": question,
                "options": raw_item.get("options") if isinstance(raw_item.get("options"), list) else ["是", "否"],
                "resolution_date": resolution_date,
                "resolution_rule": str(raw_item.get("resolution_rule", "")).strip(),
                "event_type": event.event_type,
                "event_status": event_status,
                "resolved_answer": raw_item.get("resolved_answer") if raw_item.get("resolved_answer") not in {"", "null", None} else event.resolved_answer,
                "answer_evidence": raw_item.get("answer_evidence") if raw_item.get("answer_evidence") not in {"", "null", None} else event.answer_evidence,
            }
        )
