from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

from predict_bench.domains import DomainConfig, get_domain_config, list_domains
from predict_bench.models import FutureEvent, RawNewsItem, stable_id
from predict_bench.services.llm import OpenAICompatibleLLMClient


class EventExtractor:
    def __init__(
        self,
        llm: OpenAICompatibleLLMClient | None = None,
        batch_size: int = 10,
        max_concurrent: int = 5,
        config: DomainConfig | None = None,
    ) -> None:
        self.llm = llm or OpenAICompatibleLLMClient()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.config = config

    def extract(self, raw_items: list[RawNewsItem]) -> list[FutureEvent]:
        raw_by_id = {item.id: item for item in raw_items}

        batches = []
        for dated_items in self._group_items_by_news_date(raw_items).values():
            for start in range(0, len(dated_items), self.batch_size):
                batch = dated_items[start : start + self.batch_size]
                batches.append(batch)

        results = asyncio.run(self._extract_concurrent(batches, raw_by_id))
        return results

    async def _extract_concurrent(self, batches: list[list[RawNewsItem]], raw_by_id: dict[str, RawNewsItem]) -> list[FutureEvent]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(batch: list[RawNewsItem]) -> list[FutureEvent]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate_json(
                        system=(
                            "你是新闻可问题化事件抽取器。只返回 JSON，不要 Markdown。"
                            "抽取能被改写成明确、有限、可验证预测问题的事件。"
                        ),
                        user=self._build_prompt(batch),
                    )
                )
                events = []
                for raw_event in response.get("events", []):
                    if not isinstance(raw_event, dict):
                        continue
                    event = self._coerce_event(raw_event, raw_by_id)
                    if event is not None:
                        events.append(event)
                return events

        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        events = []
        for batch_events in batch_results:
            events.extend(batch_events)
        return events

    def _group_items_by_news_date(self, raw_items: list[RawNewsItem]) -> dict[str, list[RawNewsItem]]:
        grouped: dict[str, list[RawNewsItem]] = {}
        for item in raw_items:
            grouped.setdefault(self._get_item_news_date(item), []).append(item)
        return grouped

    def _get_item_news_date(self, item: RawNewsItem) -> str:
        if item.published_at:
            try:
                published = item.published_at[:10]
                date.fromisoformat(published)
                return published
            except (IndexError, ValueError):
                pass
        return date.today().isoformat()

    def _get_news_date(self, batch: list[RawNewsItem]) -> str:
        for item in batch:
            return self._get_item_news_date(item)
        return date.today().isoformat()

    def _build_prompt(self, batch: list[RawNewsItem]) -> str:
        items = [
            {
                "raw_item_id": item.id,
                "source": item.source,
                "feed_name": item.feed_name,
                "candidate_domains": item.candidate_domains,
                "title": item.title,
                "description": item.description,
                "published_at": item.published_at,
                "news_date": self._get_item_news_date(item),
                "url": item.link,
            }
            for item in batch
        ]
        today = self._get_news_date(batch)
        domain = self.config.domain if self.config else "auto"
        if self.config:
            event_types = ", ".join(self.config.event_types)
            guidance = self.config.resolution_guidance
            domain_instruction = f"领域固定为 {domain}。"
        else:
            domain_catalog = {
                name: {
                    "event_types": get_domain_config(name).event_types,
                    "resolution_guidance": get_domain_config(name).resolution_guidance,
                }
                for name in list_domains()
            }
            event_types = json.dumps(domain_catalog, ensure_ascii=False)
            guidance = "根据事件所属领域选择对应的公开、官方或主流结算来源。"
            domain_instruction = (
                f"请为每个事件判断 domain，domain 必须是以下之一：{', '.join(list_domains())}。"
                "如果新闻跨领域，把最适合出题的领域放在 domain，其他相关领域放入 secondary_domains。"
            )
        return (
            f"新闻的发布/收录日期为 {today}，请站在该日期的视角判断以下各事件的状态。\n"
            f"{domain_instruction} 从以下新闻/快讯中抽取可问题化事件候选。\n"
            "规则：\n"
            f"  - 如果事件结果在 {today} 之前已经揭晓 → event_status 填 resolved，必须同时填 resolved_answer 和 answer_evidence。\n"
            f"  - 如果事件结果在 {today} 时尚未揭晓 → event_status 填 unresolved，不得填 resolved_answer，resolution_date 填预计揭晓日期。\n"
            "只保留明确、答案空间有限、可用公开来源验证的事件。不要抽取纯评论、历史回顾、无法验证的态度或影响。\n"
            "event_summary 必须只包含可量化的事实，不含主观描述性形容词（如：历史低位、估值低廉、显著增长、温和回升）。\n"
            f"适合的 event_type 包括：{event_types}。\n"
            f"结算来源提示：{guidance}\n"
            "返回 JSON 格式：\n"
            '{"events":[{"raw_item_id":"string","domain":"sports|politics|international|finance|economy|weather","secondary_domains":["string"],"event_type":"string","event_summary":"string","entities":["string"],"event_status":"resolved|unresolved","deadline":"YYYY-MM-DD|null","resolution_date":"YYYY-MM-DD|null","resolved_answer":"string|null","answer_evidence":"string|null"}]}\n'
            f"新闻：\n{json.dumps(items, ensure_ascii=False)}"
        )

    def _coerce_event(self, raw_event: dict[str, Any], raw_by_id: dict[str, RawNewsItem]) -> FutureEvent | None:
        raw_item_id = str(raw_event.get("raw_item_id", ""))
        item = raw_by_id.get(raw_item_id)
        if item is None:
            return None
        deadline_value = raw_event.get("deadline")
        status = str(raw_event.get("event_status", "unresolved")).strip()
        if status not in {"resolved", "unresolved"}:
            status = "unresolved"
        domain = self.config.domain if self.config else str(raw_event.get("domain") or item.domain).strip()
        if domain not in set(list_domains()):
            return None
        secondary_domains = raw_event.get("secondary_domains")
        if not isinstance(secondary_domains, list):
            secondary_domains = item.candidate_domains
        secondary_domains = [str(value).strip() for value in secondary_domains if str(value).strip() in set(list_domains()) and str(value).strip() != domain]
        payload = {
            "id": stable_id(raw_item_id, str(raw_event.get("event_summary", "")), prefix="evt"),
            "domain": domain,
            "secondary_domains": secondary_domains,
            "source": item.source,
            "source_url": item.link,
            "raw_item_id": raw_item_id,
            "event_type": str(raw_event.get("event_type", "sports_event")).strip() or "sports_event",
            "event_summary": str(raw_event.get("event_summary", "")).strip(),
            "entities": raw_event.get("entities") if isinstance(raw_event.get("entities"), list) else [],
            "deadline": deadline_value if deadline_value not in {"", "null", None} else None,
            "event_status": status,
            "resolution_date": raw_event.get("resolution_date") if raw_event.get("resolution_date") not in {"", "null", None} else None,
            "resolved_answer": raw_event.get("resolved_answer") if raw_event.get("resolved_answer") not in {"", "null", None} else None,
            "answer_evidence": raw_event.get("answer_evidence") if raw_event.get("answer_evidence") not in {"", "null", None} else None,
        }
        if not payload["event_summary"]:
            return None
        return FutureEvent.model_validate(payload)
