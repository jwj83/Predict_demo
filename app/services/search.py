from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

from app.core.config import (
    EXA_API_KEY,
    EXA_SEARCH_TYPE,
    EXA_SEARCH_URL,
    EXA_TIMEOUT_SECONDS,
    SEARCH_PROVIDER,
    SEARCH_RESULTS_PER_QUERY,
)


class SearchProviderProtocol(Protocol):
    def search(self, query: str, limit: int | None = None, published_before: str | None = None) -> list[dict]:
        ...


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_published_at(value: str | None) -> str:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed else ""


class SearchProvider:
    """
    Local deterministic provider used by tests and API-key-free development.
    """

    def search(self, query: str, limit: int | None = None, published_before: str | None = None) -> list[dict]:
        result_limit = limit or SEARCH_RESULTS_PER_QUERY
        now = _parse_datetime(published_before) or datetime.now(timezone.utc)
        safe_slug = query.lower().replace(" ", "-")
        results = []
        for index in range(result_limit):
            results.append(
                {
                    "title": f"{query} | synthetic source {index + 1}",
                    "url": f"https://example.com/{safe_slug}/{index + 1}",
                    "snippet": f"Synthetic evidence snippet for {query}. This placeholder keeps the pipeline runnable.",
                    "source_type": "synthetic_web",
                    "published_at": (now - timedelta(days=index)).isoformat(),
                    "raw": {},
                }
            )
        return results


class ExaSearchProvider:
    def __init__(self, api_key: str = EXA_API_KEY) -> None:
        if not api_key:
            raise ValueError("EXA_API_KEY is required when FORECAST_SEARCH_PROVIDER=exa.")
        self.api_key = api_key

    def search(self, query: str, limit: int | None = None, published_before: str | None = None) -> list[dict]:
        result_limit = limit or SEARCH_RESULTS_PER_QUERY
        payload: dict[str, Any] = {
            "query": query,
            "numResults": result_limit,
            "type": EXA_SEARCH_TYPE,
            "text": {"maxCharacters": 600},
            "highlights": {"numSentences": 2},
        }
        cutoff = _parse_datetime(published_before)
        if cutoff:
            payload["endPublishedDate"] = cutoff.isoformat()

        response = httpx.post(
            EXA_SEARCH_URL,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=EXA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return [self._normalize_result(item) for item in data.get("results", [])]

    def _normalize_result(self, item: dict[str, Any]) -> dict:
        highlights = item.get("highlights") or []
        text = item.get("text") or ""
        snippet = " ".join(str(part) for part in highlights if part) or text[:600] or item.get("summary") or ""
        published_at = _normalize_published_at(item.get("publishedDate") or item.get("published_at"))
        return {
            "title": item.get("title") or item.get("url") or "Untitled",
            "url": item.get("url") or "",
            "snippet": snippet,
            "source_type": "exa_web",
            "published_at": published_at,
            "raw": item,
        }


def create_search_provider(provider_name: str = SEARCH_PROVIDER) -> SearchProviderProtocol:
    if provider_name == "exa":
        return ExaSearchProvider()
    if provider_name in {"synthetic", "local", ""}:
        return SearchProvider()
    raise ValueError(f"Unsupported search provider: {provider_name}")
