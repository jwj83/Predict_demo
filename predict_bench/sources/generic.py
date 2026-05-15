from __future__ import annotations

from predict_bench.domains import DomainConfig, SourceConfig
from predict_bench.models import RawNewsItem, stable_id, utc_now_iso
from predict_bench.sources.parsers import get_parser
from predict_bench.sources.sohu_sports import SohuSportsRSSSource


class ConfiguredNewsSource:
    def __init__(
        self,
        config: DomainConfig | None = None,
        sources: list[SourceConfig] | None = None,
        timeout: float = 20.0,
    ) -> None:
        if config is None and sources is None:
            raise ValueError("ConfiguredNewsSource requires either a domain config or source configs.")
        self.config = config
        self.sources = config.sources if config is not None else sources or []
        self.fetcher = SohuSportsRSSSource(timeout=timeout)

    def fetch(self, feed_names: list[str] | None = None, max_items_per_feed: int = 10) -> list[RawNewsItem]:
        selected = self._select_sources(feed_names)
        items: list[RawNewsItem] = []
        errors: list[str] = []
        for source in selected:
            try:
                text = self.fetcher.fetch_text(source.url)
                parsed = self._parse_source_text(source, text, max_items_per_feed)
                if not parsed and source.fallback_url:
                    fallback_text = self.fetcher.fetch_text(source.fallback_url)
                    parsed = self._parse_source_text(source, fallback_text, max_items_per_feed)
                items.extend(parsed)
            except Exception as exc:  # noqa: BLE001 - keep domain runs resilient across sources.
                errors.append(f"{source.name}: {exc}")
        if not items and errors:
            label = self.config.domain if self.config else "source collection"
            raise RuntimeError(f"All sources failed for {label}: " + " | ".join(errors))
        return items

    def _select_sources(self, feed_names: list[str] | None) -> list[SourceConfig]:
        if not feed_names:
            return self.sources
        wanted = set(feed_names)
        selected = [source for source in self.sources if source.name in wanted]
        unknown = wanted - {source.name for source in selected}
        if unknown:
            label = self.config.domain if self.config else "source collection"
            raise ValueError(f"Unknown sources for {label}: {', '.join(sorted(unknown))}")
        return selected

    def _parse_source_text(self, source: SourceConfig, text: str, max_items: int) -> list[RawNewsItem]:
        parser = get_parser(source.parser)
        raw_items = parser(text, source.name, source.url, max_items)
        domain = self.config.domain if self.config else "unclassified"
        return [
            RawNewsItem(
                id=stable_id(domain, item.source, item.feed_name, item.title, item.link, prefix="raw"),
                domain=domain,
                candidate_domains=source.default_domains,
                source=source.name,
                feed_name=source.name,
                title=item.title,
                link=item.link,
                description=item.description,
                published_at=item.published_at,
                fetched_at=item.fetched_at or utc_now_iso(),
            )
            for item in raw_items
        ]
