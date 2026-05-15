from __future__ import annotations

import html
import re
from typing import Callable
from urllib.parse import urljoin, urlparse

from predict_bench.models import RawNewsItem, stable_id, utc_now_iso
from predict_bench.sources.sohu_sports import parse_html_page as parse_sohu_html_page
from predict_bench.sources.sohu_sports import parse_rss_xml
from predict_bench.sources.sohu_sports import strip_html


ParserFn = Callable[[str, str, str, int | None], list[RawNewsItem]]


def parse_rss(text: str, feed_name: str, base_url: str, max_items: int | None = None) -> list[RawNewsItem]:
    del base_url
    return parse_rss_xml(text, feed_name=feed_name, max_items=max_items)


def parse_sohu(text: str, feed_name: str, base_url: str, max_items: int | None = None) -> list[RawNewsItem]:
    del base_url
    return parse_sohu_html_page(text, feed_name=feed_name, max_items=max_items)


def parse_generic(text: str, feed_name: str, base_url: str, max_items: int | None = None) -> list[RawNewsItem]:
    return parse_generic_html_page(text, feed_name=feed_name, base_url=base_url, max_items=max_items)


PARSERS: dict[str, ParserFn] = {
    "rss": parse_rss,
    "sohu": parse_sohu,
    "generic": parse_generic,
}


def get_parser(name: str) -> ParserFn:
    try:
        return PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown source parser: {name}") from exc


def parse_generic_html_page(html_text: str, feed_name: str, base_url: str, max_items: int | None = None) -> list[RawNewsItem]:
    fetched_at = utc_now_iso()
    base_host = urlparse(base_url).netloc
    seen_links: set[str] = set()
    parsed_items: list[RawNewsItem] = []
    anchor_pattern = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    for match in anchor_pattern.finditer(html_text):
        raw_link = html.unescape(match.group(1).strip())
        title = strip_html(match.group(2))
        if not title or len(title) < 4 or title in {"更多", "阅读全文>>", "Read more"}:
            continue
        link = urljoin(base_url, raw_link)
        parsed = urlparse(link)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc and base_host and not _same_site(parsed.netloc, base_host):
            continue
        if link in seen_links:
            continue
        seen_links.add(link)
        parsed_items.append(
            RawNewsItem(
                id=stable_id(feed_name, title, link, prefix="raw"),
                domain="generic",
                source=feed_name,
                feed_name=feed_name,
                title=title,
                link=link,
                description="",
                published_at=None,
                fetched_at=fetched_at,
            )
        )
        if max_items is not None and len(parsed_items) >= max_items:
            break
    return parsed_items


def _same_site(host: str, base_host: str) -> bool:
    host = host.lower()
    base_host = base_host.lower()
    return host == base_host or host.endswith("." + base_host) or base_host.endswith("." + host)
