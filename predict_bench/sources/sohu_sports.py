from __future__ import annotations

import html
import os
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import httpx

from predict_bench.models import RawNewsItem, stable_id, utc_now_iso


SOHU_SPORTS_FEEDS = {
    "nba": "https://rss.sports.sohu.com/rss/nba.xml",
    "lanqiu": "https://rss.sports.sohu.com/rss/lanqiu.xml",
    "yingchao": "https://rss.sports.sohu.com/rss/yingchao.xml",
    "xijia": "https://rss.sports.sohu.com/rss/xijia.xml",
    "zhongchao": "https://rss.sports.sohu.com/rss/zhongchao.xml",
    "zonghetiyu": "https://rss.sports.sohu.com/rss/zonghetiyu.xml",
}

SOHU_SPORTS_PAGES = {
    "nba": "https://sports.sohu.com/nba.shtml",
    "lanqiu": "https://sports.sohu.com/lanqiu.shtml",
    "yingchao": "https://sports.sohu.com/yingchao.shtml",
    "xijia": "https://sports.sohu.com/xijia.shtml",
    "zhongchao": "https://sports.sohu.com/zhongchao.shtml",
    "zonghetiyu": "https://sports.sohu.com/zonghetiyu.shtml",
}

DEFAULT_SPORTS_FEEDS = ["nba", "lanqiu", "yingchao", "xijia", "zhongchao", "zonghetiyu"]


TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = TAG_RE.sub("", value)
    return html.unescape(text).strip()


def decode_response_text(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "charset=gb" in content_type:
        return response.content.decode("gbk", errors="replace")
    if "charset=utf-8" in content_type:
        return response.content.decode("utf-8", errors="replace")
    head = response.content[:1000].decode("ascii", errors="ignore").lower()
    if "charset=gb" in head:
        return response.content.decode("gbk", errors="replace")
    return response.content.decode("utf-8", errors="replace")


def normalize_pub_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError, IndexError, AttributeError):
        return value.strip() or None


class SohuSportsRSSSource:
    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def fetch(self, feed_names: list[str] | None = None, max_items_per_feed: int = 50) -> list[RawNewsItem]:
        selected_feeds = feed_names or DEFAULT_SPORTS_FEEDS
        items: list[RawNewsItem] = []
        errors: list[str] = []
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for feed_name in selected_feeds:
                url = SOHU_SPORTS_FEEDS.get(feed_name)
                if not url:
                    raise ValueError(f"Unknown Sohu sports feed: {feed_name}")
                try:
                    xml_text = self._get_text_with_fallback(client, url)
                    items.extend(parse_rss_xml(xml_text, feed_name=feed_name, max_items=max_items_per_feed))
                except Exception as exc:  # noqa: BLE001 - one broken feed should not stop all feeds.
                    try:
                        page_url = SOHU_SPORTS_PAGES[feed_name]
                        html_text = self._get_text_with_fallback(client, page_url)
                        items.extend(parse_html_page(html_text, feed_name=feed_name, max_items=max_items_per_feed))
                    except Exception as fallback_exc:  # noqa: BLE001
                        errors.append(f"{feed_name}: RSS failed: {exc}; HTML failed: {fallback_exc}")
        if not items and errors:
            raise RuntimeError("All Sohu sports feeds failed: " + " | ".join(errors))
        return items

    def fetch_text(self, url: str) -> str:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            return self._get_text_with_fallback(client, url)

    def _get_text_with_fallback(self, client: httpx.Client, url: str) -> str:
        try:
            response = client.get(url)
            response.raise_for_status()
            return decode_response_text(response)
        except (httpx.RequestError, httpx.HTTPStatusError):
            if url.startswith("https://"):
                try:
                    response = client.get("http://" + url[len("https://") :])
                    response.raise_for_status()
                    return decode_response_text(response)
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass
            return self._urllib_get(url)

    def _urllib_get(self, url: str) -> str:
        urls = [url]
        if url.startswith("https://"):
            urls.append("http://" + url[len("https://") :])
        last_error: Exception | None = None
        for candidate in urls:
            request = urllib.request.Request(
                candidate,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    encoding = response.headers.get_content_charset() or "gbk"
                    return raw.decode(encoding, errors="replace")
            except Exception as exc:  # noqa: BLE001 - keep fetching resilient across old RSS endpoints.
                last_error = exc
        if last_error:
            if os.name == "nt":
                return self._powershell_get(url)
            raise last_error
        raise RuntimeError(f"Unable to fetch RSS URL: {url}")

    def _powershell_get(self, url: str) -> str:
        command = (
            "$ProgressPreference='SilentlyContinue'; "
            f"(Invoke-WebRequest -UseBasicParsing '{url}').Content"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
        )
        return completed.stdout


def parse_rss_xml(xml_text: str, feed_name: str, max_items: int | None = None) -> list[RawNewsItem]:
    root = ET.fromstring(xml_text.strip())
    fetched_at = utc_now_iso()
    parsed_items: list[RawNewsItem] = []
    for item in root.findall(".//item"):
        title = strip_html(item.findtext("title"))
        link = strip_html(item.findtext("link"))
        description = strip_html(item.findtext("description"))
        published_at = normalize_pub_date(item.findtext("pubDate"))
        if not title or not link:
            continue
        parsed_items.append(
            RawNewsItem(
                id=stable_id(feed_name, title, link, prefix="raw"),
                domain="sports",
                source="sohu_sports",
                feed_name=feed_name,
                title=title,
                link=link,
                description=description,
                published_at=published_at,
                fetched_at=fetched_at,
            )
        )
        if max_items is not None and len(parsed_items) >= max_items:
            break
    return parsed_items


def parse_html_page(html_text: str, feed_name: str, max_items: int | None = None) -> list[RawNewsItem]:
    fetched_at = utc_now_iso()
    decoded = html_text
    seen_links: set[str] = set()
    parsed_items: list[RawNewsItem] = []
    anchor_pattern = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    for match in anchor_pattern.finditer(decoded):
        link = html.unescape(match.group(1).strip())
        title = strip_html(match.group(2))
        if not title or title in {"阅读全文>>", "更多"}:
            continue
        if not _looks_like_sohu_article(link):
            continue
        if link in seen_links:
            continue
        seen_links.add(link)
        parsed_items.append(
            RawNewsItem(
                id=stable_id(feed_name, title, link, prefix="raw"),
                domain="sports",
                source="sohu_sports",
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


def _looks_like_sohu_article(link: str) -> bool:
    return bool(re.search(r"https?://(?:www\.)?sohu\.com/a/\d+_", link))
