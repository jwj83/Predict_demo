from __future__ import annotations

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    name: str
    type: str
    parser: str = "generic"
    url: str
    fallback_url: str | None = None
    default_domains: list[str] = Field(default_factory=list)


class DomainConfig(BaseModel):
    domain: str
    sources: list[SourceConfig]
    event_types: list[str]
    resolution_guidance: str
    output_prefix: str


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "sohu_nba": SourceConfig(
        name="sohu_nba",
        type="html",
        parser="sohu",
        url="https://sports.sohu.com/nba.shtml",
        fallback_url="https://rss.sports.sohu.com/rss/nba.xml",
        default_domains=["sports"],
    ),
    "sohu_yingchao": SourceConfig(
        name="sohu_yingchao",
        type="html",
        parser="sohu",
        url="https://sports.sohu.com/yingchao.shtml",
        default_domains=["sports"],
    ),
    "sohu_zhongchao": SourceConfig(
        name="sohu_zhongchao",
        type="html",
        parser="sohu",
        url="https://sports.sohu.com/zhongchao.shtml",
        default_domains=["sports"],
    ),
    "bbc_zhongwen": SourceConfig(
        name="bbc_zhongwen",
        type="rss",
        parser="rss",
        url="https://feeds.bbci.co.uk/zhongwen/simp/rss.xml",
        default_domains=["politics", "international", "economy"],
    ),
    "google_news_sports": SourceConfig(
        name="google_news_sports",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=NBA%20OR%20%E8%8B%B1%E8%B6%85%20OR%20%E4%B8%AD%E8%B6%85%20OR%20%E5%86%A0%E5%86%9B%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["sports"],
    ),
    "google_news_politics": SourceConfig(
        name="google_news_politics",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=%E9%80%89%E4%B8%BE%20OR%20%E6%B3%95%E6%A1%88%20OR%20%E8%AE%AE%E4%BC%9A%20OR%20%E6%94%BF%E5%BA%9C%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["politics"],
    ),
    "google_news_international": SourceConfig(
        name="google_news_international",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=%E5%81%9C%E7%81%AB%20OR%20%E5%88%B6%E8%A3%81%20OR%20%E5%B3%B0%E4%BC%9A%20OR%20%E5%8D%8F%E8%AE%AE%20OR%20%E8%81%94%E5%90%88%E5%9B%BD%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["international", "politics"],
    ),
    "google_news_finance": SourceConfig(
        name="google_news_finance",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=%E9%87%91%E8%9E%8D%20OR%20%E8%82%A1%E5%B8%82%20OR%20%E5%88%A9%E7%8E%87%20OR%20%E6%94%B6%E7%9B%98%20OR%20%E5%B9%B6%E8%B4%AD%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["finance", "economy"],
    ),
    "google_news_economy": SourceConfig(
        name="google_news_economy",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=GDP%20OR%20CPI%20OR%20PMI%20OR%20%E5%A4%B1%E4%B8%9A%20OR%20%E5%A4%AE%E8%A1%8C%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["economy", "finance"],
    ),
    "google_news_weather": SourceConfig(
        name="google_news_weather",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=%E5%8F%B0%E9%A3%8E%20OR%20%E6%9A%B4%E9%9B%A8%20OR%20%E9%A2%84%E8%AD%A6%20OR%20%E9%99%8D%E9%9B%A8%20OR%20%E9%AB%98%E6%B8%A9%20when%3A14d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["weather"],
    ),
    "google_wsj_china": SourceConfig(
        name="google_wsj_china",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=site%3Acn.wsj.com%20%E4%B8%AD%E5%9B%BD%20when%3A30d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["politics", "economy", "international"],
    ),
    "google_wsj_international": SourceConfig(
        name="google_wsj_international",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=site%3Acn.wsj.com%20%28%E5%9B%BD%E9%99%85%20OR%20%E5%81%9C%E7%81%AB%20OR%20%E5%88%B6%E8%A3%81%20OR%20%E5%8D%8F%E8%AE%AE%29%20when%3A30d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["international", "politics"],
    ),
    "google_wsj_finance_economy": SourceConfig(
        name="google_wsj_finance_economy",
        type="rss",
        parser="rss",
        url="https://news.google.com/rss/search?q=site%3Acn.wsj.com%20%28%E9%87%91%E8%9E%8D%20OR%20%E7%BB%8F%E6%B5%8E%20OR%20%E8%82%A1%E5%B8%82%20OR%20%E5%88%A9%E7%8E%87%29%20when%3A30d&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        default_domains=["finance", "economy"],
    ),
    "jiemian_kuaibao": SourceConfig(
        name="jiemian_kuaibao",
        type="html",
        parser="generic",
        url="https://www.jiemian.com/lists/20.html",
        default_domains=["finance", "economy"],
    ),
    "jiemian_finance": SourceConfig(
        name="jiemian_finance",
        type="html",
        parser="generic",
        url="https://www.jiemian.com/lists/2.html",
        default_domains=["finance"],
    ),
    "jiemian_business": SourceConfig(
        name="jiemian_business",
        type="html",
        parser="generic",
        url="https://www.jiemian.com/lists/3.html",
        default_domains=["economy", "finance"],
    ),
    "china_weather": SourceConfig(
        name="china_weather",
        type="html",
        parser="generic",
        url="http://www.weather.com.cn/alarm/",
        default_domains=["weather"],
    ),
    "china_weather_news": SourceConfig(
        name="china_weather_news",
        type="html",
        parser="generic",
        url="http://www.weather.com.cn/news/",
        default_domains=["weather"],
    ),
}


DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "sports": DomainConfig(
        domain="sports",
        sources=[
            SOURCE_CONFIGS["sohu_nba"],
            SOURCE_CONFIGS["sohu_yingchao"],
            SOURCE_CONFIGS["sohu_zhongchao"],
            SOURCE_CONFIGS["google_news_sports"],
        ],
        event_types=[
            "champion",
            "series_winner",
            "match_winner",
            "award",
            "retirement",
            "transfer",
            "injury_return",
            "coach_change",
            "qualification",
        ],
        resolution_guidance="以赛事官方公告、联盟/俱乐部公告、官方赛程赛果或主流媒体一致报道为准。",
        output_prefix="sports",
    ),
    "politics": DomainConfig(
        domain="politics",
        sources=[
            SOURCE_CONFIGS["bbc_zhongwen"],
            SOURCE_CONFIGS["google_news_politics"],
            SOURCE_CONFIGS["google_wsj_china"],
        ],
        event_types=["election", "bill_passage", "resignation", "diplomatic_visit", "policy_approval", "sanction"],
        resolution_guidance="以政府官网、议会/选举委员会公告、法院文件或 BBC 等主流媒体明确报道为准。",
        output_prefix="politics",
    ),
    "international": DomainConfig(
        domain="international",
        sources=[
            SOURCE_CONFIGS["bbc_zhongwen"],
            SOURCE_CONFIGS["google_news_international"],
            SOURCE_CONFIGS["google_wsj_international"],
        ],
        event_types=["ceasefire", "treaty", "summit", "sanction", "un_resolution", "diplomatic_visit"],
        resolution_guidance="以联合国、相关政府/国际组织公告或 BBC 等主流媒体明确报道为准。",
        output_prefix="international",
    ),
    "finance": DomainConfig(
        domain="finance",
        sources=[
            SOURCE_CONFIGS["jiemian_kuaibao"],
            SOURCE_CONFIGS["jiemian_finance"],
            SOURCE_CONFIGS["google_news_finance"],
            SOURCE_CONFIGS["google_wsj_finance_economy"],
        ],
        event_types=["price_threshold", "index_move", "earnings", "merger", "commodity_price", "stock_move"],
        resolution_guidance="必须指定价格/指数/商品口径、日期和公开行情来源；以交易所、公司公告或主流行情数据为准。",
        output_prefix="finance",
    ),
    "economy": DomainConfig(
        domain="economy",
        sources=[
            SOURCE_CONFIGS["jiemian_kuaibao"],
            SOURCE_CONFIGS["jiemian_business"],
            SOURCE_CONFIGS["google_news_economy"],
            SOURCE_CONFIGS["google_wsj_china"],
            SOURCE_CONFIGS["google_wsj_finance_economy"],
        ],
        event_types=["cpi", "gdp", "employment", "central_bank_rate", "pmi", "trade_data"],
        resolution_guidance="以统计局、央行、财政部、海关、IMF/OECD 等官方或权威机构首次发布数据为准。",
        output_prefix="economy",
    ),
    "weather": DomainConfig(
        domain="weather",
        sources=[
            SOURCE_CONFIGS["china_weather"],
            SOURCE_CONFIGS["china_weather_news"],
            SOURCE_CONFIGS["google_news_weather"],
        ],
        event_types=["typhoon_landfall", "rain_warning", "temperature_threshold", "rainfall_threshold", "disaster_warning"],
        resolution_guidance="必须包含地点、日期和气象指标/预警等级；以国家或地方气象部门公开观测/预警为准。",
        output_prefix="weather",
    ),
}


def get_domain_config(domain: str) -> DomainConfig:
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}")
    return DOMAIN_CONFIGS[domain]


def list_domains() -> list[str]:
    return list(DOMAIN_CONFIGS.keys())


def get_source_config(source_name: str) -> SourceConfig:
    if source_name not in SOURCE_CONFIGS:
        raise ValueError(f"Unknown source: {source_name}")
    return SOURCE_CONFIGS[source_name]


def list_sources() -> list[str]:
    return list(SOURCE_CONFIGS.keys())


def get_source_configs(source_names: list[str] | None = None, domain: str | None = None) -> list[SourceConfig]:
    if source_names:
        return [get_source_config(name) for name in source_names]
    if domain and domain != "all":
        get_domain_config(domain)
        return [source for source in SOURCE_CONFIGS.values() if domain in source.default_domains]
    return list(SOURCE_CONFIGS.values())
