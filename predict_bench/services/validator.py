from __future__ import annotations

from datetime import date

from predict_bench.models import BenchmarkItem, ValidationIssue


AUTHORITATIVE_SOURCES = {
    "wsj", "bbc", "reuters", "路透", "彭博", "bloomberg",
    "nyt", "金融时报", "ft.com", "economist", "经济学人",
    "imf", "oecd", "央行", "国家统计局", "海关总署",
    "cnbc", "cnn", "ap", "afp",
}

SPORTS_TEAM_MARKERS = (
    "队", "队", "俱乐部", "球队", "火箭", "湖人", "热火", "勇士",
    "鹈鹕", "雷霆", "魔术", "凯尔特人", "快船", "太阳", "马刺",
    "雄鹿", "掘金", "76人", "篮网", "步行者", "公牛", "黄蜂",
    "灰熊", "爵士", "森林狼", "老鹰", "活塞", "骑士", "奇才",
    "国王", "尼克斯", "猛龙", "独行侠", "开拓者",
    "皇马", "巴萨", "曼联", "曼城", "切尔西", "利物浦", "阿森纳",
    "拜仁", "尤文", "AC米兰", "巴黎", "恒大", "上港", "国安",
    "中国", "日本", "韩国", "美国", "德国", "巴西", "阿根廷",
)

FINANCE_MARKERS = (
    "指数", "股价", "美元", "欧元", "英镑", "日元", "汇率",
    "上证", "深证", "纳斯达克", "道琼斯", "标普", "恒生", "日经",
    "期货", "黄金", "原油", "布伦特", "WTI", "铜", "铝",
    "涨", "跌", "收", "开盘", "最高", "最低", "交易所", "行情",
)

ECONOMY_MARKERS = (
    "CPI", "GDP", "PMI", "PPI", "利率", "就业", "失业",
    "央行", "美联储", "欧洲央行", "统计局", "海关", "财政部",
    "IMF", "OECD", "发布", "数据", "预期", "实际",
)

POLITICS_MARKERS = (
    "政府", "议会", "国会", "参议院", "众议院", "选举", "投票",
    "法院", "总统", "总理", "首相", "部长", "议案", "法律",
    "联合国", "官方", "BBC", "主流媒体",
)

WEATHER_MARKERS = (
    "城市", "地区", "气温", "温度", "降雨", "雨量", "台风",
    "预警", "气象", "登陆", "毫米", "℃", "高温", "低温",
    "暴雨", "暴雪", "洪水", "干旱",
)

SOURCE_AUTHORITY_CACHE: dict[str, bool] = {}


def is_authoritative_source(source: str) -> bool:
    if source not in SOURCE_AUTHORITY_CACHE:
        lower = source.lower()
        SOURCE_AUTHORITY_CACHE[source] = any(marker in lower for marker in AUTHORITATIVE_SOURCES)
    return SOURCE_AUTHORITY_CACHE[source]


def get_domain_markers(domain: str) -> tuple[str, ...]:
    return {
        "sports": SPORTS_TEAM_MARKERS,
        "finance": FINANCE_MARKERS,
        "economy": ECONOMY_MARKERS,
        "politics": POLITICS_MARKERS,
        "international": POLITICS_MARKERS,
        "weather": WEATHER_MARKERS,
    }.get(domain, ())


class BenchmarkValidator:
    OPEN_ENDED_MARKERS = ("怎么看", "如何看待", "为什么", "是否影响", "是否值得关注", "有什么影响", "如何")
    SUBJECTIVE_ADJECTIVE_MARKERS = (
        "低廉", "温和", "重大", "显著", "明显", "轻微", "强劲",
        "低迷", "火热", "火爆", "疲软", "萎缩", "回暖", "过热",
        "历史低位", "历史高位", "历史新高", "历史新低", "相对低位",
        "显著增长", "温和回升", "温和上升", "温和下降",
        "大幅增长", "大幅下降", "小幅", "轻微",
    )

    def validate_one(self, item: BenchmarkItem) -> tuple[str | None, float]:
        reason = self._basic_validate(item)
        if reason is not None:
            return reason, 0.0

        score = self._score(item)
        if score < 55.0:
            reason = f"quality_score {score:.1f} below minimum 55"
            return reason, score

        return None, score

    def _basic_validate(self, item: BenchmarkItem) -> str | None:
        if item.event_status not in {"resolved", "unresolved"}:
            return "event_status must be resolved or unresolved"
        if len(item.options) < 2:
            return "options must contain at least two values"
        if len(item.options) > 10:
            return "options must contain at most ten values"
        if len(set(item.options)) != len(item.options):
            return "options must be unique"
        if not item.resolution_rule.strip():
            return "missing resolution_rule"
        if not item.source_url.strip():
            return "missing source_url"
        if any(marker in item.question for marker in self.OPEN_ENDED_MARKERS):
            return "question is subjective or open-ended"
        if any(marker in item.question for marker in self.SUBJECTIVE_ADJECTIVE_MARKERS):
            return "question contains vague subjective adjective"
        if item.event_status == "resolved":
            if not item.resolved_answer:
                return "resolved item missing resolved_answer"
            if item.resolved_answer and item.resolved_answer not in item.options:
                return "resolved_answer must be one of options"
        if item.event_status == "unresolved":
            if item.resolved_answer:
                return "unresolved item must not include resolved_answer"
        return None

    def _score(self, item: BenchmarkItem) -> float:
        score = 0.0
        now = date.today()

        if item.resolution_date:
            score += 30.0

        text = f"{item.question} {item.resolution_rule}"
        if any(marker in text for marker in (
            "官方", "官网", "交易所", "委员会", "统计局", "BBC", "WSJ", "央行",
            "联盟", "赛事", "nba", "fifa", "cba", "新浪", "搜狐", "新华社",
            "气象局", "气象部门", "气象中心", "预报",
        )):
            score += 25.0

        if item.event_status == "resolved" and item.answer_evidence and item.answer_evidence.strip():
            score += 15.0

        if is_authoritative_source(item.source):
            score += 10.0

        domain_markers = get_domain_markers(item.domain)
        if domain_markers and any(marker in text for marker in domain_markers):
            score += 10.0

        if 2 <= len(item.options) <= 4:
            score += 10.0

        return score

    def validate_many(
        self, items: list[BenchmarkItem], limit: int
    ) -> tuple[list[BenchmarkItem], list[ValidationIssue]]:
        accepted: list[BenchmarkItem] = []
        rejected: list[ValidationIssue] = []
        seen_questions: set[str] = set()
        seen_source_questions: set[tuple[str, str]] = set()

        for item in items:
            now = date.today()
            item.event_status = "unresolved"
            if item.resolution_date and item.resolution_date < now:
                item.event_status = "resolved"
                if item.resolved_answer is None:
                    rejected.append(ValidationIssue(item_id=item.id, reason="resolved item missing resolved_answer"))
                    continue

            reason, score = self.validate_one(item)
            item.quality_score = score

            key = item.question.strip()
            source_key = (item.source_url.strip(), key)
            if reason is None and key in seen_questions:
                reason = "duplicate question"
            if reason is None and source_key in seen_source_questions:
                reason = "duplicate question for same source news"

            if reason is not None:
                rejected.append(ValidationIssue(item_id=item.id, reason=reason))
                continue

            accepted.append(item)
            seen_questions.add(key)
            seen_source_questions.add(source_key)

            if len(accepted) >= limit:
                break

        return accepted, rejected