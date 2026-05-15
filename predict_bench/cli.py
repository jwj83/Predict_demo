from __future__ import annotations

import argparse
import json

from predict_bench.agents.question_agent import SourceFirstQuestionAgent
from predict_bench.domains import get_source_configs, list_domains


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="predict_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate benchmark questions")
    generate.add_argument("--domain", default="sports", choices=[*list_domains(), "all"])
    generate.add_argument("--sources", default=None, help="Comma-separated source names")
    generate.add_argument("--feeds", default=None, help="Deprecated alias for --sources")
    generate.add_argument("--limit", type=int, default=10)
    generate.add_argument("--max-items-per-feed", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        source_names = parse_csv(args.sources) or parse_csv(args.feeds)
        sources = get_source_configs(source_names=source_names, domain=args.domain)
        domain_filter = None if args.domain == "all" else args.domain
        result = SourceFirstQuestionAgent(sources=sources).run(
            limit=args.limit,
            source_names=None,
            domain_filter=domain_filter,
            max_items_per_feed=args.max_items_per_feed,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    return 1
