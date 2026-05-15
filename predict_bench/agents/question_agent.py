from __future__ import annotations

from collections import defaultdict

from predict_bench.domains import DomainConfig, SourceConfig, get_domain_config, get_source_configs
from predict_bench.models import AgentRunResult, BenchmarkItem, SourceAgentRunResult, ValidationIssue
from predict_bench.services.event_extractor import EventExtractor
from predict_bench.services.question_generator import QuestionGenerator
from predict_bench.services.storage import JsonStorage
from predict_bench.services.validator import BenchmarkValidator
from predict_bench.sources.generic import ConfiguredNewsSource
from predict_bench.sources.sohu_sports import SohuSportsRSSSource


class QuestionAgent:
    def __init__(
        self,
        config: DomainConfig | None = None,
        source: ConfiguredNewsSource | SohuSportsRSSSource | None = None,
        extractor: EventExtractor | None = None,
        generator: QuestionGenerator | None = None,
        validator: BenchmarkValidator | None = None,
        storage: JsonStorage | None = None,
    ) -> None:
        self.config = config or get_domain_config("sports")
        self.source = source or ConfiguredNewsSource(self.config)
        self.extractor = extractor or EventExtractor(config=self.config)
        self.generator = generator or QuestionGenerator(config=self.config)
        self.validator = validator or BenchmarkValidator()
        self.storage = storage or JsonStorage()

    def run(
        self,
        limit: int = 10,
        feeds: list[str] | None = None,
        max_items_per_feed: int = 10,
    ) -> AgentRunResult:
        print(f"[1/5] Fetching {self.config.domain} news...")
        raw_items = self.source.fetch(feed_names=feeds, max_items_per_feed=max_items_per_feed)
        raw_path = self.storage.save_raw([item.model_dump(mode="json") for item in raw_items], domain=self.config.domain)
        print(f"[1/5] Saved {len(raw_items)} raw news items to {raw_path}")

        print("[2/5] Extracting question-worthy events with LLM...")
        events = self.extractor.extract(raw_items)
        events_path = self.storage.save_events([event.model_dump(mode="json") for event in events], domain=self.config.domain)
        print(f"[2/5] Saved {len(events)} event candidates to {events_path}")

        print("[3/5] Generating benchmark questions with LLM...")
        generated = self.generator.generate(events, limit=limit * 2)
        print(f"[3/5] Generated {len(generated)} candidate questions")

        print("[4/5] Validating questions...")
        accepted, rejected = self.validator.validate_many(generated, limit=limit)

        print("[5/5] Saving benchmark questions...")
        benchmark_path = self.storage.save_benchmark(
            [item.model_dump(mode="json") for item in accepted],
            domain=self.config.domain,
        )
        print(f"[5/5] Saved {len(accepted)} benchmark questions to {benchmark_path}")

        return AgentRunResult(
            raw_path=str(raw_path),
            events_path=str(events_path),
            benchmark_path=str(benchmark_path),
            raw_count=len(raw_items),
            event_count=len(events),
            item_count=len(accepted),
            rejected=rejected,
            items=accepted,
        )


class SportsQuestionAgent(QuestionAgent):
    def __init__(
        self,
        source: SohuSportsRSSSource | None = None,
        extractor: EventExtractor | None = None,
        generator: QuestionGenerator | None = None,
        validator: BenchmarkValidator | None = None,
        storage: JsonStorage | None = None,
    ) -> None:
        config = get_domain_config("sports")
        super().__init__(
            config=config,
            source=source,
            extractor=extractor,
            generator=generator,
            validator=validator,
            storage=storage,
        )


class SourceFirstQuestionAgent:
    def __init__(
        self,
        sources: list[SourceConfig] | None = None,
        source: ConfiguredNewsSource | None = None,
        extractor: EventExtractor | None = None,
        generator: QuestionGenerator | None = None,
        validator: BenchmarkValidator | None = None,
        storage: JsonStorage | None = None,
    ) -> None:
        self.source_configs = sources or get_source_configs()
        self.source = source or ConfiguredNewsSource(sources=self.source_configs)
        self.extractor = extractor or EventExtractor(config=None)
        self.generator = generator
        self.validator = validator or BenchmarkValidator()
        self.storage = storage or JsonStorage()

    def run(
        self,
        limit: int = 10,
        source_names: list[str] | None = None,
        domain_filter: str | None = None,
        max_items_per_feed: int = 10,
    ) -> SourceAgentRunResult:
        selected_label = ",".join(source_names) if source_names else "configured sources"
        print(f"[1/5] Fetching news from {selected_label}...")
        raw_items = self.source.fetch(feed_names=source_names, max_items_per_feed=max_items_per_feed)
        raw_path = self.storage.save_raw([item.model_dump(mode="json") for item in raw_items], domain="sources")
        print(f"[1/5] Saved {len(raw_items)} raw news items to {raw_path}")

        print("[2/5] Extracting question-worthy events and classifying domains with LLM...")
        events = self.extractor.extract(raw_items)
        events_path = self.storage.save_events([event.model_dump(mode="json") for event in events], domain="sources")
        print(f"[2/5] Saved {len(events)} event candidates to {events_path}")

        if domain_filter:
            events = [event for event in events if event.domain == domain_filter]
            print(f"[2/5] Kept {len(events)} events for domain={domain_filter}")

        events_by_domain: dict[str, list] = defaultdict(list)
        for event in events:
            events_by_domain[event.domain].append(event)

        print("[3/5] Generating benchmark questions with LLM...")
        benchmark_paths: dict[str, str] = {}
        all_items: list[BenchmarkItem] = []
        all_rejected: list[ValidationIssue] = []
        for domain, domain_events in sorted(events_by_domain.items()):
            generator = self.generator or QuestionGenerator(config=get_domain_config(domain))
            generated = generator.generate(domain_events, limit=limit * 2)
            print(f"[3/5] Generated {len(generated)} candidate questions for {domain}")

            print(f"[4/5] Validating {domain} questions...")
            accepted, rejected = self.validator.validate_many(generated, limit=limit)
            all_items.extend(accepted)
            all_rejected.extend(rejected)

            print(f"[5/5] Saving {domain} benchmark questions...")
            benchmark_path = self.storage.save_benchmark(
                [item.model_dump(mode="json") for item in accepted],
                domain=domain,
            )
            benchmark_paths[domain] = str(benchmark_path)
            print(f"[5/5] Saved {len(accepted)} {domain} benchmark questions to {benchmark_path}")

        return SourceAgentRunResult(
            raw_path=str(raw_path),
            events_path=str(events_path),
            benchmark_paths=benchmark_paths,
            raw_count=len(raw_items),
            event_count=len(events),
            item_count=len(all_items),
            rejected=all_rejected,
            items=all_items,
        )
