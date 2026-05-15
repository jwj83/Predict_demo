from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .case_loader import load_cases_jsonl, load_default_sample_cases
    from .rule_search import RuleSearchRunner, result_to_dict
except ImportError:  # pragma: no cover - supports direct script execution.
    from case_loader import load_cases_jsonl, load_default_sample_cases
    from rule_search import RuleSearchRunner, result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated scoring-rule search demo.")
    parser.add_argument("--cases", default="", help="Path to JSONL evaluation cases. Defaults to bundled sample cases.")
    parser.add_argument("--domain", default="politics_governance", help="Domain to adapt the generic rubric to.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--output", default="outputs/search_report.json")
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases, domain=None) if args.cases else load_default_sample_cases()
    if args.domain:
        domain_cases = [case for case in cases if case.domain in {args.domain, "politics", "politics_governance"}]
        cases = domain_cases or cases
    result = RuleSearchRunner().run(
        cases=cases,
        domain=args.domain,
        iterations=args.iterations,
        candidates_per_round=args.candidates_per_round,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    print("Automated scoring-rule search finished.")
    print(f"Domain: {result.domain}")
    print(f"Cases: {result.validation_summary.case_count}")
    print(f"Best rule set: {result.best_rule_set.rule_set_id}")
    print(f"Validation score: {result.validation_summary.validation_score}")
    print(f"Correlation with resolved probability: {result.validation_summary.correlation_with_resolved_probability}")
    print(f"Correlation with Brier score: {result.validation_summary.correlation_with_brier}")
    print(f"Report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
