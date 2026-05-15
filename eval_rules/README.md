# Eval Rules: Automated Scoring-Rule Search

`eval_rules` is the automated trajectory-scoring rule search module inside
`predict_bench`. It demonstrates the benchmark idea:

```text
generic seed rubric
-> LLM/mock candidate rule generation
-> score ReAct forecast trajectories
-> validate rule candidates against resolved outcomes
-> keep the best rule set for further iteration
```

The first version is intentionally lightweight. It can run without an API key
using a deterministic mock generator, while preserving a path to use the
existing OpenAI-compatible LLM client in `predict_bench`.

## Run

```powershell
cd F:\predict_bench\eval_rules
F:\miniconda\envs\predict\python.exe run_demo.py
```

Output:

```text
outputs/search_report.json
```

## Optional LLM Candidate Generation

If `LLM_API_KEY` or `OPENAI_API_KEY` is configured, `rule_generator.py` will try
to use the existing `predict_bench.services.llm.OpenAICompatibleLLMClient` to
generate rule candidates. If the request fails, it falls back to deterministic
mock candidates.

## Case Format

The demo expects JSONL evaluation cases with resolved outcome metrics and saved
forecast traces:

```json
{
  "case_id": "case_001",
  "domain": "politics_governance",
  "question": "...",
  "resolved_answer": "...",
  "candidate_probabilities": [],
  "scoring_metrics": {
    "accuracy": 1.0,
    "brier_score": 0.12,
    "resolved_option_probability": 0.78
  },
  "evidence_items": [],
  "sub_agent_results": [],
  "round_snapshots": [],
  "markdown_report": "..."
}
```
