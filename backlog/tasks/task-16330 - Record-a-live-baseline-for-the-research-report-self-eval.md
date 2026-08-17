---
id: TASK-16330
title: Record a live baseline for the research report self-eval
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 16:46'
updated_date: '2026-08-15 20:56'
labels:
  - research
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The self-eval baseline (task-16327) is synthetic: it pins metric definitions but measures nothing about the real pipeline. Add a committed helper that runs real research questions through the engine, scores the resulting verification payloads, and records the aggregate live baseline; then run it and record the numbers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A committed helper script runs N research questions through the local engine with the configured pipeline settings, collects each run's verification payload, and prints or writes the aggregated live metrics,The baseline doc gains a live section with real numbers and the questions used,Script defaults keep spend bounded (small result counts, no subquery fan-out) and are documented,Scoring uses the existing scorer unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD the aggregation logic (mean metrics over scored payloads) as a small function reused by the script
2. Write Helper_Scripts/Benchmarks/record_research_baseline.py: builds a temp LocalResearchService, assembles search params from the tool settings, runs N bounded questions through the real engine defaults, scores each run's verification payload, prints aggregate metrics + JSON
3. Run it live against the configured pipeline (google + both LLMs, bounded spend: 5 results, no subquery fan-out, 3 questions)
4. Record the live numbers in the baseline doc; tests plus lint plus task close
ADR required: no - offline tooling over existing seams
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `aggregate_metrics` added to the scorer (TDD) — mean metrics + sample_count over payload lists; the script is I/O glue over it (mirroring the RAG benchmark precedent).
- `Helper_Scripts/Benchmarks/record_research_baseline.py`: assembles engine params exactly like the `web_deep_search` tool, preflights engine credentials (fails fast with actionable output; `--engine duckduckgo` documented as keyless), supports `--academic` (keyless arXiv+S2 lane), `--llm/--llm-base-url` (primes `api_settings.<provider>.api_url`/`api_timeout` in-process — no config file writes; local models need a bumped request timeout), exits non-zero when no run produced a scorable payload, and writes aggregate JSON via `--json-out`.
- **Live baseline recorded** (2026-08-15, duckduckgo+academic lane, local llama.cpp Qwen3.8-27B): citation_accuracy **1.00** (20/20 markers), claim_support **1.00**, cited_sentence_ratio **0.68** (n=2 scored of 3 questions; 1 run rejected entirely by the relevance gate — recorded as the observed weak link, not hidden). quote_grounding 0.00 because the model emitted no quoted spans (nothing checked — untested, not failing). Numbers + command + configuration in `Docs/Development/research-report-eval-baseline.md`.
- Deviations/limitations (documented): web-lane engines unreachable from this network (DDG/Baidu bot challenge; all keyed engines unconfigured), so the live baseline measures the ACADEMIC-lane configuration; sample size 2 is thin but honestly reported with the one-command procedure to extend.
- Live verification surfaced and fixed three real bugs on the way (each its own commit): OpenAI-dict normalization at `chat_api_call` (local providers returned raw dicts breaking every string consumer — also unlocked REAL token-usage recording for task-16329), arXiv phrase-quoting (token queries surfaced off-topic papers), and question→topic normalization for paper queries. Default questions are academic topics since the evidence pool (with the web lane blocked) is papers.
<!-- SECTION:NOTES:END -->
