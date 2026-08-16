---
id: TASK-16794
title: Record a live biomedical-lane baseline
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-16 14:31'
updated_date: '2026-08-16 14:37'
labels:
  - research
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The recorded live baseline covers the academic lane's arXiv path only; the biomedical and repository providers have no live measurement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] The baseline script gains a --providers flag accepting ids or categories,Biomedical questions are run through the lane with providers configured accordingly and scored,The baseline doc records the new live numbers alongside the existing academic-lane table
<!-- AC:END -->

## Implementation Notes (partial)

- `--providers` shipped in this PR: accepts source ids or category names (expanded via the catalog), binding the lane to that set. AC 1 complete.
- **Blocked on the local LLM endpoint**: the recorded baselines use the llama.cpp server at 127.0.0.1:52864, which is currently down (connection refused; no other local LLM found on common ports, and no cloud keys are configured). The relevance and synthesis LLMs are hard prerequisites for any scored run. Re-run once the endpoint is back:
  `python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --providers biomedical --llm-base-url http://127.0.0.1:<port>/v1`
  then record the numbers in `Docs/Development/research-report-eval-baseline.md` alongside the existing academic-lane table (ACs 2-3).
