---
id: TASK-16794
title: Record a live biomedical-lane baseline
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 14:31'
updated_date: '2026-08-16 15:45'
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
- [x] #1 The baseline script gains a --providers flag accepting ids or categories,Biomedical questions are run through the lane with providers configured accordingly and scored,The baseline doc records the new live numbers alongside the existing academic-lane table
<!-- AC:END -->



## Implementation Notes (partial)

- `--providers` shipped in this PR: accepts source ids or category names (expanded via the catalog), binding the lane to that set. AC 1 complete.
- **RESOLVED (endpoint on :9191)**; the original blocker note follows:
- ~~Blocked on the local LLM endpoint~~: the recorded baselines use the llama.cpp server at 127.0.0.1:52864, which is currently down (connection refused; no other local LLM found on common ports, and no cloud keys are configured). The relevance and synthesis LLMs are hard prerequisites for any scored run. Re-run once the endpoint is back:
  `python3 Helper_Scripts/Benchmarks/record_research_baseline.py --questions 3 --engine duckduckgo --academic --providers biomedical --llm-base-url http://127.0.0.1:<port>/v1`
  then record the numbers in `Docs/Development/research-report-eval-baseline.md` alongside the existing academic-lane table (ACs 2-3).

## Implementation Notes (completion)

- Live run (2026-08-16, PubMed lane via `--providers biomedical`, local Qwen3.8-27B on :9191): citation_accuracy **1.00 (36/36 markers)**, claim_support 1.00, gate_pass 0.93, cited_sentence_ratio 0.51 — all three questions scored. Comparison table against the academic lane added to the baseline doc.
- Real bug found and fixed by the live run: the script launched runs with the service-default `checkpointed` autonomy, so every run parked at plan review and produced no report — it now launches `autonomous` (the baseline measures the pipeline, not the checkpoint UX).
- quote_grounding 0.00 on this run set: the model emitted no quoted spans (untested, not failing) — same pattern as the pre-hardening academic runs.
