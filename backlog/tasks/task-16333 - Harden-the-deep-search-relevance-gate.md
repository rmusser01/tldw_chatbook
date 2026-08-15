---
id: TASK-16333
title: Harden the deep-search relevance gate
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-15 21:32'
updated_date: '2026-08-15 21:32'
labels:
  - research
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The recorded live baseline (task-16330) measured citation integrity at 1.00 but showed the relevance gate is the pipeline's weak link: the prompt demands results comprehensively answer the question, the judgment runs at temperature 0.7 (a classification call), and when the gate rejects every result the run produces NO report at all. 1 of 3 baseline questions died this way and verdicts flipped between identical runs. Make the gate judge usefulness, run deterministically, and degrade to a flagged report instead of nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gate prompt judges usefulness (partial coverage counts, False only for essentially unrelated results) instead of requiring comprehensive coverage,The relevance judgment LLM call runs at a classification temperature at or below 0.2 while the summarization call temperature is unchanged,When the gate rejects every result but raw results exist the pipeline proceeds with the top-ranked results flagged gate-unverified - the flag reaches the evidence entries, the verification summary gains a gate block (relevant and raw counts plus fallback marker), and the web_deep_search footer discloses unverified evidence,A genuinely empty result set keeps today's explanatory no-results path,Gate pass-rate is computed by the self-eval scorer when the payload carries gate counts and the aggregate averages per-key over payloads that have it,Tests pin the temperature the fallback flagging the footer disclosure and the scorer metric
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD prompt plus temperature: rewrite websearch.result_relevance_eval instruction to usefulness semantics and pin the judgment call to a low classification temperature (constant), summarization temp unchanged
2. TDD zero-relevant fallback in search_result_relevance: track evaluated results, and when all are rejected but raw results exist, return the top-ranked few as gate-unverified entries (snippet-level content, no scrape or summarize spend)
3. TDD flag propagation: aggregate_results carries gate_unverified into evidence entries, analyze_and_aggregate attaches a gate block (relevant, raw, fallback) to the final answer, the web_deep_search footer discloses fallback evidence, and the engine verification summary includes the gate block
4. TDD scorer: unwrap citation_verification when present, compute gate_pass_rate from gate counts, aggregate averages per-key over payloads that carry the key
5. Full suites plus lint; re-run the live baseline against the local llamacpp endpoint to measure the improvement; close task
ADR required: no - behavior tuning and a degradation path inside the existing deep-search tool contract; decision recorded in the task and code comments
<!-- SECTION:PLAN:END -->
