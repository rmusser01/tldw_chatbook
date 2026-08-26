---
id: TASK-16325
title: Persist structured claims and enable follow-up Q&A over run evidence
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
updated_date: '2026-08-15 14:36'
labels:
  - research
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Runs currently would store only a final markdown blob. Persist claims with source id and verbatim quote and confidence as a JSON artifact so follow-up questions can be answered from stored evidence without re-spending on search, mirroring tldw_server follow_up_json bounded seed contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extracted claims with source id and quote and confidence are persisted as a run artifact in JSON
- [x] #2 Follow-up questions are answered from stored evidence without new searches when evidence suffices
- [x] #3 Insufficient evidence triggers an explicit fallback to a new search or run rather than a fabricated answer
- [x] #4 The seed shape is bounded (outline plus key claims plus unresolved questions) matching the server follow-up contract
- [x] #5 Tests cover retrieval and the fallback boundary
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD extend deep_search_citations.verify_citations with per-claim detail (sentence-level claims: text, resolved source ids, unknown ids, per-sentence quote verdicts, supported or unverified status)
2. TDD engine packaging: persist claims.json artifact from the citation verification claims
3. TDD engine.answer_follow_up: build the bounded seed (question plus outline <=7 plus key claims <=5 plus unresolved <=5 plus verification counts, matching the server follow_up_json contract), answer via injectable answer_fn (default: synthesis LLM strictly from the seed, INSUFFICIENT marker parse), return an explicit insufficient-evidence fallback verdict instead of a fabricated answer when the seed cannot support the question or no claims artifact exists
4. Tests plus lint plus task close
ADR required: no - additive artifacts and a read-only follow-up path inside the ADR-068 engine contract
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `verify_citations` now returns per-claim detail under `claims`: every sentence of the ORIGINAL answer carrying at least one `[n]` marker becomes `{claim_id, text, source_ids, unknown_marker_ids, quotes_checked, quotes_verified, status}` with status `supported` only when all markers resolve AND every quote in the sentence verified (sentences are taken from the original text — the annotated `[n?]` form would not match the marker regex and the record should quote what was written). The pipeline's `citation_verification` payload whitelist carries `claims` through to consumers.
- Engine packaging persists `claims.json` (`{claims, claim_count, supported_claim_count, unverified_claim_count}`) whenever the synthesis verdict includes claims.
- `LocalResearchEngine.answer_follow_up(run_id, question, answer_fn=...)`: builds the bounded seed from stored artifacts (outline = plan sub-questions ≤7, key_claims = supported-first ≤5, unresolved = bundle remaining_gaps ≤5, plus verification/source-trust counts — field names match the server `follow_up_json` contract), then answers via an injectable `answer_fn`. The default uses the synthesis LLM with a strict ONLY-from-seed prompt and an `INSUFFICIENT_EVIDENCE` marker; no LLM configured → honestly insufficient. Every insufficient path (no claims artifact — the answerer is never even called — LLM marker, call failure) returns `{status: "insufficient_evidence", answer: None, reason, suggestion}` instead of a fabricated answer. Exchanges record `follow_up_answered`/`follow_up_insufficient` events.
- Mid-task correction: this branch was missing the task-16331 module (it lived on the sibling `feat/deep-search-citation-verification` branch), so that branch was merged in (da110e093) before the work continued — the stack is now linear.
- Verified TDD: 3 citation claims tests + 4 engine tests written first and watched failing; combined `Tests/Web_Scraping/ + Tests/Tools/test_web_deep_search.py + Tests/Research/` = 312 passed, 3 skipped; ruff clean. Files: `deep_search_citations.py`, `WebSearch_APIs.py` (whitelist), `local_research_engine.py`, plus tests.
<!-- SECTION:NOTES:END -->
