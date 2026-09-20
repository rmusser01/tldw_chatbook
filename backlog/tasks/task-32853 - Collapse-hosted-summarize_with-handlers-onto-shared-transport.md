---
id: TASK-32853
title: Collapse the nine hosted summarize_with handlers onto shared transport
status: In Progress
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-19642.10
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The summarization layer never received the transport collapse: `LLM_Calls/Summarization_General_Lib.py` carries nine hosted `summarize_with_*` handlers (openai :904, anthropic :1109, cohere :1381, groq :1631, openrouter :1813, huggingface :2046, deepseek :2236, mistral :2424, google :2646 — 1,936 LOC of handlers), each re-rolling "session + Retry adapter + post + stream_generator". The 2026-09-17 review documented the drift: only 2 of 16 total copies close the response on abandon; `summarize_with_deepseek` hardcodes `https://api.deepseek.com/chat/completions` (:2257, :2316) ignoring `api_base_url`; three handlers double-yield the accumulated text after the deltas; retry loops are hand-rolled per handler.

Summarization is a chat call + prompt template + `choices[0]` extraction — ~15-30 provider-specific LOC survive per provider. OpenAI-compatible providers should route through `hosted_chat.py`; the non-OpenAI wire shapes (anthropic, google, cohere) get one shared `_post_with_retry` helper with their payload assembly kept.

Constraint: `test_summarization_diagnostic_privacy.py` freezes per-function log call sites, so consolidation re-keys the diagnostic ledger — do it coherently per the task-17387 pattern, not by deleting assertions. Error-STRING return shape is a caller contract; preserve it. ADR required: no — ADR-062-consistent; the ledger re-key follows task-17387's precedent.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-17`-adjacent evidence in `slices/LLM.md`; cascade framing in `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The nine hosted handlers share one transport path (hosted_chat for OpenAI-compatible providers; one `_post_with_retry` for the anthropic/google/cohere wire shapes); each handler is a profile + prompt assembly
- [ ] #2 No hardcoded endpoint URLs remain; `api_base_url` is honored everywhere it exists for chat
- [ ] #3 Abandoned streaming consumers close the response in all nine; the double-yield defect class is gone and pinned
- [ ] #4 Retry-After honoring goes through the shared, bounded policy (no uncapped sleeps)
- [ ] #5 The diagnostic ledger is re-keyed coherently and its tests updated to the new call sites
- [ ] #6 Caller-visible error-string returns are unchanged; ingest analysis paths still work
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase A (safety-net restoration) landed 2026-09-19 via PR #2746 (commits on `fix/cascade-wave1`): the two summarization suites joined `keep_bootstrap_profile`, restoring 106 admission-red tests (the TASK-19642.10 signature -- every `summarize_with_*` reads provider settings through `get_cli_setting` and the broad excepts turn the admission failure into an error STRING); the ten stream-laziness tests unmasked by that fix now accept `recovery_review._OpenAIStream` (TASK-32628's retention wrapper); and the manifest boundary is re-pinned. Both suites are green for the first time since 2026-09-16 -- the collapse work can proceed against a live safety net. Phase B (the six OpenAI-compatible handlers onto `owned_json_post`) and Phase C (anthropic/google/cohere behind one `_post_with_retry`) are not started.
<!-- SECTION:NOTES:END -->
