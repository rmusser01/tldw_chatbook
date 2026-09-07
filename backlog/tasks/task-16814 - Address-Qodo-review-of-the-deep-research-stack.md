---
id: TASK-16814
title: Address Qodo review of the deep-research stack
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 04:39'
updated_date: '2026-08-16 04:47'
labels:
  - research
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo reviewed the merged deep-research stack (PRs 1707/1708/1670) and filed 8 bugs and 4 rule violations across them: zero-doc batch raises, estimate path omitting system_message and over-counting multimodal payloads, worker-side _set_status mutation, unknown-citation sentences counted uncited, search reservations never released when fan-out stops early, serial paper search, truthy-string config cast for the academic toggle, tokens_estimated always true, plus docstring/import-order/transaction-compliance findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 allot_docs(0) returns 0 even with an exhausted budget,Estimated prompt tokens include the system message and do not explode on list-shaped (multimodal) message content,Status updates from the engine worker dispatch through the UI message pump rather than mutating widgets directly,uncited_sentences is computed on the original answer so unknown-marker sentences are not miscounted as uncited,Unused search reservations are released when the pipeline stops fan-out early so budgets reflect executed searches,Paper search runs both providers concurrently with per-provider degradation preserved,String config values parse as booleans correctly for the academic toggle default,tokens_estimated reflects whether settled usage was exact or estimated,update_run_progress external-DB branch follows the service transaction pattern,Docstring and import-order findings fixed,Tests cover each behavioral fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Bugs fixed**: (1) `allot_docs(0)` returns 0 on an exhausted budget (nothing to process consumes nothing). (2) Estimated prompt tokens now include `system_message` via `_estimate_prompt_text`, which also takes only text parts of list-shaped multimodal content so base64 payloads cannot explode estimates. (3) Engine-worker status errors dispatch via `call_later` (UI message pump) instead of direct widget mutation. (4) `uncited_sentences` computed on the ORIGINAL answer, so unknown-marker sentences (`[n?]` in the annotated form) are no longer miscounted as uncited. (5) `_collect_round` settles EXECUTED searches (parsing the pipeline's "searched N of M queries" early-stop warning) and `release_searches()` returns unused reservations, so early fan-out stops no longer exhaust `max_searches` prematurely. (6) `search_papers` runs both providers via `asyncio.gather` (per-provider degradation preserved inside each lane). (7) `_parse_config_bool` handles string config values for the academic toggle ("false"/"0"/"no"/"off" are False). (8) `tokens_estimated` reflects reality: `UsageTokenRecorder` splits exact (`record_usage`) vs estimated (`record_exchange`) counts, the engine settles with `exact=`, and the snapshot flag is true only when settled usage includes estimates.
- **Rules fixed**: `update_run_progress`'s external-DB branch now follows delete_run's `transaction()` + raw-statement precedent (with version bump, fake-external-DB test); Google-style docstrings added to `match_quote_in_sources`, `execute_run`, `record_usage`, `resolve_semantic_scholar_api_key`, and ResearchScreen's `compose_content`/`save_state`/`restore_state`; the WebSearch_APIs import block re-sorted (ruff isort).
- Two behavioral pins updated to the corrected semantics (record_usage counts are exact -> `tokens_estimated` False; snapshot test settles estimate tokens so its flag stays True).
- Verified TDD: 9 new tests (zero-doc allot, exactness flag, reservation release, unknown-marker uncited, system-message estimate, multimodal cap, string-bool parse, provider concurrency via deadlock-proof events, external-DB transaction) all written first and watched failing; full remediation sweep 303 passed; ruff clean.
<!-- SECTION:NOTES:END -->
