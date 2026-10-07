---
id: TASK-34418
title: Provider HTTP session reuse and payload deepcopy removal ADR-214
status: Done
created_date: 2026-10-07 02:41
dependencies:
- TASK-34417
updated_date: 2026-10-07 06:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 / F8b: full message-history payload is deepcopied per POST attempt and every provider call opens a fresh requests Session paying TLS handshake per turn
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-214 written before code,Per-thread session registry with same-key reuse,No cross-thread session sharing,Payload passed by reference with mutation probe test,TLS handshake count evidence recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 6 (T6)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- ADR: `backlog/decisions/222-provider-http-session-reuse.md` (numbered
  222 at creation; the brief's "ADR-214" was a placeholder, 221 was the
  highest canonical number -- verified per lessons-backlog-hygiene).
- New module `tldw_chatbook/LLM_Calls/provider_sessions.py`:
  `get_session(key, factory)` per-thread (`threading.local`) registry,
  `close_all_for_current_thread()`, `close_session(key)`, plus guarded
  `trust_setting_fragment()` / `default_timeout_fragment()` key
  fragments (guarded because cold test sandboxes refuse the config
  bootstrap the factory's own read uses; production reads succeed).
- Call sites swapped to the registry: `hosted_chat.owned_json_post`,
  `qwencloud.chat_with_qwencloud`, `Summarization_General_Lib
  ._post_with_retry`, and in `LLM_API_Calls.py`: OpenAI embeddings,
  Anthropic, Cohere, Google Gemini, HuggingFace streaming +
  non-streaming. Adapter mounts moved into per-site factories so cached
  sessions keep their warm pool; retry-budget and trust fragments are
  part of each key so settings changes get a new session.
- Deliberate exclusions (ADR-222 section 5): OpenAI chat streaming /
  non-streaming stay per-call (`recovery_review.openai_post` transfers
  session ownership to the guarded recovery `_Operation`, which closes
  it at operation end and sets `trust_env=False` in recovered mode);
  local-provider paths (`LLM_API_Calls_Local.py`,
  `Local_Summarization_Lib.py`) are localhost and unchanged.
- Streams now own only their response: `OwnedSSEStream` /
  `QwenCloudStream` take an optional session (provider calls pass None).
- Payload deepcopy removed in `owned_json_post` (`json=payload`);
  response-side `deepcopy(dict(result))` kept. Mutation-probe + golden
  wire-body tests pin it.
- `Tests/conftest.py` gained an autouse `reset_provider_session_registry`
  fixture (registry hermeticity across tests); existing session-close
  lifecycle pins in test_hosted_chat / test_qwencloud* updated to the
  registry contract; within-test re-patch loops reset the registry.
- Evidence: local HTTP/1.1 counting server, 3 back-to-back
  `owned_json_post` calls -- pristine HEAD 361e80235b: 3 TCP
  connections; this branch: 1 (10 calls: still 1). Live provider smoke
  skipped honestly: no provider API keys configured in this environment
  (config + env checked for presence only).
- Tests: `Tests/LLM_Calls` before 9 failed / 2011 passed / 46 skipped,
  after 9 failed / 2025 passed / 46 skipped -- identical failure list
  (all pre-existing `RecoveryRequired: raw_source_selection_changed`
  sandbox-state failures). New: `Tests/LLM_Calls
  /test_provider_session_reuse.py` (14 tests, green). Ruff: no new
  findings in edited files (per-file counts equal HEAD); new files clean.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
