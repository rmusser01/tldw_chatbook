---
id: TASK-34387
title: Pool the default egress session; drop redundant deepcopies on hosted streams
status: Done
created_date: 2026-10-11 21:43
labels:
- performance
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-11 23:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
create_default_session() is constructed per LLM request (LLM_API_Calls.py and 6+ other sites), so no HTTP keep-alive ever happens: every turn pays fresh DNS+TCP+TLS. Also MoonshotStream/ZAIStream __next__ deepcopy an event that HostedChatStream already returned as a fresh copy, and hosted_chat_request deepcopies the full request payload per retry attempt. Builds on the default-timeout posture from task-19830 (Done).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 create_default_session reuses one process-wide session per (default-timeout, verify) key instead of building a new TCP+TLS connection per request,Context-manager exit no longer closes the shared session, and an explicit close path exists for tests/shutdown,MoonshotStream.__next__ and ZAIStream.__next__ mutate the received event instead of deepcopying it,hosted_chat_request sends the payload without a per-attempt deepcopy,Existing egress + hosted-stream tests pass unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Utils/egress.py create_default_session/DefaultTimeoutSession and its tests
2. Add process-wide session pool keyed on (default-timeout, verify); context exit stops closing the shared session; expose close for tests/shutdown
3. Remove wrapper deepcopies in MoonshotStream/ZAIStream and the per-attempt payload deepcopy in hosted_chat_request
4. Extend/add targeted tests; run egress + hosted stream suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Pooled sessions keyed (timeout,verify,retry-signature,prefixes); no-op close on pooled instances; 44 mount sites migrated; 3 deepcopies removed; 65 test patch sites converted; pool-contract tests added
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

- `Utils/egress.py`: `create_default_session()` now returns a process-wide pooled
  session (`PooledDefaultTimeoutSession`, a `DefaultTimeoutSession` subclass) keyed on
  `(resolved default timeout, TLS verify policy, full mounted-retry signature,
  mount prefixes)` — distinct retry/trust configurations never share a session.
  Context-manager exit and `close()` are no-ops on pooled instances (the pool owns
  lifetime; a caller closing "its" session cannot break sharers — qwencloud's
  non-streaming `_best_effort_close` now becomes a harmless no-op). New
  `create_pooled_default_session(mount_retry=..., mount_prefixes=...)` mounts the
  retry adapter once at pool creation, and `close_pooled_default_sessions()` is the
  explicit release path for tests/shutdown. Pool is capped at 8 entries.
- Migrated all 44 adapter-mount sites (LLM_API_Calls ×9 incl. OpenAI/Anthropic/
  Cohere/DeepSeek/Google/Groq/HuggingFace/Mistral/OpenRouter, Summarization_General
  ×16, Local_Summarization ×15, LLM_API_Calls_Local ×2, qwencloud ×1, hosted_chat ×1)
  from `HTTPAdapter` + `session.mount` onto shared sessions to the pooled factory;
  a scripted rewrite was verified by py_compile plus a scope-aware forward-reference
  checker (branch-local `Retry` definitions required relocating several blocks).
- Removed the double `deepcopy` per streamed chunk (`MoonshotStream.__next__`,
  `ZAIStream.__next__` — `HostedChatStream` already hands out a fresh copy per event)
  and the full request-payload deepcopy per attempt in `hosted_chat_request`.
- Tests: pooled-contract tests added to `Tests/Utils/test_egress.py` (identity,
  timeout/retry key separation, mounted adapter persistence, close semantics; plus an
  autouse pool-reset fixture because tests mount recording adapters onto the shared
  session). 65 test monkeypatch sites across 11 files converted to patch both seams
  (`create_default_session` kept for plain paths, `create_pooled_default_session`
  with arity-tolerant wrappers added). Qwencloud real-server tests' session-close
  assertions updated to the pooled contract (sessions are no longer closed per call).
- Verification: Tests/Utils/test_egress.py + test_tls_trust.py 151 passed;
  Tests/LLM_Calls/ 1111+ passed with only pre-existing failures; Chat suites rerun
  after conversion. Pre-existing failures on this branch (verified by stash-running
  at HEAD): `test_manifest_boundary_*` ×3 in test_summarization_diagnostic_privacy
  and the `Utils/tls_trust.py:273` bare-session finding in the task-19830
  architecture guard. The full 7-file Chat set finishes 687/687 after the seam
  conversions (the earlier huggingface/chat_functions failures were also stale-seam
  effects, now fixed).
- Known follow-up: `Web_Scraping/WebSearch_APIs.py:3732` still builds a private
  HTTPAdapter on its own session (out of LLM scope, left untouched).
- Deviation: AC said "context-manager exit no longer closes the shared session";
  implemented additionally as no-op `close()` — qwencloud's deliberate close-on-exhaust
  made a close-only-on-exit design unsafe for sharing.
