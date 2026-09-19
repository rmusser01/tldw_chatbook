---
id: TASK-32851
title: Migrate groq and openrouter chat handlers onto the hosted_chat engine
status: To Do
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
  - backlog/decisions/062-hosted-chat-completions-provider-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-062 built `hosted_chat.py` as the provider-neutral transport and named deepseek/groq/mistral/openrouter as awaiting "separably tested migrations". Groq and openrouter are the least test-pinned (1 test file each) and the most boilerplate: `chat_with_groq` (`LLM_Calls/LLM_API_Calls.py:4205`, 257 LOC) and `chat_with_openrouter` (`:5229`, 257 LOC) are ~75-80% verbatim transport that the engine already owns. The proven target shape is `moonshot.py::chat_with_moonshot` (87 lines: resolver → payload builder → `hosted_chat_request` → wrap).

Migrating fixes three live defect classes for free: the `yield "data: [DONE]"`-in-`finally` stream leak (groq `:4370-4373`; openrouter same shape `:5389-5392`), dropped streamed usage (neither requests `stream_options.include_usage`), and wrong metric labels (groq logs `openrouter_api_*` streaming / `mistral_api_*` non-streaming at `:4338-4428`).

Depends on TASK-19642.10 (the engine's own 25 contract tests are currently red — migrating onto an unverified engine inverts the evidence order). ADR required: no — this executes ADR-062's stated direction; the engine gains only neutral capabilities (an `extra_headers` hook for openrouter's `HTTP-Referer`/`X-Title`, which `owned_json_post` currently hardcodes away), never provider-specific flags that weaken the hosted contract.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `chat_with_groq` and `chat_with_openrouter` route transport through `hosted_chat.py`; each handler is a provider profile (credential/base/model resolution, payload builder, finish policy) with no hand-rolled session/HTTPAdapter/SSE relay loop
- [ ] #2 Engine additions are neutral (e.g. an extra-headers hook) with no provider-specific weakening of the hosted contract; ADR-062's parity bar is met
- [ ] #3 Streaming Stop closes the transport without yielding after GeneratorExit — the OpenAI pinning-test shape exists for both providers
- [ ] #4 Streamed usage is requested and forwarded so the gateway usage ledger records groq/openrouter streamed turns
- [ ] #5 Metrics name the right provider, and any consumer keying on the old (wrong) labels is identified and updated in the same PR
- [ ] #6 Both providers are covered by the provider-neutral contract suite and existing per-provider tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prep landed 2026-09-19 on branch `fix/cascade-prep` (worktree `tldw-cascades`, commit `533c8ef84f`): `Tests/LLM_Calls/test_groq_openrouter_migration_characterization.py` — 7 pins of CURRENT behavior, each defect pin annotated with what this task must flip. Verified beyond the review's claims while writing them: (a) every normally-completed groq stream ends with a DUPLICATE `data: [DONE]` (the provider's own relayed sentinel, then the synthetic one from the `finally` yield) — consumers tolerate it today, the engine path emits exactly one; (b) groq's EFFECTIVE default temperature is 0.7 (the default config template overrides the code literal 0.2 — two defaults for one knob; the profile should name one); (c) the relayed-line contract is newline-terminated raw lines, which any engine shim must preserve or migrate consumers in the same PR. The blocking dependency is clear: TASK-19642.10 is Done — the engine's contract suite is green (248/248).
<!-- SECTION:NOTES:END -->
