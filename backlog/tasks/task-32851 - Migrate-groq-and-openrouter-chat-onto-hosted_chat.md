---
id: TASK-32851
title: Migrate groq and openrouter chat handlers onto the hosted_chat engine
status: Done
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
ADR-062 built `hosted_chat.py` as the provider-neutral transport and named deepseek/groq/mistral/openrouter as awaiting "separately tested migrations". Groq and openrouter are the least test-pinned (1 test file each) and the most boilerplate: `chat_with_groq` (`LLM_Calls/LLM_API_Calls.py:4205`, 257 LOC) and `chat_with_openrouter` (`:5229`, 257 LOC) are ~75-80% verbatim transport that the engine already owns. The proven target shape is `moonshot.py::chat_with_moonshot` (87 lines: resolver → payload builder → `hosted_chat_request` → wrap).

Migrating fixes three live defect classes for free: the `yield "data: [DONE]"`-in-`finally` stream leak (groq `:4370-4373`; openrouter same shape `:5389-5392`), dropped streamed usage (neither requests `stream_options.include_usage`), and wrong metric labels (groq logs `openrouter_api_response_time` streaming / `mistral_api_response_time` non-streaming at `:4338-4428`).

Depends on TASK-19642.10 (the engine's own 25 contract tests are currently red — migrating onto an unverified engine inverts the evidence order). ADR required: no — this executes ADR-062's stated direction; the engine gains only neutral capabilities (an `extra_headers` hook for openrouter's `HTTP-Referer`/`X-Title`, which `owned_json_post` currently hardcodes away), never provider-specific flags that weaken the hosted contract.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `chat_with_groq` and `chat_with_openrouter` route transport through `hosted_chat.py`; each handler is a provider profile (credential/base/model resolution, payload builder, finish policy) with no hand-rolled session/HTTPAdapter/SSE relay loop
- [x] #2 Engine additions are neutral (e.g. an extra-headers hook) with no provider-specific weakening of the hosted contract; ADR-062's parity bar is met
- [x] #3 Streaming Stop closes the transport without yielding after GeneratorExit — the OpenAI pinning-test shape exists for both providers
- [x] #4 Streamed usage is requested and forwarded so the gateway usage ledger records groq/openrouter streamed turns
- [x] #5 Metrics name the right provider, and any consumer keying on the old (wrong) labels is identified and updated in the same PR
- [x] #6 Both providers are covered by the provider-neutral contract suite and existing per-provider tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD red: engine extra_headers contract test; flip the characterization defect pins to the fixed contracts.
2. Engine: neutral extra_headers on HostedHTTPTransportConfig (Authorization/Content-Type not overridable).
3. Provider profiles: LLM_Calls/groq.py + openrouter.py mirroring the moonshot exemplar (resolution, payload allowlist, finish policy, legacy GroqStream/OpenRouterStream line shims, GroqResponse/OpenRouterResponse dicts); LLM_API_Calls re-exports keep entry points stable.
4. Green + blast radius: characterization, engine contract suites, dispatcher mapping; classify any adjacent failures as mine vs pre-existing via a pristine-worktree A/B.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prep landed 2026-09-19 on branch `fix/cascade-prep` (worktree `tldw-cascades`, commit `533c8ef84f`): `Tests/LLM_Calls/test_groq_openrouter_migration_characterization.py` — 7 pins of CURRENT behavior, each defect pin annotated with what this task must flip. Verified beyond the review's claims while writing them: (a) every normally-completed groq stream ends with a DUPLICATE `data: [DONE]` (the provider's own relayed sentinel, then the synthetic one from the `finally` yield) — consumers tolerate it today, the engine path emits exactly one; (b) groq's EFFECTIVE default temperature is 0.7 (the default config template overrides the code literal 0.2 — two defaults for one knob; the profile should name one); (c) the relayed-line contract is newline-terminated raw lines, which any engine shim must preserve or migrate consumers in the same PR. The blocking dependency is clear: TASK-19642.10 is Done — the engine's contract suite is green (248/248).

PR: #2746 (wave 1, base dev). Migration landed 2026-09-19, commit `08ee9c54563` (rebased) (pushed to `origin/fix/cascade-prep`). TDD red→green throughout: 6 tests failed for the right reasons pre-implementation, all green after.

- **Engine (neutral only):** `HostedHTTPTransportConfig.extra_headers` — a validated str→str mapping; `Authorization`/`Content-Type` are rejected in it (case-insensitive) and merged after the engine's own pair. Pinned by `test_owned_json_post_forwards_extra_headers_without_core_overrides`.
- **Profiles:** `tldw_chatbook/LLM_Calls/groq.py` (365 lines) and `openrouter.py` (378) mirror the moonshot exemplar: resolution from the existing `api_settings` tables (unchanged semantics, including groq's effective temp default), payload allowlist + `stream_options.include_usage` when streaming, per-provider finish policy (standard OpenAI reasons + `content_filter` as a classified terminal state), groq/openrouter-labelled metrics, typed-error passthrough (no broad except — parity with the exemplar; the engine raises typed, redacted errors only). OpenRouter passes its `HTTP-Referer`/`X-Title` through `extra_headers`.
- **Legacy surfaces kept stable:** `GroqStream`/`OpenRouterStream` yield newline-terminated `data: ...` lines with exactly one trailing `data: [DONE]\n\n` (close is a method forward — no yield-after-exit is structurally possible); non-streaming returns the legacy choices/usage dict (`GroqResponse`/`OpenRouterResponse`) with the normalized terminal turn attached. Entry points re-exported from `LLM_API_Calls` (`app.py`/`Chat_Functions` imports unchanged).
- **Defects fixed and pinned** (the flipped characterization tests): clean Stop with response closed exactly once; requested + forwarded streamed usage (the trailing usage chunk reaches consumers as a data line the gateway ledger parses); exactly one `[DONE]` per stream; metric series name groq/openrouter. AC#5 consumer check: no consumer of the old (wrong) `openrouter_api_response_time`/`mistral_api_response_time` labels emitted by groq exists outside `LLM_Calls/` (grep verified).
- **LOC honesty:** `LLM_API_Calls.py` −520 lines; the two new profile modules add 743 (policies, shims, docstrings). Net production +246 for the first two providers — the pattern's fixed cost. The deletions compound from here: deepseek+mistral (TASK-32852) are nearly pure subtraction, and their slice should FIRST extract one shared `LegacyLineStream` from the two identical ~60-line shims (noted for 32852).
- **Verification:** 276 passed / 0 failed across the characterization file, `test_hosted_chat.py`, `test_qwencloud.py`, `test_hosted_chat_streaming.py`, and `test_dispatcher_status_mapping.py`. Ruff clean on both new modules (repo config). Adjacent failures classified via a pristine worktree at the branch base: 3 `test_sensitive_llm_logging.py` sentinel failures (drive `chat_with_openai`, untouched) fail identically at `cebe68148b` — pre-existing on dev, not this change.
- **Merge notes:** branch is 2 ahead / 22 behind `origin/dev` — rebase before PR. `Tests/LLM_Calls/` carries 119 pre-existing summarization failures (same admission root cause as TASK-19642.10's notes) — owned by TASK-32853/32854.
<!-- SECTION:NOTES:END -->
