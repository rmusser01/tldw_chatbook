---
id: TASK-28229
title: Rate-limit header telemetry on provider responses
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 06:39'
labels:
  - providers
  - observability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C21's cheap residue, promoted by TASK-26041: the display surface now exists (console cost tracker + cost chip), but zero x-ratelimit-* headers are read anywhere in LLM_Calls/. Parse the standard rate-limit headers off provider responses and surface remaining-budget alongside cost.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Placement (owner, 2026-10-03): the remaining budget goes in the Console context/cost chip's mouse-over, at the bottom of its stack of info lines (after "On next send"), above the "Open Conversation Inspector" hint.
1. Capture at the one requests factory: `create_default_session` gets a response hook that keeps only rate-limit headers (`x-ratelimit-*`, `anthropic-ratelimit-*`) when a context variable names the provider, storing the latest set per provider. Nothing is recorded outside a Console provider call.
2. Console gateway sets that provider scope around its provider calls (the non-stream sync call and the stream worker), so a lazily iterated stream is covered.
3. Parse and format at display time in a module imported only when an entry exists (keeps the UI-ready module census unchanged): OpenAI-style durations, Anthropic RFC 3339 resets, bare seconds/epoch values, per-window variants; numbers only, absolute clock times (the tooltip is rebuilt on refresh, not on hover).
4. The cost chip state builder gets the line from the session's provider key; no line when nothing was captured (no fake zeros; providers without headers unchanged).
5. Tests: parser units, the hook through a real session with a fake adapter, gateway stream scope, tooltip placement and absence; User Guide update.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rate-limit remaining/reset from provider response headers is captured per call where the provider sends them
- [x] #2 The Console cost surface shows remaining budget when known, absent otherwise (no fake zeros)
- [x] #3 Providers without the headers behave exactly as today
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Placement per the owner (2026-10-03): the Console context/cost chip's mouse-over, as the last info line (after the next-send estimate), above the "Open Conversation Inspector" hint. Example: `Rate limit at 14:32:05: 4,999/5,000 requests left (resets 14:32:53) · 39,200/40,000 tokens left (resets 14:32:06)`.
Capture: `Utils/egress.create_default_session` -- the factory every hosted chat handler and the hosted engine use -- now installs a `requests` response hook. It records only while `capture_rate_limits_for(provider_key)` is active, which the Console gateway enters around both provider-call paths: `_complete_sensitive_sync` (auxiliary calls spend the same budget) and the stream worker, for the whole consume, because a stream may send its request lazily. Only `x-ratelimit-*` / `anthropic-ratelimit-*` headers are kept (values cut to 64 chars), as the latest set per provider with its wall-clock time; a reply without them leaves the last reading. Outside that scope nothing is recorded, so every other session user is unchanged. The gateway's own httpx client only serves llama.cpp (local, no such headers), so it is not hooked.
Display: `Chat/provider_rate_limits.py` parses numbers and times only (header text never reaches the tooltip): OpenAI-style Go durations (`6m0s`, `20ms`), Anthropic RFC 3339 resets, bare seconds / epoch seconds / epoch ms, per-period variants (`-day`, `-minute`). A metric without a remaining value, a negative or non-numeric value, or a reset beyond 40 days is dropped, not guessed. Times are absolute clock times because the tooltip is rebuilt on refresh, not on hover. The module is imported only once an entry exists, so the UI-ready module census is unchanged (its only importer is the lazy call in `console_spend_projection.console_rate_limit_line`).
chat_screen.py: one line (passes `spend.console_rate_limit_line(provider_key)`). It was already 75 over its screen budget on dev (25,293 vs 25,218); this adds one wiring line rather than splitting the screen.
Tests: `Tests/Chat/test_provider_rate_limits.py` (parser, the hook through a real session with a fake adapter, scope on and off, both gateway paths including a lazy stream, placement and the unchanged tooltip without a reading); a mounted screen test in `Tests/UI/test_console_cost_chip_screen.py` (that file's mounted tests fail locally with RecoveryRequired on dev too, and it is not in the UI PR gate, so it is CI evidence only). Nine mutations: eight turned a test red; the ninth exposed a redundant regex lookahead, which was removed.
Qodo round (#2981), all four real: (1) header values now pass constrained Pydantic types (counts 0..1e12, reset offsets 0..40 days; NaN/inf fail the bounds) before they become a window; (2) the Hugging Face streaming POST used module-level requests.post, bypassing the session hook -- it now posts through create_default_session (closed after the POST, as requests.post does, so the open stream is unaffected); (3) a reset six or more days out shows its calendar date, since a weekday alone is ambiguous for weekly/monthly windows; (4) side calls recorded under kwargs['api_endpoint'], the execution key, so a custom endpoint's readings landed in the shared custom-hosted bucket and never reached its own tooltip -- _complete_sensitive_sync now takes the session provider. Each fix has a test that a mutation turns red; a redundant allow_inf_nan flag was removed after its mutant survived.
Not live-verified: no provider keys are available (TASK-33640); the header shapes come from the providers' documented formats.
Files: Utils/egress.py, Chat/provider_rate_limits.py (new), Chat/console_provider_gateway.py, LLM_Calls/LLM_API_Calls.py (Hugging Face stream), UI/Console_Modules/console_spend_projection.py, Widgets/Console/console_context_controls.py, UI/Screens/chat_screen.py, Docs/User_Guide/console/context-and-rag.md, tests above.
<!-- SECTION:NOTES:END -->
