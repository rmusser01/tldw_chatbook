---
id: TASK-25901
title: 'Agent loop: classified retry with bounded backoff on model errors'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:07'
updated_date: '2026-08-31 20:16'
labels:
  - agents
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single transient provider error currently discards an entire agent run along with every tool result in it. Verified on origin/dev: Agents/agent_runtime.py:1291-1302 turns any call_model exception into a STEP_MODEL_ERROR trace and re-raises, which agent_service catches as RUN_ERROR; a named grep for fallback_model, retry_model, max_retries and backoff across tldw_chatbook/Agents/ returns zero hits. Local-first users run flaky local servers and hit rate limits routinely, so this is the highest damage-per-occurrence gap found in the 2026-08-31 parity pass. Scope is retry only; cross-provider failover is separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A transient model error (429, 5xx, connection reset, read timeout) is retried inside the loop rather than ending the run
- [x] #2 Retries are bounded by a configurable attempt count and use jittered backoff; a Retry-After header is honored when present
- [x] #3 Errors classified as terminal (auth failure, invalid request, content policy) are NOT retried and end the run immediately with their existing copy
- [x] #4 Every retry emits a trace step so the attempt is visible in the run log rather than silent
- [x] #5 Total retry time is bounded by the run's remaining wall budget and cannot extend a run past max_wall_seconds
- [x] #6 Tests cover: transient error recovers, terminal error does not retry, budget exhaustion during retry ends the run honestly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Adds bounded retry at an existing failure point; no new dependency or seam.

1. Classify by exception TYPE, never by matching message text. The repo already has the taxonomy (Chat_Deps).
2. Fail closed on anything unrecognised: one lost run costs less than a retry storm against a provider telling us to stop.
3. Retry IN PLACE around the existing call, not by rebuilding LoopDeps -- that would reset the wall-budget origin TASK-25913's tool clamp reads from (flagged in that task's review as the trap this lane would hit).
4. Inject sleep through LoopDeps so backoff is testable without real waiting; append the field last, per the class's stated positional-slot convention.
5. Bound total retry time by the run's remaining wall budget, so retry cannot do what TASK-25913 just stopped tool calls doing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A transient provider failure no longer discards the run. `run_agent_loop` retries in place around the existing `call_model` call; terminal failures raise on the first attempt exactly as before.

**Classification is by exception type, never by message text.** The repo already had the taxonomy in `Chat/Chat_Deps.py`, so `is_transient_model_error` keys off it: `ChatRateLimitError` and 5xx/408/425/529 `ChatProviderError` are transient, `ChatAuthenticationError` / `ChatBadRequestError` / `ChatConfigurationError` are terminal, and `ConnectionError` / `TimeoutError` cover the transport shapes. Order matters in a way that is easy to get wrong: the terminal types are checked BEFORE the generic status test, because `ChatConfigurationError` carries a 500 default and would otherwise read as retryable.

Anything unrecognised is terminal. Failing closed costs one run; failing open is a retry storm against a provider that is telling us to stop.

**Backoff.** Exponential with full jitter over `[capped/2, capped]`, capped at 30s. Jitter is not optional: without it every client that failed at the same moment retries at the same moment and rebuilds the stampede that caused the rate limit. A usable `Retry-After` wins over the computed delay -- the provider knows when it will serve us -- but is still capped, so a hostile or broken value cannot park the run, and an unusable one (string, negative, NaN) falls back to backoff.

**Retry happens in place, deliberately.** The reviewer of TASK-25913 flagged that any future path rebuilding `LoopDeps` mid-run would reset `run_started` and silently make that task's tool-timeout clamp permissive again. Retrying around the existing call rather than reconstructing deps avoids exactly that; the comment at the retry site says so, so the next person to touch it knows why.

**Bounded by the wall budget (AC#5).** A retry is only taken if its delay fits in `max_wall_seconds - elapsed`. Otherwise the run gives up rather than sleeping past its own deadline -- retry must not do what TASK-25913 just stopped tool calls doing.

**Configurable (AC#2).** `RunBudget.max_model_retries`, default 2, surfaced through Console as `agent_max_model_retries` alongside the existing budget knobs. **0 reproduces the pre-retry behaviour exactly**, and a test pins that. `LoopDeps.sleep` is injected for testability and appended last, per that class's stated convention for preserving legacy positional slots.

**Verification.** 30 tests: 24 on the classifier and backoff (every transient and terminal type, cap, jitter, `Retry-After` honoured/capped/unusable), 6 driving the real loop (transient recovers, terminal raises immediately, retries bounded, each retry traced, backoff cannot pass the wall budget, retries-off unchanged). Verified non-vacuous by mutation: disabling the retry branch fails 2 of the 6 loop tests.

`Tests/Agents/` holds at the same 15 baseline failures, verified by diffing sorted failure names (2237 passing, up 32). `Tests/App/`, `Tests/Metrics/` and `Tests/MCP/` unchanged at the 2 known MCP baselines.

**Files:** `tldw_chatbook/Agents/model_retry.py` (new), `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_models.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/config.py`, `Tests/Agents/test_model_retry.py` (new), `Tests/Agents/test_model_retry_loop.py` (new).
<!-- SECTION:NOTES:END -->
