---
id: TASK-42532
title: "A streamed in-place retry after a failed non-streaming resend in the same chain is refused at trace reservation"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 23:14
labels:
- follow-up
- console
- agents
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_provenance.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/refused_stream.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by code reading during TASK-42472's round-24 work (2026-11-24); **not probed**. Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Between a failed call and its retry, `Chat/console_trace_service.py::_retry_generation_matches` allows only one change in generation settings: streaming True -> False, proven by the stream-fallback overlay. Two paths make a failed call a non-streaming resend *inside the run's own chain* (a TOOL_LOOP call with `streaming=False`). After either one, the agent's next in-place retry streams again, is compared with `streaming=False`, and would be refused with `trace_tool_chain_unavailable`, ending on the trace card instead of the provider's error:
- TASK-37421: a self-hosted or Custom OpenAI-compatible endpoint that refuses a stream (400/406/415/422/501, or an empty stream; `Chat/refused_stream.py`) is sent once more without streaming. If that resend then fails transiently (a 429, a 5xx, a timeout), the agent retries.
- llama.cpp with tools in the request (`has_tools`): `Chat/console_trace_provenance.py::stream_fallback_route_provenance` projects the resend onto TOOL_LOOP in the chain instead of its own `llama_fallback` run.

TASK-42472 AC#8 covered only the no-tools llama.cpp resend, which has its own run id.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A probe or test establishes, for each of the two paths, whether a streamed in-place retry after a failed non-streaming resend is refused today
- [x] #2 Where it is refused, a captured run retries the streamed call in place and ends on the provider's own error when retries run out; tests are red before the fix
- [x] #3 A retry that changes target, settings other than this streaming flip, or surface is still refused
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#1: drive both paths through the real gateway and trace factory (handler seam with `chat_api_call`'s signature): a self-hosted provider (vLLM) whose stream is refused (HTTP 400), and llama.cpp with tools in the request; the in-chain non-streaming resend fails with a 503, then the agent's streamed in-place retry. Record whether it is refused today.
2. Where refused: in `ConsoleTraceService.persist_request`'s failed-call header comparison, a streamed retry after a failed in-chain resend is compared with the streamed call that resend repeated (the call before it), not with the resend; the resend is recognised as such only when it is that call's verified True -> False flip. Same principle as TASK-42472 AC#8.
3. Tests: gateway-level for both paths (admitted; changed model, temperature, surface still refused), and a Console run through the real controller whose retries run out and ends on the provider's error, red before the fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**AC#1: both paths are refused today**, and they are one path in the code. llama.cpp with tools in the request does not take the direct llama.cpp branch (`and not prepared.tools`); it streams through `_stream_generic_chat` like every self-hosted provider, and `retries_refused_stream` counts llama.cpp as self-hosted. So in both cases a refused (HTTP 400) or empty stream is resent without streaming as the run's next TOOL_LOOP call. `test_a_streamed_retry_after_a_failed_in_chain_resend_repeats_the_stream` drives each through the real gateway and trace factory (handler seam with `chat_api_call`'s signature): vLLM with no tools (`self_hosted`) and llama.cpp with a tool schema (`llama_tools`). At the TASK-42530 commit, the streamed in-place retry (`valid`) is refused for both with `TraceCallPersistenceError`; a non-streaming retry of the resend (`unchanged_resend`) is admitted.
- The refusal is at dispatch binding, not reservation: `ConsoleTraceService.persist_request` compares a call after a failed call with that call's header, and `_retry_generation_matches` allows only a True -> False streaming change, so the resend (`streaming=False`) refused the retry (`streaming=True`), `trace_tool_chain_unavailable` at `phase=trace_dispatch_commit`. In a Console run the run ended on the dispatch-recovery card (`DISPATCH_STARTED`) instead of the provider's error.

**Fix.** In `persist_request`, when the failed previous call is a non-streaming call and the new call streams, read the call before it. When that call failed too, streamed, and its generation settings equal the previous call's apart from `streaming` True -> False, the previous call was its resend (the only way such a pair is admitted is the stream-fallback overlay check), and the new call is compared with that streamed call instead. Every other target, setting and surface change is still refused; the factory still requires an unchanged surface. Same principle as TASK-42472 AC#8: the retry repeats the streamed call, not its failed resend. ADR-337: "Integration recorded 2026-11-24: stream resends inside a run" (shared with TASK-42530).

**Tests** (`Tests/Chat/test_console_failed_call_retry_ownership.py`).
- `test_a_streamed_retry_after_a_failed_in_chain_resend_repeats_the_stream`, both paths: `valid` (admitted, the provider is asked once more) and `unchanged_resend` (admitted, as before); `changed_model`, `changed_temperature`, `changed_surface` refused with no provider call (AC#3).
- `test_a_captured_run_retries_a_refused_stream_whose_resend_failed`, the real controller, bridge, gateway and trace factory, Capture On, vLLM through the handler seam, every stream refused with 400:
  - `recovered`: the first resend fails with 503, the retried stream's resend answers; the run completes, lineage `agent_first` error then `tool_loop` error, error, complete;
  - `exhausted`: every resend fails; three rounds (two retries), and the run ends `failed` on the provider's own 503 copy, not the trace card, with no dispatch recovery (AC#2).
- Red before (the service file at the TASK-42530 commit): 4 failed, both `valid` gateway cases (`TraceCallPersistenceError`) and both run cases (`assert ConsoleDispatchRecoveryState(kind=DISPATCH_STARTED ...) is None`, `trace_tool_chain_unavailable` at `trace_dispatch_commit`). Green after.
- Not pinned: the generation-settings equality that proves the previous call was the streamed call's flip. Through the gateway a non-streaming call after a failed streamed call inside one run is only ever that resend, so no test can reach the other case; the check states the proof and stays as defence in depth.

**Size row.** `console_trace_service.py` 6,774 -> 6,809 (+35), exact measurement, with a dated comment in `Tests/Architecture/test_module_size_ratchet.py`.

**Files.** `tldw_chatbook/Chat/console_trace_service.py`; `Tests/Chat/test_console_failed_call_retry_ownership.py`; `Tests/Architecture/test_module_size_ratchet.py`; ADR-337.

**Runs.** Same branch and runs as TASK-42534's notes.
<!-- SECTION:NOTES:END -->
