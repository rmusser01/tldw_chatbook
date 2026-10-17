---
id: TASK-35133
title: Engine stalls 30s+ in compose_run_mcp_provider when a provider call is parked
  mid-run
status: In Progress
created_date: 2026-10-17 01:58
updated_date: 2026-10-17 15:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified against dev base (b912d334 stash): parking a provider call mid-run stalls the engine's second agent step inside compose_run_mcp_provider -> UnifiedMCPControlPlaneService.local_external_catalog, 30s+ under the stall watchdog, status STREAMING. Surfaced during the 2026-10-16 burndown T2 scoping test; NOT a regression of that branch (stash-verified). Needs a dedicated investigation task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A bounded reproduction test exists in Tests/Web_Server/test_mobile_service.py: a gateway double whose FIRST provider call genuinely parks (arrival awaited before release), then the SECOND (post-tool) provider call must arrive within 10 s; on timeout the failure message dumps every asyncio task's await stack and every thread's stack.
- [x] The stall no longer reproduces within the bound: releasing the parked call through the model-call lifeline loop's threadsafe door (`call_soon_threadsafe`) brings the second provider call in <1 s (pre-fix: exactly the stall-watchdog deadline, measured 90 s at the default window).
- [x] Root cause documented in Implementation Notes with the evidence chain (dump + single-variable measurements).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- **Outcome:** test-only fix. The engine (Console + mobile Binding-1 construction) needed no change; the stall was a harness-class defect in how a parked provider call is released, now guarded by a regression test.
- **Root cause (evidence chain):**
  1. The engine consumes provider streams on its private model-call lifeline loop: `ConsoleAgentBridge._chat_call_impl` (Chat/console_agent_bridge.py:3942) submits `_consume()` with `asyncio.run_coroutine_threadsafe(...)` onto `_submit_loop` and the turn's worker thread blocks in `future.result(timeout=_CHAT_CALL_TIMEOUT_SECONDS)` (line 3947). The thread dump showed that worker parked exactly there, with the `console-agent-loop` lifeline thread idle in `select()`.
  2. The gated-gateway double parks inside `stream_chat` on an `asyncio.Event` whose waiter future belongs to the *lifeline* loop. Releasing it from the test's loop with a plain `Event.set()` resolves the waiter via non-threadsafe `call_soon` — no self-pipe write, so the lifeline loop's selector is never woken.
  3. The parked call therefore resumes only when the lifeline loop's next timer fires: the stall watchdog's deadline. Measured (twice, plus the original red runs): release at T=0, approval raised at exactly T=90.0s (`DEFAULT_STALL_TIMEOUT_SECONDS`), second provider call at T=90.4s. The T2 report's "30s+" is the same mechanics under a 30 s window.
  4. Single-variable confirmation: the same shape with the gate set *before* the call parks wakes instantly (0.00 s) — the pre-existing `test_send_returns_while_the_turn_is_pending` passes only because it wins that race; a timing flip would turn it into the same 90 s stall/flake.
- **Engine not at fault:** every production cross-thread entry into the lifeline loop is threadsafe (`run_coroutine_threadsafe` at console_agent_bridge.py:3942, `call_soon_threadsafe` at 2537); real provider streams carry their own socket on that loop, and the mobile approval-decision path measured 0.4 s end-to-end. `compose_run_mcp_provider` / `local_external_catalog` complete in <0.5 s once woken (the T2 stack attribution placed the stall one frame too deep — the parked window is spent blocked in `_chat_call_impl`'s `future.result`, upstream of any second-step composition).
- **Fix:** `Tests/Web_Server/test_mobile_service.py` gains `_LoopSafeGate` (records the loop inside `wait()`, releases via `loop.call_soon_threadsafe(event.set)`); the new regression test `test_second_agent_step_not_stalled_by_parked_provider_call` uses it, and the existing `_GatedAgentGateway` test was converted to it (same defect class, kills the latent race-flake). The timeout dump (`_dump_asyncio_stacks`: all tasks' await stacks + all threads' stacks) is kept as the failure diagnostic.
- **Files:** `Tests/Web_Server/test_mobile_service.py` only. No production code, no new/changed diagnostic log statements (TASK-494 pin: N/A).
- **Verification:** regression test 3x green (~2.3 s each, bound 10 s; pre-fix red at 90 s with dump); mobile service suite 30 passed (29 baseline names unchanged + 1 new test); mobile api 31 / page 23 green, api names identical pre/post change (its 31-vs-30 delta vs the #3142 note is pre-existing dev drift, untouched by this change).
- **ADR:** none required — test-only change, no architectural decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
