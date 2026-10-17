---
id: TASK-35133
title: Engine stalls 30s+ in compose_run_mcp_provider when a provider call is parked
  mid-run
status: Done
created_date: 2026-10-17 01:58
updated_date: 2026-10-17 16:39
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
RESOLVED 2026-10-17 (fix/served-burndown-residuals @ 0004010): engine EXONERATED by evidence. Root cause: the gated test-double released a provider-park asyncio.Event cross-loop with a plain set() — the engine's private lifeline loop never woke, so the turn worker blocked in ConsoleAgentBridge._chat_call_impl future.result until the 90s stall watchdog fired. Every production cross-thread entry into that loop is already threadsafe (run_coroutine_threadsafe console_agent_bridge.py:3942; call_soon_threadsafe :2537; adversarial repo sweep found no counterexample). Fix: _LoopSafeGate (call_soon_threadsafe release) in the test file; latent 90s flake in _GatedAgentGateway converted too. The original compose_run_mcp_provider/local_external_catalog attribution was one frame too deep — second-step composition completes in 0.4s once woken. Regression test red at exactly 90.0s (with task/thread stack dump instrument), green 3x at ~2.3s.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
