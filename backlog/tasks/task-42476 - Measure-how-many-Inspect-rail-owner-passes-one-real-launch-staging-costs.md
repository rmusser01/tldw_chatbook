---
id: TASK-42476
title: "Measure how many Inspect rail owner passes one real launch staging costs"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 09:42
updated_date: 2026-11-24 01:49
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/right_rail.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42474's review. Checked on `fu/r22-railrecon` (the round-22 stack a999508ae7 plus TASK-42474); paths are under `tldw_chatbook/`.

**Why.** Staging a launch refreshes every Inspector reader of the pending launch through `_sync_console_pending_launch_surfaces` (`UI/Screens/chat_screen.py:14363-14411`). That syncs the staged-context tray and the settings summary at once, in the same call, so their reconcile requests can share one outer owner pass. The run inspector is synced later, through the deferred control-bar sync (`_request_console_control_bar_sync()`, a `call_after_refresh` into `_run_coalesced_control_bar_sync`), and the live-work card swap is queued with `call_later` and requests its own outer reconcile when it finishes. So one staging may cost two or more outer owner passes (`ConsoleInspectorRail._outer_owner_reconcile_count`), each a measure and refold of the rail.

TASK-42474's sibling test showed the shape: after it staged a launch and synced only the tray, a Console sync that landed later synced the settings summary and run inspector for that launch and added a second owner pass (`assert 3 == (1 + 1)`). Nobody has measured what a real staging costs, or whether the extra passes are worth folding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record how many outer owner passes, and how much UI-loop time, one real launch staging costs on a mounted Console with the Inspect rail open and closed, measured with the startup syncs settled
- [x] #2 If a staging costs more than one owner pass, the notes say which surfaces cause the extra passes and either the product folds them into one (shown by a test that counts passes per staging) or the notes record why it should not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a temporary (uncommitted) mounted production-CSS probe that settles startup with TASK-42475's accessor, stages one real launch through `_stage_console_library_rag_launch` with the Inspect rail open and closed, and counts outer owner passes, the requests behind each, and UI-loop busy time (every loop callback timed, minus an idle window of the same length).
2. Run it through `run.sh`, several repetitions per state, and record numbers, method and load in the notes.
3. If a staging costs more than one owner pass, name the surfaces behind the extra passes and either fold them (with a pass-counting test) or record why not.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Result: one real launch staging costs exactly one outer owner pass, with the Inspect rail open or closed.** No fold is needed; nothing in the product changed.

**Method.** A temporary pytest probe (never committed; run through `run.sh` at `-n 4`, the file deleted afterwards) mounted the real `ChatScreen` in `_ProductionConsoleHarness` (the full production CSS) at 160x45 with a send-ready local provider. It waited for the startup work to settle with TASK-42475's `console_pending_startup_work()` (plus the rail's outer and section reconciles and the live-work card swap, on two polls 50 ms apart), with the rail opened through `_open_production_inspector` or left closed. It then staged one real launch through the production entry a Library RAG result uses, `screen._retrieval._stage_console_library_rag_launch(...)`, with a three-reference evidence bundle (`_launch(3)` from `test_console_staged_evidence_strip.py`), and waited for quiet again. It counted `ConsoleInspectorRail._outer_owner_reconcile_count` across the window, then watched an idle window of the same length for late passes. UI-loop time is the summed run time of every event-loop callback in the window (`asyncio.events.Handle._run` timed), minus the same sum over the idle window. The probe also timed the synchronous staging call, the deferred control-bar sync, the live-work card swap and the outer reconcile's two hops, and logged every local and outer reconcile request with its callers. 20 stagings, 10 per state, two batches; load average 36-47 (other agents' runs).

**Numbers (median, range).**
- Rail open: owner passes 1 in 10 of 10 stagings, late passes 0. The pass lands 547-1858 ms after the staging call. Loop time over idle 405 ms (166-709): window 1258 ms (696-2204), busy 754 ms against an idle 386 ms. Synchronous staging call 32 ms (5.5-53), card swap 72 ms wall (44-485, its awaits included), control-bar sync 4.5 ms, outer reconcile 38 ms run plus 4.4 ms finish.
- Rail closed: owner passes 1 in 10 of 10, late passes 0 (a closed rail still completes its owner generation). The pass lands 54-136 ms after the call. Loop time over idle 141 ms (-125 to 503; the idle control is noisy at this load). Staging call 8 ms, card swap 52 ms, control-bar sync 5.5 ms, outer reconcile 2.6 ms plus 3.5 ms.

**Why one pass.** The staging call syncs the staged-context tray and the settings summary in the same call. The control-bar sync (run inspector) and the card swap land later, but while the outer generation is still pending: `_run_scheduled_outer_reconcile` re-defers while any section reconcile is scheduled, and a request made while a generation is scheduled marks it dirty instead of starting another. So every reader of the launch folds into one owner pass. TASK-42474's sibling test saw two passes because it synced only the tray, and a later Console sync synced the other owners after the tray's pass had finished. A full staging does not do that.

**Where the time goes instead.** With the rail open, one staging made 101-172 local section reconcile requests. 23-38 of them were the live-work section's own `_wait_for_reconcile_layout` re-requests while its height re-converged on the swapped-in card. With the rail closed it made 23-24, one of them live-work. The owner pass itself is about 42 ms. Recommendation: leave the owner passes alone. If staging latency with the rail open matters, look at the live-work section's height convergence (listed as a proposed follow-up in the round report, not filed).

**AC #2.** A staging costs one owner pass, so neither the fold nor the "why not" record is needed.

**Files.** This task file only. The probe stayed in the session scratchpad.
<!-- SECTION:NOTES:END -->
