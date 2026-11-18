---
id: TASK-41607
title: Move saved Console result discovery off the startup UI loop
status: In Progress
created_date: 2026-11-18 03:35
assignee:
- '@codex'
labels:
- console
- performance
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
updated_date: 2026-11-18 03:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Launching Console with a saved AgentRuns database performs a synchronous SQLite discovery before the first usable composer frame. Keep launch input responsive during that read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A held real launch discovery does not stall the app loop or construct the runtime while its result is pending.
- [x] #2 Delivery still uses the current database, automatic-wake gate and durable recovery audit. Quit retires the admitted read, and a changed source or cancelled startup publishes nothing.
- [x] #3 Matched private macOS PTY runs report the discovery thread, launch to usable input, full observation windows and ordinary resource retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze the PR head and trace startup in the sealed 120×50 macOS PTY. Compare the disposable browser projection candidate and keep its results.
2. Reproduce the blocking startup SQLite read through the real app scheduling entry point. Use the existing app-owned deferred task and finite-read lifetime for one authoritative discovery, keeping delivery on the app loop. Guard publication against source changes, disabled automatic wake and shutdown.
3. Verify durable recovery, in-flight cancellation, exact source and native retirement. Compare paired launches and both Library windows without changing timing limits.
ADR required: no
ADR path: backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md and backlog/decisions/097-boot-budget-ratchets.md
Reason: This moves a read to the existing finite worker boundary and preserves the durable recovery and delivery policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing app-owned launch task now reads the captured AgentRuns path on a finite worker. The previous preliminary discovery duplicated the delivery coroutine's authoritative lookup when results were present. The original SQLite query remains read-only, and delivery checks the selected store, quit state and auto-wake switch before the original durable recovery audit.

The held native-read regression failed on the baseline because scheduling opened SQLite on the app loop. The candidate passed all 37 targeted checks, including old-source refusal, gate changes, cancellation retirement, and the original delivery tests.

Across three alternating PTY pairs, launch scheduling spent a median 76.63 ms on the UI thread before the change and 7.32 ms after it. Every candidate lookup ran on a worker. The launch-to-usable median moved from 10.64 s to 10.74 s, and two pairs improved. The full results and the failed Library baseline are retained. The composition-sharing experiment removed only a sub-millisecond duplicate and stays outside the production patch.

This is a local follow-up while PR #3610 CI runs on the published parent. See [the paired results](../../Docs/QA/console-library-responsiveness/README.md#startup-saved-agent-discovery) and [ADR-135](../decisions/135-fleet-completion-delivery-and-crash-recovery.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
