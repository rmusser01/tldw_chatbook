---
id: TASK-22863
title: Coordinate durable Watchlists check and briefing operations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-28 01:19'
labels:
  - watchlists
  - console
  - async
  - briefings
dependencies:
  - TASK-22859
  - TASK-22860
  - TASK-22861
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accept source checks and briefing generation quickly, execute them under application-owned supervision, and expose exact durable receipt status without tying work to a screen lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Domain services expose public accept/execute APIs that commit or resolve the durable receipt before returning; tools do not call private briefing helpers or drive Textual widgets.
- [x] #2 `watchlists_check_sources` accepts at most 50 canonical source IDs or one collection of at most 50 members, returns existing active receipts on duplicates, and executes no more than four checks concurrently.
- [x] #3 `watchlists_generate_briefing` acknowledges one durable generating receipt, returns the existing active receipt on a duplicate, and continues generation independently of Console navigation/tool timeout.
- [x] #4 Every accepted response contains the exact poll tool/arguments, suggested bounded backoff, canonical receipt/entity IDs, and terminal-state set.
- [x] #5 The app-owned coordinator schedules work on the running application loop, retains strong task references, consumes terminal exceptions, and never uses `asyncio.run()` against app-owned services.
- [x] #6 Shutdown stops acceptance, applies bounded cancellation/reconciliation, and leaves every accepted receipt in a truthful terminal or restart-recoverable state.
- [x] #7 Console receipt cards refresh from durable status and expose Runs/Artifacts inspection plus Retry/Cancel only when the domain can honor those actions.
- [x] #8 Concurrency, navigation survival, timeout, shutdown, restart reconciliation, scrubbed failure, provider dispatch, and Textual card tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED tests and split durable source-check and briefing acceptance from execution behind public domain APIs; acceptance must commit or resolve the winning receipt before returning.
2. Implement one app-loop-owned WatchlistsOperationCoordinator with a four-check semaphore, strong task references, duplicate suppression, terminal exception consumption, startup reconciliation, and bounded shutdown.
3. Add Console-only watchlists_check_sources and watchlists_generate_briefing descriptors that validate bounds/effects and return exact receipt-keyed polling instructions immediately.
4. Route direct Check Now and Generate actions through the same coordinator and preserve navigation-independent execution and truthful failure recovery.
5. Add the production-shaped Console receipt card/following state with phase-aware polling, durable refresh, exact Runs/Artifacts destinations, and only domain-supported Retry/Cancel actions; rebuild and verify consolidated CSS.
6. Run task-targeted domain/coordinator/tool/Textual tests, Ruff, CSS integrity, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 already defines durable receipt-before-acknowledgement and application-owned long-operation execution for TASK-22863; this task implements that accepted boundary rather than creating a duplicate ADR. ADR-019 remains the sole scheduler authority but is not changed by this manual-operation coordinator.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented application-owned durable Watchlists operations. Public domain acceptance commits or resolves exact check and briefing receipts before execution; Console-only commands validate source bounds and return canonical polling metadata. The coordinator runs on the application loop with four-check concurrency, strong ownership, definitive terminal reconciliation, bounded shutdown, captured-boundary restart recovery, and cross-process-safe duplicate following. Direct UI actions share the coordinator. Console receipt cards persist canonical IDs only, poll durable status, survive navigation, support domain-authorized actions, and deep-link to exact Runs or Briefing artifacts. Independent review required five rounds and closed shutdown bounding, terminal-write recovery, unfollow resurrection, exact briefing selection, provider replay, cross-process winner corruption, and timing-sensitive evidence. Verification: 375 task-targeted behavior tests and 30 CSS integrity tests passed; Ruff and diff checks passed. The known Requests dependency warning remains. Three unrelated feed-server socket failures, eight pre-existing unmounted run-context failures, and one scheduler-race baseline failure were verified/excluded; scheduler code remains owned by ADR-019 and unchanged. ADR decision: existing ADR-032 governs receipt-before-acknowledgement and application-owned execution; no new ADR was required. Added the generalized two-coordinator ownership lesson to backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
