---
id: TASK-32010
title: Release Console validation state when the readiness probe raises
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 00:16'
updated_date: '2026-09-08 00:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An exception during provider readiness validation leaves a failed Console send marked busy, prevents retry, and keeps transcript polling active. Restore terminal state for the owning conversation while preserving cancellation and draft recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A readiness exception releases the failed conversation slot and permits retry; the unsent echo remains excluded from provider history.
- [x] #2 Cancellation propagates and session close or shutdown does not revive state or disturb another active conversation.
- [x] #3 Mounted Enter sends retain existing diagnostic evidence, restore failed drafts, stop idle transcript polling, and settle without repeated full-screen redraws at narrow and wide sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (existing ownership follows backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; existing diagnostics follow ADR-029).
Reason: routine bug fix within the existing readiness exception handler; no new runtime, persistence, UI or diagnostic interface.
1. Extend the existing readiness exception regression to prove terminal state, slot release, retry, cancellation and owning-session isolation. Preserve the failing mounted Enter evidence from TASK-31977.
2. Release only the owning validation state in its existing exception cleanup, retaining exception propagation and existing failed-echo handling.
3. Run focused controller lifecycle and mounted Enter tests, check actual compositor output and existing diagnostic privacy, lint changed code and self-review. Document the reproduced defect separately from the reporter-specific flicker, which remains unconfirmed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed a reproducible pre-provider send lifecycle defect: readiness exceptions failed the optimistic user echo but left its session VALIDATING, occupying a send slot and keeping the transcript timer polling every 200ms. The existing exception handler now releases manual/queued validation state to BLOCKED, or STOPPED for cancellation, while preserving exception propagation and failed-echo history exclusion. It targets only the live owning session and does not overwrite an already-terminal state.

Scope: no new capture tool, logging surface, dependency or UI timer workaround. AGENT_WAKE retains its separate coordinator retry lifecycle: applying cleanup to wakes changed immediate retry ordering and failed its existing fairness regression, so that independent lifecycle remains unchanged. ADR required: no; routine cleanup under existing ADR-094 controller ownership and ADR-029 diagnostic privacy.

Extended the existing controller exception regression with real retry/provider-payload assertions and inactive-session isolation; added cancellation, close and shutdown cases. Extended the existing mounted diagnostic test with actual Enter, capture enabled, four persisted conversations, 80x24 and 160x45 layouts, failed-draft restoration, timer termination and the existing actual compositor counter. Added an incident-backed testing lesson.

Evidence: before the fix, three mounted readiness-exception cases left the timer alive; focused exception and cancellation tests left VALIDATING/occupied slots. After the fix, 46 focused lifecycle checks, 20 mounted/diagnostic checks and 5 queue regression checks passed (71 total). The earlier 46 layout/caret investigation checks also passed. Mounted cases observed zero repeated full-screen redraws over a settled 1.2-second window both before and after the cleanup: this proves the polling defect, not the cause of the external reporter's terminal flicker. The supplied startup log does not identify a send-stage exception; reporter-specific root cause remains unresolved.

Verification limits: two broader checks fail identically with the untouched HEAD controller: test_no_unbounded_resolve_for_send_awaits_remain (two pre-existing direct awaits) and test_stop_active_run_cancels_only_viewed_sessions_task (terminal generation persistence fixture failure). The wake fairness check passes both baseline and final patch. Ruff reports zero introduced findings (controller 178 baseline/current; controller tests 17 baseline/current); mounted test lint and formatting pass, changed controller/test ranges format cleanly, task-ID guard and git diff --check pass. Task remains In Progress because the repository Definition of Done requires all-green checks. Full suite was not run. Self-review completed; no PR or merge is part of this follow-up investigation yet.
<!-- SECTION:NOTES:END -->
