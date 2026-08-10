---
id: TASK-14806
title: Guard Console queues across close, navigation, and quit
status: To Do
created_date: 2026-08-10 06:04
dependencies:
- TASK-14805
labels:
- console
- agents
- ux
priority: high
references:
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 06:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect process-memory queued prompts at every user-initiated loss boundary before the mounted queue interface makes them reachable.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One queue-aware lifecycle projection reports exact live-run, queued-session, and unsent-prompt counts, including a claimed pre-accept entry, without describing paused queues as running agents.
- [ ] #2 Session close uses one combined transcript, live-turn, and queued-prompt confirmation, including an empty transcript with live or queued work; Stay preserves all state and confirmed close tombstones before Stop or cancellation.
- [ ] #3 Leaving Console consults the same count-aware projection and proceeds only after approval, while Stay preserves runs, queues, claims, and recovery state.
- [ ] #4 TldwCli.action_quit remains a thin non-blocking dispatcher into one exclusive asynchronous pre-quit confirmation worker guarded against repeated requests.
- [ ] #5 Quit approval marks controller shutdown before cancellation and cleanup; Stay or confirmation errors fail closed, clear the reentrancy guard, and preserve the mounted Console and all unsent state.
- [ ] #6 Lifecycle tests prove exact counts, one dialog and cleanup pass, empty-transcript validation windows, cancellation-after-tombstone suppression, repeated quit handling, and failure preservation.
- [ ] #7 The lifecycle integration adds no queue persistence and unsent prompt text remains absent from snapshots, prompt history, diagnostics, logs, and application-level state.
- [ ] #8 Lifecycle counts are a revisioned immutable aggregate derived from the controller activity projection, not a second mutable owner; session close is pinned to its requested session ID and close, leave, and quit revalidate after confirmation, failing closed if impact changed.
- [ ] #9 Every user-initiated in-app exit entry point uses the async quit guard; startup password cancellation and signal or forced termination remain explicitly outside the interactive guarantee.
- [ ] #10 Approved quit keeps blocking cache and configuration persistence and timed joins off the Textual event loop, marshals app-owned state and final exit correctly, preserves cleanup ordering, and remains responsive with exactly one cleanup pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
