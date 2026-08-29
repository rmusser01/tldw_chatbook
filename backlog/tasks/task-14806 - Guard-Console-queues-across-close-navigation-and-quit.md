---
id: TASK-14806
title: Guard Console queues across close, navigation, and quit
status: Done
created_date: 2026-08-10 06:04
dependencies:
- TASK-14805
labels:
- console
- agents
- ux
priority: high
references:
- backlog/decisions/098-visible-bounded-console-prompt-queue.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 16:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect process-memory queued prompts at every user-initiated loss boundary before the mounted queue interface makes them reachable.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One queue-aware lifecycle projection reports exact live-run, queued-session, and unsent-prompt counts, including a claimed pre-accept entry, without describing paused queues as running agents.
- [x] #2 Session close uses one combined transcript, live-turn, and queued-prompt confirmation, including an empty transcript with live or queued work; Stay preserves all state and confirmed close tombstones before Stop or cancellation.
- [x] #3 Leaving Console consults the same count-aware projection and proceeds only after approval, while Stay preserves runs, queues, claims, and recovery state.
- [x] #4 TldwCli.action_quit remains a thin non-blocking dispatcher into one exclusive asynchronous pre-quit confirmation worker guarded against repeated requests.
- [x] #5 Quit approval marks controller shutdown before cancellation and cleanup; Stay or confirmation errors fail closed, clear the reentrancy guard, and preserve the mounted Console and all unsent state.
- [x] #6 Lifecycle tests prove exact counts, one dialog and cleanup pass, empty-transcript validation windows, cancellation-after-tombstone suppression, repeated quit handling, and failure preservation.
- [x] #7 The lifecycle integration adds no queue persistence and unsent prompt text remains absent from snapshots, prompt history, diagnostics, logs, and application-level state.
- [x] #8 Lifecycle counts are a revisioned immutable aggregate derived from the controller activity projection, not a second mutable owner; session close is pinned to its requested session ID and close, leave, and quit revalidate after confirmation, failing closed if impact changed.
- [x] #9 Every user-initiated in-app exit entry point uses the async quit guard; startup password cancellation and signal or forced termination remain explicitly outside the interactive guarantee.
- [x] #10 Approved quit keeps blocking cache and configuration persistence and timed joins off the Textual event loop, marshals app-owned state and final exit correctly, preserves cleanup ordering, and remains responsive with exactly one cleanup pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize existing controller activity, session close, Console navigation, and application quit boundaries with focused tests.
2. Add a revisioned immutable lifecycle-impact aggregate derived only from controller activity, including claimed entries and paused queues.
3. Integrate revision-pinned combined session-close and Console-leave confirmations that revalidate before destructive actions and fail closed.
4. Refactor application quit into one guarded async dispatcher, tombstone shutdown before cancellation, and keep blocking cleanup off the Textual event loop.
5. Inventory user-initiated quit entry points and add reached lifecycle, privacy, responsiveness, and mutation-checked regression tests.
6. Run focused and reached verification, static checks, architecture ratchets, self-review, and complete Backlog notes and DoD.

ADR required: no
ADR path: backlog/decisions/098-visible-bounded-console-prompt-queue.md and backlog/decisions/033-application-session-state-ownership.md
Reason: The accepted ADRs already define transient queue ownership, count-aware loss guards, and thin application coordination without a new root state owner.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented one immutable, content-free `ConsoleLifecycleImpact` derived from the
existing controller activity projection, with fleet and per-session revision fences.
Session close and Console leave now show combined exact counts, hide prompt bodies,
pin the requested session, and re-present updated impact when activity changes.

Refactored application quit into a guarded exclusive worker. Approval tombstones
Console queue work before cancellation, audio/timer cleanup stays on the app loop,
and timed cache/config persistence runs through `asyncio.to_thread`; Stay and
confirmation errors preserve state. The only interactive in-app exit is the global
`ctrl+q`/`action_quit` route. Startup password-app exits and process signal
`os._exit` remain outside this interactive contract.

Added controller, mounted close/navigation, and quit tests covering claimed and
paused counts, empty transcripts, prompt-text privacy, stale confirmation windows,
close/shutdown tombstone ordering, repeated requests, failure preservation,
off-loop persistence, and loop responsiveness. Ruff and `git diff --check` pass.
Reached suites passed (56 queue/controller tests, 11 final quit/close/config tests,
4 mounted close tests, and 4 navigation tests).
The screen ratchet was run; its pre-existing budget is already red at 17,727 versus
the 19,211-line branch baseline. This change reduces `chat_screen.py` to 19,168
lines (-43) and does not raise the ratchet. The full button-routing file also retains
one unrelated existing section-collapse expectation failure; all four reached close
tests pass independently.

ADR required: no. ADR-098 and ADR-033 already define the transient queue ownership,
loss-boundary confirmation, and application coordination used here.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console queued prompts are now protected across session close, Console navigation,
and application quit by exact, revision-pinned loss confirmations and tombstone-first
teardown. Approved quit remains responsive and exactly-once; rejected or failed
confirmation preserves the mounted Console and all unsent work.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] All acceptance criteria completed.
- [x] Implementation plan followed; verification exceptions documented above.
- [x] Automated lifecycle, controller, mounted UI, navigation, and persistence tests added and passed.
- [x] Ruff and whitespace/static checks passed.
- [x] Relevant prompt-queue plan and task documentation updated.
- [x] Self-review completed; no prompt bodies added to projections, logs, or persistence.
- [x] No feature regression found; known baseline ratchet and unrelated browser-toggle failures documented.
- [x] No generalizable new lesson was produced beyond the existing testing and live-verification guidance.
- [x] ADR check completed against ADR-098 and ADR-033; no new ADR required.
<!-- DOD:END -->
