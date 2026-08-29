---
id: TASK-14805
title: Coordinate sequential Console queued turns
status: Done
created_date: 2026-08-10 06:04
dependencies:
- TASK-14802
- TASK-14803
- TASK-14804
labels:
- console
- agents
- architecture
priority: high
references:
- backlog/decisions/098-visible-bounded-console-prompt-queue.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 16:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the bounded queue with Console run ownership so accepted follow-up prompts drain sequentially with explicit pause and recovery behavior while preserving per-session and global-cap guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The controller coordinator admits queue work only behind an accepted live turn or existing queue, drains separate FIFO turns in one per-session chain, and retains exactly one global agent reservation until the queue empties or pauses.
- [x] #2 Manual and queued submission origins are explicit: the legacy no-argument acceptance hook fires only for manual sends, while queued acceptance emits a separate content-free session-and-entry event.
- [x] #3 Failure, Stop, context mismatch, staged-rider conflict, pre-accept refusal, and unexpected exceptions pause without prompt loss, duplication, reordering, or silent context adoption.
- [x] #4 Retry, Skip and resume, Resume next, Retry stopped turn, Keep draining, and Use current context and resume follow the approved recovery semantics and reacquire the global slot visibly when required.
- [x] #5 One controller activity projection drives cap accounting, busy counts, run markers, fleet summaries, polling, navigation warnings, and completion notification eligibility; intermediate queued turns do not look finished or emit completion toasts.
- [x] #6 Queue-aware gates prevent unrelated Continue, Regenerate, Edit and resend, Summarize, or hands-free actions from bypassing older queued work while allowing non-generating history mutations to force context review.
- [x] #7 Session close and shutdown tombstone chains before cancellation can wake terminal callbacks, and the queue-empty admission race produces either one queued entry or one normal send, never a stranded or duplicate prompt.
- [x] #8 Joined controller tests prove ordered multi-turn drains, slot retention and reacquisition, session isolation, approval waits, authority revalidation, recovery paths, boundary races, and shutdown suppression against production signatures.
- [x] #9 Queue recovery crosses the generation gate only through a narrow internal typed authorization unavailable to unrelated actions; activity state distinguishes slot occupancy, preparation/validation, accepted live work, approval waits, queue presence, and pause state.
- [x] #10 Accepted queued prompts enter normal persistence and prompt history exactly once in accepted order; admission, edit, reorder, refused starts, and recovery selection do not write history, and the queue-empty admission race emits one final notification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the controller's accepted-send, terminal-outcome, run-slot, marker/notification, recovery, shutdown, and prompt-history seams with joined red tests against real signatures. 2. Add explicit submission origin, content-free queued-acceptance and activity models, plus a focused coordinator that owns per-session admission, chain draining, context checks, claims, reservation/recovery decisions, and shutdown suppression through the pure registry. 3. Integrate the coordinator into ConsoleChatController while preserving the manual acceptance hook, immutable owning-session turn contexts, live authority checks, exact persistence/history semantics, and existing non-queue behavior. 4. Gate competing generation and hands-free paths through one activity projection, then run focused/reached suites, mutation checks, architecture/static checks, and self-review. 5. Complete ACs, evidence, and task hygiene. ADR required: yes. ADR path: backlog/decisions/098-visible-bounded-console-prompt-queue.md. Reason: this task implements ADR-098's accepted controller/coordinator boundary and introduces no different architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-098's controller-owned queue coordinator with explicit manual/queued origins, content-free queued acceptance, accepted-turn identity/terminal metadata, one queue-aware activity projection, FIFO claim/settle drains, retained/reacquired slot reservations, typed internal recovery authority, recovery/context/rider/Stop/failure pauses, origin-aware owning-session RAG capture, competing-generation gates, exactly-once persistence/history, and close/shutdown tombstones with accepted-boundary revalidation. Added console_prompt_queue_coordinator.py, extended the queue registry recovery epoch transition, integrated ConsoleChatController/models, added joined coordinator tests, and corrected live Auto-RAG setting capture in ConsoleSessionController. ADR required: yes; implemented existing backlog/decisions/098-visible-bounded-console-prompt-queue.md; no new ADR. Verification: 217 focused queue/controller tests passed; 168 reached queue/run-marker/Stop/attachment/history/MCP tests passed; 44 agent-runtime tests passed; 18 joined coordinator tests passed; mounted same-send Auto-RAG proof passed. Ruff check (scoped, ignoring two pre-existing E721 exact-type checks), Ruff format check for new/focused files, py_compile, and diff check passed. Mutations disabling shutdown accepted-boundary revalidation and inverting the context-epoch comparison both made their focused tests fail, then restored green. Broader Architecture suite has unrelated existing blockers: stale blocking-I/O baseline entries and a UTF-8 decode failure in the diagnostic inventory caused by another workspace file; full Auto-RAG suite has one pre-existing one-argument fake-hook compatibility failure already present in HEAD. No lesson entry added: no new generalizable incident beyond ADR/spec rules.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console can now coordinate bounded per-session prompt queues as sequential real turns with explicit origins, one-slot chain ownership, visible pause/recovery semantics, queue-aware fleet activity, safe shutdown/close ordering, and exactly-once normal persistence. Direct controller APIs and joined tests are ready for TASK-14806 lifecycle loss guards and TASK-14808 mounted queue UX.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked
- [x] #2 Implementation follows ADR-098 and the recorded plan
- [x] #3 Joined unit and integration tests cover new behavior
- [x] #4 Scoped static analysis, formatting, compile, and diff checks pass
- [x] #5 Relevant Superpowers plan documentation is updated
- [x] #6 Implementation notes and final summary record approach and evidence
- [x] #7 Self-review and mutation checks are complete
- [x] #8 Unrelated workspace changes and known baseline failures are documented
<!-- DOD:END -->
