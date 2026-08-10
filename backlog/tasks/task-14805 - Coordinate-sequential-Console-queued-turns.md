---
id: TASK-14805
title: Coordinate sequential Console queued turns
status: To Do
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
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 06:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the bounded queue with Console run ownership so accepted follow-up prompts drain sequentially with explicit pause and recovery behavior while preserving per-session and global-cap guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The controller coordinator admits queue work only behind an accepted live turn or existing queue, drains separate FIFO turns in one per-session chain, and retains exactly one global agent reservation until the queue empties or pauses.
- [ ] #2 Manual and queued submission origins are explicit: the legacy no-argument acceptance hook fires only for manual sends, while queued acceptance emits a separate content-free session-and-entry event.
- [ ] #3 Failure, Stop, context mismatch, staged-rider conflict, pre-accept refusal, and unexpected exceptions pause without prompt loss, duplication, reordering, or silent context adoption.
- [ ] #4 Retry, Skip and resume, Resume next, Retry stopped turn, Keep draining, and Use current context and resume follow the approved recovery semantics and reacquire the global slot visibly when required.
- [ ] #5 One controller activity projection drives cap accounting, busy counts, run markers, fleet summaries, polling, navigation warnings, and completion notification eligibility; intermediate queued turns do not look finished or emit completion toasts.
- [ ] #6 Queue-aware gates prevent unrelated Continue, Regenerate, Edit and resend, Summarize, or hands-free actions from bypassing older queued work while allowing non-generating history mutations to force context review.
- [ ] #7 Session close and shutdown tombstone chains before cancellation can wake terminal callbacks, and the queue-empty admission race produces either one queued entry or one normal send, never a stranded or duplicate prompt.
- [ ] #8 Joined controller tests prove ordered multi-turn drains, slot retention and reacquisition, session isolation, approval waits, authority revalidation, recovery paths, boundary races, and shutdown suppression against production signatures.
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
