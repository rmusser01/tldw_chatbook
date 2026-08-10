---
id: TASK-14804
title: Add the pure bounded Console prompt queue registry
status: To Do
created_date: 2026-08-10 06:04
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
Provide a deterministic process-memory owner for per-session queued prompt state before controller scheduling or widgets depend on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The registry owns immutable text-only entries and render-safe snapshots with stable IDs, FIFO order, per-session isolation, and a hard capacity of ten across waiting plus claimed entries.
- [ ] #2 Revision-checked admission, edit, move, remove, clear-waiting, claim, settle, return-to-head, pause, resume, reservation, closing, shutdown, and session-removal transitions reject stale or invalid intents without partial mutation.
- [ ] #3 Claimed or starting entries cannot be edited, moved, removed, or cleared as waiting work, and new entries append behind all older work while paused.
- [ ] #4 Queue state records the active-chain context baseline needed for first-admission and later context-review decisions without storing provider payloads or authority.
- [ ] #5 All transitions are synchronous and event-loop-thread confined; foreign-thread access is rejected or marshalled by callers rather than protected by widget locks.
- [ ] #6 The registry has no Textual, provider, database, snapshot, prompt-history, diagnostics, or logging dependency, and queued prompt bodies are never serialized.
- [ ] #7 Pure tests cover capacity, FIFO, revisions, lifecycle, every pause and claim transition, and mutation checks for the ten-entry and stale-revision guards.
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
