---
id: TASK-14802
title: Add a Console conversation-context epoch
status: In Progress
created_date: 2026-08-10 06:04
labels:
- console
- agents
- architecture
priority: high
references:
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
- backlog/decisions/033-application-session-state-ownership.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 06:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console a store-owned, per-session signal for detecting provider-relevant conversation changes so deferred turns can fail closed without treating ordinary streaming or linear appends as context invalidation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ConsoleChatStore exposes an isolated monotonic conversation-context epoch for every live session and removes its state when the session closes.
- [ ] #2 The epoch advances for effective active-path content, lineage, selected textual variant, summary-boundary, rewind, delete, edit-resend, and regeneration changes.
- [ ] #3 The epoch remains stable for ordinary linear appends, streaming and terminal status updates, persistence, feedback, display overlays, off-path edits, and idempotent same-value selections or writes.
- [ ] #4 The epoch is process-memory-only and introduces no schema, persistence, snapshot, configuration, or logging changes.
- [ ] #5 Focused store tests cover every provider-relevant mutation seam, session isolation, lifecycle cleanup, and mutation checks that prove the guards can fail.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red tests in Tests/Chat/test_console_conversation_context_epoch.py that characterize the current store mutation seams: session isolation and cleanup; active-path edit, selected variant, summary, leaf, branch, rewind, and delete changes; and stable linear append, stream/status, persistence, feedback, off-path, and idempotent operations.
2. Add a process-memory per-session epoch map plus conversation_context_epoch(session_id) and one internal increment seam to ConsoleChatStore, initializing it for create/restore and purging it on close without changing persistence or snapshots.
3. Wire the increment seam into the existing content, active-leaf, summary, variant, branch, and delete mutation boundaries using before/after effective-state checks so each semantic change increments once and no-op or off-path changes remain stable.
4. Run the new tests red-to-green, then the existing Console store tree, summary, variant, retry/regenerate, rewind, persistence, and session-close suites reached by those mutations.
5. Mutation-check the active-path edit and same-value guards with bytecode caches disabled or cleared, recording the exact new tests that turn red.
6. Run Ruff on the changed Python files, full-tree pytest collection, and the architecture/runtime-policy inventories reachable from the store; compare any ambient failures using identical commands rather than counts.
7. Self-review every ConsoleChatStore mutation that can change provider-visible history, update TASK-14802 acceptance criteria and implementation notes, and document any new generalizable testing lesson only if the implementation exposes one.

ADR required: yes
ADR path: backlog/decisions/046-visible-bounded-console-prompt-queue.md
Reason: This is a direct implementation of the accepted store-owned conversation-context epoch in ADR-046, consistent with ADR-033 application-session state ownership; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
