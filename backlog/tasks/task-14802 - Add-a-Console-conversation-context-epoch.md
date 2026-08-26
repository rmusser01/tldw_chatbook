---
id: TASK-14802
title: Add a Console conversation-context epoch
status: Done
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
updated_date: 2026-08-10 07:02
modified_files:
- tldw_chatbook/Chat/console_chat_store.py
- Tests/Chat/test_console_conversation_context_epoch.py
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console a store-owned, per-session signal for detecting provider-relevant conversation changes so deferred turns can fail closed without treating ordinary streaming or linear appends as context invalidation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ConsoleChatStore exposes an isolated monotonic conversation-context epoch for every live session and removes its state when the session closes.
- [x] #2 The epoch advances for effective active-path content, lineage, selected textual variant, summary-boundary, rewind, delete, edit-resend, and regeneration changes.
- [x] #3 The epoch remains stable for ordinary linear appends, streaming and terminal status updates, persistence, feedback, display overlays, off-path edits, and idempotent same-value selections or writes.
- [x] #4 The epoch is process-memory-only and introduces no schema, persistence, snapshot, configuration, or logging changes.
- [x] #5 Focused store tests cover every provider-relevant mutation seam, session isolation, lifecycle cleanup, and mutation checks that prove the guards can fail.
- [x] #6 A failed-assistant retry advances the epoch exactly once when the row becomes provider-visible complete or stopped history, including same-text recovery; a failed or refused retry remains excluded and does not advance or authorize adoption.
- [x] #7 Adding, removing, reordering, or selecting attachments on an active-path message advances the epoch when the future provider payload changes; equivalent off-path attachment mutations remain stable.
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
Implemented a process-local, store-owned per-session conversation-context epoch with lifecycle initialization and cleanup. Semantic guards advance it only for effective active-path text, branch/leaf, summary, variant, failed-retry recovery, and generation-attachment changes; ordinary append/stream/status, off-path changes, metadata, persistence, and same-value operations remain stable. Added 16 focused contract tests plus manual mutation checks proving the active-path and same-value guards fail when inverted. Verification: 699 focused/reached tests passed; Ruff and git diff --check passed. The broader RuntimePolicy run had 343 passes and 12 skips plus six unrelated ambient failures (four MediaScreen mount timeouts and two AppleDouble source-scan decode failures). Full-tree collection is independently blocked by two existing Confluence imports requiring the absent optional playwright dependency. ADR: backlog/decisions/046-visible-bounded-console-prompt-queue.md. No schema, persistence, snapshot, configuration, or logging changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and verified the Console conversation-context epoch required for safe deferred prompt adoption, including active-path semantic guards, failed-retry visibility handling, attachment changes, lifecycle cleanup, and focused regression coverage.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
