---
id: TASK-16246
title: Preserve realtime replies when delayed transcripts land
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 11:56'
updated_date: '2026-08-14 12:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep realtime assistant replies attached when the provider delivers the user transcript after reply start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delayed realtime user content is finalized without purging its assistant descendant.
- [x] #2 Realtime row ordering and provenance regressions pass.
- [x] #3 The affected store and realtime test modules pass.
- [x] #4 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a narrow correction inside the existing Console store/realtime boundary.

1. Retain the two failing realtime integration tests as RED evidence.
2. Add a strict store operation for filling an initially blank user turn without applying edit-descendant invalidation semantics.
3. Route final and empty realtime transcripts through that operation.
4. Run focused, module, static, and checkpoint verification; self-review and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a strict `ConsoleChatStore.finalize_deferred_user_message_content` operation for the realtime-only case where a blank user turn is completed after its assistant child already exists. Ordinary edits retain their existing descendant-invalidation behavior.
- Routed both final and explicitly empty realtime transcripts through the new operation, preserving reply ordering, metadata, deferred persistence, and the assistant descendant.
- TDD evidence: the two original chunk failures were RED, then GREEN. Direct store characterization proves the reply child survives. Full verification: 172 store tests and 75 realtime tests passed.
- `git diff --check` passed. Scoped Ruff lint/format reproduce the exact HEAD baseline in legacy `chat_screen.py`, `console_chat_store.py`, and realtime-test formatting; the changed store test is formatted and no new lint finding was introduced.
- The combined checkpoint rerun is intentionally paired with the immediately following rail-contract task because chunk 50 still contains four already-isolated, unrelated rail failures.
- ADR check: no ADR was required because the existing store/realtime ownership boundary is unchanged.
<!-- SECTION:NOTES:END -->
