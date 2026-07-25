---
id: TASK-562
title: Keep saved Library prompt clean after source snapshot
status: Done
assignee: []
created_date: '2026-07-25 18:34'
updated_date: '2026-07-25 18:43'
labels:
  - library
  - prompts
  - workers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the Library prompt editor from being marked dirty again when a successful create or create-conflict overwrite refreshes and recomposes the local source snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A successful new-prompt save remains clean after all source-snapshot workers and recompositions complete.
- [x] #2 A successful create-conflict overwrite retry remains clean after all source-snapshot workers and recompositions complete.
- [x] #3 The saved prompt remains selected and its persisted editor values and prompt-count refresh remain correct.
- [x] #4 Focused prompt-canvas tests and static checks pass.
- [x] #5 A genuine prompt edit present when a source snapshot lands remains visible and dirty after the recompose.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: This is a routine worker/recompose lifecycle repair within the existing Library workbench boundary; ADR-011 already governs that boundary.

1. Strengthen the normal-create and conflict-overwrite regressions to await all app workers and verify the editor stays clean, selected, and persisted after the source snapshot lands.
2. Run the focused tests red and trace the lifecycle through the source-snapshot recompose.
3. Sequence prompt-editor re-arming after the create-triggered snapshot finishes, without changing update-save or navigation contracts.
4. Run focused prompt-canvas coverage, lint/format checks, and resume TASK-546 fail-fast verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the Library prompt editor's worker/recompose lifecycle. Broad source-snapshot application now captures live prompt fields into the active render source, preserves the existing dirty bit, disarms dirty tracking for the remount, and re-arms only after child mount messages drain. The create-conflict overwrite path now removes its conflict banner before starting the count/list refresh, preventing concurrent recomposes from re-arming early. Applied the deferred re-arm contract consistently to every prompt-editor recompose site.

Regression coverage now waits for all workers in both normal-create and conflict-overwrite success paths, verifies clean selected persisted state, and proves a genuine live edit remains visible and dirty across a source refresh.

Verification: Tests/UI/test_library_prompts_canvas.py: 66 passed with one existing RequestsDependencyWarning; focused final lifecycle gate: 3 passed; scoped Ruff check, Ruff format check, and git diff --check passed. ADR required: no; backlog/decisions/011-chatbook-workbench-ui-system.md remains the governing boundary. Modified: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_prompts_canvas.py, and this task file.
<!-- SECTION:NOTES:END -->
