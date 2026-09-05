---
id: TASK-31659
title: Defer Console command modal imports until first use
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 17:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep command-only modal modules off the Console boot path while preserving first-use commands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Style and rewind commands retain behavior
- [ ] #2 Command-only modal imports are deferred without changing boot budgets
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Implements existing first-use import discipline without changing runtime interfaces.
1. Confirm modal imports are used only by command methods.
2. Move imports to first-use methods and remove obsolete screen aliases.
3. Run focused command tests and report boot census to root.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Moved style-picker and rewind-modal imports into the four command methods that use them, with choice/row annotations under TYPE_CHECKING; removed the obsolete screen import aliases. No controller construction or runtime behavior changed. Fresh first-use modal-open smoke: 2 passed. The broader style/rewind selection ran 27 passed before 3 existing style fixture failures, reproduced identically on origin/dev 93388ba69b and assigned to the root repair. Parent measured the pre-change UI-ready count at 978 versus the unchanged ceiling 972. Final boot census and baseline fixture repair remain pending, so this task is not marked Done.
