---
id: TASK-15510
title: Clarify File Notes secondary action copy
status: Done
assignee: []
created_date: '2026-08-11 23:24'
updated_date: '2026-08-11 23:28'
labels: []
dependencies: []
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
modified_files:
  - tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
  - Tests/UI/test_library_file_notes_workspace.py
  - backlog/tasks/task-15510 - Clarify-File-Notes-secondary-action-copy.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the File Notes secondary-action disclosure immediately recognizable and use input-neutral confirmation language for delete so visible guidance serves mouse and keyboard users equally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The collapsed secondary-action control is labeled More file actions and its expanded state is labeled Hide file actions.
- [x] #2 The first Delete activation asks the user to Activate Delete again to confirm without mouse-specific wording.
- [x] #3 The revised labels remain fully visible, keyboard operable, and focus-stable at compact and desktop sizes.
- [x] #4 Focused tests cover copy, action behavior, geometry, and focus retention, and focused static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize current secondary-action and delete-confirmation copy plus compact layout coverage.

2. Update focused mounted tests first for recognizable labels, input-neutral guidance, geometry, and retained focus.

3. Replace the user-facing copy without changing action IDs, state ownership, or disclosure behavior.

4. Run focused tests, Ruff, mutation verification, and scoped diff review.

5. Complete acceptance criteria and record implementation notes and verification evidence.

ADR required: no

ADR path: N/A

Reason: This is copy-only UI clarification within the existing disclosure and confirmation behavior governed by ADR-011.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed the retained secondary-action disclosure to More file actions and Hide file actions, and changed the delete safety prompt to the input-neutral Activate Delete again to confirm. The longer disclosure label revealed a compact two-column clipping edge, so stacked editor actions now give the disclosure control both columns while preserving its identity and focus. Added mounted coverage at 120x40 and 40x20 for exact rendered copy, geometry, keyboard operation, and focus retention, plus a delete-status assertion through the real two-step confirmation path. Verification: 8 focused action, layout, and service-flow tests passed; Ruff check passed; diff check passed. The copy assertions failed before the implementation, and removing the compact column span reproduced truncation to Hide file, so both the behavior and layout guards are non-vacuous. ADR check: no new ADR; ADR-011 governs the existing disclosure behavior. No reusable lesson was added because compact neighbor geometry is already documented in the testing lessons.
<!-- SECTION:NOTES:END -->
