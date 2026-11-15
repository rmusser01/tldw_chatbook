---
id: TASK-41142
title: "The workspace create dialog's binding-retry error repeats each failed folder's path"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- settings
- workspaces
- copy
priority: low
references:
- tldw_chatbook/Widgets/workspace_create_modal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A long path printed twice per failed folder pushes the retry action further down a dialog that already needed a scroll fix at 80x24 (TASK-40318).

**Evidence.** `workspace_create_modal.py:541-545` builds the error as `f"{folder}: {message}"` per failed folder, and `message` is `str(exc)` from the binding attempt (`:519`), which the round-14 Settings stream reports already contains the path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failed folder's path appears once in the retry error
- [ ] #2 A test with a failing binding checks the error text, red before the change
<!-- AC:END -->
