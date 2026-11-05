---
id: TASK-38740
title: "Audit every Button rule that can leave a label zero content width in a grid"
status: To Do
assignee: []
created_date: 2026-11-04 03:09
labels:
- follow-up
- ui
- crash
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-38710 and its review (2026-11-04). In Textual 8.2.8 a grid sizes an auto row with `get_content_height(..., column_width - gutter)`, which ignores `min-width`. `Content.get_height` then subtracts both of Button's `line-pad` cells. A Button styled `min-width: 0; padding: 0 1` in a 4-cell grid column therefore wraps its label at width 0, and Rich's `chop_cells(word, 0)` raises `ValueError: range() arg 3 must not be zero` during layout, which exits the app. Only a content width of exactly 0 raises; a negative width does not.

TASK-38710 fixed the File Notes action-button rule and the Session Git panel's button rule by making their labels truncate (`text-wrap: nowrap; text-overflow: ellipsis`). Other Button rules in the app with the same shape (`min-width: 0`, horizontal padding, inside a `1fr` grid or a narrow column) have not been checked, and the crash only shows at particular widths and heights, so a fixed-size test does not find it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Button rule in the app's stylesheets that sets `min-width: 0` (or a min-width below its padding plus line-pad) is listed in the task notes with its verdict: guarded, unreachable at width 0, or fixed
- [ ] #2 No screen reachable at a terminal width of 40 or more raises during layout because a Button label reached zero content width; a test sweeps the fixed rules' screens across widths and heights and is red before the fix
- [ ] #3 A check (test or preflight) fails when a new Button rule reintroduces the unguarded shape, or the notes record why a static check cannot express it
<!-- AC:END -->
