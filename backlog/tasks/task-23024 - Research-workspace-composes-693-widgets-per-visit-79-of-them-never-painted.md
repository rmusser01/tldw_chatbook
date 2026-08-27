---
id: TASK-23024
title: >-
  Research workspace composes 693 widgets per visit, 79% of them never painted
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - ui
  - screens
priority: high
---

## Description

`ResearchWorkspaceScreen` (new 2026-08-24, reached by F10) is now the most expensive screen in the
app: **693 widgets constructed per visit**, 4.4x Library and 1.3x Console. **544 of the 691 mounted
(79%) sit inside `display=False` subtrees** - constructed, mounted and CSS-matched on every visit,
never painted. One whole-screen recompose: **1.73 s**.

Cause is eager slot pools: `Research_Workspace_Modules/source_list.py:196` composes
`MAX_VISIBLE_SOURCE_ROWS = 25` slots, each yielding ~13 widgets (`:41-88`), **on an empty profile** -
325 widgets before any data exists - plus 20 receipt slots (`source_receipt.py:91`).

The slot-pool pattern is defensible; allocating the full pool at compose is what makes the empty case
pay the maximum case, on every visit. Screens are never cached, so this is paid every time.

## Acceptance Criteria

- [ ] Widgets composed on an empty profile scale with content, not with the maximum
- [ ] Widget count per visit and recompose wall time measured before and after, interleaved
- [ ] Scrolling and row recycling still work at the maximum row count - the pool exists to avoid mount/unmount churn and that benefit must survive
- [ ] The `display=False` proportion is reported after the change

## Evidence

693 constructed / 691 mounted per visit, identical on every lap (screens are never cached).
Composition measured: 265 Buttons, 201 Statics, 91 Horizontals. Recompose 1735/1731/1793 ms over 3
trials.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
