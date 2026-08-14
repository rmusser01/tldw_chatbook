---
id: TASK-15992
title: 'Selection dialogs crash on push: on_mount calls nonexistent Vertical.clear()'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both the Note and Conversation selection dialogs call `Vertical.clear()` in `on_mount`; no such method exists on Textual containers (`remove_children` is the idiom), so pushing either dialog still ends in AttributeError from the dialog's own code. This is the THIRD pre-existing defect in these dialogs: TASK-15450 fixed their invalid `font-size: 10` (which poisoned the whole app stylesheet) and deliberately left this one un-papered-over — the mounted test in `Tests/UI/test_widget_css_consolidation.py` documents it and currently tolerates the AttributeError via a scope fence. Fixing this should also let that test drop its `"clear" not in str(raised)` escape hatch (see TASK-15994). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both selection dialogs push and mount without raising
- [ ] #2 The consolidation test's AttributeError tolerance is removed (the mounted pin asserts a clean open)
- [ ] #3 Born-red evidence for the fix (test fails on the current dev behavior)
<!-- AC:END -->
