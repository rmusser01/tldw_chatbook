---
id: TASK-31975
title: >-
  Library pins: one that cannot pass and one that rests on centre-fold
  arithmetic
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Census findings from PR O's batch-3 review. (1) `Tests/UI/test_library_honesty_accessibility.py` (~1126) asserts `str(export_btn.label) == "Export selected"` against the real `LibraryMediaCanvas`, whose base label has been the short padded `Export` since PR J — red on dev and on every branch since. (2) The conversations column pin's equality (`test_conversations_export_label_holds_its_column_across_the_first_selection`) holds only because the disabled `○` centred and the padded `  Exp` both put their first glyph at cell 71 in a 9-cell `1fr` button; a CSS width change would turn a correct implementation red with a bare tuple.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The honesty/accessibility pin asserts the current Media label contract and passes on dev
- [ ] #2 The conversations column pin states the centre-fold regime in its docstring and fails with a message that names it
<!-- AC:END -->
