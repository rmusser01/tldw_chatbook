---
id: TASK-38697
title: "Widget CSS: BaseAppScreen's scoped block is emitted after its subclass KanbanBoardScreen"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- css
- tests
priority: low
references:
- Tests/UI/test_widget_css_consolidation.py
- tldw_chatbook/css/widget_defaults_scoped.tcss
- tldw_chatbook/css/build_css.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36523's checks, PR #3370). Checked on dev 9bbd5c2515.

`Tests/UI/test_widget_css_consolidation.py::test_base_class_blocks_precede_their_subclasses` fails with `[scoped] BaseAppScreen is a base of KanbanBoardScreen but is emitted after it, inverting the tie-breaker Textual gave them`. In `widget_defaults_scoped.tcss`, KanbanBoardScreen's block starts at :43 and BaseAppScreen's at :748. Within one stream an exact specificity tie goes to the later rule, so a base block must come first (the test docstring explains why).

KanbanBoardScreen's scoped block arrived in ab991c7117 (2026-10-31, "Repair Kanban first-use creation and column identity"). The two blocks share no selector today (`BaseAppScreen #screen-content` against `KanbanBoardScreen #kanban-*`), so nothing renders differently yet. The pin exists so that a later shared selector does not silently flip.

**Evidence.** Locally the file errors at setup with the `RecoveryRequired` trap (TASK-38111). The red above was seen with the test run under a temporary `@private_profile_test` marker, since removed. The census lists only `test_class_level_css_stays_within_the_allowlist` from this file, so no PR lane runs this test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_base_class_blocks_precede_their_subclasses passes on dev because the generated scoped stream emits every base block before its subclasses' blocks, and the generated sheets reproduce from build_css.py
- [ ] #2 The test runs on a PR lane (a census entry for this node), or the task notes record why it cannot
<!-- AC:END -->
