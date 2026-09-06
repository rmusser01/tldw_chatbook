---
id: TASK-31900
title: >-
  Dev regressions: vertical rail-handle geometry and compact-access rail display
status: To Do
assignee: []
created_date: '2026-09-06 09:00'
labels: [console, tests, regression]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while baselining PR #2453 against origin/dev @ 5894f4755e (paired arms in a
throwaway worktree): five failures reproduce IDENTICALLY on the untouched dev tree.
Two are the known wall-clock flakes, but two look like genuine dev regressions:
test_console_rail_handle vertical-handle geometry fails on content_region.width
3 == 1, and test_console_inspector_compact_access x2 fail on rail.display is False
at widths where the rail should show. Neither is caused by the burn-down branch
(verified base-vs-branch byte-identical failure sets across three commits).
Bisect against dev history; the 472-commit window between 7e904737c7 and
5894f4755e contains the culprit.

UPDATE (same PR, while fixing the two ADR-097 ratchets): EIGHT more inherited
failures found, same method, same verdict -- identical failing sets at
457a350e0a (branch, pre-ratchet-work) and at dev @ 5894f4755e, so none are the
branch's. They fall in two groups:

- `test_console_run_inspector.py` x7 -- four parametrised
  `test_inspector_row_status_class_has_a_stylesheet_rule` cases plus
  `test_session_settings_title_is_styled_as_a_heading`,
  `test_inspector_group_heading_shares_a_left_edge_with_its_rows` and
  `test_disabled_inspector_action_has_a_legible_style_in_the_app_stylesheet`.
  All the same root cause, and it is a STALE TEST, not missing CSS: these
  assertions read only the pre-split bundle (`tldw_cli_modular.tcss`) while
  the rules they look for live in `screen_agentic_console.tcss` since the CSS
  split. Verified directly: `.console-inspector-row-ready` and
  `Button.console-inspector-action:disabled` are each present exactly once in
  `screen_agentic_console.tcss` and absent from the modular bundle. This is
  the same defect already filed as TASK-31860 (stale CSS coverage contract);
  fix these with it, not separately.
- `test_console_rail_color_grammar.py::test_fresh_rail_compose_applies_the_
  agent_status_class` -- `TypeError: Widget.__init__() got an unexpected
  keyword argument 'agent_fleet_section_state'`. A test harness passing a
  kwarg the widget no longer accepts: dev-side API drift, unrelated to CSS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Both failures bisected to their introducing commit(s) on dev and either fixed or the tests re-pinned with a documented behavior ruling
- [ ] #2 The known wall-clock flakes in the same files are annotated or stabilized so they stop polluting baseline sweeps
- [ ] #3 The seven test_console_run_inspector stylesheet-coverage failures are resolved by pointing those assertions at the split bundles (with TASK-31860), and a negative control proves the assertion can still fail when a rule is genuinely absent
- [ ] #4 test_console_rail_color_grammar's agent_fleet_section_state TypeError is fixed at whichever side drifted, and a dev-tree run of both files is green
<!-- AC:END -->
