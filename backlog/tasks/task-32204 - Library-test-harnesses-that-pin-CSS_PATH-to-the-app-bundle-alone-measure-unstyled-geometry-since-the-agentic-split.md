---
id: TASK-32204
title: >-
  Library test harnesses that pin CSS_PATH to the app bundle alone measure
  unstyled geometry since the agentic split
status: To Do
assignee: []
created_date: '2026-09-10 15:20'
labels:
  - library
  - tests
  - css
  - test-health
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-25812 (PR #2281, 2026-08-31) split the Library, Console and Settings rules out of the boot bundle into lazily loaded screen sheets (`screen_agentic_library.tcss` and siblings), and TASK-24459 did the same for Evals and Scheduling. Production loads those sheets through the owning screen's `CSS_PATH` or `TldwCli._SCREEN_OWNED_ROUTE_CSS`, so the real app is styled correctly. Any test harness that sets `CSS_PATH` to the bundle alone has been rendering Library widgets with no Library rules since that day: `Button` falls back to `width: auto`, `Vertical` to `height: 1fr`, and every width, wrap and scroll assertion in such a test measures an unstyled layout.

Three Notes tests were found red on `dev` for exactly this reason and repaired in PR #2564 (`test_compact_pagers_paint_full_wrapped_copy_at_80x24`, `test_nested_pager_paints_projected_depth_indentation`, `test_long_history_keeps_paging_actions_pinned_with_scroll_cue`): bisected to `b62407e258`, not to any product change; `grep -c library-notes-tree-pager tldw_cli_modular.tcss` returns 0; a live probe showed the pager buttons resolving `width = auto` instead of the sheet's `width: 100%`. `Tests/UI/consolidated_css.py` already documents the trap and exposes `APP_STYLESHEETS` as the correct pin. PR #2564 fixed the two harness files it touched and did not sweep the rest of `Tests/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test harness under `Tests/` that composes a Library, Console, Settings, Evals or Scheduling widget loads the same stylesheet set the app loads for that screen (`APP_STYLESHEETS` or the owning screen's sheet), and the sweep that found them is recorded in the notes with its command and hit list
- [ ] #2 A guard test fails when a harness class sets `CSS_PATH` to the bundle path alone while composing a widget whose rules live in a screen-owned sheet, naming the harness and the sheet
- [ ] #3 Any test whose expected geometry changes once it is styled is re-pinned to the styled truth with equal or better assertion strength, and each such re-pin cites this task
<!-- AC:END -->
