---
id: TASK-36191
title: >-
  Tests: the screen-size ratchet and the timer-path update inventory are red on
  origin/dev
status: To Do
assignee: []
created_date: '2026-10-22 18:22'
labels:
  - tests
  - dev-red
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while rebasing TASK-33620.15.7's branch onto origin/dev 7700c63a68 (2026-10-22). Two architecture pins fail on a detached origin/dev worktree with the same test ids as on that branch, which touches neither file, so they are dev's own reds. No PR-required lane runs either file (the full-tree core-tests job in test.yml runs on main pushes and dispatch only), so PR CI does not report them.

- `Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/library_screen.py]`: library_screen.py measures 37062 lines against its 36968 budget (+94).
- `Tests/Architecture/test_timer_path_static_update_inventory.py::test_no_timer_path_update_defaults_to_layout_true` and `::test_clock_root_set_is_pinned`: `ProjectIndexBindingCard` (`tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py`, added by 2e9018b86d "feat(settings): manage optional workspace keyword indexes") arms `set_interval(2, self._poll)`, a new repeating-clock root, and the `.update(` calls it reaches in `_paint`, `_feedback`, `_launch` and `run` pass no `layout=`, so each poll can request a whole-screen layout.

The module-size ratchet's dev reds are a separate list (TASK-36161, filed on another branch the same day).

Reproduce: run each file alone from a detached origin/dev worktree (`.venv/bin/python -m pytest <file> -q`). Both are deterministic (a line count and a static call-graph census), not timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 library_screen.py is back under its screen-size budget, or the row is re-pinned at the measured value with a dated owner-decision comment
- [ ] #2 Each ProjectIndexBindingCard update reached from its poll passes layout= or has a CLASSIFIED_SITES row saying why it needs none, and the new clock root is in EXPECTED_CLOCK_ROOTS
- [ ] #3 Both files pass on origin/dev, and each is on a PR-required lane or the reason it is not is recorded
<!-- AC:END -->
