---
id: TASK-36191
title: >-
  Tests: the timer-path update inventory is red on origin/dev
  (ProjectIndexBindingCard poll)
status: To Do
assignee: []
created_date: '2026-10-22 18:22'
updated_date: '2026-11-12 19:38'
labels:
  - tests
  - dev-red
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while rebasing TASK-33620.15.7's branch onto origin/dev (2026-10-22; first 7700c63a68, then 28beeb9ed5). `Tests/Architecture/test_timer_path_static_update_inventory.py::test_no_timer_path_update_defaults_to_layout_true` and `::test_clock_root_set_is_pinned` fail on a detached origin/dev worktree with the same test ids as on that branch, which does not touch the card, so they are dev's own reds. No PR-required lane runs the file (the full-tree core-tests job in test.yml runs on main pushes and dispatch only), so PR CI does not report them.

`ProjectIndexBindingCard` (`tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py`, added by 2e9018b86d "feat(settings): manage optional workspace keyword indexes") arms `set_interval(2, self._poll)`, a new repeating-clock root, and the `.update(` calls it reaches in `_paint`, `_feedback`, `_launch` and `run` pass no `layout=`, so each poll can request a whole-screen layout.

(The same rebase first also found `test_screen_size_ratchet`'s library_screen.py row over budget, 37062 against 36968; dev re-pinned that row at 37277 in c3b3aa8e0a, so it is not part of this task. The module-size ratchet's dev reds are TASK-36161, filed on another branch the same day.)

Reproduce: run the file alone from a detached origin/dev worktree (`.venv/bin/python -m pytest <file> -q`). It is deterministic (a static call-graph census), not timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each ProjectIndexBindingCard update reached from its poll passes layout= or has a CLASSIFIED_SITES row saying why it needs none
- [ ] #2 The card's poll is in EXPECTED_CLOCK_ROOTS
- [ ] #3 The file passes on origin/dev, and is on a PR-required lane or the reason it is not is recorded
<!-- AC:END -->

## Implementation Notes

**2026-11-12 (round-13 closeout), from TASK-39129's notes.** After TASK-39129 the inventory file is 2 failed / 15 passed, and both reds are this task's: the `_poll` root and its four card sites. With this task stood in temporarily (`_feedback` and `_paint` rows, the `_poll` root, and `layout=True` on the two "Keyword index unavailable" writes) the file went 17 passed / 0 failed. **Trap:** the census names the nested `run` closure's calls under both `_launch` and `run` (the class-level `ast.walk` in `test_timer_path_static_update_inventory.py:2193-2196` registers the nested `run` as a method), so a `_launch` row then reads as stale; the two closure writes need `layout=`, not rows. Sites at 8b633d0101: `project_index_binding_card.py:140` (`set_interval(2, self._poll)`), `:167`, `:194`, and `:356` / `:365` inside `async def run()` (`:293`, nested in `_launch`, `:284`). The census has no ProjectIndexBindingCard row yet, and the file still runs on no PR lane (AC#3).
