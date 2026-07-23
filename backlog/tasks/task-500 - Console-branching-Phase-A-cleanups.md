---
id: TASK-500
title: >-
  Console branching Phase A cleanups: dead resolver, ancestry cycle guards,
  multi-root resume test
status: To Do
assignee: []
created_date: '2026-07-23'
labels:
  - console
  - tech-debt
  - test
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor follow-ups surfaced by the Console branching Phase A final review (PR #799), none blocking merge:
- `_iter_console_tree_messages` (chat_screen.py) is now dead in production (only a unit test calls it) and its docstring is misleading ("most-recent-branch leaf resolver"); the real fallback resolver is the store's `_most_recent_leaf_native`/`_leaf_under`. Delete it (and its test) or correct the docstring.
- The ancestry walks in `ConsoleChatStore._recompute_active_path` and `active_path_message_ids` have no cycle guard (unlike `_nearest_persisted_ancestor_id`). Unreachable via real DB (unique PKs), so defensive-only — add a `seen` set for consistency/hardening against malformed test doubles.
- Multiple-root resume handling is correct-by-construction but untested; add a real-DB test with two root threads where the active leaf is under the first root, asserting the second root loads off-path but is not shown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dead `_iter_console_tree_messages` is removed (with its test) or its docstring corrected to reflect its actual role
- [ ] #2 `_recompute_active_path` and `active_path_message_ids` carry a visited-set cycle guard
- [ ] #3 A real-DB multiple-root resume test exists and passes
<!-- AC:END -->
