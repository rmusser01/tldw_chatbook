---
id: TASK-23149
title: Library focus tests stub a focus path production no longer takes
status: To Do
assignee: []
created_date: '2026-08-28'
updated_date: '2026-11-19 15:20'
labels:
  - tests
  - library
priority: medium
dependencies: []
---

## Description

6 tests in `Tests/UI/test_screen_navigation.py` fail from two distinct, both-stale causes.

**Four** fail because production switched from a deferred `row.focus()` to an immediate
`self.set_focus(row)` and renamed the post-refresh callback to a generation-guarded variant.
`Screen.set_focus` reads `widget.focusable`, which the fake row button does not define.

**Two** fail because they capture *every* `call_after_refresh` and now catch a **layout** sync that
is not a focus grab at all. The assertion is simply over-broad — production is correct.

## Acceptance Criteria

- [ ] #1 `test_focus_library_list_entry_prefers_still_checked_row_in_select_mode` passes: its fake row
  carries every attribute the focus path reads (`focusable`, `is_attached`, `has_class`, and any
  further one the run shows), and focus is asserted by the widget `set_focus` targets, not by a
  `.focus()` stub that production no longer calls
- [ ] #2 `test_focus_library_list_entry_checked_row_preference_is_media_only` passes on the same terms

## Evidence

Immediate focus at `tldw_chatbook/UI/Screens/library_screen.py:9215`, `:9230`; the renamed callback
is `_focus_library_list_entry_if_current`. Introduced by `04e29673a2` (2026-08-27) "Close amended
Console decomposition ratchet (#2137)" — its diff shows `-row.focus()` / `+self.set_focus(row)`.

The extra captured callable is `LibraryScreen._sync_library_ordinary_rail_width_contract`, added to
`compose_content` at `library_screen.py:12949` by `9de08fa193` (2026-08-26) "feat(library): bound
ordinary rail geometry", on dev via merge `6bed8d6f59` (PR #2124).

**2026-11-15 (round-14 closeout).** The fake row still fails at the round-14 stack top (c7d4a41717), now one attribute further on: two `test_screen_navigation.py` nodes raise `AttributeError: '_FakeMediaRowButton' object has no attribute 'is_attached'` at `UI/Navigation/base_app_screen.py:208` (run.sh, `-n 4`, with a scratch `bootstrap_profile` stamp so the plain file reaches its assertions). The file's other off-lane reds are TASK-41127's.

**2026-11-19 (TASK-41529): criteria narrowed to the two fake-row nodes.** The rest of this task
was done by cbf233b179 "test(library): re-pin stale screen-navigation nodes (TASK-41127)", merged
through PR #3603 (b60535b101). It was first committed as 83c1eb5458, which is not on dev; the two
have the same `git patch-id --stable`. That commit made both compose tests count focus callbacks
only (`test_compose_content_reapplies_pending_list_entry_focus_on_every_recompose` keeps the
callbacks whose name contains `focus_library_list_entry`;
`test_compose_content_leaves_focus_alone_without_a_pending_request` asserts no `focus_library`
callback), which was the old third criterion. It also moved the entry-focus assertions to the
generation-guarded `_focus_library_list_entry_if_current`, and gave the fake row `focusable`. The
third fake-row test, `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection`,
stubs `screen.set_focus` and passes.

At c176f553b3 (run.sh, `-k "focus_library_list_entry or compose"`, `-n 4`, load 20-25): 2 failed,
4 passed. The two failures are the nodes in the criteria above, both
`AttributeError: '_FakeMediaRowButton' object has no attribute 'is_attached'` at
`tldw_chatbook/UI/Navigation/base_app_screen.py:208`, the guard 29a03afb5a (TASK-34000.26) added to
`BaseAppScreen.set_focus`. With `is_attached = True` added to the fake row as a temporary edit
(reverted), both fail one step later on `has_class`, at `UI/Screens/library_screen.py:4546`: the
focus write refreshes the Notes footer context, which reads the focused widget's classes. So the
fix is still a fuller fake row (or a real mounted row), not a product change.

Old criteria, for the record: (1) the fake row carries `focusable` and `has_class`; (2) focus is
asserted by the widget `set_focus` targets; (3) the two compose tests filter captured callbacks to
focus callbacks only. (3) is met as above; (1) and (2) are carried into the new #1 and #2.

