---
id: TASK-23149
title: Library focus tests stub a focus path production no longer takes
status: To Do
assignee: []
created_date: '2026-08-28'
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

- [ ] The fake row button carries the attributes `Screen.set_focus` reads (`focusable`, `has_class`)
- [ ] Focus is asserted by the widget `set_focus` targets, not by a `.focus()` stub that production
  no longer calls
- [ ] The two compose tests filter captured callbacks to focus callbacks only, so an unrelated
  layout callback entering `compose_content` cannot fail them again

## Evidence

Immediate focus at `tldw_chatbook/UI/Screens/library_screen.py:9215`, `:9230`; the renamed callback
is `_focus_library_list_entry_if_current`. Introduced by `04e29673a2` (2026-08-27) "Close amended
Console decomposition ratchet (#2137)" — its diff shows `-row.focus()` / `+self.set_focus(row)`.

The extra captured callable is `LibraryScreen._sync_library_ordinary_rail_width_contract`, added to
`compose_content` at `library_screen.py:12949` by `9de08fa193` (2026-08-26) "feat(library): bound
ordinary rail geometry", on dev via merge `6bed8d6f59` (PR #2124).
