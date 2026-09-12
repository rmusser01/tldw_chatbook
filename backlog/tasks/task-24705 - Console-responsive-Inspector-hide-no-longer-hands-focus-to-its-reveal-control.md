---
id: task-24705
title: >-
  Console: responsive Inspector hide no longer hands focus to its reveal
  control
status: To Do
assignee: []
created_date: '2026-08-30'
labels:
  - console
  - inspector
  - regression
  - dev-red
priority: high
---

## Description (the why)

`Tests/UI/test_console_inspector_navigation.py::test_responsive_hide_and_
reveal_hands_off_focus_without_losing_offset` is red on `dev`.

When the terminal widens past the responsive threshold and the Inspector rail
is hidden, focus is supposed to hand off to `#console-inspector-rail-open` —
the control that brings the rail back. The test resizes 128→129 columns with
focus held inside an overflowing bounded section and waits for that handoff.
It now times out: the rail does not end up hidden with focus on the reveal
control.

This matters beyond the red test. Focus handoff on responsive hide is the
mechanism that stops a resize from stranding keyboard focus on a widget that
is no longer on screen, and the reveal control is the way back to a rail that
just vanished. TASK-24600 was filed for the same class of defect at a
different width — a collapsed rail with no on-screen way back.

Found while rebasing PR #2220 (Inspect rail burn-down). It is NOT caused by
that branch: it reproduces on a pristine checkout with none of that work
present.

## Evidence

Bisected to a single commit, by running the one test in clean worktrees at
each commit:

| Commit | Result |
| --- | --- |
| `605b8f91de` "stop the Context rail repeating provider and model" | **1 passed** |
| `d5e1a26ab3` "close the 118-128 column dead zone that deleted Context" | **1 failed** |

`d5e1a26ab3` is the parent-to-child step, so it is the introducing commit.
It also still fails at dev head `0ec518610c` and later, so nothing since has
repaired it.

Failure mode: the wait for
`focused.id == "console-inspector-rail-open" and rail.display is False`
times out after 5s at 129 columns.

The suspected interaction is that closing the 118–128 dead zone changed which
panes are budget-eligible at 129 columns, so the Inspector is no longer
hidden at the width the test steps to — in which case the fix may be to the
threshold, to the test's chosen width, or to both. Confirm which before
changing either: if the rail legitimately stays visible at 129 now, the test
is asserting the old layout and should move to a width that still hides it;
if the rail does hide and only the focus handoff was lost, that is a
production defect.

## Acceptance Criteria (the what)

- [ ] Determine whether, at 129 columns on current dev, the Inspector rail is
      expected to hide at all — and record the answer with a capture, not an
      assumption
- [ ] If it hides: focus lands on `#console-inspector-rail-open` when the
      rail is hidden by a resize, with the section's scroll offset preserved
- [ ] If it no longer hides at that width: the test is re-pinned to a width
      that does hide it, and the change notes which threshold moved and why
- [ ] `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset`
      passes on dev
- [ ] A resize that hides the rail never leaves focus on a widget inside the
      hidden rail (the property the test exists to protect), asserted at
      whatever width is chosen

## Notes

Not the only red on dev at the time of filing. These also fail on a pristine
current-dev checkout and are unrelated to this one — worth triaging
separately rather than folding in here:

- `test_console_inspector_navigation.py::test_staged_owner_sync_drives_ten_
  eleven_ten_cue_and_clamp`
- `test_console_live_work_handoffs.py` — four tests
- `test_console_narrow_layout.py::test_console_retry_speech_button_routes_
  without_resuming`
- `test_console_right_rail.py::test_authority_focus_f1_preserves_literal_
  rich_markup`
