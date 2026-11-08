---
id: TASK-39169
title: "Settings can take back a focus the user moved: a same-category pane rebuild restores its captured control, and the category landing's deferred focus overrides a press that lands first"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- settings
- ui
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_panel_scoped_updates.py
dependencies:
- TASK-39046
- TASK-39135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39135 and its reviews, round-12 branch fu/r12-settings). Read on the round-12 stack 9091820bf2, 2026-11-08. Round 12 changes this code, so it is named by symbol, not line number.

**Same-category rebuild.**
- The category swap captures `focus_id = self._focused_id_in_category_panes()` at swap start and rebuilds both panes. By review r12s-rev2's measure the rebuild takes 81-1,348 ms, and input keeps running meanwhile.
- When no category landing is pending, the swap then calls `_restore_category_pane_focus(focus_id)`.
- A same-category rebuild has no pending landing. Examples are a Workspaces row click, the show-archived toggle and `_refresh_settings_workspaces_pane`. So `_focus_left_pending_origin()` never withdraws anything.
- A focus the user moved during the rebuild, to the rail filter with "/" for example, is then taken back to the captured control.
- TASK-39135's notes leave this out of scope: "a same-category rebuild is outside this task".

**Deferred landing.**
- The landing calls `_focus_category(category_value)`, which focuses a rail `Button`.
- Textual 8.2.8's `Widget.focus()` does `self.app.call_later(set_focus, self)` (`textual/widget.py:4579-4598`), so the focus lands one callback later.
- A key or click handled between `_focus_category` and that callback is overridden.
- TASK-39046's case, a Review press about 40 ms after a navigation, may partly be this window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focus the user moves during a same-category rebuild is kept; a test with a real key and a real click during a held rebuild is red before the change
- [ ] #2 A focus the user moves between the landing's `_focus_category` and its deferred `set_focus` is kept; a test is red before the change
- [ ] #3 With no input, a same-category rebuild still restores the captured control and a category switch still lands on its rail row; the existing pins stay green
- [ ] #4 The tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
