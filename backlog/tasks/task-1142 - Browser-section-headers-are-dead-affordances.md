---
id: TASK-1142
title: Browser section headers are dead affordances
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:05'
updated_date: '2026-07-28 02:44'
labels:
  - console
  - ui
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F4): the conversation browser's top-level section headers (Starred/Workspaces/Chats) render collapse carets (▾/▸) but do not respond to clicks (caret column, caret+1, and label all inert), while workspace group rows toggle fine. This is a misleading affordance and makes TASK-912's collapsed-section marker aggregation unreachable through live interaction. Either wire the headers for click-toggle (persisting like group state) or remove the caret glyph from non-interactive headers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Section headers either toggle on click (state persisted, aggregate glyph shown when collapsed) or carry no collapse affordance.
- [x] #2 A mounted test drives the chosen behavior through the real click path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the existing section-header/group-header toggle implementation in `console_workspace_context.py` + `chat_screen.py`'s `on_button_pressed` handler and the collapse-state persistence seam (`_console_conversation_browser_collapse_preferences` / `_set_console_conversation_browser_group_collapsed`) to confirm what already exists.
2. Write a failing mounted test using the real `pilot.click()` path (not `.press()`) against the section header toggle.
3. Root-cause why the real click misses: measure widget geometry vs what the compositor actually resolves at that screen position.
4. Apply the minimal, defensible fix and re-verify with the real-click test.
5. Add the task-912 aggregate-marker-reachable test (collapse Workspaces via real click with a busy group beneath).
6. Run the full gate suite and record findings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Round 0 (superseded -- see Round 1 below)

The section-header `Button` + `on_button_pressed` dispatch + collapse-state
persistence already existed and, per a mounted test calling `Button.press()`
directly, looked wired. `.press()` invokes the handler directly and never
exercises Textual's actual click routing, which is why no prior test caught
this. Round 0 shipped an inline `toggle.styles.width/min_width/max_width = 3`
on both toggle buttons, theorizing the app-tier CSS width rule wasn't
reliably beating Textual's `Button` default. Round-1 review found this fix
**inert**: removing it left every round-0 test passing, because those tests
called `pilot.click(selector)`, which clicks at the widget's own CENTER --
never at the coordinates a live user's mouse actually lands on, which is the
caret glyph's rendered screen position. The inline styles were reverted.

### Round 1 (evidenced root cause)

Reopened with a coordinate-honest method: locate the caret glyph in the
RENDERED pane text (`compositor.render_strips()`, cell-width-aware index of
`▾`/`▸` in each row -- the same thing a tmux capture shows), then
`pilot.click(offset=(x, y))` at exactly that position, no selector.

Reviewer's Hypothesis A (the caret paints inside the non-interactive title
`Static`, not the `Button`) does **not** hold: a scan across 100-220 column
terminal widths, on a freshly-mounted tray, found the glyph's rendered
position resolving to the toggle `Button` every time.

The actual, reproduced mechanism: `ConsoleWorkspaceContextTray.
_conversation_browser_list_height` -- the tray's own auto-height estimate
for the `#console-workspace-conversations` container -- assumed every
section/group `empty_copy` line ("No starred conversations." / "No
workspace conversations.") always renders as exactly one row
(`_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1`, used as a flat constant).
Unlike a row title, that Static is **not** reduced by the star-column
chrome width, so at the tray's real content width it silently wraps to two
lines while the heuristic still counted one. Confirmed directly: at
`size=(160, 48)` with only "Chats" populated (Starred/Workspaces both
empty), collapsing "Chats" via a real glyph-coordinate click succeeds, but
searching the freshly rendered pane for "Chats"'s own (now `▸`) caret
raises `AssertionError` -- **the header is not merely mispositioned, it is
not painted at all**, clipped out of the tray's own `#console-workspace-
conversations` box because two wrapped empty-copy lines (Starred's and
Workspaces') pushed the real content two rows past what the heuristic
budgeted. Verified this exact test genuinely fails on 168f61ed8 (round 0's
commit) and passes after the round-1 fix below.

Fix: `_conversation_browser_list_height` now takes both `row_title_budget`
(unchanged, for row titles) and a separate `empty_copy_budget` (the tray's
full `_row_content_width`, no star-column reduction -- the budget an
empty-copy Static actually renders at). A new `_empty_copy_line_count`
helper reuses the same `wrap_console_conversation_title` word-wrap row
titles already use, so the estimate can no longer disagree with a plausible
real render the way the flat constant did. This is a real, in-scope fix for
the click-affordance bug (a header whose caret is clipped out of the visible
pane cannot be clicked, full stop), not a cosmetic height tweak.

The round-0 inline width styling was reverted (confirmed inert per the
review; no scenario found where it changes real behavior once the CSS
bundle is loaded, which every production run does).

### Tests

- `test_section_header_caret_is_clickable_at_its_rendered_screen_coordinates`
  (new, round 1): the coordinate-honest test the review required. Locates
  the caret in rendered text, clicks at those exact screen coordinates
  (not widget-center), for both a collapse and a subsequent expand read
  from the freshly re-rendered pane. Fails on 168f61ed8, passes after the
  round-1 fix.
- `test_console_rail_list_height_matches_rendered_rows` (pre-existing,
  updated): its own expected-height math assumed a flat one row per
  empty-copy widget -- exactly the stale assumption this fix corrects.
  Updated to sum each empty-copy widget's own settled `region.height`
  (ground truth: what `Static` actually painted) instead.
- `test_section_header_toggles_via_real_click_and_persists_across_rebuild`
  and `test_collapsing_workspaces_via_real_click_reveals_aggregate_marker_
  from_busy_group` (round 0, kept): widget-center real clicks proving
  collapse/expand/persist-across-rebuild and TASK-912's aggregate marker
  reachability. Still valuable as coverage of the click-handling and
  persistence logic even though they could not, by construction, have
  caught the round-0 defect.

Modified files:
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` -- reverted
  the inert inline width styling; added `_empty_copy_line_count` and
  widened `_conversation_browser_list_height`'s empty-copy budget.
- `Tests/UI/test_console_workspace_context_rail.py` -- new coordinate-honest
  test + helpers; updated `test_console_rail_list_height_matches_rendered_
  rows`'s expected-height math; docstring correction on
  `_click_conversation_browser_toggle` noting its widget-center limitation.

Verification: `Tests/UI/test_console_workspace_context_rail.py` +
`Tests/Workspaces/test_console_conversation_browser_state.py` +
`Tests/UI/test_console_parallel_runs.py` in one run -- 142 passed, 1
pre-existing failure (`test_console_workspace_context_syncs_active_
conversation_marker`, a `TypeError` on `_sync_console_workspace_context`'s
signature, reproduced identically on the unmodified base commit via `git
stash`).
<!-- SECTION:NOTES:END -->
