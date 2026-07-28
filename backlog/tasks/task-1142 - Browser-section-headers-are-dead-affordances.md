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
The section-header `Button` + `on_button_pressed` dispatch + collapse-state
persistence (`_console_conversation_browser_collapse_preferences`,
`"section:<id>"` keys read by `build_console_conversation_browser_state`)
already existed and, per a mounted test calling `Button.press()` directly,
looked wired. `.press()` invokes the handler directly and never exercises
Textual's actual click routing (`get_widget_at` hit-testing against the
compositor), which is exactly why no prior test caught this and why the
live UAT session could still see the caret as dead.

Root cause, found by driving the real `pilot.click()` path against a fully
styled mount (production CSS bundle + a real transcript message, so the
onboarding setup-card backdrop that otherwise swallows every click isn't in
the way): the section/group toggle `Button`s relied solely on the app-tier
CSS rule `.console-workspace-action.console-workspace-conversations-toggle
{ width: 3; ... }` to beat Textual's own `Button` default (`width: auto;
min-width: 16` -- the exact failure mode TASK-712 already hit for the
Switch/New buttons in this same rail). Whenever that CSS rule doesn't
apply (confirmed: every OTHER mounted test in this file runs without the
CSS bundle loaded at all, per `ConsoleWorkspaceContextTray`'s "loads
widget DEFAULT_CSS but not the built bundle" test-harness note), the
button's real layout width stays 16, pushing its actual (unclipped) hit
region past the rail's clipped right edge -- the caret glyph paints inside
the visible column, but the widget the compositor resolves at that screen
position (matching what a real mouse click hits) is whatever sits behind
the rail. Fix: set `toggle.styles.width/min_width/max_width = 3` inline on
both the section-header and group-header toggle (mirroring the legacy
single-section toggle's own existing belt-and-suspenders inline styling),
removing the dependency on the CSS bundle having loaded/cascaded
correctly at all.

Also found (documented as a discovered-but-out-of-scope follow-up, not
fixed here): `ConsoleWorkspaceContextTray._conversation_browser_list_height`
assumes every section's `empty_copy` line ("No starred conversations." /
"No workspace conversations.") renders as exactly one row
(`_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1`), but at a narrow rail width
that text wraps to two lines in the real render -- undershooting the
tray's auto-height and, in a narrow-enough terminal, clipping a
lower section's toggle out of its own container's visible bounds. Sidestepped
in the new tests by using a wider harness size (`220x52`, matching the
precedent in `test_console_workspace_action_row_geometry.py`), not by
fixing the estimator; a real user on a narrow terminal with multiple empty
sections could still hit this independently of TASK-1142's fix.

Added two mounted regression tests to `Tests/UI/test_console_workspace_
context_rail.py`, both against `StyledConsoleHarness` (real CSS bundle) and
a helper (`_click_conversation_browser_toggle`) that scrolls the target
into view and waits for `get_widget_at` to agree with the widget's own
region before calling `pilot.click` (never for this defect specifically,
but because the tray's own `_fit_height_to_content` settles its auto-height
over more than one deferred pass -- a real, pre-existing and unrelated
timing race in that machinery, worked around by never re-toggling the same
tray instance twice in quick succession, not fixed):

- `test_section_header_toggles_via_real_click_and_persists_across_rebuild`:
  real click collapses (rows unmount, caret flips to `▸`), state persists
  across a rail rebuild (workspace switch away and back), then a real
  click on the freshly-rebuilt tray expands it back (rows remount, caret
  flips to `▾`).
- `test_collapsing_workspaces_via_real_click_reveals_aggregate_marker_from_
  busy_group`: a real background session with a live run in its own
  workspace group, collapsed via the real click path, surfaces TASK-912's
  aggregate glyph (`Workspaces ●`) on the header -- proving that marker is
  now reachable through live interaction, not just the empty-Chats default
  collapse.

Modified files:
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` -- inline
  width hardening on both toggle buttons.
- `Tests/UI/test_console_workspace_context_rail.py` -- two new real-click
  mounted tests plus shared helpers.

Verification: `Tests/UI/test_console_workspace_context_rail.py` +
`Tests/Workspaces/test_console_conversation_browser_state.py` +
`Tests/UI/test_console_parallel_runs.py` in one run -- 141 passed, 1
pre-existing failure (`test_console_workspace_context_syncs_active_
conversation_marker`, a `TypeError` on `_sync_console_workspace_context`'s
signature, reproduced identically on the unmodified base commit).
<!-- SECTION:NOTES:END -->
