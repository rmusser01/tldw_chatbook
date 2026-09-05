---
id: TASK-31661
title: Environment 10s poll silently resets rail focus when the file set changes
status: Done
assignee: []
created_date: '2026-09-05 07:00'
updated_date: '2026-09-05 19:17'
labels:
  - console
  - inspector
  - ux
  - critique-2026-09-05
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1, live-measured: with focus parked on a rail row, an external
file change makes the next 10s poll recompose the section and throw focus
to a widget above the section header (invisible focus, no indicator moves;
two Tabs to recover). Fires repeatedly during agent runs — the panel's core
workflow. The activation path already restores focus by row_id
(_request_console_environment_row_focus); the poll/sync path does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A poll-driven recompose restores focus to the row with the same row_id when it still exists, else to the nearest surviving row in the same section
- [x] #2 Focus never lands on a widget with no visible indication as a result of a background sync
- [x] #3 Wiring test: park focus on a row, land a snapshot that changes the row set, assert focus location after the sync
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _land_console_environment, the activation path
   (_console_environment_row_is_focused / _request_console_environment_row_focus /
   _focus_console_environment_row / _handle_console_environment_row), and
   ConsoleInspectorSection.sync_state/_structural_key to understand the
   capture-before-mutate + call_next discipline already used for the
   expand/collapse (activation) focus restore.
2. Write RED wiring tests in Tests/UI/test_console_environment_wiring.py:
   (a) a poll landing that ADDS a row while focus is on a surviving row_id,
   (b) a poll landing that REMOVES the focused row_id (a whole PR sub-tree
   disappearing), asserting the nearest-surviving-row fallback, and
   (c) a negative control with focus outside the rail (composer) that must
   not be touched. Confirm (a)/(b) fail against the current code.
3. Implement the fix in chat_screen.py: capture each section's focused
   row_id (if any) synchronously in _land_console_environment BEFORE
   calling sync_state; after both sections sync, schedule a restore via
   section.call_next for any section that had a captured row_id. The
   restore tries the exact row_id first (reusing the activation path's
   widget lookup, refactored into a shared helper), then falls back to
   the nearest surviving row (same index, then index-1, then the first
   row) when the exact row is gone.
4. Re-run the wiring tests (green), then the adjacent suites (controller,
   section, inspector-section, right-rail, fleet panel, native chat flow,
   destination shells, product-maturity gate1) to confirm no regressions.
5. Update the task file's ACs + Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: `_land_console_environment` (the 10s Environment/Tasks poll's
landing path) called `sync_state` on both rail sections with no regard for
which row currently held focus. When an external change (new commit, PR
closing) altered a section's row SET, `sync_state` took the structural
recompose branch, unmounting the focused row; Textual's own focus reset
then landed the caret on the rail's outer body -- a widget with no visible
focus indicator, matching the live-measured defect exactly.

Fix mirrors the existing activation-path discipline
(`_handle_console_environment_row` -> `_request_console_environment_row_
focus` -> `_focus_console_environment_row`): before either section's
`sync_state` call, synchronously capture which row (if any, per section)
owns focus via new `_console_environment_focused_row_in_section`
(row.section_id match -- zero overhead when focus is elsewhere, e.g. the
composer). After both syncs, schedule a restore via the SECTION's own
`call_next` (same ordering guarantee as the activation path -- runs after
any queued recompose). The restore
(`_focus_console_environment_row_after_sync`) tries the exact row_id
first (via a new shared lookup, `_focus_console_environment_row_by_id`,
factored out of `_focus_console_environment_row`); if that row is gone,
it falls back to the nearest surviving row -- same index, then index-1,
then the section's first row -- three fixed candidates per the plan's
"keep it simple" guidance, never a widening search.

Confirmed via `ps aux`/git log that the Agent fleet section is a genuinely
separate path (`_sync_console_agent_section`, its own tick) and out of
scope; documented explicitly rather than left as a silent gap.

Tests (Tests/UI/test_console_environment_wiring.py, TDD): two new RED
cases reproduced the live defect exactly against the real screen/harness
(focus landed on `_InspectorOuterBody`) -- an ADD-a-row case (exact row_id
survives) and a REMOVE-the-focused-row case (whole PR sub-tree
disappears, exercising the first-row fallback rung), plus a negative
control (focus in the composer is never touched). All three green after
the fix; discovered along the way that `env-file-N` rows are not
`clickable` and can therefore never hold focus, so the removal scenario
uses the PR row sub-tree instead of a file row.

Full suite check: Tests/UI/test_console_environment_wiring.py (31),
test_console_environment_controller.py, test_console_environment_section.py,
Tests/Chat/test_console_environment_state.py, test_console_right_rail.py
-- 157 passed / 158 collected in one combined run; the one failure
(test_console_inspector_section.py::test_inspector_section_css_is_styled_
in_source_and_bundle) is a pre-existing CSS-bundle/source drift this diff
never touches (git diff --stat shows zero CSS files changed).

Files: tldw_chatbook/UI/Screens/chat_screen.py,
Tests/UI/test_console_environment_wiring.py. Full report:
.superpowers/sdd/2026-09-05-inspect-rail-critique-burndown/task-31661-report.md

--- Round-1 review fixes (2026-09-05) ---

Review found two Importants (both probe-verified) plus ride-alongs against
the original implementation:

I1 (AC#2 gap): UNBOUND/ERROR/PENDING/"No git workspace" Environment
projections (task-31660) render a row explaining the state but never a
clickable one, so when the row set collapses into one of those and the
previously-focused row is gone, the nearest-survivor search had nothing
to land on and fell through to the defect widget. Fixed:
`_focus_console_environment_row_after_sync` now falls back to the
section's own collapse chevron (`_focus_console_environment_section_
toggle`) -- a real focusable Button -- as the last resort. Pinned with
two tests (UNBOUND + ERROR variants); confirmed RED without the fallback.

I2 (guard at the wrong window): the original docstring claimed the
synchronous pre-sync capture protected against a mid-flight user click,
but `sync_state`'s own recompose (unmount, then remount) is two awaits,
so real time -- and real input -- passes between `call_next` scheduling
the restore and it actually running. Probe: click the composer in that
window -> focus got silently stolen back to the rail row. Fixed: the
restore now checks, AT CALLBACK TIME, whether focus has left the rail
body (`_console_environment_focus_left_the_rail`, using the same
`owner in focused.ancestors_with_self` idiom already used elsewhere in
this file) and yields immediately if so. Pinned with two tests (a
same-tick move via a direct focus() call before any await, and a
one-tick-later move injected after draining `asyncio.sleep(0)` cycles
until Textual's own unmount reset has fired but this task's restore has
not yet run -- both confirmed RED with the guard disabled).

M4 (fallback wasn't genuinely nearest): the original 3-candidate ladder
(same index, index-1, first row) could overshoot a much closer surviving
neighbor straight to a distant row whenever more than one row vanished
at once. Replaced with `_nearest_surviving_console_environment_row_id`,
which walks OUTWARD from the removed row's old index (distance 1, 2, 3,
...) for the first id still present. Updated the removal test's expected
outcome from "env-changes" (distance 5) to the true nearest survivor
"env-branch" (distance 3) -- pins the outward walk against a regression
to a fixed "always land on row 0" shortcut.

M5 (skip no-op syncs): `_land_console_environment` now also captures each
section's pre-sync rows/summary and skips scheduling a restore entirely
when the new state is identical (matching `sync_state`'s own early-return
condition) -- one less redundant `call_next` on a landing that didn't
change that section at all.

#3 (docstring correction): removed the false "never in response to a
user gesture" claim (`_handle_console_environment_row`'s row-expansion
branch also calls this method) and documented the resulting benign
double-schedule (two `call_next` restores queued for the same row when
that path also calls `_request_console_environment_row_focus`) as a
deliberate, harmless redundancy rather than fixing it away.

M7 (test-only): `Tests/UI/test_console_inspector_section.py`'s CSS-bundle
guard hard-coded `tldw_cli_modular.tcss`, but the CSS build's
screen-owned split (`build_css.py`) moved every
`.console-inspector-section*` rule out of that monolithic bundle into
`screen_agentic_console.tcss` (loaded directly by app.py/chat_screen.py)
-- 0 matches vs 17, the exact pre-existing red this task's original
report baselined. Repointed the check at the bundle that actually ships
the rules; verified it goes green now and red again when a rule is
removed from that file (temporary local mutation + restore, no diff
left behind).

BOOKKEEPING: appended AC#13 to task-31665 flagging that the fleet
section's `_sync_console_agent_section` has the same focus-stealing
defect (its rows ARE focusable) -- out of this task's scope, filed as a
follow-up per the review.

Files: tldw_chatbook/UI/Screens/chat_screen.py,
Tests/UI/test_console_environment_wiring.py (2 new tests + 2 existing
ones' expected outcomes corrected),
Tests/UI/test_console_inspector_section.py (M7 path fix).
Full suite: Tests/UI/test_console_environment_wiring.py (35 passed),
Tests/UI/test_console_environment_controller.py,
Tests/UI/test_console_environment_section.py,
Tests/Chat/test_console_environment_state.py,
Tests/UI/test_console_right_rail.py, Tests/UI/test_console_inspector_
section.py -- all green together; ./scripts/preflight.sh clean.
<!-- SECTION:NOTES:END -->
