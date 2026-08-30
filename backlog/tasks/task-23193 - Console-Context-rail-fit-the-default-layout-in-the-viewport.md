---
id: TASK-23193
title: 'Console Context rail: fit the default layout in the viewport'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:55'
updated_date: '2026-08-29 23:02'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Context rail's default configuration renders 51 rows into a 32-row viewport at 160x48, and overflows at every one of ten measured terminal geometries including 200x60. Three of seven sections are entirely below the fold on a fresh install. Reduce per-section chrome and open fewer sections by default so the shipped default fits without scrolling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Context rail default content height fits the viewport at 160x48 without the outer overflow hint
- [x] #2 Default open sections are Sessions and Conversations only
- [x] #3 Every section header is reachable without scrolling at 160x48 (revised from "reduce header chrome" - see Implementation Notes)
- [x] #4 A regression test pins default content height against viewport height at 160x48
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a failing UI test asserting the Context rail's outer content height fits the viewport at 160x48 with default preferences\n2. Change ConsoleRailPreferences/ConsoleRailState defaults to open Sessions and Conversations only\n3. Remove the redundant border-top separator from .console-rail-section-header (the raised background already separates sections) and regenerate the CSS bundle\n4. Re-measure with the UAT harness and capture a screenshot\n5. Confirm no overflow hint at 160x48
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the overflow by changing the shipped defaults only. ConsoleRailPreferences and ConsoleRailState now open Sessions and Conversations; Workspaces, Model and Character ship closed (Agent and Details already did).

Measured with the UAT harness in output/ux-review-console (uat_context.py, cleared sandbox so persisted prefs cannot mask defaults):

  geometry   before          after
  200x60     50/44 overflow  45/45 fits
  160x48     51/32 overflow  33/33 fits, all 7 headers visible
  140x40     51/24 overflow  30/24 overflow (3 headers hidden)
  110x30     51/16 overflow  30/16 overflow

AC #3 was revised mid-task. The original wording assumed reducing header chrome. I built that variant - dropping the redundant border-top rule and the 2-row min-height, which made 140x40 fit too (25/25) - but an existing contract test (test_context_section_headers_match_inspector_title_band) deliberately ties Context section headers to the Inspector's title band and pins header height at 2. On review of both screenshots the reviewer chose to keep the rule, so the header change was reverted and AC #3 was rewritten to the outcome actually delivered and tested: all seven headers reachable without scrolling at 160x48. 140x40 and below still overflow; that is follow-up work, not silently dropped.

Two things the audit that produced this task got wrong, corrected here:
- The waste is not a uniform '2 blank rows + 1 separator' gutter. Headers are 1 content row plus 1 border row, and Textual's Widget.size excludes borders; the rest of the slack is inside section bodies.
- ConsoleBoundedSection.allocation is not dead. It is granted only to the ACTIVATED section, and only when the rail overflows. A rail that fits reports allocation None, meaning 'shown in full'.

Test fallout, all 'the test encoded the old default':
- Tests/Chat/test_console_rail_state.py - three default-value expectations updated.
- Tests/UI/test_console_left_rail.py - disclosure-independence test now opens Workspaces explicitly rather than relying on it.
- Tests/UI/test_console_character_avatar.py - fixtures open Character and wait for layout; one geometry assertion re-pinned. Its 'holder < body' claim depended on the crowded default: with fewer sections open the portrait legitimately fills the rail width. Replaced with a direct guard on the task-1661 regression (measured width must not be pinned at the 16-column minimum).
- Tests/UI/test_console_new_workspace.py - the helper captured its Button before ConsoleWorkspaceContextTray re-mounted, so press() no-opped on a detached widget and read as a broken handler. Now waits on and re-acquires the button after every await.
- New: Tests/UI/console_rail_section_helpers.py, a shared open_rail_section helper.

Not addressed, pre-existing on dev and verified by re-running them with only console_rail_state.py reverted: 4 failures in test_console_session_settings.py and test_console_inspector_compact_access.py. preflight reports diagnostic-inventory drift in library_skills_browse_controller.py from commit 40af5ba07; not regenerated, since that would fold another change's unreviewed diagnostics into this one.

Files: tldw_chatbook/Chat/console_rail_state.py; Tests/UI/test_console_context_rail_fits.py (new); Tests/UI/console_rail_section_helpers.py (new); 4 test files updated.
<!-- SECTION:NOTES:END -->
