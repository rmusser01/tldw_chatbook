---
id: TASK-32362
title: >-
  Library: two disabled controls without an adjacent reason (Export bundle; '○
  Find' on the Analysis tab)
status: Done
assignee: []
created_date: '2026-09-11 06:19'
updated_date: '2026-09-11 07:49'
labels:
  - library
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Export bundle (.zip)' carries no inline reason — 'No destination chosen' sits three rows above (B D12 cap 32); '○ Find' is disabled on the Analysis tab with no reason beside it (A cap 42). Both break the screen's own ○-plus-reason rule. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every disabled control on the Export canvas and in the Media viewer carries its reason on the same line
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Expose submit_blocked_reason on the export state from the same source export_button_tooltip uses.
2. Yield an inline reason Static under the Export submit button (library-media-action-reason class).
3. Yield the Find reason inline under the Reader primary toolbar, matching the task-31981 Analysis precedent.
4. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both blocked controls now print their reason on the line below them, in the library-media-action-reason grammar task-31981 established for the Reader's blocked Generate.

Export button: LibraryExportFormState gained a submit_blocked_reason property that returns export_button_tooltip(self) when the gate is closed and "" when it is open -- one source, so the inline line and the tooltip are literally the same string and cannot drift. apply_library_export_submit_gate (the single helper the compose path and both in-place patchers already shared for disabled/label/tooltip) now flips the Static too, so counts landing cannot leave a stale reason. A test parametrises all four blocked predicates against export_button_tooltip, and a Pilot test asserts the line sits at button.region.y + height.

Reader "○ Find": _compose_primary_toolbar yields the Static AFTER the ds-toolbar Horizontal closes, not inside it -- mixing a Static in with that toolbar's Buttons is this canvas's documented non-rendering failure mode, and the Generate precedent in the same file yields outside for the same reason. The string still comes from analysis_find_unavailable_reason, so it tracks whatever that function returns.

Live (tmux 235x52, seeded profile): Analysis tab with no analysis renders "○ Find" over "No analysis to search yet.", directly above the existing "○ Generate" / "No analysis provider is configured · ..." pair. Confirmed at the same time that the Info tab's Find is still ENABLED today (analysis_find_unavailable_reason returns "" for mode != "analysis"); Task 1 owns that function, and once it lands the Info case renders through this same branch with no further change here.

Files: tldw_chatbook/Library/library_export_state.py, tldw_chatbook/Widgets/Library/library_export_canvas.py, tldw_chatbook/Widgets/Library/library_media_viewer.py (_compose_primary_toolbar only -- _compose_active_body untouched, it belongs to Task 1); Tests/UI/test_library_crit10_export.py; Docs/User_Guide/library/import-and-export.md, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
