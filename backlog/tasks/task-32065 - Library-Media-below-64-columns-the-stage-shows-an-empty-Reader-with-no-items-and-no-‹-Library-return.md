---
id: TASK-32065
title: >-
  Library Media below 64 columns: the stage shows an empty Reader with no items
  and no '‹ Library' return
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 19:55'
labels:
  - library
  - media
  - layout
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 60x24 activating Media paints 'Select a media item to read it here.' with two '›' grips, no items list and no return control, although library.md says the single-stage mode returns via '‹ Library'. The media page notes a pending follow-up for this exit. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Below 64 columns the Media stage shows the Items list
- [x] #2 A '‹ Library' (or '< Library' in ASCII mode) control is present and works by keyboard
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED at 60x24: the Media stage resolves library_open=False AND items_open=False, so it paints only the empty Reader placeholder and two grips -- no list, no way back to the rail.
2. Resolver: when the work pane holds no item and the width cannot seat a list beside it, the LIST wins the stage (list-only layout, reader gets what is left). Opt-in per destination profile (list_first_when_empty), Media only, following the list_grows precedent.
3. Add a '‹ Library' control to the Media Items pane, shown only while the Library pane is closed below the 64-column ordinary floor; it posts the same PaneToggleRequested('library') the grip does, so preference persistence is unchanged.
4. GREEN; live-verify at 60x24; docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two changes, both scoped to the widths where the defect exists.

AC#1 (resolver): a new opt-in profile flag, `list_first_when_empty`, set only on the Media profile (the `list_grows` precedent). Below the ordinary single-stage floor (64 -- `LIBRARY_EMERGENCY_WIDTH`, the constant the rail-and-canvas layouts already use), with NOTHING open in the work pane, the LIST wins the stage instead of being dropped for an empty Reader; the Reader keeps the remainder for its placeholder. Opening an item (`reader_has_item=True`) resolves the ordinary way and hands the width straight back. Bounded to <64 deliberately: at 80x24 the Items pane is dropped on purpose and focus evacuates to its grip, which three existing tests pin.

AC#2: a '‹ Library' Button at the top of the Media Items pane, shown only while the Library pane is closed below that same floor. It posts the grip's own `PaneToggleRequested('library')`, so pane state and preference persistence are unchanged and no new action/gate exists. Deliberately NOT `LibraryEmergencyReturn`: that widget's visibility belongs to the ordinary-route emergency stage, which force-hides every one of them whenever an adaptive reader shell is mounted. Its display is computed at compose time AND after each layout sync -- live at 60x24 the sync-only version came back hidden after the recompose a pane collapse triggers (pinned by test).

Live at 60x24 on the seeded profile: the Items list paints 11 rows with '‹ Library' above it; pressing it brings the rail back; collapsing the rail brings the control back. Captures: caps/32065-60x24-*.txt.

Files: tldw_chatbook/Utils/adaptive_reader_state.py, tldw_chatbook/Library/library_media_reader_state.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_crit8_polish_media.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
