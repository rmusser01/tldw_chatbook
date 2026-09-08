---
id: TASK-31979
title: >-
  Library media: the wide layout truncates a long title while the reader pane
  sits empty
status: Done
assignee:
  - '@claude'
created_date: '2026-09-07 22:48'
updated_date: '2026-09-08 03:25'
labels:
  - library
  - media
  - ux
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P2, both assessors. At 235x52 a 98-character title is cut to 47 characters in a fixed 52-column list pane while roughly 120 columns of reader pane hold the single sentence `Select a media item to read it here.`. At 100x30 the same title wraps across three fully readable lines. The wide layout is the one that loses information: the two-pane split reserves reader width that nothing uses when no item is open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When no media item is open, the list pane is allowed enough width to show substantially more of a long title (or the title wraps rather than truncating)
- [x] #2 Opening an item restores the reader pane's width
- [x] #3 A pin asserts a long title paints more characters at 235x52 with no item open than the current fixed 52-column pane allows
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add reader_has_item param to resolve_adaptive_reader_layout (default True = current behavior); when items open, reader empty, and not custom, give freed reader columns to items down to work_min floor.\n2. Thread through resolve_media_reader_layout wrapper.\n3. Wire library_screen _sync_library_media_reader_layout_from_shell to pass reader_has_item = (view == viewer).\n4. Update the list-view resize test to expect the widened layout; add resolver unit tests (RED) + painted pin (235x52 no-item shows more chars, opening restores reader).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a reader_has_item flag (default True = unchanged) to resolve_adaptive_reader_layout: when the work/Reader pane has no item open and the Items list is open (automatic widths only, matching the list_grows gate), the freed Reader columns go to the list down to work_min_width. Threaded through resolve_media_reader_layout; the Media call site (_sync_library_media_reader_layout_from_shell) passes reader_has_item = (_library_media_view == viewer).

Widths at 235x52: item open 34/56/143 (unchanged pin); no item open 34/153/46 -> a 98-char title paints in full (ellipsized at the pane edge, cap 120) instead of ~56 cells. 100x30 is width-starved (freed=0) so it is byte-for-byte unchanged.

Opening restores the split via a new side-effect-free _restore_library_media_reader_width_on_open (shell.sync_layout only, NO presentation-epoch advance / NO return-settlement re-arm) because the full sync's epoch machinery corrupted the media-return fence mid-open. Because list-view and viewer-view Media layouts now differ, _library_media_layout_signature was made invariant to the widening (derived from the canonical item-open reader width) so an exact scroll return survives the list<->viewer transition.

RULING (scope): the empty-reader widening is applied only on the resolver's main (non-starved) path; the width-starved priority early-return has no surplus to reallocate. Interaction with task-31970's two comfort clamps: unchanged -- the new widening is gated on `not custom_widths_enabled` (Custom obeys), so it never widens a hand-typed width; task-31968/31970 left separate.

Files: tldw_chatbook/Utils/adaptive_reader_state.py, tldw_chatbook/Library/library_media_reader_state.py, tldw_chatbook/UI/Screens/library_screen.py, Docs/User_Guide/library/media-and-conversations.md; tests Tests/Library/test_library_adaptive_reader_state.py (3 new unit tests), Tests/UI/test_library_media_reader_shell.py (painted pin at 235x52 + 100x30, resize-test expectation updated), Tests/UI/test_library_media_return_settlement.py (signature pin updated).
<!-- SECTION:NOTES:END -->
