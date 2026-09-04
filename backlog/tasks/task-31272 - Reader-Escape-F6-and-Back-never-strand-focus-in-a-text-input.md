---
id: TASK-31272
title: 'Reader Escape, F6 and Back never strand focus in a text input'
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 19:58'
labels:
  - library
  - media-ux
  - a11y
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P1: leaving the Reader takes three Escapes and the second lands inside the rail's Search input (A cap 20); `s` typed at the F6 rail stop lands in `Search Library…` (B cap_79); F6's content stop is real (first candidate in `_MEDIA_WORKBENCH_FOCUS_TARGETS`) but invisible after the task-31221 outline suppression, so a keyboard user cannot tell it landed (B cap_57); Escape does not close the More menu though the label promises `close more` (B cap_106); `‹ Back` flips `_library_media_view` to list without changing the visible Reader, so ]/[ die on identical pixels (A cap 25/31/39). Eight distinct `esc …` labels were seen live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape from the Reader focuses the loaded list row, never an Input
- [x] #2 The F6 content stop has a visible focus treatment that does not paint over content (painted-text test proves content still paints; a focus test proves the treatment)
- [x] #3 Escape closes the More menu
- [x] #4 In the three-pane shell, Back either is removed or makes list mode visible (dimmed Reader or a parked label) so the key-map change is explained on screen
- [x] #5 Distinct `esc …` labels reduced to four or fewer, table recorded in the notes
- [x] #6 Live-verified
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: Escape ladder → loaded row → rail row; More closes from content focus; F6 content stop visible with content still painting; Back absent at 235x52 / present at 100x30; labels ∈ {close, focus Items, focus Library, back}
2. GREEN: one _library_media_reader_exit_available predicate behind Back, the rail-Escape branch and the chip; view flag no longer flipped in three-pane (the More gate read it); focus tint on the existing border with outline:none; _library_media_list_surface_active keys s / list footer / choice strips / list refresh on the Items region; Escape gate = bool(label)
3. Live tmux 235x52 + 100x30
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Escape from the Reader focuses the loaded Items row and leaves the Reader live; Escape from the Items region focuses the Media rail ROW (never #library-search-input). The More menu did close before — the check_action gate refused because Back/rail-Escape had flipped _library_media_view to 'list'; the flip is gone in three-pane, so ]/[ stay bound. '‹ Back' is not composed in the side-by-side layout (viewer back_visible, threaded through construction and the in-place sync compare/assign) and stays in the compact layout with the existing exit. F6's content stop tints the box's existing border (outline: none, painted-text proven). Escape vocabulary 8 → 4: close / focus Items / focus Library / back (table in the SDD report); the shared choice strip still says 'esc cancel'. With Back gone, everything keyed on view=='list' was re-keyed on the Items region holding focus (_library_media_list_surface_active): the s gate, the list footer set, the type/sort strips (Escape closes an open strip first), the background list refresh; on a plain row the Reader footer stays with s added. Follow-ups filed after the wave: below ~92 cols (both panes collapsed) the exit flips the flag with nothing visible — the fix is to open the Items pane; select mode surviving an Escape hop to the rail; F6's first two stops are text inputs; 100x30 list view advertises 'esc focus rail' but the rail target is not focusable there; a _library_media_focus_region() refactor; 'library-media-back' still an F6 candidate. The three #2367-era shell tests that waited on an always-mounted Find bar were converted; the large-document proxy's residual is its pre-existing parse-count pin.
<!-- SECTION:NOTES:END -->
