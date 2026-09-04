---
id: TASK-31233
title: Review selected opens the review it creates
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 03:05'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1: the Select-mode "Review" bulk action toasts "Reviewing N items." then leaves the user in select mode with unchecked boxes and a blank reader pane — the invoked feature is invisible at the moment of invocation, and the user guide's promise ("Creating a set activates it and opens its first item in the Reader") is false for this path. Root cause verified: both create paths call _open_library_media_viewer(items[0]) (library_screen.py:39305), but nothing on the selection path exits select mode (_library_media_select_mode is cleared only by Done/bulk-delete, lines 24249/24526), so the viewer never surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Creating a set from Select mode exits select mode and opens the set's first item in the Reader, exactly like "Review these"
- [x] #2 The review banner and walk footer are armed immediately after the create
- [x] #3 A pinning test covers the select-path create end state (not select mode, viewer open at item 1)
<!-- AC:END -->

## Implementation Plan

1. RED: pin the create-from-selection end state (exit before open, no discard notice, canvas sync)
2. GREEN: exit select mode + canvas sync in _create_and_open_review_set before the viewer open
3. Live tmux verify at 200x50

## Implementation Notes

One guard in the shared _create_and_open_review_set (covers Review-selected AND any future creator invoked over armed select mode): if select mode is active, _exit_library_media_select_mode(announce_discard=False) then _sync_library_canvas(self, "media"), then the viewer open. The canvas sync is load-bearing and was found live: _open_library_media_viewer ends in the viewer-scoped seam, which updates only the VIEWER in place — without the canvas sync the select toolbar stayed mounted (stale "2 selected" rows) while the banner armed. Live-verified: Select → 2 checked → Review → browse toolbar restored, banner "2 selected items — 1 of 2", Reader at item 1. Files: library_screen.py, test_review_set_walker.py, user guide.
