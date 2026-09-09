---
id: TASK-32170
title: >-
  Library phase-C media: extract the 20 deferred canvas-origin handler bodies
  onto LibraryMediaController so their @on rows can graduate
status: To Do
assignee: []
created_date: '2026-09-09 10:03'
labels:
  - library
  - library-decomposition
  - phase-c
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase-C task 3's region-ownership migration moved 16 of media's 79 @on rows onto LibraryMediaCanvas, but 20 canvas-origin rows are DEFERRED: the control is composed by the canvas, yet the handler body still lives on LibraryScreen, outside the media series' _MEDIA_CLUSTER_METHOD_NAMES (140 names). Moving the @on row alone would make the canvas call back through the screen, which the named-constructor-dependency canon retired; moving the bodies is a phase-A-style pure extraction that phase C rides after, never instead of. This corrects the graduation criterion 'the subsystem's phase-A series is fully landed' -- true of the series, false of every handler it left screen-native. Until they move, the canvas owns 16 of the 36 rows it structurally could (44% of its own subtree's routing), and the heaviest media interactions (the media row itself, the whole bulk-delete and analyze flows) stay screen-native. The 20 are enumerated with their blocker in Tests/UI/test_library_phase_c_region_ownership.py as _MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS: the analyze family (overwrite, receipt_dismiss, retry, selected, skip), the bulk_delete family (cancel, confirm, receipt_dismiss, undo), delete_selected, the empty-state pair (clear_type, import), export_selected, the review-dismiss pair (receipt_close, undo), the media row, select_toggle, trash_open, and the type filter pair (choice, filter_pressed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 20 deferred handler bodies (the _MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS set) move from LibraryScreen onto LibraryMediaController as a pure move, byte-for-byte, with _MEDIA_CLUSTER_METHOD_NAMES and the media-wiring delegator pins updated
- [ ] #2 Each of the 20 @on rows then migrates onto LibraryMediaCanvas following task 3's mechanism (the actions= constructor dependency and refusal via _media_actions_for_press), and its row in test_library_phase_c_region_ownership.py moves from the deferred set to the migrated set
- [ ] #3 The screen size ratchet is lowered in the same landing commit; the recompose-site and preimport ratchets stay neutral
- [ ] #4 Both dual-receiver guards, all residency guards, and the acceptance and storm pins stay green; the media battery shows zero branch-unique failures paired against the base
- [ ] #5 The deferred set in the region-ownership test is empty at close, or any residual row is re-classified as permanent with its reason, so 'deferred' is a transient state and not a permanent bucket
<!-- AC:END -->
