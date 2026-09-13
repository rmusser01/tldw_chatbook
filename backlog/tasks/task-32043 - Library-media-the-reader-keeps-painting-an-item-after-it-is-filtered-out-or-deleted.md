---
id: TASK-32043
title: >-
  Library media: the reader keeps painting an item after it is filtered out or
  deleted
status: Done
assignee: []
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 16:59'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P2. After a filter that yields zero results, and after deleting the item currently loaded in the reader, the reader keeps painting the now-absent item's content instead of reflecting that nothing is selected. The reader should track the visible set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the loaded media item leaves the visible set (0-result filter or deletion), the reader clears to its no-selection state (or a 'this item was deleted' note with Undo)
- [x] #2 Opening or re-filtering to a present item still shows it normally
- [x] #3 A pin drives a 0-result filter and a delete-of-loaded-item and asserts the reader no longer paints the absent item
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Media reader kept painting an item after a 0-result filter or after the loaded item was deleted. Fix: one shared `_reset_library_media_reader_to_no_selection` (mirrors the single-item-delete reset) called from two paths that lacked reconciliation: the filter-apply seam `_sync_library_media_browse_state` (only on a SETTLED page with zero retained rows that still holds a LOCAL reader item -- mid-load, page-turns, and filter-with-results are structurally excluded), and the bulk-delete worker (when the reader's loaded/pending local id is in `succeeded_ids`). The reader paints from `_media_state.detail`, so the helper drops `detail`/`composed_detail`/`highlights`, sets `view=list`, rebuilds an empty session (all id slots None, request-generation bumped to reject in-flight settles), and clears `selected_media_id`. Undo preserved: the reset runs BEFORE the receipt is written and never touches the receipt fields (pinned -- receipt survives, Undo re-adds and re-opens normally). Server items respected via external_detail. The one new read on the shared bulk-mutation worker uses `getattr(self._media_state, 'reader_session', None)` (the multiselect SimpleNamespace fakes are partial). Two new pins (0-result filter clears; bulk-delete of the open item clears), red first; 8/8 crit6 scroll-restore + ladder + Undo regression pins green. The two `test_library_media_reader_scroller_resolution.py` pins are baseline-red on dev for an UNRELATED harness backend gap (`Local quiz backend is unavailable`), not this change. Files: library_screen.py, Tests/UI/test_library_shell.py (or the media pins file), Docs/User_Guide, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
