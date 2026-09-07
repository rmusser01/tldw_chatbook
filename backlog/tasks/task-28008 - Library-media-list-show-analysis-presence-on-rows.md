---
id: TASK-28008
title: Library media list - show analysis presence on rows
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 04:10'
updated_date: '2026-09-06 23:45'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The browse summary projection is contractually exactly five keys (library_media_state.py:52-54 validator), so a row cannot show whether the item has an analysis - the product's core artifact is invisible at every list-level decision point. Extend the projection with has_analysis, bump the key-set contract, and render a one-glyph marker in the row secondary line and the preview pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rows visibly distinguish items with an analysis from items without
- [x] #2 The summary key-set contract and its tests are updated
- [x] #3 No per-row extra DB round-trips (presence comes from the existing summary query)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1: bump the media summary contract from five to exactly seven keys (has_analysis, reviewed); has_analysis projected in SQL (EXISTS over DocumentVersions, newest live version with non-empty analysis, deleted = 0 on both legs) with the query plan pinned against the existing index; every producer/fake/shape test moved in one commit; one summary_row test helper; Trash stays five keys.
2. Task 2: render `analysed` on the secondary line from has_analysis; one-cell state slot (select mode's ☑/☐ replaces it).
3. Task 3: keyword-only hits carry the matched keyword as a per-query side channel (not a contract key): one extra SELECT per page, ` · keyword: <term>` on the secondary line, truncated after 10 characters at the 36-cell floor.
4. SDD per task (review + scoped re-review), final whole-branch review + fix round, PR I.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design: backlog/docs/design-library-row-state-markers.md (task-31278, Option A, approved 2026-09-04). Contract: seven keys, validated exact-type at the boundary (SQLite 1/0 narrowed once at the service seam). `has_analysis` comes from SQL — no per-row round trip (AC#3): the EXISTS uses the existing DocumentVersions(media_id, version_number) index, plan captured from the EMITTED query with sqlite_stat1 absent (two SEARCH steps, no SCAN, no TEMP B-TREE), no new index. The brief's SQL missed soft-deleted versions (the Reader filters deleted = 0) — fixed in the task's fix round with fixtures for a soft-deleted newest version and a zero-version row. Rendering: `type · age · analysed` (24 cells fits the 36-cell floor, pinned at the floor via custom widths); keyword-only hits append ` · keyword: <term>` from a per-page mapping (ruling: per-query side channel, not an eighth key). Task 3 rides `MediaBrowseResult.match_reasons` (one SELECT per page over the page's ids with the keyword leg's LIKE escaping) and bakes the suffix into row.secondary at build time — zero canvas/screen/controller changes; the probe re-evaluates the LIKE half of the text branch, so it under-reports at worst, never labels a title/content hit; truncation is unconditional at 10 chars (the secondary is baked once; rider if wide lists should show more). Riders: restore/undo placeholder row carries has_analysis=False until the next fetch (no 'unknown' in the contract); recompose-ratchet count red on dev (66 vs 63; this PR removes one site).
Files: tldw_chatbook/Library/library_media_state.py, tldw_chatbook/DB/Client_Media_DB_v2.py, tldw_chatbook/Media/media_reading_scope_service.py, tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py, tldw_chatbook/Widgets/Library/library_media_canvas.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/library_media_rows.py (new), Tests/DB/test_client_media_pagination.py, Tests/Media/test_media_reading_scope_service.py, Tests/UI/test_library_media_render_fixes.py + the shape/fake files, Docs/User_Guide/library/media-and-conversations.md.
Riders (to be filed in the wave-5 close-out docs PR): restore/undo placeholder row carries has_analysis=False until the next fetch; keyword truncation counts code points not cells; no pin that the analysed/keyword suffix survives a density/select toggle (traced to hold); review-set enumeration loop runs the probe for nothing.
<!-- SECTION:NOTES:END -->
