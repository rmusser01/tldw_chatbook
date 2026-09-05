---
id: TASK-31555
title: Library Notes test fake lacks the tree filter seam
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:04'
updated_date: '2026-09-05 01:24'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore current Library Notes filter coverage after the shared UI fake fell behind the production search_note_tree_placements contract, causing submitted filters to return silently without recomposing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared Library Notes fake implements the exact placement-page filter seam and records calls.
- [x] #2 Filter results include matching note records with correct paging metadata.
- [x] #3 Library Notes capability and filter tests pass against the current tree UI.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare the shared Notes fake with the production tree-filter service signature and return model.
2. Implement case-insensitive title/content matching plus offset/limit placement paging while preserving call evidence.
3. Update the stale destination inventory assertion for the already-shipped Collections reader and run the focused Notes filter/capability modules.

ADR required: no
ADR path: N/A
Reason: This updates test doubles and assertions to existing Library contracts; it changes no production boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the production-compatible `search_note_tree_placements` fake seam with case-insensitive title/body matching, offset/limit paging, placement records, paging metadata, and call evidence.
- Refreshed the already-shipped Collections inventory expectation and current placement mutation identity payloads exposed by the capability test.
- Evidence: the three exact capability/filter CI regressions pass together. One separate wide-editor deep-link test still times out before the Notes filter path and remains a residual, not a skipped assertion.
- ADR required: no; only test doubles and current contract assertions changed.
<!-- SECTION:NOTES:END -->
