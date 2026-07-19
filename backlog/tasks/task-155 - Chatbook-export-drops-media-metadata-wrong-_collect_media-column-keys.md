---
id: TASK-155
title: Chatbook export drops media metadata (wrong _collect_media column keys)
status: Done
assignee: []
created_date: '2026-07-11 22:01'
updated_date: '2026-07-11 23:52'
labels:
  - follow-up
  - chatbooks
  - export
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChatbookCreator._collect_media reads media_type/created_at/updated_at from dict keys that do not exist on the Media row (actual columns: type/ingestion_date/last_modified), plus summary/prompt/media_keywords which live in join/analysis tables. All resolve to None, so media type and timestamps are silently lost across every export round-trip. Pre-existing; became reachable once F4 fixed media export to actually run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exported media entries carry the correct type and timestamps
- [ ] #2 Round-trip test asserts media type survives export+import
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the quick-wins batch (branch claude/followups-quickwins). See Docs/superpowers/plans/2026-07-11-followups-quickwins.md.
<!-- SECTION:NOTES:END -->
