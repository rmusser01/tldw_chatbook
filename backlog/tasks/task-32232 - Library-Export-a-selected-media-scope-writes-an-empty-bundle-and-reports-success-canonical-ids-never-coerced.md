---
id: TASK-32232
title: >-
  Library Export: a selected-media scope writes an empty bundle and reports
  success (canonical ids never coerced)
status: To Do
assignee: []
created_date: '2026-09-10 14:51'
labels:
  - library
  - export
  - bug
  - data-loss
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Select mode carries canonical display ids (`local:media:<n>`); the selected-scope branch of `resolve_export_selections` (`library_export_scope.py`, `if scope.ids: return {...: list(scope.ids)}`) passes them through untouched while the everything scope normalises with `str(int(id))`; `ChatbookCreator._collect_media` then does `int(media_id)` inside a broad `except Exception` that logs `Error collecting media local:media:10: invalid literal for int()` and continues, so the zip holds README + a manifest with `content_items: []` and the UI paints 'Last export: …'. The pinning tests feed bare '1','2','3' and `str(selected_id)`, so they never see the real id shape. Pre-existing since the canonical id landed (2026-08-16); data-loss class for a local-first product. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected-media export from select mode contains every selected item (zip `content/media/*` and manifest `content_items`), pinned by a test that drives the real select-mode id shape
- [ ] #2 Ids are coerced once at the scope seam with the existing backing-id owner (`library_media_state.py` coercion), not per consumer
- [ ] #3 The creator raises (or the run reports failure) when a non-empty selection collects zero items; the UI shows `✗ export produced no content · N items were selected` with Retry
- [ ] #4 The export receipt is read back from the written artifact (`✓ exported · N items · X KB · path`), never from the intent
<!-- AC:END -->
