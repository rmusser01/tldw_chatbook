---
id: TASK-32232
title: >-
  Library Export: a selected-media scope writes an empty bundle and reports
  success (canonical ids never coerced)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 14:51'
updated_date: '2026-09-10 15:03'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: roundtrip test drives a selected-media scope with canonical local:media:<n> ids through resolve_export_selections -> real LocalChatbookService -> zip; assert content/media/*.txt + manifest content_items carry both items (fails today with 0).
2. GREEN: coerce ids once in resolve_export_selections' scope.ids branch for kind='media' via the existing owner library_media_int_backing_id (unparseable ids pass through untouched so the creator's guard still sees them).
3. Creator guard: ChatbookCreator.create_chatbook raises ChatbookExportEmptyError when a non-empty collectable selection yields zero manifest.content_items; returns (False, message, dependency_info with empty_export_requested=N). Library export controller renders '✗ export produced no content · N items were selected' in the error line (Export button stays enabled = Retry).
4. Receipt readback: _run_library_export_via_service reopens the written zip, counts manifest content_items and stats size; the receipt renders '✓ exported · N items · X KB · <path>' from those artifact facts (old 'Last export: …' line kept only as the no-facts fallback for restored sessions).
5. Docs: Docs/User_Guide/library/import-and-export.md Export section + Verified-against stamp.
6. Live-verify in tmux on a seeded scratch profile: select 2 media rows, export, unzip -l the artifact, read the receipt.
7. Run the covering test files.
<!-- SECTION:PLAN:END -->
