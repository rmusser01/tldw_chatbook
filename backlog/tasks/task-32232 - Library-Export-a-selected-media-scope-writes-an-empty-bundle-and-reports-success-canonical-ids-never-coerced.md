---
id: TASK-32232
title: >-
  Library Export: a selected-media scope writes an empty bundle and reports
  success (canonical ids never coerced)
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 14:51'
updated_date: '2026-09-10 15:28'
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
- [x] #1 A selected-media export from select mode contains every selected item (zip `content/media/*` and manifest `content_items`), pinned by a test that drives the real select-mode id shape
- [x] #2 Ids are coerced once at the scope seam with the existing backing-id owner (`library_media_state.py` coercion), not per consumer
- [x] #3 The creator raises (or the run reports failure) when a non-empty selection collects zero items; the UI shows `✗ export produced no content · N items were selected` with Retry
- [x] #4 The export receipt is read back from the written artifact (`✓ exported · N items · X KB · path`), never from the intent
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selected-media exports now contain what was selected, and the run can no
longer report success over an empty bundle.

**Root cause (AC#1/#2).** `resolve_export_selections`' explicit-ids branch
passed select mode's canonical `local:media:<n>` display ids straight
through, while the whole-source branch normalised with `str(int(...))`;
`ChatbookCreator._collect_media` then failed `int(media_id)` per item
inside a broad `except`, so the zip held README + `content_items: []`.
Fixed at the one seam, through the existing backing-id owner
(`library_media_int_backing_id`) — no second parser. An id carrying no
backing id is passed through UNCHANGED rather than dropped: dropping it
would shrink the selection silently, which is the failure mode being
fixed; keeping it lets the guard below report honestly.

**Honest failure (AC#3).** The guard belongs in the creator, not the
Library runner — every caller (Library, the creation wizard) routes
through `create_chatbook`, and every collector logs-and-continues per
item, so any of them could collect nothing. `create_chatbook` now raises
`ChatbookExportEmptyError` when a non-empty *collectable* selection yields
zero `manifest.content_items`, before any archive is written, and returns
the selection size in `dependency_info["empty_export_requested"]` so the
canvas can render `✗ export produced no content · N items were selected`
without parsing a message. A PARTIAL collection still succeeds. The
submit button relabels itself "Retry export" while a failure line shows.

**Receipt from the artifact (AC#4).** After a successful write the run
reopens the zip, counts its manifest's `content_items` and stats its size;
the receipt renders `✓ exported · N items · X KB · <path>` from those
facts. An unreadable manifest degrades to the old path-only line rather
than flipping a real export into a failure.

**Fallout worth knowing:** three existing tests patched collectors to bare
no-ops (they collect nothing by construction) and one asserted the old
"still succeeds with no conversations" behaviour — that assertion WAS the
bug. All four updated.

Live-verified on a seeded scratch profile (235x52): two media rows
selected in select mode, exported; zip holds 6 entries including
`content/media/media_10.txt` and `media_11.txt`, manifest `content_items`
names both titles, and the canvas reads
`✓ exported · 2 items · 4 KB · …/exports/p0.zip`.

Files: `Library/library_export_scope.py`, `Library/library_export_state.py`,
`Chatbooks/chatbook_creator.py`,
`UI/Library_Modules/library_export_controller.py`,
`UI/Library_Modules/library_export_state.py`, `UI/Screens/library_screen.py`,
`Widgets/Library/library_export_canvas.py`,
`Docs/User_Guide/library/import-and-export.md`, plus the tests above.
<!-- SECTION:NOTES:END -->
