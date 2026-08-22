---
id: TASK-19902
title: >-
  Harden the auto re-chunk re-stamp: one outer transaction over row replacement
  + config re-stamp
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22 15:59'
labels:
  - chunking
dependencies:
  - TASK-19901
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Outcome: wrap `_replace_chunk_rows` and `_restamp_auto_chunking_config` in one outer `media_db.transaction()` on the auto re-chunk path in `tldw_chatbook/Library/library_rechunk_service.py` (currently two independent transactions at `:252`/`:225`, called back-to-back at `:573`/`:600`). The DB's `transaction()` already supports nesting — only the outermost commit/rollback matters (`tldw_chatbook/DB/Client_Media_DB_v2.py:1100-1119`) — so an outer wrap is the whole fix.

This also fixes the accepted sub-effect that a re-stamp raise after row replacement counts the item failed (`summary["failed"] += 1` via the per-item handler) while its rows were already replaced.

Filed from the final review of the Chunking Auto-Selection sub-project #3 (TASK-19901; Task-5 review Minor: "two-transaction re-stamp window … nested-outer-transaction hardening noted as trivially closable follow-up").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A crash or raise between row replacement and config re-stamp leaves no stale `template` key: both writes commit together or neither does (one outer `media_db.transaction()` around `_replace_chunk_rows` + `_restamp_auto_chunking_config`, nesting verified against `Client_Media_DB_v2.transaction()`'s outermost-commit semantics)
- [x] #2 A re-stamp failure after row replacement no longer miscounts: the item either fails atomically (rows rolled back, counted failed) or lands fully (rows + re-stamp, counted rechunked) — never rows-replaced-but-counted-failed
- [x] #3 The existing re-stamp tests stay green: the three flip tests in `Tests/Library/test_library_rechunk_service.py` (template→plan without a template key, template→plain, template-still-wins restamps the new winner)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Wrap the `_replace_chunk_rows` + `_restamp_auto_chunking_config` call pair in `rechunk_legacy_items` in one outer `with media_db.transaction():` (nesting verified: only the outermost commit/rollback matters).
2. Keep the per-item try/except semantics: a raise inside now rolls BOTH writes back together and the item is still counted failed with no partial state.
3. Add a red-first test forcing the re-stamp's UPDATE to raise after row replacement; assert the old rows AND the old config both survive and the item is counted failed (never rows-replaced-but-counted-failed).
4. Run the four targeted suites (`Tests/Chunking/test_auto_selection.py`, `Tests/Chunking/test_chunking_interop_v7.py`, `Tests/Library/test_library_rechunk_service.py`, `Tests/UI/test_library_ingest_template_picker.py`) plus the neighboring suites touching the modified modules.

ADR required: no
ADR path: N/A
Reason: mechanical atomicity hardening inside one module's existing transaction pattern — no schema, contract, or boundary change; the DB's nesting semantics were already an ADR-level given.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: exactly the filed outcome — ONE outer `with media_db.transaction():` in `rechunk_legacy_items` (`tldw_chatbook/Library/library_rechunk_service.py`) now wraps both the chunk-row replacement and the auto re-stamp. `MediaDatabase.transaction()` nests (`Client_Media_DB_v2.py` — outermost-commit semantics), so each helper's own `transaction()` becomes a nested no-commit scope on this path and the pair commits or rolls back together. The per-item try/except is untouched: a re-stamp raise now rolls the replacement back too, and the item is still counted `failed` — closing the accepted sub-effect (replaced-rows-without-config is no longer a reachable state). `summary["rechunked"] += 1` sits after the outer block, so it fires only when both writes landed.
- Features implemented or modified: `tldw_chatbook/Library/library_rechunk_service.py` (outer transaction + comment/docstring notes); `Tests/Library/test_library_rechunk_service.py` (new `test_rechunk_restamp_failure_rolls_back_row_replacement`: monkeypatched re-stamp whose UPDATE raises after replacement — asserts old rows intact, old config intact, `failed == 1`, `rechunked == 0`; verified red before the fix, green after).
- Technical decisions and trade-offs: no new transaction helper or refactor — the minimal outer wrap the task filed; the helpers stay independently transactional for any other caller.
- Evidence: the three existing flip tests stay green; targeted suites `Tests/Chunking/test_auto_selection.py`, `Tests/Chunking/test_chunking_interop_v7.py`, `Tests/Library/test_library_rechunk_service.py`, `Tests/UI/test_library_ingest_template_picker.py` — 75 passed; neighboring suites for the modified modules (Local_Ingestion template trio, UI ingest canvas/rechunk action, RAG_Admin, integration ingest flow, template_runtime) — 589 + 30 + 13 passed, 1 skipped (embeddings_rag deps not installed).
- Delivered in commit `fix(chunking): address Qodo review — atomic rechunk restamp (TASK-19902), case-insensitive auto reservation` on `feat/chunking-auto-selection` (PR #1952), together with the sibling Qodo #4 case-insensitive Auto-name reservation fix.
- Modified or added files: `tldw_chatbook/Library/library_rechunk_service.py`, `Tests/Library/test_library_rechunk_service.py` (this task); `tldw_chatbook/Chunking/chunking_interop_library.py`, `tldw_chatbook/Chunking/auto_selection.py`, `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` and their tests (sibling Qodo #4 fix, same commit).
<!-- SECTION:NOTES:END -->
