---
id: TASK-19902
title: 'Harden the auto re-chunk re-stamp: one outer transaction over row replacement + config re-stamp'
status: To Do
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - chunking
dependencies: [TASK-19901]
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
- [ ] #1 A crash or raise between row replacement and config re-stamp leaves no stale `template` key: both writes commit together or neither does (one outer `media_db.transaction()` around `_replace_chunk_rows` + `_restamp_auto_chunking_config`, nesting verified against `Client_Media_DB_v2.transaction()`'s outermost-commit semantics)
- [ ] #2 A re-stamp failure after row replacement no longer miscounts: the item either fails atomically (rows rolled back, counted failed) or lands fully (rows + re-stamp, counted rechunked) — never rows-replaced-but-counted-failed
- [ ] #3 The existing re-stamp tests stay green: the three flip tests in `Tests/Library/test_library_rechunk_service.py` (template→plan without a template key, template→plain, template-still-wins restamps the new winner)
<!-- AC:END -->
