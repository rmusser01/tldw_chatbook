---
id: TASK-20014
title: Remove retired Library diagnostic from reviewed metadata ledger
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 20:15'
updated_date: '2026-08-23 20:24'
labels:
  - testing
  - architecture
  - diagnostics
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the persistent-diagnostic architecture suite after the lasting Notes sync cutover removed a reviewed diagnostic but left its test ledger entry stale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reviewed metadata-only ledger matches the current Library diagnostics.
- [x] #2 The retired diagnostic is removed only after source history proves the production path was intentionally deleted.
- [x] #3 The full persistent-diagnostic architecture module passes.
- [x] #4 No production diagnostic or privacy allowlist is changed.
- [x] #5 Static checks and task hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the architecture ledger failure and trace the retired Library diagnostic through source history.
2. Prove the lasting Notes sync cutover intentionally removed the production path and that no replacement diagnostic needs review.
3. Remove only the stale test-ledger entry; do not change production diagnostics or privacy allowlists.
4. Run the full persistent-diagnostic architecture module and related inventory/privacy checks.
5. Complete independent review and task hygiene.

ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: existing ADR-029 governs the metadata-only diagnostic review ledger; this is test-ledger reconciliation with no new decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed only the stale `Failed to persist a Library notes setting` key from the reviewed metadata-only test ledger. RED evidence was the full architecture module at 1 failed / 64 passed, with exactly that missing label. Source history at `d9722bbf3` proved the lasting Notes sync cutover intentionally deleted `_save_library_notes_sync_setting`, its five legacy config-persistence callers, and the config-backed legacy sync panel; current source has no equivalent `save_setting_to_cli_config("notes", ...)` path or replacement diagnostic requiring this ledger. The standalone inventory checker remained exact at 520 owners / 1226 TASK-492 calls / 7190 TASK-494 calls / 8 sink files, proving no production diagnostic or sink change. Verification passed: full architecture module 65 tests; persistent diagnostic/privacy matrix 50 tests; summarization diagnostic privacy 257 tests; focused ledger and generated-inventory nodes; Ruff lint, Python compilation, diff hygiene, and task-ID checks. Whole-file Ruff format reports two unrelated pre-existing expressions that it would reflow identically at HEAD; they were intentionally left untouched to preserve the one-line test-only repair. No production file, generated inventory, review fixture, privacy allowlist, dependency, or license changed. ADR required: no; existing `backlog/decisions/029-local-private-data-boundary.md` governs this reconciliation. No lesson was added because TASK-19642.20 and the existing diagnostic-inventory lessons already record the incident and review workflow.
<!-- SECTION:NOTES:END -->
