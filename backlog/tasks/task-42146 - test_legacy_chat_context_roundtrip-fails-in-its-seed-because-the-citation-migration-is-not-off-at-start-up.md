---
id: TASK-42146
title: "test_legacy_chat_context_roundtrip fails in its seed because the citation migration is not off at start-up"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:18
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_legacy_chat_context_roundtrip.py
- Tests/Backup_Recovery/test_complete_roundtrip.py
- .github/workflows/backup-recovery-qualification.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backup stream (round 18). At d85e90bc9e `Tests/Backup_Recovery/test_legacy_chat_context_roundtrip.py::test_native_version1_chat_context_roundtrip_uses_real_message_ids` fails (serial and `-n 4`, through the test wrapper). Its seed child exits 1 (`_run_profile_child`, `Tests/Backup_Recovery/test_complete_roundtrip.py:651`) at line 29 of the generated `seed.py`, `assert migration.writes_enabled is False and migration.ready is False`, right after `TldwCli()` (the `_SEED` script, test file line 51). So at least one of the two is already true for `app.citation_legacy_migration_service`. Not diagnosed: the product may now enable the migration earlier (and the test is stale), or the migration may be wrongly ready. The file is listed in `.github/workflows/backup-recovery-qualification.yml` (lines 55 and 277).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named: which of `writes_enabled` and `ready` is true at that point, what sets it, and since which commit
- [x] #2 The cause is fixed in the product or in the test, as the cause shows, and the node passes
- [x] #3 If the product changed on purpose, the notes say why the test's expectation was stale
<!-- AC:END -->

## Implementation Plan

1. Show which of `writes_enabled` / `ready` is true at the seed (temporary assertion message), and find the commit that changed it.
2. If the product changed on purpose, update the test's stale expectations to the shipped start-up state and keep its legacy-fallback, no-journal and preserved-sidecar checks; otherwise fix the product.
3. Run the node green through the wrapper.

## Implementation Notes

**Cause (AC#1).** At the seed's line 29 `writes_enabled` is `True` and `ready` is `False` (a temporary assertion message printed `('DIAG', True, False)`). `CitationLegacyMigrationService.writes_enabled` reads `CitationTraceRepository.canonical_writes_enabled`, which comes from `get_rag_citation_canonical_writes_enabled()`. Since f7d5b91542 (TASK-33620.8, 2026-10-29) that switch ships on: the default `[rag_citations] canonical_writes_enabled` went from `false` to `true`, and a missing key reads as on. The test's config has no `[rag_citations]` section, so it gets the shipped default. `ready` stays `False` because the same commit defers the fingerprint key to first use (`defer_key=True`), and `legacy_migration_ready` needs that key.

**Stale test, not a product bug (AC#2, AC#3).** The change was on purpose (owner decision 2026-10-27: citations persist by default; boot never touches the keyring). The test was written when writes shipped off, so its three checks (seed, capture, read) now expect the shipped state: writes on, migration not ready. Its real subject still holds: only the app's idle unit migrates a sidecar, and these children never run the app, so the sidecar stays the legacy fallback with no journal. A comment above `_SEED` says so.

**A second rot behind the first.** With the first check fixed, the capture child failed at `preview.complete`: `research.local` (the fixture's explicitly selected `custom/research.db`) was `missing_required`. TASK-21105 opens that store on first use, and an explicitly selected database that is absent is never `unused`, which is fail-closed by design. The seed now initializes the store with `app.local_research_service.list_sessions()==[]`, the same fix TASK-42010 gave the RAG and MCP roundtrips (4d64a8baa2).

**Evidence.** Red at base dc371f0842: `1 failed` at `seed.py` line 29. After the expectation fix only: `1 failed` at the capture child's `missing_required` (research.local). Both fixes: `1 passed in 35.85s` (`-n 4 --timeout=1500`, through the wrapper).

**Files.** `Tests/Backup_Recovery/test_legacy_chat_context_roundtrip.py`.
