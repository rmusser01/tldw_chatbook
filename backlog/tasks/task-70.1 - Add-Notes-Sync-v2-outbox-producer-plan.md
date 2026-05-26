---
id: TASK-70.1
title: Add Notes Sync v2 outbox producer plan
status: Done
labels:
- sync
- sync-v2
- notes
- local-first
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
modified_files:
- tldw_chatbook/Sync_Interop/notes_outbox_producer.py
- tldw_chatbook/Sync_Interop/envelope_builder.py
- tldw_chatbook/Sync_Interop/__init__.py
- tldw_chatbook/Notes/notes_scope_service.py
- Tests/Sync_Interop/test_notes_outbox_producer.py
- Tests/Notes/test_notes_scope_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first content-producing Sync v2 path for Notes so local note create, update, and delete operations can enqueue durable encrypted outbox envelopes for manual sync without requiring server availability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes create, update, and delete operations enqueue Sync v2 outbox envelopes for the active server profile personal dataset.
- [x] #2 Local Notes operations remain successful and recoverable when the server is unavailable.
- [x] #3 Tests cover envelope identity, domain scope, idempotency, encryption boundary, and pending profile summary counts.
- [x] #4 No background sync, automatic push, or broad domain sync is introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a pure Notes Sync v2 outbox producer and NotesScopeService post-success enqueue hooks.
2. Add a note delete envelope builder while preserving existing encrypted note upsert behavior.
3. Implement a small NotesSyncV2OutboxProducer that reads the configured local-first profile, requires a dataset key, builds Notes envelopes, and persists them to the durable outbox.
4. Wire local Notes create/update/delete paths to the producer only when an explicit Sync v2 profile context and producer are injected.
5. Run focused Sync/Notes verification and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `NotesSyncV2OutboxProducer` as the first content-producing Notes Sync v2 boundary. The producer reads the active local-first profile from `SyncStateRepository`, requires device/dataset identity and an in-memory dataset key, encrypts note title/body upserts, builds clear delete tombstone envelopes, and persists them through the existing durable outbox API. Non-local-first, unconfigured, or missing-key profiles return explicit skipped states and do not dispatch remote sync.

Extended `SyncEnvelopeBuilder` with `build_note_delete()` and wired `NotesScopeService` local save/delete paths to call an injected producer only after successful local mutations. The hook is opt-in, preserves existing local note behavior by default, and does not add background sync, automatic push/pull, or broad domain sync.

PR review follow-up isolated best-effort enqueue failures from primary local note operations, validated Sync v2 profile scope strings through centralized text validation/sanitization utilities, propagated note keywords to outbox `tag_ids`, and expanded public producer docstrings to Google-style `Args`/`Returns` coverage.

Verification:
- Red tests failed on the missing producer module and missing `NotesScopeService` producer injection.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_notes_outbox_producer.py Tests/Notes/test_notes_scope_service.py::test_scope_service_enqueues_local_note_upsert_after_successful_save Tests/Notes/test_notes_scope_service.py::test_scope_service_does_not_enqueue_local_note_upsert_after_failed_save Tests/Notes/test_notes_scope_service.py::test_scope_service_does_not_enqueue_local_note_upsert_after_failed_create Tests/Notes/test_notes_scope_service.py::test_scope_service_ignores_incomplete_sync_v2_profile_after_local_save Tests/Notes/test_notes_scope_service.py::test_scope_service_enqueues_local_note_delete_after_successful_delete --tb=short` passed with 8 tests.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_notes_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_sync_state_repository.py Tests/Notes/test_notes_scope_service.py --tb=short` passed with 56 tests.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_restore_service.py --tb=short` passed with 42 tests.
- `../../.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique --tb=short` passed with 1 test.
- `../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/notes_outbox_producer.py tldw_chatbook/Sync_Interop/envelope_builder.py tldw_chatbook/Notes/notes_scope_service.py` passed.
- `../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_scope_service.py::test_scope_service_local_save_survives_sync_v2_enqueue_failure Tests/Notes/test_notes_scope_service.py::test_scope_service_passes_keywords_to_sync_v2_note_upsert Tests/Notes/test_notes_scope_service.py::test_scope_service_ignores_invalid_sync_v2_profile_after_local_save Tests/Notes/test_notes_scope_service.py::test_scope_service_local_delete_survives_sync_v2_enqueue_failure --tb=short` passed with 4 tests.
- `../../.venv/bin/python -m pytest -q Tests/Sync_Interop/test_notes_outbox_producer.py Tests/Notes/test_notes_scope_service.py --tb=short` passed with 33 tests.
- `git diff --check` passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
`TASK-70.1` adds the first Notes content-producing Sync v2 path. Local Notes create/update/delete can now produce durable pending outbox envelopes when an active local-first profile and dataset key are explicitly provided, while all remote dispatch and background sync remain out of scope.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 No background sync or automatic dispatch added
<!-- DOD:END -->
