# TASK-600: STT Provenance and Retry Lineage Implementation Plan

> **For Codex:** Execute this plan test-first in the isolated
> `codex/task-600-stt-provenance` worktree. Keep the persistence format small
> and reject unknown or malformed data at write/import boundaries.

**Goal:** Persist normalized STT provenance and retry history atomically with
media transcripts, and preserve it through existing job, sync, search, API,
and Chatbook boundaries.

**Architecture:** Add one dependency-light STT persistence module that
validates a version-1 JSON document and sanitized failed-attempt snapshots.
The module owns one explicit `FailedTranscriptionAttempt` persistence DTO
because the runtime `TranscriptionFailure` intentionally lacks dependency and
language context; callers must supply the complete resolved attempt rather
than having persistence synthesize missing fields.
Store the canonical compact JSON in a nullable Media column and pass it through
the existing transaction and sync payload. Add nullable ingest-job columns for
navigation (`retry_of_job_id`), the job's own structured failure, and the
failed-attempt snapshot carried into a retry. Existing rows remain valid with
null values.

**Tech stack:** Python 3.11, dataclasses/stdlib JSON, SQLite migrations,
Pydantic API schemas, pytest.

**ADR required:** yes
**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Reason:** ADR-025 already fixes the persistence and retry-lineage boundary;
TASK-600 implements that accepted decision without introducing a new
architecture choice.

---

## Task 1: Define the persisted provenance contract

**Files:**

- Create: `tldw_chatbook/STT/persistence.py`
- Modify: `tldw_chatbook/STT/__init__.py`
- Create: `Tests/STT/test_persistence.py`

1. Write failing tests for a complete version-1 document built from
   `TranscriptionResult`, including artifact dependencies, capabilities,
   warnings, and optional retry lineage.
   Define and test `FailedTranscriptionAttempt` with attempt/job identity,
   provider/model, exact root and dependency artifact identities, precision,
   requested/effective device, requested/effective/detected language, task,
   and stable error code. It is the validated join point for a runtime failure
   plus its resolved request/artifact context; no field is inferred.
2. Write failing tests that reject unknown versions, unknown fields, malformed
   artifact identities, oversized documents, raw exception/path fields, and
   non-JSON values. Test every failed-snapshot field and nullable job IDs.
3. Run `pytest Tests/STT/test_persistence.py` and confirm the intended failures.
4. Implement the smallest strict validator/serializer/deserializer and builder
   needed by the tests. Keep the failed-attempt snapshot fixed-shape,
   sanitized, and size-bounded.
5. Re-run the focused test until green.

## Task 2: Persist provenance atomically with Media

**Files:**

- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/DB/sql_validation.py`
- Modify as needed: `tldw_chatbook/DB/Sync_Client.py`
- Modify: `Tests/Media_DB/test_media_db_v2.py`
- Modify or create focused migration tests under `Tests/Media_DB/`
- Modify: focused sync tests under `Tests/DB/`

1. Write failing tests for schema v4-to-v5 migration, old null rows, valid
   create/overwrite round trips, search projection, sync-payload preservation,
   parser rejection before any row mutation, and writer rollback after the
   content update starts.
   Add a fault-injection migration rollback test proving both the schema
   version and table/data remain at v4 when migration fails. Add a real
   sender-payload to receiver-apply/read sync round trip, not only a payload
   unit assertion.
2. Confirm the focused tests fail for missing schema/write support.
3. Add nullable `Media.transcription_provenance_json`, schema version 5, and
   the SQL identifier allowlist entry.
4. Validate and serialize provenance before entering
   `add_media_with_keywords`' existing transaction. Include the canonical JSON
   in insert, overwrite, same-content metadata updates, and `_media_payload`;
   leave `transcription_model` as the compatibility summary.
5. Re-run the focused Media DB tests until green.

## Task 3: Persist ingest-job failure context and retry navigation

**Files:**

- Modify: `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Modify: `Tests/DB/test_library_ingest_jobs_db.py`
- Modify: `Tests/Library/test_library_ingest_jobs.py`
- Modify: `Tests/Library/test_library_ingest_jobs_restore.py`

1. Write failing tests for schema v4-to-v5 migration, null compatibility,
   structured failure-provenance persistence/restore, and requeue setting
   `retry_of_job_id` while carrying the sanitized failed snapshot. Add a
   fault-injection migration rollback test proving schema version, table, and
   existing rows remain at v4.
2. Confirm the tests fail before implementation.
3. Add nullable `retry_of_job_id`, `stt_failure_provenance_json`, and
   `retry_source_failure_provenance_json` columns and thread their values
   through the dataclass, DB upsert, restore, `mark_failed`, and `requeue`.
   The first JSON field describes that job's own failed attempt; requeue copies
   it into the second immutable field on the new job. A later `mark_failed`
   records the retry's own failure without overwriting its retry source. Do not
   add an arbitrary history collection.
4. Validate structured failure data with the STT persistence helper. Never put
   STT provenance into `Transcripts.whisper_model`, and do not rewrite prior
   jobs.
5. Test that successful retry provenance remains interpretable after the
   failed job row is deleted/pruned, and that a non-Library retry works with
   nullable job IDs.
6. Re-run the focused job tests until green.

## Task 4: Preserve provenance at external boundaries

**Files:**

- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Modify: `tldw_chatbook/tldw_api/media_reading_schemas.py`
- Modify as needed: `tldw_chatbook/tldw_api/schemas.py`
- Modify: focused tests in `Tests/Chatbooks/`, `Tests/Media/`, and
  `Tests/tldw_api/`

1. Write failing round-trip tests proving Chatbook export/import preserves the
   structured document and rejects malformed imported provenance without
   creating or partially updating a Media row.
2. Write failing local-service/API schema tests for structured provenance in
   detail/search results and retry/failure fields in ingest-job projections.
3. Confirm failures, then add only the necessary mapping/schema fields.
4. Decode canonical database JSON at the local read boundary; do not expose
   storage-format details as a second public field.
5. Re-run each affected focused suite until green.

## Task 5: Verification and handoff

**Files:**

- Modify: `backlog/tasks/task-600 - Persist-STT-provenance-and-retry-lineage.md`

1. Run the affected STT, Media DB, ingest-job, Chatbook, local Media, API
   schema, and sync tests.
2. Run `git diff --check` and inspect the full diff for schema/migration,
   rollback, privacy, and backward-compatibility mistakes.
3. Add concise implementation notes and check acceptance criteria only after
   the evidence supports them. Record any repository-wide verification blocker
   rather than weakening the task's gate.
4. Request code review before merging or marking the task Done.
