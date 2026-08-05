---
id: TASK-600
title: Persist STT provenance and retry lineage
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-07-28 20:04'
labels:
  - stt
  - database
  - provenance
dependencies:
  - TASK-599
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make normalized transcription provenance and explicit faster-whisper retry history durable across media writes, export and import, sync, API boundaries, and bounded ingest-job retention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned nullable Media transcription provenance document is validated and written atomically with transcript content while transcription_model remains a compatibility summary.
- [x] #2 Provenance records attempt identity, provider, model, root and dependency artifact revisions, precision, requested and effective device, requested, effective, and detected language, task, capabilities, warnings, and retry lineage.
- [x] #3 Library ingest jobs persist retry_of_job_id and structured STT failure provenance without repurposing Transcripts whisper_model or rewriting historical rows.
- [x] #4 A successful retry embeds a bounded sanitized failed-attempt snapshot so lineage remains interpretable after the failed job is pruned or for non-Library callers.
- [x] #5 Export, import, sync, API schemas, and search projections preserve the versioned document; old records remain readable with null provenance.
- [x] #6 Migration rollback and transaction tests prove that parser or writer failure cannot leave transcript content and provenance out of sync.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define one strict, versioned JSON provenance document plus an explicit complete FailedTranscriptionAttempt persistence DTO; validate artifact, execution, language, capability, warning, retry, and stable failure fields without inference.\n2. Add Media schema v5 and validate/serialize provenance before the existing transaction so transcript content, compatibility model, provenance, sync payload, and projections commit together; test parser rejection, writer rollback, migration rollback, and a real sender-to-receiver sync round trip.\n3. Add Library ingest-job schema v5 fields for retry_of_job_id, the job's own structured STT failure, and an immutable retry-source failure snapshot; preserve them through restore, failure, requeue, pruning, and non-Library retry construction without overwriting ancestor context.\n4. Preserve the document through Chatbook export/import, local/API detail, job projections, and search results while keeping old null rows readable.\n5. Prove migrations, rollback, round trips, bounded sanitization, and retry lineage with focused tests, then run broader affected suites and self-review.\n\nADR required: yes\nADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md\nReason: ADR-025 already governs the persisted STT provenance and retry-lineage boundary; this task implements that accepted decision, so no new ADR is needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a strict version-1 STT provenance document and complete sanitized failed-attempt DTO, including bounded canonical JSON, artifact identity validation, retry cross-checks, and local-path exclusion. Added nullable Media schema v5 persistence plus atomic create/overwrite/same-content writes, sync validation, search/local/API projections, and Chatbook export/import round trips while preserving `transcription_model` as the compatibility summary.

Added Library ingest-job schema v5 retry navigation, own-failure provenance, and first-write-preserved retry-source snapshots. Registry boundaries defensively copy the snapshots so callers cannot mutate stored lineage; retry success remains interpretable after the failed job is pruned and for callers without Library job IDs. Both schema migrations and transcript/provenance writes have fault-injection rollback coverage.

PR review follow-up moved ingest-job upserts onto the shared transaction context and made retry creation a single durable source-plus-retry transaction. A failed pair write now leaves the source visible and retryable, exposes no in-memory retry, and does not consume a job id. Public provenance DTO/build/load/dump APIs now carry complete Google-style documentation. The dependency-free strict validator and each database class's existing self-versioned inline migration convention remain intentional; production audio/video coordinator provenance and failure capture remain owned by TASK-602 AC6.

ADR check: existing [ADR-025](../decisions/025-shared-stt-artifacts-and-runtime-routing.md) applies; no new ADR was required. Post-review verification against the latest `dev`: 841 affected tests passed and 6 environment-dependent sync integration cases skipped; Ruff, focused mypy, compileall, and `git diff --check` passed. Repository-wide collection remains blocked on current `dev` by two unrelated existing imports: removed `StreamDone` in `Tests/Event_Handlers/test_worker_events_contract.py` and removed `TabState` in `Tests/UI/test_chat_shell_bar.py` (21,960 tests otherwise collected). Task status remains In Progress until that mandatory repository-wide gate is green.
<!-- SECTION:NOTES:END -->
