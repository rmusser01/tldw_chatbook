---
id: TASK-513
title: Persist STT provenance and retry lineage
status: To Do
assignee: []
created_date: '2026-07-24 01:03'
labels:
  - stt
  - database
  - provenance
dependencies:
  - TASK-512
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
- [ ] #1 A versioned nullable Media transcription provenance document is validated and written atomically with transcript content while transcription_model remains a compatibility summary.
- [ ] #2 Provenance records attempt identity, provider, model, root and dependency artifact revisions, precision, requested and effective device, requested, effective, and detected language, task, capabilities, warnings, and retry lineage.
- [ ] #3 Library ingest jobs persist retry_of_job_id and structured STT failure provenance without repurposing Transcripts whisper_model or rewriting historical rows.
- [ ] #4 A successful retry embeds a bounded sanitized failed-attempt snapshot so lineage remains interpretable after the failed job is pruned or for non-Library callers.
- [ ] #5 Export, import, sync, API schemas, and search projections preserve the versioned document; old records remain readable with null provenance.
- [ ] #6 Migration rollback and transaction tests prove that parser or writer failure cannot leave transcript content and provenance out of sync.
<!-- AC:END -->
