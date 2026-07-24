---
id: TASK-518
title: Promote Parakeet ONNX defaults and remove legacy providers
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - migration
  - release
dependencies:
  - TASK-509
  - TASK-510
  - TASK-511
  - TASK-515
  - TASK-516
  - TASK-517
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the gated migration to Parakeet ONNX by switching persisted and semantic defaults, removing overlapping NeMo and MLX Parakeet implementations, and retaining safe recovery paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Default promotion occurs only when every approved platform, artifact, quality, throughput, memory, buffer, cancellation, crash, migration, privacy, and package gate has passed for the same release candidate.
- [ ] #2 Versioned idempotent configuration migration maps persisted parakeet and parakeet-mlx selections to semantic default, preserves explicit languages and saved auto, and resolves missing language to en with backup and one-time notice.
- [ ] #3 NeMo Parakeet and parakeet-mlx provider code, registrations, provider-specific settings, privacy entries, dependency declarations, and live tests are removed while unrelated NeMo capabilities remain.
- [ ] #4 Historical transcript provenance is unchanged; new explicit API or CLI use of removed provider IDs fails with ProviderRemoved guidance rather than being silently rewritten.
- [ ] #5 External Hugging Face, NeMo, and MLX caches are never deleted automatically and rollback retains the prior default and providers until gates pass.
- [ ] #6 No live UI or service code retains old provider IDs outside migration fixtures, and release documentation accurately describes en, v2, v3, auto, faster-whisper retry, INT8, F32, and optional transcribe.cpp behavior.
<!-- AC:END -->
