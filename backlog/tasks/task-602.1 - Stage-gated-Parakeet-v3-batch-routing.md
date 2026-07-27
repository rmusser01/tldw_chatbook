---
id: TASK-602.1
title: Stage gated Parakeet v3 batch routing
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 17:47'
updated_date: '2026-07-27 19:14'
labels:
  - stt
  - onnx
  - ingestion
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
parent_task_id: TASK-602
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the working batch transcription path with explicit Parakeet v3 support and deterministic language routing while preserving the artifact/default-promotion gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explicit parakeet-onnx plus en selects `nemo-parakeet-tdt-0.6b-v2`; explicit supported non-English selects `nemo-parakeet-tdt-0.6b-v3`; both require a user-selected existing local directory with the required filenames; explicit auto, unsupported languages, and translation fail clearly with Retry with faster-whisper guidance.
- [x] #2 Compatible semantic provider=default requests resolve to faster-whisper until the Parakeet promotion gate is enabled, while non-English translation targets fail; the approved en/v2, supported non-English/v3, and auto-or-unsupported/faster-whisper policy is implemented without silently crossing engines.
- [x] #3 Parakeet v3 never receives a decoder language constraint and returns requested_language, effective_language=auto, detected_language=null, and requested_language_not_enforced.
- [x] #4 Batch audio and video preserve the resolved provider, model, language, model directory, and INT8 precision without downloading inside transcription workers.
- [x] #5 Focused routing, service, audio, video, and app-option tests pass; parent TASK-602 remains open for managed executor, provenance, VAD, retry-action, and platform gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a dependency-free, test-first batch STT routing policy that keeps semantic defaults on faster-whisper while the promotion gate is closed and resolves exact Parakeet v2/v3 requests deterministically.
2. Extend the existing Parakeet ONNX service seam for explicit v3 INT8 inference without passing a decoder language constraint, and normalize requested/effective/detected language plus warnings.
3. Resolve audio/video routes once at the app option boundary and preserve provider/model/language/model_dir through workers without downloads.
4. Run focused tests and static checks, update routing documentation, record remaining parent gates, and complete only TASK-602.1.

Detailed plan: Docs/superpowers/plans/2026-07-27-gated-parakeet-v3-batch-routing.md

ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs routing, v3 language transparency, INT8, explicit recovery, and the promotion gate; no new decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added deterministic local INT8 batch routing, explicit Parakeet ONNX v2/v3
  execution, transparent v3 language metadata, and end-to-end audio/video
  option preservation. Updated the Library-facing routing documentation and
  kept compatible semantic `default` requests on faster-whisper while the
  promotion gate is closed.
- Corrected the plan's invalid faster-whisper `en` to `fr` example to a
  translation-to-English request. Runtime validation rejects all non-English
  faster-whisper targets. Also allowed a bounded, non-symlink receipt metadata
  read of at most 64 KiB solely to reject repository and revision metadata that
  identify v2 when v3 is selected. The receipt is not authenticated and does
  not verify file contents or v3 eligibility; no download or graph parsing is
  performed.
- Runtime changes are in the Library capability/app boundary and
  `Local_Ingestion` routing, service, audio, and local-file seams. Focused
  coverage is in the App, Library, Local_Ingestion, Transcription, and Library
  UI and Console dictation test modules. User documentation is in
  `Docs/Features/TRANSCRIPTION.md` and
  `Docs/Features/TRANSCRIPTION_PROVIDERS.md`.
- Fresh post-rebase expanded verification command: 227 passed, 11 skipped, and
  3 warnings.
  Ruff lint passed on all cumulative changed runtime/test files; compileall
  passed on changed runtime modules; routing mypy reported no issues;
  `git diff --check` passed. The known whole-file `app.py` Ruff-format baseline
  was not reformatted or rewritten.
- ADR-025 remains the governing decision. Parent TASK-602 remains open for
  promoted/managed artifact eligibility, the app-owned `LocalSTTExecutor`,
  durable normalized provenance and retry lineage, managed long-form VAD and
  cancellation, the interactive retry action, and Windows/Linux/platform
  matrix evidence. Full managed v3 artifacts and retry UI are not part of this
  child task.
- Current route/model values are selection labels, not attestation of a local
  directory's identity or contents. The Library offers the verified v2
  installer but no supported in-app v3 acquisition; manually selected
  directories receive required-filename checks and the bounded receipt
  metadata mismatch check only.
<!-- SECTION:NOTES:END -->
