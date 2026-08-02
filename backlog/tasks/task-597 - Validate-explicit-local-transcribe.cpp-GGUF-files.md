---
id: TASK-597
title: Validate explicit local transcribe.cpp GGUF files
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-02 14:55'
labels:
  - stt
  - gguf
  - validation
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
parent_task_id: TASK-596
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate one user-selected local GGUF for direct use by the optional transcribe.cpp provider without copying, registering, or activating it in the managed artifact store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An explicit `.gguf` selection is validated through the project path boundary, rejects symlinks and irregular files, and is opened with identity checks without invoking a native runtime.
- [ ] #2 A bounded standard-library GGUF v3 parser validates magic, version, typed metadata, tensor information, alignment, offsets, and all approved resource limits without reading tensor payload.
- [ ] #3 Admission accepts exactly the pinned transcribe.cpp 0.1.3 architecture declaration and five wheel-supported platform pairs, rejecting near misses and unsupported systems before provider load.
- [ ] #4 A successful result contains only the explicit path, bounded metadata, source identity snapshot, and normalized platform pair; it creates no descriptor, copy, hash, stage, install, activation, or routing state.
- [ ] #5 Missing, malformed, incompatible, or replaced-during-admission files fail with typed path-safe errors suitable for **Choose another GGUF…** recovery.
- [ ] #6 Focused tests cover valid admission, parser bounds and truncation, symlinks, irregular files, path replacement, identity checks, platform admission, and import boundaries excluding artifact-store, native-runtime, and UI dependencies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Revised design: Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md

ADR required: yes
ADR path: backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md
Reason: direct local paths now precede managed GGUF acquisition.

1. Retain the reviewed bounded GGUF parser and pinned runtime/platform admission.
2. Remove descriptor and managed-store preparation from TASK-597.
3. Add safe direct-file validation and a compact admission result.
4. Verify parser, path-boundary, platform, and dependency-import tests.
5. Complete review and hand TASK-604 the validated-path API.

Detailed plan will replace the superseded store-first plan after revised spec review.
<!-- SECTION:PLAN:END -->
