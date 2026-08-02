---
id: TASK-597
title: Validate explicit local transcribe.cpp GGUF files
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-02 16:28'
labels:
  - stt
  - gguf
  - validation
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
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
- [ ] #3 Admission accepts exactly the pinned transcribe.cpp 0.1.3 architecture declaration and five released OS/CPU wheel-candidate pairs, rejecting near misses while leaving final wheel/ABI availability to the provider probe.
- [ ] #4 A successful result contains only the explicit non-repr path, bounded metadata, source identity snapshot, and normalized platform pair; it creates no descriptor, copy, hash, stage, install, activation, or routing state.
- [ ] #5 Missing, malformed, incompatible, or replaced-during-admission files fail with typed path-safe errors suitable for **Choose another GGUF…** recovery.
- [ ] #6 Focused tests cover valid admission, parser bounds and truncation, symlinks, irregular files, path replacement, identity checks, platform-candidate admission, path-safe representations/errors, and import boundaries excluding artifact-store, native-runtime, and UI dependencies.
- [ ] #7 The existing managed-import descriptor prototype is preserved in a private module explicitly deferred to TASK-1915, with an empty public surface and no export, registration, production import, call site, or active TASK-597 behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-08-01-task-597-local-gguf-import.md

ADR required: yes
ADR path: backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
Reason: ADR-041 accepts direct local GGUF paths before managed acquisition while preserving the prior store-facing prototype as private deferred TASK-1915 reference code.

1. Keep the renamed active gguf_admission.py parser/compatibility module store-free, recover the prior descriptor prototype from fd9956903^ into private _deferred_gguf_managed_import.py, and prove it is unexported and unreachable.
2. Add one no-follow, same-handle validate_local_gguf boundary with typed path-safe errors, source identity evidence, and normalized wheel-candidate platform output.
3. Verify focused and full tests, static checks, active/deferred scope scans, and independent code review before Backlog closeout.
<!-- SECTION:PLAN:END -->
