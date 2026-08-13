---
id: TASK-2062
title: 'Model browser Phase 3: adopt GGUF models and retire the legacy downloader'
status: In Progress
assignee: []
created_date: '2026-08-03 20:11'
updated_date: '2026-08-13 01:32'
labels:
  - models
  - ui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-03-task-2062-model-browser-phase-3-design.md
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete Model-browser Phase 3 without forcing managed-only model usage: add safe content-addressed local GGUF import, let llama.cpp and llamafile choose managed or external GGUF authority, preserve llamafile embedded mode and unchanged vLLM/MLX behavior, then remove the obsolete unverified Models download surfaces after their replacement is available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can import an unmanaged GGUF as a path-private immutable managed copy with locally recorded integrity while Chatbook never writes renames or deletes the original
- [ ] #2 llama.cpp supports mutually exclusive Managed and External GGUF sources; llamafile supports Embedded Managed and External sources; arbitrary external GGUF files outside the store remain first-class
- [ ] #3 Managed runtime launches retain the exact artifact lease until the exact process is confirmed dead and external launches never mutate the managed store
- [ ] #4 The legacy Widgets/HuggingFace and Transformers direct-write Models download paths are removed only after Import and External source flows are usable
- [ ] #5 vLLM MLX the Hugging Face inference provider existing model IDs external directories and legacy unmanaged discovery remain unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md (amendment)
Reason: TASK-2062 extends the shared artifact service into LLM GGUF ownership and defines the managed-versus-external runtime and process-lease boundary.

1. Complete TASK-2062.1: managed local GGUF import and Installed-view recovery.
2. Complete TASK-2062.2 after 2062.1: managed/external llama.cpp and embedded/managed/external llamafile launch authority.
3. Complete TASK-2062.3 after 2062.1 and 2062.2: remove obsolete direct-write Models downloaders.
4. Run focused TDD and mutation checks in each child; finish with affected regression, native file/process lifecycle, static, privacy, dead-reference, and mounted 80-column UI gates.
5. Close the parent only after all child tasks satisfy their Definition of Done.
<!-- SECTION:PLAN:END -->
