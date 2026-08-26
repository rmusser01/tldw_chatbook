---
id: TASK-2062
title: 'Model browser Phase 3: adopt GGUF models and retire the legacy downloader'
status: Done
assignee: []
created_date: '2026-08-03 20:11'
updated_date: '2026-08-14 05:33'
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
- [x] #1 A user can import an unmanaged GGUF as a path-private immutable managed copy with locally recorded integrity while Chatbook never writes renames or deletes the original
- [x] #2 llama.cpp supports mutually exclusive Managed and External GGUF sources; llamafile supports Embedded Managed and External sources; arbitrary external GGUF files outside the store remain first-class
- [x] #3 Managed runtime launches retain the exact artifact lease until the exact process is confirmed dead and external launches never mutate the managed store
- [x] #4 The legacy Widgets/HuggingFace and Transformers direct-write Models download paths are removed only after Import and External source flows are usable
- [x] #5 vLLM MLX the Hugging Face inference provider existing model IDs external directories and legacy unmanaged discovery remain unchanged
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Model-browser Phase 3 through TASK-2062.1, TASK-2062.2, and TASK-2062.3, merged in PRs #1578, #1610, and #1628. Added path-private content-addressed local GGUF import that never mutates the source; added exact managed-artifact and arbitrary external-GGUF runtime authority for llama.cpp and llamafile while preserving embedded llamafile; retained managed leases until proven process death; and removed the legacy Widgets/HuggingFace browser plus Transformers direct-write downloader only after the replacement flows were usable. Preserved vLLM, MLX, Hugging Face inference and provider caching, Remote acquisition, external directories, and legacy unmanaged discovery. Key implementation areas were Model_Artifacts admission/service, Installed and Models UI, GGUF source and server lifecycle ownership, and removal of obsolete downloader owners. Verification across the children included focused and mutation-tested suites, mounted 80-column production-CSS behavior, static/privacy/dead-reference checks, governed diagnostic inventory, and exact Linux/macOS/Windows evidence; all task-specific native lanes passed. Final reviews found no remaining correctness or minimality issues. ADR-025 remains the governing decision; no new ADR was required. Child-specific Windows stat, compositor timing, and generated-inventory incidents were handled in their task evidence and existing lessons. Parent closeout changes only this task metadata.
<!-- SECTION:NOTES:END -->
