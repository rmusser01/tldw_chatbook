---
id: TASK-1915
title: Add managed transcribe.cpp GGUF acquisition after provider
status: To Do
assignee: []
created_date: '2026-08-02 14:54'
updated_date: '2026-08-03 21:37'
labels:
  - stt
  - gguf
  - artifacts
dependencies:
  - TASK-596
  - TASK-597
  - TASK-604
references:
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add curated downloads and managed local-file import only after direct-path transcribe.cpp transcription works, while reusing the same provider load path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A representative curated transcribe.cpp catalog declares immutable sources, sizes, digests, licenses, capabilities, Q8_0 as the default, and an explicit full-precision option where upstream publishes one.
- [ ] #2 Users can explicitly download a curated GGUF or copy an existing compatible local GGUF into the managed artifact store with verified final bytes.
- [ ] #3 Managed GGUF installation, activation, recovery, deletion, and browser status reuse the shared artifact core without changing provider inference behavior.
- [ ] #4 The transcribe.cpp provider accepts either the existing validated direct path or a managed path, and direct-path users are not forced to migrate.
- [ ] #5 Managed artifacts receive precise curated, integrity-verified, or local-integrity provenance and never enter semantic default routing.
- [ ] #6 Focused tests cover download/import failure cleanup, disk preflight, immutable promotion, activation, recovery, and all supported wheel platforms.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cross-link 2026-08-03: TASK-2062 (model browser Phase 3, in progress on feat/task-2062-model-browser-phase-3) is building the generic copy-into-store import engine as Model_Artifacts/local_import.py — stream-copy+hash into marker-owned staging temps, install(consume_source=True) verification, content-addressed revisions, with consumer/runtime/format parameterizable (LLM defaults). When 1915 activates, review REUSING that engine plus the live gguf_admission parser rather than activating _deferred_gguf_managed_import.py wholesale; the deferred prototype's descriptor constants may still be the right transcribe.cpp values, but the copy/verify/promote path should be shared. Also note: descriptor file:// source_url and unknown-license accommodation landed via 2062 Task 1 (cross-field-gated in service.py), which 1915's local-copy AC #2 will need.
<!-- SECTION:NOTES:END -->
