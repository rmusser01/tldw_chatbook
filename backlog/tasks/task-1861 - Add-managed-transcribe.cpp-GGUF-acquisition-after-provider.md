---
id: TASK-1861
title: Add managed transcribe.cpp GGUF acquisition after provider
status: To Do
assignee: []
created_date: '2026-08-02 14:54'
labels:
  - stt
  - gguf
  - artifacts
dependencies:
  - TASK-596
  - TASK-597
  - TASK-604
references:
  - backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md
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
- [ ] #1 A representative curated transcribe.cpp catalog declares immutable sources, sizes, digests, licenses, capabilities, and reviewed precision options.
- [ ] #2 Users can explicitly download a curated GGUF or copy an existing compatible local GGUF into the managed artifact store with verified final bytes.
- [ ] #3 Managed GGUF installation, activation, recovery, deletion, and browser status reuse the shared artifact core without changing provider inference behavior.
- [ ] #4 The transcribe.cpp provider accepts either the existing validated direct path or a managed path, and direct-path users are not forced to migrate.
- [ ] #5 Managed artifacts receive precise curated, integrity-verified, or local-integrity provenance and never enter semantic default routing.
- [ ] #6 Focused tests cover download/import failure cleanup, disk preflight, immutable promotion, activation, recovery, and all supported wheel platforms.
<!-- AC:END -->
