---
id: TASK-2062
title: 'Model browser Phase 3: adopt GGUF models and retire the legacy downloader'
status: To Do
assignee: []
created_date: '2026-08-03 20:11'
labels:
  - models
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 of the TASK-596 spec (Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md, 'Phasing'). Phases 1-2 are merged (PRs #1175, #1185, #1190, #1210, #1245): Curated/Remote/Installed views over the acquisition layer, with unmanaged GGUF files listed as 'Unmanaged -- integrity unknown' and no actions. Phase 3 completes the migration: (1) Import turns an unmanaged file into a managed model (LOCAL_INTEGRITY_RECORDED, digests computed at import); (2) local-server launch paths (llama.cpp, llamafile, vLLM, MLX) resolve model paths from ModelArtifactService instead of user-typed paths; (3) retire Widgets/HuggingFace/ (~2,200 lines) and the 'Download Models' rail row. Rollback note from ADR-025 applies: keep new installs disabled rather than reverting to unverified direct writes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An unmanaged GGUF file can be imported as a managed model with locally recorded digests
- [ ] #2 Local-server launch paths resolve from the service; user-typed path fields are retired or become an explicit unmanaged escape hatch
- [ ] #3 Widgets/HuggingFace and the download-models rail row are removed; no unverified direct-write download path remains
<!-- AC:END -->
