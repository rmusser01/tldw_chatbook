---
id: TASK-503
title: >-
  RAG settings + profiles SP3: RAG settings screen (profile manager + full editor)
status: To Do
assignee: []
created_date: '2026-07-23 21:00'
updated_date: '2026-07-23 21:00'
labels:
  - rag
  - profiles
  - settings
dependencies:
  - task-483
  - task-502
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final sub-project of the RAG settings screen + profiles program (overview: Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md; SP3 spec: Docs/superpowers/specs/2026-07-21-rag-settings-screen-design.md; plan: Docs/superpowers/plans/2026-07-23-rag-settings-screen-sp3.md). Makes the profile system user-facing: the "Library & RAG" settings category becomes "RAG", editing the ACTIVE PROFILE (the engine's real config source after SP2b) instead of the dead AppRAGSearchConfig.rag.* keys. Depends on SP2a (task-483, PR #780) and SP2b (task-502, PR #795), both merged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The RAG category loads from and saves to the active profile file (via ConfigProfileManager); no AppRAGSearchConfig.rag.* writes remain (dead builders retired).
- [ ] Profile-manager region: builtins-vs-user grouped list with active marker; set-active (off-thread, dirty-draft Save/Discard prompt), clone, rename, delete; builtin active → read-only editor + "Clone to edit".
- [ ] Editor covers every ProfileConfig section in collapsible groups (search + embedding + chunking + vector-store + reranking, ~21 fields); index-determining fields marked; rerank toggle controls reranking_config presence and provably reaches the service flag (integration test).
- [ ] RAGConfig.validate() wired into category validation (first caller) plus rerank checks.
- [ ] Index status readout (built/empty/absent + provenance) with Backfill action; "new empty index — Backfill" warning on BOTH set-active and index-field save.
- [ ] QA captures via textual-serve; user screen approval gates the PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-23-rag-settings-screen-sp3.md — 5 TDD tasks: (1) adapter seam load/save→active profile + retitle; (2) profile-manager region; (3) extended editor groups + validation + rerank presence; (4) index status/backfill/warnings; (5) dead-code retirement + regression + QA captures. New headless module settings_rag_profile_adapter.py. SDD-executed, user-gated PR.
<!-- SECTION:PLAN:END -->
