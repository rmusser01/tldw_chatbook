---
id: TASK-483
title: >-
  RAG settings + profiles SP2a: profile storage foundation
status: To Do
assignee: []
created_date: '2026-07-22 01:15'
updated_date: '2026-07-22 01:15'
labels:
  - rag
  - profiles
dependencies:
  - task-476
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First half of SP2 (profile system) in the RAG settings screen + profiles program (overview: Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md; SP2 spec: Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md; plan: Docs/superpowers/plans/2026-07-22-rag-profile-system-sp2a-storage.md). Turns the existing `ConfigProfileManager` into a correct, file-backed, user-facing profile store. SP2b (config-resolution unification onto the active profile, reset seam, first-run import, parity test) is a separate follow-up task/PR. Depends on SP1 (task-476, merged PR #771).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `ProfileConfig.from_dict` reconstructs nested dataclasses correctly (routes rag_config through `RAGConfig.from_dict`), so file-per-profile round-trips.
- [ ] Builtin profiles use the real dataclass fields and valid enum values (chunk_size/chunk_overlap/default_top_k/default_search_mode; store type memory not "in_memory"; keyword→plain), so fast/balanced/accuracy actually differ; `validate_profile` reads real fields.
- [ ] `ProfileConfig` has a stable, rename-safe `id` and a `read_only` marker; builtins are read-only.
- [ ] User profiles are stored one JSON file per profile under `rag_profiles/`; a legacy `custom_profiles.json` blob is migrated to per-file once (idempotent).
- [ ] CRUD: `save_profile` / `delete_profile` / `rename_profile` / `clone_profile`, all refusing to mutate builtins; clone deep-copies (incl. a builtin) into a fresh writable profile; rename keeps id + file.
- [ ] No config-resolution / active-pointer / reset-seam / first-run-import changes (those are SP2b). No DB migration.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-22-rag-profile-system-sp2a-storage.md — 6 TDD tasks: (1) round-trip fix; (2) builtin field/value fix + validate; (3) stable id + read_only marker; (4) file-per-profile storage + legacy migration; (5) CRUD with read-only guards; (6) regression gate + docs. SDD-executed, user-gated PR. All changes in RAG_Search/config_profiles.py.
<!-- SECTION:PLAN:END -->
