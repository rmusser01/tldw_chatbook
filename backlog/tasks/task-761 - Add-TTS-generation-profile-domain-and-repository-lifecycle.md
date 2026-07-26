---
id: TASK-761
title: Add TTS generation profile domain and repository lifecycle
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 21:55'
updated_date: '2026-07-26 22:01'
labels:
  - tts
  - profiles
  - sqlite
dependencies: []
references:
  - TASK-710
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the durable local storage and concurrency boundary for reusable TTS generation profiles so later STTS and character-assignment slices can consume them without coupling profiles to provider connection settings or character databases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A dedicated versioned SQLite store initializes and migrates safely while rejecting unsupported or corrupt schemas.
- [ ] #2 Profiles enforce immutable identity normalized unique names exact bounded generation data timestamps and optimistic revisions.
- [ ] #3 Authority-scoped assignments enforce referential integrity and provide bounded counts lists and one joined immutable profile read.
- [ ] #4 All repository operations use one serialized off-loop lifecycle lane and stale pre-restore work cannot publish after generation changes.
- [ ] #5 The profile database path and Backup All integration use repository-owned consistent backup semantics.
- [ ] #6 Deterministic tests cover schema validation CRUD concurrency interprocess exclusion backup restore and stale-generation behavior.
- [ ] #7 Online backup and bounded exclusive restore preserve the current store on quiescence validation lock or replacement failure and report unavailable without recreating data after a post-replacement reopen failure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record ADR-028 and the approved Slice 2A scope in the task and Superpowers plan.
2. Add safe immutable profile-domain validation with TDD.
3. Add the dedicated versioned SQLite schema and fail-closed migration validation with TDD.
4. Add bounded shared/exclusive interprocess store locking.
5. Add the one-worker repository lifecycle and generation-checked operation results.
6. Add transactional profile CRUD optimistic revisions bounded pagination assignments counts and joined reads.
7. Add SQLite online backup and generation-safe bounded atomic restore with recovery coverage.
8. Add the profile DB path one lazy app-owned repository and Backup All integration.
9. Update documentation run focused broad static and baseline-aware repository verification then request independent review.

Full plan: Docs/superpowers/plans/2026-07-26-tts-profile-domain-repository-lifecycle.md
ADR required: yes
ADR path: backlog/decisions/028-character-tts-generation-profile-ownership.md
Reason: Slice 2A establishes a new versioned store data ownership authority-scoped assignment and backup/restore lifecycle boundary.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and concise implementation notes are recorded.
- [ ] #2 Focused unit integration concurrency and lifecycle tests pass.
- [ ] #3 Ruff formatting compileall focused typing and git diff checks pass.
- [ ] #4 ADR-028 and relevant TTS backup documentation are current.
- [ ] #5 Self-review confirms the slice adds no STTS library character assignment runtime routing portability or managed audio.cpp behavior.
<!-- DOD:END -->
