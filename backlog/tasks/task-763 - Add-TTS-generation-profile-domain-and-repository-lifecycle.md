---
id: TASK-763
title: Add TTS generation profile domain and repository lifecycle
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 21:55'
updated_date: '2026-07-27 04:18'
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
- [x] #1 A dedicated versioned SQLite store initializes and migrates safely while rejecting unsupported or corrupt schemas.
- [x] #2 Profiles enforce immutable identity normalized unique names exact bounded generation data timestamps and optimistic revisions.
- [x] #3 Authority-scoped assignments enforce referential integrity and provide bounded counts lists and one joined immutable profile read.
- [x] #4 All repository operations use one serialized off-loop lifecycle lane and stale pre-restore work cannot publish after generation changes.
- [x] #5 The profile database path and Backup All integration use repository-owned consistent backup semantics.
- [x] #6 Deterministic tests cover schema validation CRUD concurrency interprocess exclusion backup restore and stale-generation behavior.
- [x] #7 Online backup and bounded exclusive restore preserve the current store on quiescence validation lock or replacement failure and report unavailable without recreating data after a post-replacement reopen failure.
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

## Implementation Notes

Implemented the approved Slice 2A persistence foundation: immutable and
bounded profile-domain values, a dedicated versioned SQLite schema,
authority-scoped assignments, one serialized generation-aware repository
lifecycle, cooperative process locking, optimistic CRUD, and repository-owned
online backup plus bounded atomic restore. The application now owns one lazy
repository, the profile path is configurable, and Backup All obtains a
consistent profile snapshot without copying an open database.

ADR-028 records profile ownership, privacy, assignment authority, lifecycle,
and rollback decisions. The developer and Speech Services guides now document
the local store, normalized names and revisions, process-lock behavior,
per-database backup consistency, restore recovery/fail-closed behavior, and the
explicit deferred scope.

Verification evidence before the required final rebase:

- The seven-file focused suite passed: 615 passed, 3 warnings.
- The broad TTS suite passed: 1527 passed, 14 skipped, 13 warnings, adding 598
  passing tests over the recorded 929-pass/14-skip TTS baseline.
- Task-scoped Ruff, Ruff format, compileall, focused mypy, and
  `git diff --check` passed. The five diagnostics from the exact broad Ruff
  command are pre-existing `app.py` baseline findings also present on
  `origin/dev`; the task introduces no Ruff regression.
- Direct pytest collection aborts in the installed optional MLX dependency on
  this host. The recorded suites used the normal fixtures with the optional
  MLX modules made unavailable so the application's supported optional-
  dependency path could be exercised.
- Privacy and scope review found authority identifiers only in assignment
  contracts/persistence and no task-added credential, provider-origin, raw
  profile path, message-content, or managed-process behavior.

Slice 2B profile-management services, Slice 3 character assignment and
authority acquisition, Slice 4 runtime/roleplay routing, portability/sync,
legacy-provider execution, provider connection details, and managed audio.cpp
behavior remain deferred.

Status remains In Progress. A superseded pre-rebase repository-wide run was
stopped at 8% (exit 143, with no failures reported); its `origin/dev` comparison
was intentionally not started. Post-rebase repository-wide verification,
independent review, final DoD confirmation, and the Done transition remain
pending.
