---
id: TASK-763
title: Add TTS generation profile domain and repository lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 21:55'
updated_date: '2026-07-27 14:50'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the approved Slice 2A persistence foundation: immutable bounded
profile-domain values, a dedicated versioned SQLite schema, authority-scoped
assignments, one serialized generation-aware repository lifecycle,
interprocess locking, optimistic CRUD, and repository-owned online backup plus
bounded atomic restore. The application owns one lazy repository, its path is
configurable, and Backup All obtains a consistent profile snapshot without
copying an open database.

ADR-028 records profile ownership, privacy, assignment authority, lifecycle,
and rollback decisions. The developer and Speech Services guides document the
store, normalized names and revisions, locking, backup consistency, bounded
restore recovery, fail-closed behavior, and deferred scope.

Final delivery evidence:

- The branch was rebased conflict-free onto `origin/dev`
  `20ff9928622de58f2adc96212258876e5a5d06a6`; all 57 implementation commits
  remained patch-identical across the rebase.
- The final eight-file profile unit/integration/concurrency/lifecycle gate
  passed: 656 passed, 3 warnings.
- The broad TTS gate passed on the same patch-identical implementation:
  1552 passed, 14 expected optional skips, 13 dependency/deprecation warnings.
- Task-scoped Ruff and Ruff format, compileall, focused mypy across the five
  profile modules, and exact-range `git diff --check` passed.
- Independent whole-range and post-rebase integration reviews found no
  Critical, Important, or Minor issues. The final upstream overlap in
  `path_validation.py` is non-conflicting: upstream Windows-path handling and
  this slice's value-free privacy diagnostic are both retained.
- Privacy and scope review found no task-added credential, provider endpoint,
  message-content, managed-process, STTS-library, character-editor, runtime
  routing, or portability behavior.

Slice 2B profile services/STTS library, Slice 3A character identity and
assignment, Slice 3B roleplay resolution/runtime routing, optional Slice 4
portability, and managed audio.cpp launch/supervision remain separate work.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and concise implementation notes are recorded.
- [x] #2 Focused unit integration concurrency and lifecycle tests pass.
- [x] #3 Ruff formatting compileall focused typing and git diff checks pass.
- [x] #4 ADR-028 and relevant TTS backup documentation are current.
- [x] #5 Self-review confirms the slice adds no STTS library character assignment runtime routing portability or managed audio.cpp behavior.
<!-- DOD:END -->
