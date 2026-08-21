---
id: TASK-19053
title: Add local Persona Visual pack foundation
status: Done
assignee: []
created_date: '2026-08-20 17:12'
updated_date: '2026-08-21 02:24'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
  - tldw_server dev commit 385afa951922c8a9dc2002c675bb6cad65e4ac23
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a profile-local Persona Visual runtime compatible with the pinned server's sprite-frame contract so local Personas can own immutable operational-state visuals without merging them into Shared Visual Identity reactions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local persistence supports Persona Visual packs, immutable versions, assets, and one active binding per eligible local Persona without changing existing Persona records.
- [x] #2 Validation matches the pinned server contract: manifest version 1 for `sprite_frames`, nine reserved built-ins including `wake_armed`, five required resolvable states, bounded safe custom states, fallback chains, frames, regions, timing, authored triggers, validated static fallback selection, and reduced-motion rendering that stops animation.
- [x] #3 Activatable packs resolve `idle`, `listening`, `thinking`, `speaking`, and `error`; runtime misses fall through validated manifest fallbacks, then `idle`, then Persona portrait with a stable reason.
- [x] #4 Assets use validated profile-owned storage, MIME/decode/dimension/frame budgets, and immutable full-identity cache keys; public resolver/result objects, user-facing errors, logs, and diagnostic inventory expose stable identifiers and reasons only, never private paths.
- [x] #5 Repository and publication paths enforce optimistic binding/version authority, rollback, and pinned orphan cleanup, and return stable old/new full identities for later targeted consumer invalidation.
- [x] #6 Frozen fixtures derived from server commit `385afa...` pin supported and unsupported renderer/manifest behavior.
- [x] #7 No Workbench authoring UI, floating Buddy, provider generation, or server write path is introduced.
- [x] #8 Focused migration/repository/validator/asset/resolver/publication coverage runs born-RED then GREEN; mutation proof covers authority, validation, fallback, rollback, orphan cleanup, full-identity/cache, and privacy guards; assigned-worktree import provenance is asserted; real SQLite migration/repository tests run in an isolated HOME/XDG/config/data profile; and scoped Ruff, format, compile, `git diff --check`, diagnostic, privacy, architecture, and governance checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze the pinned sprite_frames manifest-v1 capability, state, fallback, trigger, static-preview, and reduced-motion contracts with born-RED vectors.
2. Add the separate ChaChaNotes V40-to-V41 Persona Visual schema and repository for immutable packs, versions, assets, and optimistic local-Persona bindings.
3. Add profile-owned raster validation/loading with bounded MIME, decode, dimension, frame, digest, confinement, and privacy checks.
4. Add the path-free operational-state resolver with manifest fallback, idle, portrait, static/reduced-motion, and full-identity cache behavior.
5. Add synchronous immutable publication with authority revalidation, pinned atomic replacement, rollback, stable old/new identities, and identity-scoped orphan cleanup.
6. Verify only touched components with born-RED-to-GREEN and mutation evidence, isolated HOME/XDG/config/data roots, focused migration/repository/runtime/publication tests, scoped static/privacy/architecture/governance checks, then complete the task.

ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: ADR-074 already governs the separate Persona Visual runtime, storage ownership, compatibility, and authority boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the separate profile-local Persona Visual foundation governed by ADR-074: pinned sprite_frames manifest-v1 contracts and server-derived fixtures; V40-to-V41 immutable pack/version/asset/binding persistence; bounded profile-owned raster admission; path-free state resolution; and synchronous authority-pinned publication, rollback, and orphan cleanup.

ADR required: no. ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md. ADR-074 already defines the storage, compatibility, and runtime boundary; no new architectural decision was introduced.

Verification: focused Persona Visual plus migration 473 passed; touched ChaChaNotes and Shared Visual Identity adjacency 563 passed and 1 Windows-only skip; architecture guard born-RED 1 failed/2 passed under a temporary forbidden UI import, then GREEN 3 passed. Isolated HOME/XDG/config/data verification migrated real SQLite to schema 41, published and resolved a pack from this assigned worktree, asserted six module provenances, zero writes outside the isolated root, and zero private-root token leaks. Scoped Ruff format/check, compileall, git diff --check, scope exclusions, and relevant privacy gates passed (17 ChaChaNotes owner cases; 40 database-path and Persona architecture cases).

Born-RED/mutation evidence across Tasks 1-5 and the closeout guard covers required-state validation, recursive fallback then idle, repository/publication authority CAS and ABA, late-write transaction rollback, source/final inode substitution, cleanup reference races and pinned candidate substitution, every full graph/cache identity field, and path/exception redaction. The focused owning regressions are included in the 473-pass gate; the architecture boundary mutation was rerun during closeout.

The prescribed broad diagnostic/privacy command also produced 187 passes and 7 proven unrelated baseline failures: one stale generated diagnostic inventory mismatch and six Client_Media_DB_v2 exception-chaining cases. Read-only semantic review found no Persona Visual diagnostics and no sink/classification change; unrelated delta was one new library_media_browse_controller owner, Client_Media_DB_v2 354 to 338 calls, library_screen 110 to 109 calls, plus digest-only console_chat_controller/enhanced_file_picker changes. The generated inventory was therefore intentionally left untouched. The two plan paths for visual identity migration/repository were stale; their canonical Tests/ChaChaNotesDB files ran within the 563-pass adjacency gate.

Touched files: tldw_chatbook/Persona_Visual; ChaChaNotes_DB.py and the V41 migration; focused Persona Visual/migration and architecture tests; this task and its execution plan. Self-review confirmed no Workbench UI, Buddy, provider generation, server writes, Shared Visual Identity schema reuse, Persona JSON mutation, archive import/export flow, or new dependency. Per explicit user instruction, no full repository suite was run; verification was limited to touched components.
<!-- SECTION:NOTES:END -->
