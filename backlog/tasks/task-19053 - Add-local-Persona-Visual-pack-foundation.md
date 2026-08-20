---
id: TASK-19053
title: Add local Persona Visual pack foundation
status: To Do
assignee: []
created_date: '2026-08-20 17:12'
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
- [ ] #1 Local persistence supports Persona Visual packs, immutable versions, assets, and one active binding per eligible local Persona without changing existing Persona records.
- [ ] #2 Validation matches the pinned server contract: manifest version 1 for `sprite_frames`, nine reserved built-ins including `wake_armed`, five required resolvable states, bounded safe custom states, fallback chains, frames, regions, timing, authored triggers, validated static fallback selection, and reduced-motion rendering that stops animation.
- [ ] #3 Activatable packs resolve `idle`, `listening`, `thinking`, `speaking`, and `error`; runtime misses fall through validated manifest fallbacks, then `idle`, then Persona portrait with a stable reason.
- [ ] #4 Assets use validated profile-owned storage, MIME/decode/dimension/frame budgets, and immutable full-identity cache keys; public resolver/result objects, user-facing errors, logs, and diagnostic inventory expose stable identifiers and reasons only, never private paths.
- [ ] #5 Repository and publication paths enforce optimistic binding/version authority, rollback, and pinned orphan cleanup, and return stable old/new full identities for later targeted consumer invalidation.
- [ ] #6 Frozen fixtures derived from server commit `385afa...` pin supported and unsupported renderer/manifest behavior.
- [ ] #7 No Workbench authoring UI, floating Buddy, provider generation, or server write path is introduced.
- [ ] #8 Focused migration/repository/validator/asset/resolver/publication tests plus scoped static and diagnostic-governance checks pass in an isolated profile.
<!-- AC:END -->
