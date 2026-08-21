---
id: TASK-19056
title: Enable Shared Visual Identity for Persona actors
status: To Do
assignee: []
created_date: '2026-08-20 18:01'
updated_date: '2026-08-20 18:02'
labels: []
dependencies:
  - TASK-16319
references:
  - backlog/decisions/067-bundled-samira-visual-identity-pack.md
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the already-declared Persona actor path in Shared Visual Identity so local Personas can own reaction/expression packs without merging those expressions into Persona Buddy operational states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible local Personas, identified by exact local source/id plus current profile/editor revision, can create, replace, clear, publish, and resolve Shared Visual Identity bindings using the existing immutable pack/version model; inactive, disabled, deleted, or missing Personas cannot author, publish, or render. Restore, replacement, or concurrent update re-resolves authority, and stale authority cannot mutate active state.
- [ ] #2 Personas Workbench exposes path-free Shared Visual Identity metadata, lazy selected preview, staged edits, visible manual labels where applicable, Save, and Cancel with full session/actor/binding fences; declining dirty navigation preserves the draft and staging, while accepted navigation or Cancel signals and drains in-flight work, discards only the unpublished candidate/draft, and preserves the active version.
- [ ] #3 Persona resolution uses exact full actor and cache identities, deterministic fallback, targeted actor invalidation, and source-only change detection.
- [ ] #4 Console/persona-chat consumers render the active Persona expression without giving Persona Buddy operational states any reaction semantics.
- [ ] #5 Server-backed Personas require Save Local Copy first; source, session, actor, binding, version, and profile-revision authority is revalidated after every await, and any stale change fails closed without publication or repaint.
- [ ] #6 Existing Character creation, authoring, Console rendering, publication, cache, and four-state operational behavior remain unchanged.
- [ ] #7 No schema/runtime merge with Persona Visual, Actor Pack archive workflow, or server write path is introduced.
- [ ] #8 Labelled actions are keyboard-operable and compact and normal layouts paint usable controls; user-facing errors, logs, and diagnostics remain path-free. Focused real SQLite repository/resolver/Workbench/Console/race/invalidation/lifecycle tests pass in an isolated profile with born-RED → GREEN evidence and mutation proof for authority, cancellation, and invalidation guards. Evidence records assigned-worktree provenance and establishes isolated `HOME`, XDG, config, and data roots before imports; scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance checks, including ADR-067 gates, pass.
<!-- AC:END -->
