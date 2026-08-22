---
id: TASK-19056
title: Enable Shared Visual Identity for Persona actors
status: Done
assignee: []
created_date: '2026-08-20 18:01'
updated_date: '2026-08-22 16:31'
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
- [x] #1 Eligible local Personas, identified by exact local source/id plus current profile/editor revision, can create, replace, clear, publish, and resolve Shared Visual Identity bindings using the existing immutable pack/version model; inactive, disabled, deleted, or missing Personas cannot author, publish, or render. Restore, replacement, or concurrent update re-resolves authority, and stale authority cannot mutate active state.
- [x] #2 Personas Workbench exposes path-free Shared Visual Identity metadata, lazy selected preview, staged edits, visible manual labels where applicable, Save, and Cancel with full session/actor/binding fences; declining dirty navigation preserves the draft and staging, while accepted navigation or Cancel signals and drains in-flight work, discards only the unpublished candidate/draft, and preserves the active version.
- [x] #3 Persona resolution uses exact full actor and cache identities, deterministic fallback, targeted actor invalidation, and source-only change detection.
- [x] #4 Console/persona-chat consumers render the active Persona expression without giving Persona Buddy operational states any reaction semantics.
- [x] #5 Server-backed Personas require Save Local Copy first; source, session, actor, binding, version, and profile-revision authority is revalidated after every await, and any stale change fails closed without publication or repaint.
- [x] #6 Existing Character creation, authoring, Console rendering, publication, cache, and four-state operational behavior remain unchanged.
- [x] #7 No schema/runtime merge with Persona Visual, Actor Pack archive workflow, or server write path is introduced.
- [x] #8 Labelled actions are keyboard-operable and compact and normal layouts paint usable controls; user-facing errors, logs, and diagnostics remain path-free. Focused real SQLite repository/resolver/Workbench/Console/race/invalidation/lifecycle tests pass in an isolated profile with born-RED → GREEN evidence and mutation proof for authority, cancellation, and invalidation guards. Evidence records assigned-worktree provenance and establishes isolated `HOME`, XDG, config, and data roots before imports; scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance checks, including ADR-067 gates, pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/067-bundled-samira-visual-identity-pack.md and backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: Existing ADRs already define Persona actor bindings and require Shared Visual Identity expressions to remain separate from Persona Visual/Buddy operational states.

Executable plan: Docs/superpowers/plans/2026-08-22-task-19056-persona-shared-visual-identity.md

1. Freeze exact local Persona authority and generalize deterministic Shared Visual Identity resolution with Character non-regression.
2. Support unbound Persona candidates and authority-fenced immutable publication through the existing repository/filesystem boundary.
3. Add the distinct Personas Workbench Shared Visual Identity editor with lazy preview, staging, Save/Cancel, dirty-navigation, and cancellation drain.
4. Extend the existing Console actor-scoped reaction/cache path to eligible local Personas without Buddy coupling.
5. Prove lifecycle, race, targeted invalidation, privacy, architecture, and Character-equivalence contracts.
6. Run isolated touched-component/static/governance evidence and close the task only if every scoped gate is green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Persona actors on the existing Shared Visual Identity boundary: exact local authority and immutable resolution/publication, Workbench staging with Save/Cancel and lifecycle fences, Console expression rendering, and affected-only invalidation while retaining Character and Persona Buddy separation.

Verification: assigned-worktree provenance and architecture gate 4 passed; complete affected-component gate 984 passed with 1 Windows-only skip and 3 dependency warnings; diagnostic, privacy, architecture, and governance gate 68 passed; focused Console gate 99 passed with 314 deselected; lifecycle and mutation matrix 88 passed. Born-RED to GREEN and mutation evidence covered source, Persona revision, editor generation, binding version, post-decode identity, cancellation drain, and affected-only invalidation guards. All changed Python files pass Ruff, format, py_compile, and diff checks; the forbidden-boundary scan found no Persona Buddy, Persona Visual, Actor Pack archive, or server-write coupling. No full repository suite was run.

ADR: no new ADR was required; the implementation follows ADR-067 and ADR-074. The executable plan at Docs/superpowers/plans/2026-08-22-task-19056-persona-shared-visual-identity.md was completed without architectural deviation. No reusable lesson entry was warranted.
<!-- SECTION:NOTES:END -->
