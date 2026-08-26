---
id: TASK-19054
title: Author and import Persona Visual packs
status: Done
assignee: []
created_date: '2026-08-20 17:29'
updated_date: '2026-08-21 08:34'
labels: []
dependencies:
  - TASK-19053
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users review, edit, import, stage, and explicitly publish Persona Visual packs for local Personas while keeping active runtime visuals unchanged until Save.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Personas Workbench shows all nine baseline state slots, bounded safe custom states, path-free validation inventory, and one selected lazy preview for an eligible local Persona.
- [x] #2 Replace, Clear, Add Custom State, and import mutate only an isolated draft; Save revalidates Persona, binding, and draft authority, publishes exactly one immutable version, then invalidates both stable old/new full identities while preserving unrelated cache entries; failed or cancelled publication invalidates nothing; Cancel discards the draft and leaves authoritative metadata unchanged.
- [x] #3 `.tldw-persona-vpack` import validates full pinned sprite-frame archives into review drafts in bounded private staging and never activates before explicit Save; it rejects traversal, links, nested/encrypted archives, undeclared/external files, duplicate/colliding paths, bomb/budget violations, MIME/digest mismatches, and archive replacement races; failure or cancellation removes only identity-pinned staging and never changes active authority.
- [x] #4 Unsupported renderer/manifest capabilities, malformed assets, stale Persona/binding/session authority, and import cancellation fail closed without changing the active version.
- [x] #5 Server-backed Personas show Save Local Copy first; legacy expression-set and Actor Pack import remain separate, honestly labelled actions.
- [x] #6 Preview inventory/resolve/decode work is screen-owned, serialized across navigation, drained on cancellation, weak-targeted, and fenced after every await.
- [x] #7 No image-generation provider, recipe workflow, Shared Visual Identity merge, or Buddy window is added.
- [x] #8 Labelled actions are keyboard-operable, preserve focus, and add no forbidden bindings; compact and normal layouts paint usable controls; untrusted archive text renders as plain text; user-facing errors, logs, and diagnostics remain path-free. Evidence includes born-RED→GREEN tests and mutation proof for draft, Save, Cancel, authority, archive, cancellation, and invalidation guards; assigned-worktree provenance; real SQLite publication/repository tests where touched; isolated HOME/XDG/config/data roots; focused widget/screen/race/import/publication tests; Ruff, format, compile, and diff checks; diagnostic, privacy, architecture, and governance gates; and Impeccable review after the final visible change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an immutable, path-free authoring draft contract over the existing Persona Visual manifest/publication boundaries.
2. Add a pinned-server-compatible `.tldw-persona-vpack` importer with bounded private staging and identity-pinned cleanup.
3. Add the Persona Visual browser/editor section and typed actions to the existing Persona profile editor.
4. Wire screen-owned loading, selected-only preview, isolated edits/import, cancellation drain, navigation guards, and honest import labels.
5. Publish one immutable version through the existing authority boundary, invalidate exact old/new identities, and refresh authoritatively.
6. Run touched-component, mutation, isolated-profile, SQLite, UI, privacy, architecture, Impeccable, and static gates; then record concise implementation evidence.

ADR required: no

ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

Reason: ADR-074 already defines the separate local Persona Visual runtime, review-first import, immutable publication, authority, and scope boundaries implemented by this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added immutable authoring drafts, bounded `.tldw-persona-vpack` review import, and identity-pinned private workspaces over the existing ADR-074 repository/publication boundary.
- Added the Persona Workbench pack browser with nine baseline slots, bounded custom states, selected-only preview, Replace/Clear/Import/Save/Cancel actions, compact layout, honest legacy labels, and screen-owned authority/cancellation/navigation orchestration.
- Save publishes once after exact local Persona and draft revalidation, invalidates only the old/new full identities, reloads authoritatively, and preserves active state on failure or cancellation. No provider, Shared Visual Identity, Buddy, Actor Pack, or server-backed authoring scope was added.
- TDD and mutation evidence covered hostile archives, draft isolation, workspace substitution, duplicate Save, Cancel/drain, stale authority, preview cancellation, import cancellation, old/new invalidation, and path-free error handling. Final touched gates were 120 authoring/import/workspace/publication/widget/screen tests, 10 legacy Workbench compatibility tests, 60 CSS/profile tests, 524 Persona Visual package tests, and 319 Workbench tests; Ruff, formatting, compilation, diff checks, isolated-profile provenance, and the one required Impeccable detector run passed.
- Per user direction, the full repository suite was not run. The scoped diagnostic/privacy command had 107 passing tests and one generated-inventory mismatch caused partly by unrelated `retrieval.py`, `workspace.py`, and `chat_screen.py` drift; topology, classifications, and exclusions were unchanged, so unrelated generated inventory was not rewritten. Six unrelated `Client_Media_DB_v2` privacy-baseline failures reported by the preceding foundation task were also left outside this task.
- ADR check: no new ADR; ADR-074 remains the governing decision. No reusable lesson was added because the implementation followed the existing cancellation, worktree, and testing guidance without uncovering a new cross-task trap.
<!-- SECTION:NOTES:END -->
