---
id: TASK-19054
title: Author and import Persona Visual packs
status: To Do
assignee: []
created_date: '2026-08-20 17:29'
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
- [ ] #1 Personas Workbench shows all nine baseline state slots, bounded safe custom states, path-free validation inventory, and one selected lazy preview for an eligible local Persona.
- [ ] #2 Replace, Clear, Add Custom State, and import mutate only an isolated draft; Save revalidates Persona, binding, and draft authority, publishes exactly one immutable version, then invalidates both stable old/new full identities while preserving unrelated cache entries; failed or cancelled publication invalidates nothing; Cancel discards the draft and leaves authoritative metadata unchanged.
- [ ] #3 `.tldw-persona-vpack` import validates full pinned sprite-frame archives into review drafts in bounded private staging and never activates before explicit Save; it rejects traversal, links, nested/encrypted archives, undeclared/external files, duplicate/colliding paths, bomb/budget violations, MIME/digest mismatches, and archive replacement races; failure or cancellation removes only identity-pinned staging and never changes active authority.
- [ ] #4 Unsupported renderer/manifest capabilities, malformed assets, stale Persona/binding/session authority, and import cancellation fail closed without changing the active version.
- [ ] #5 Server-backed Personas show Save Local Copy first; legacy expression-set and Actor Pack import remain separate, honestly labelled actions.
- [ ] #6 Preview inventory/resolve/decode work is screen-owned, serialized across navigation, drained on cancellation, weak-targeted, and fenced after every await.
- [ ] #7 No image-generation provider, recipe workflow, Shared Visual Identity merge, or Buddy window is added.
- [ ] #8 Labelled actions are keyboard-operable, preserve focus, and add no forbidden bindings; compact and normal layouts paint usable controls; untrusted archive text renders as plain text; user-facing errors, logs, and diagnostics remain path-free. Evidence includes born-RED→GREEN tests and mutation proof for draft, Save, Cancel, authority, archive, cancellation, and invalidation guards; assigned-worktree provenance; real SQLite publication/repository tests where touched; isolated HOME/XDG/config/data roots; focused widget/screen/race/import/publication tests; Ruff, format, compile, and diff checks; diagnostic, privacy, architecture, and governance gates; and Impeccable review after the final visible change.
<!-- AC:END -->
