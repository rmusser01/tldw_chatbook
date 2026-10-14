---
id: TASK-31757
title: Prepare pixel-migu portable Buddy art pack
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 22:27'
updated_date: '2026-09-05 22:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the supplied pixel-migu sprite sheets into a portable animated Buddy visual pack for Chatbook and tldw_server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transparent aligned PNG frames preserve the original eight animations and additional supplied operational/expression artwork.
- [x] #2 Portable pixel-migu archive passes Chatbook import and server preview validation with all baseline states resolved.
- [x] #3 Animation preview and import instructions accompany the Buddy pack and a separate pixel-migu character with a Visual Identity expression pack.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract transparent frames from the user-approved source artwork, normalize alignment, and retain all eight animation sequences.
2. Assemble a sprite_frames v1 Buddy archive, plus a character card and separate Shared Visual Identity expression pack using the existing character publication APIs.
3. Verify alpha, frame bounds, archive checksums, Chatbook import/publication/runtime, and server import-preview compatibility in disposable storage.
4. Provide a local animation preview, source frames, reproduction script, and import instructions.

ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md; backlog/decisions/067-bundled-samira-visual-identity-pack.md
Reason: art/data package implementing the existing portability contract; no runtime, storage, interface, or dependency change.

User approved the faithful adaptation on 2026-09-05. Image generation returned RGB checkerboards twice; user explicitly approved Python source-image processing. User clarified the added expressions must support a separate emoting character avatar, in addition to the Buddy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared pixel-migu as two distinct artifacts: an animated Persona Buddy pack (64 transparent 128×128 sprites; 31 mapped states) and a V2 character PNG/JSON plus an 18-expression Shared Visual Identity pack. Retained original sources, extraction coordinates/hashes, an offline HTML gallery, static expression gallery, and animated GIF preview. A focused installer uses the existing character/Visual Identity APIs and creates no runtime changes; it refuses duplicate names and rolls back failed publication. Character operational states update through the existing runtime; other emotions remain selectable reactions, with no new sentiment or emote parser.

Verification: two installer tests failed before implementation, then passed against real SQLite, checking every expression, four operational states, preservation, duplicate refusal, and publication rollback. Chatbook imported all 64 Buddy assets, published/reopened the binding, and resolved/rendered all 31 states with normal and reduced motion (62 cases). V2 embedded card matches its JSON; four-state legacy expression ZIP imports. Server Persona Visual manifest/import preview passed without warnings; server expression ZIP imported 18 slots, activated a character binding, and resolved all 18 through the service. Scoped Ruff E4/E7/E9/F/I, formatter, compilation and ZIP/checksum verification passed. Reports and source commits are in output/pixel-migu/verification-*.json.

All changes are artifact files in output/pixel-migu and this task. No application runtime files, live profiles, or running servers were modified; no full test sweep or commit. Browser policy blocked local HTML navigation, so no browser-interaction verification is claimed; PNG/GIF previews were checked independently. The image-generation tool returned fake transparency twice; user explicitly approved direct source-image extraction and the generated attempts were discarded.

ADR check: existing ADR-074 and ADR-067 (bundled-samira-visual-identity-pack) supply the separate Buddy/expression contracts; no new architectural decision. No new generalized lesson was needed: the disposable macOS temp-path alias issue followed an already-known strict-path constraint, and no application behavior was changed.
<!-- SECTION:NOTES:END -->
