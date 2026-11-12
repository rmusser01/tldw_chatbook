---
id: TASK-40304
title: "Actor Pack activation writes into captured persona trees without a storage participant, so a capture or restore cannot pause it or account for it"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- backup
- recovery
- persona
priority: medium
references:
- tldw_chatbook/Actor_Packs/activation.py
- tldw_chatbook/Actor_Packs/import_controller.py
- tldw_chatbook/Backup_Recovery/persona_visual_participants.py
- tldw_chatbook/Backup_Recovery/visual_identity_participants.py
dependencies:
- TASK-39116
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream, TASK-39116 made `persona.actor_packs` a capture owner). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Every other persona writer takes a storage participant scope, so a capture or restore can pause it and account for its files. Actor Pack activation takes none. A capture or restore running beside an Actor Pack import can copy a half-written asset, or a file whose `persona_visual_assets` row is not committed yet.

**Evidence.**
- `Actor_Packs/activation.py:401-436` writes `visual_identities/actor_packs/`, and `:505-560` writes `actor_packs/<digest>/persona-runtime/`, both through a raw `_publish_immutable` (`:714`, `os.open`/`O_EXCL`/`fsync`) and `_cleanup_published` (`:774`). The call path is `import_controller.py:282`. No `Actor_Packs/*.py` module imports a Backup_Recovery participant or `acquire_storage`.
- The mechanism other writers use: `Backup_Recovery/persona_visual_participants.py:174` `request()` and `:378` `files(...)` (used by `Persona_Visual/publication.py:209`, `:502-505`), and `visual_identity_participants.py:244`/`:448` (used by `Character_Chat/visual_identity.py:1405`, `:2373`).
- The owner inventory rows for activation say "no participant token".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Activating an Actor Pack while a capture or restore holds storage waits or refuses with a named reason, and writes nothing under actor_packs/ or visual_identities/actor_packs/ meanwhile; a test is red before the fix
- [ ] #2 The owner inventory's activation rows no longer say 'no participant token'
<!-- AC:END -->
