---
id: TASK-40304
title: "Actor Pack activation writes into captured persona trees without a storage participant, so a capture or restore cannot pause it or account for it"
status: Done
assignee: ['@claude']
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 02:44
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
- [x] #1 Activating an Actor Pack while a capture or restore holds storage waits or refuses with a named reason, and writes nothing under actor_packs/ or visual_identities/actor_packs/ meanwhile; a test is red before the fix
- [x] #2 The owner inventory's activation rows no longer say 'no participant token'
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd: activation still publishes with raw `_publish_immutable`/`_cleanup_published` and takes no storage admission between `secure_private_directory` and the DB commit (confirmed: no `Actor_Packs/*.py` imports Backup_Recovery).
2. Red test: activate a persona Actor Pack (shared-visual + persona-runtime sections) on a configured file-backed profile; begin the in-process capture/restore pause (`storage._begin_local_pause`) once the first asset is published, and assert the pause's drain cannot report idle while the activation is in flight and that the activation leaves nothing under `actor_packs/` or `visual_identities/actor_packs/`; plus a pause held before activation refuses with a named reason and writes nothing.
3. Follow the fixed-profile-directory precedent (`Backup_Recovery/agent_worktree_storage.py`, ADR-126 TASK-35582): a small Backup_Recovery owner function that takes ordinary storage admission for the activation's two publication roots, held from before the first write through the DB commit and any cleanup, so a pause refuses new activations and its drain waits for one in flight.
4. Activation translates a storage refusal (RecoveryRequired) into a named, path-free `ActorPackActivationError` category after cleaning up its published files; the Personas screen maps the category to copy.
5. Update the owner inventory's activation rows and the `persona.actor_packs` limits paragraph; ADR-126 note; run covering tests, census, preflight and ratchets.

## Implementation Notes

Premise re-checked on base 26b2e677dd: the cited lines are unchanged (`_prepare_character_sections` :401-436, `_prepare_persona_sections` :505-560, `_publish_immutable` :714, `_cleanup_published` :774) and no `Actor_Packs/*.py` module touched storage admission. Two further gaps showed up on the way: a pause makes the importer's store reads fail, which it reports as `actor_pack_import_review_stale` ("The actor or staged pack changed"), and a `RecoveryRequired` (a `RuntimeError`) escaping the body skipped `_cleanup_published`.

Approach (ADR-126 fixed-root precedent, TASK-35582; no new source bridge): `ActorPackActivationService.activate` now takes ordinary storage admission (`acquire_storage`) on `visual_identities/actor_packs/<digest>/` with `actor_packs/<digest>/` as a related path, held from before the first write through the commit or the removal of the attempt's own files. The old body moved unchanged into `_activate_admitted`, which also maps `RecoveryRequired` to the new path-free category `actor_pack_import_storage_unavailable` after cleanup. A refused admission maps to the same category, and so does any failure while this process holds a local pause (its refused store reads otherwise read as a stale review). The Personas screen gives the category copy; the User Guide import paragraph says so.

Result: a capture's or restore's local pause refuses a new activation before it writes; a pause that begins mid-activation cannot drain until the activation has removed its uncommitted files and closed its lease; another process's maintenance makes it wait for native admission.

Decisions: the Persona Visual / Shared Visual Identity bridges were not reused. They authenticate their own modules' callsites and candidate objects, and activation's immutable, call-owned files gain nothing from them (ADR-126 note 2026-11-13).

Red/green (`Tests/Architecture/test_backup_owner_inventory.py`):
- `test_actor_pack_activation_refuses_while_a_capture_or_restore_holds_storage`: red at base, `assert 'actor_pack_import_review_stale' == 'actor_pack_import_storage_unavailable'` (nothing was written either way). Green after the fix; it also pins the screen copy and the six inventory rows (AC2).
- `test_actor_pack_activation_in_flight_keeps_a_capture_drain_waiting`: red at base, the pause drained and left `visual_identities/actor_packs/<digest>/asset-0001.png` with no committed row (`assert [...asset-0001.png] == []`). Green after. Its early-drain check was already False at base, because the worker thread's own ChaChaNotes connection lease keeps a drain waiting; it stays as a guard.

The tests run in a lane-run file, so the census needed no change. Activation must also run under a configured profile; the worker uses `run_finite_local_worker` so its own DB connection retires, as runtime maintenance requires.

Files: `tldw_chatbook/Actor_Packs/activation.py`, `tldw_chatbook/UI/Screens/personas_screen.py` (+3 copy lines; `Tests/Architecture/test_module_size_ratchet.py` row raised 16835 -> 16837 by the owner rule "raise by the change's own lines"), `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md` (six activation rows and the `persona.actor_packs` paragraph), `backlog/decisions/126-complete-local-backup-and-recovery.md` (note), `Docs/User_Guide/roleplay-chat-dictionaries.md`.

Covering runs (load 64-88): the 3 Architecture files serially, 246 passed. `Tests/Actor_Packs` with `-n 4`: 239 passed, 2 failed. The two failures are `test_actor_pack_export.py::test_archive_matches_independent_golden_bytes[character|persona]`, export golden bytes, which fail the same against base activation.py. The UI/Packaging/Character_Chat activation consumers: 73 passed, 17 failed. All 17 are in `Tests/UI/test_actor_pack_creation_workflow.py` (`RecoveryRequired: raw_source_selection_changed`, the known local trap) and are identical at base.

