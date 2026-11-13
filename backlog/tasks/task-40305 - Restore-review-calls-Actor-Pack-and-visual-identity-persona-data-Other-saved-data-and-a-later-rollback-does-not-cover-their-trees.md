---
id: TASK-40305
title: "Restore review calls Actor Pack and visual-identity persona data 'Other saved data', and a later rollback does not cover their trees"
status: Done
assignee: ['@claude']
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:36
labels:
- follow-up
- backup
- recovery
- persona
priority: low
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- tldw_chatbook/Backup_Recovery/later_rollback.py
dependencies:
- TASK-39116
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream, TASK-39116). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Restore review shows Actor Pack persona images as "Other saved data", so the user cannot tell what a restore replaces. A later rollback of a restore that changed these trees does not get the persona-tree preservation checks persona artwork gets.

**Evidence.**
- `Backup_Recovery/recovery_service.py:77-104`: `_OWNER_LABELS` names `persona.assets` ("Persona artwork") and `persona.visual_identity_builtin`, but not `persona.actor_packs` or `persona.visual_identity`; `owner_label` (`:107-109`) falls back to "Other saved data".
- `later_rollback.py` handles only `persona.visual_identity_builtin` and `persona.assets` (`:247`, `:293`, `:371`, `:635`, `:667`, `:785`, `:1350`, `:1420`); nothing there, in `publication.py` or in `replacement.py` names `actor_packs`.
- The owner inventory states both limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore and rollback review name persona.actor_packs and persona.visual_identity with their own labels
- [x] #2 A later rollback of a restore that changed actor_packs/ or the visual-identity tree restores the prior files byte-equal or refuses with a named reason; a test covers it
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd (`_OWNER_LABELS` lacks both owners; later_rollback names only persona.assets / builtin).
2. AC1: add `persona.actor_packs` and `persona.visual_identity` to `recovery_service._OWNER_LABELS` (the single label source for restore review, recovery-copy summary and portable relocation review); test red first.
3. AC2: build a real later-rollback test of a replace restore that changed the actor_packs/ and visual-identity trees (Conversations group only, child processes as in the plugin and persona rollback precedents), and assert prior files come back byte-equal or the rollback refuses with a named reason. Fix later_rollback only if the observed behavior is neither.
4. Update the owner inventory limits paragraph; run covering tests, preflight and ratchets.

## Implementation Notes

Premise re-checked on base 26b2e677dd. `_OWNER_LABELS` lacked both owners, so `owner_label` returned "Other saved data". `later_rollback.py` names only `persona.visual_identity_builtin` and `persona.assets`.

AC1: added `persona.visual_identity` "Reaction artwork" and `persona.actor_packs` "Actor Pack persona artwork" to `recovery_service._OWNER_LABELS`. `owner_label` is the one label source for restore review (`summary`: group labels and `owner_labels`), recovery-copy inspection (the rollback review's contents) and portable-relocation review. Test: `Tests/Architecture/test_backup_owner_inventory.py::test_actor_pack_and_reaction_trees_have_their_own_review_labels`. It was red at base (`['Other saved data', 'Other saved data', 'Persona artwork', 'Bundled persona artwork']`) and is green now. The AC2 test also checks both labels in a real archive's `summary()['owner_labels']`.

AC2: no product change was needed; a test now covers the behaviour. `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py::test_later_rollback_of_a_restore_that_changed_actor_pack_trees` runs in one child process, as the selected-group service tests do. Each activation runs in its own process, so the orchestrator never imports config (TASK-40132 lesson). The steps:
1. Activate a persona Actor Pack (shared-visual and persona-runtime sections) through the real services.
2. Back up the Conversations and Workspaces groups. Workspaces is required: the populated workspace and agent-run stores link to the ChaChaNotes core, and a Conversations-only restore refuses `required_target_group_unavailable:workspaces`.
3. Activate a second pack.
4. Replace-restore. The second pack's files leave both trees.
5. Later-roll the restore back through `preview_rollback` and `RecoveryService.start_rollback`, with the credential-review abort/retry flow.

Every file under `actor_packs/` and `visual_identities/` comes back with identical SHA-256, and the final preview is complete. The two owners do not need the persona safety-scope checks, because they are ordinary restored tree owners.

Red evidence, under temporary edits to `later_rollback._preview` (the invariant dev already keeps), reverted afterwards:
- Dropping the trees' restore pairs made review refuse `target_changed`. The first draft of the test accepted any refusal and passed, so the test now pins the rollback.
- Also recomputing the fingerprint made `start_rollback` refuse `rollback_dependency_selection_required`, and the test failed.

The product's own validation stops a silent partial rollback before the byte check. The byte check stays as the end-to-end guard.

Not covered, and proposed as a follow-up: a restore that creates either tree in a profile that never had it. Its later rollback goes through `_created_destination_target` / `_known_absences`.

Placement: the test lives beside the persona.assets later-rollback precedent, which no PR lane runs. It takes 7-10 min at load 55-70, too slow for a PR lane. Adding it to `backup-recovery-qualification.yml` is a proposed follow-up; workflow edits were outside this brief.

Files: `tldw_chatbook/Backup_Recovery/recovery_service.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`, `backlog/docs/backup-recovery-owner-inventory.md` (the `persona.actor_packs` paragraph).

Runs: the label test serially, 1 passed. The AC2 test: green twice (585.9 s at load ~70, 440.2 s at load ~60) and red under both negative controls. `Tests/Backup_Recovery/test_recovery_service.py` and `test_group_issues.py` with `-n 4`: 40 passed.

