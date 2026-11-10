---
id: TASK-39116
title: "About 110 Plugins, Actor_Packs and Personal Context producer rows sit in the conservative unsupported default with no per-owner backup or exclusion decision"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 18:23
labels:
- follow-up
- backup
- needs-owner-decision
priority: low
references:
- backlog/docs/backup-recovery-owner-inventory.md
- tldw_chatbook/Plugins/
- tldw_chatbook/Actor_Packs/
- tldw_chatbook/Personal_Context/key_protector.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-backupinv, whose TASK-39053 notes list these cohorts as "observations for later cohort work". Counts are from the branch's inventory doc; line numbers are dev's.

| Package | Producer rows | In `unsupported` / miscellaneous | Other rows |
|---|---|---|---|
| Plugins | 66 | 61 | 2 `plugins.registry`, 2 memory, 1 process artifact |
| Actor_Packs | 33 | 30 | 3 memory |
| Personal_Context | 19 | 15 | 3 `personal_context.repository`, 1 `personal_context.interview_drafts` |

- `unsupported` is the review's conservative default, not a decision about each owner.
- For Personal Context the decision has a constraint: its database is encrypted with a keyring-held key (`KeyringProfileKeyProtector`, `Personal_Context/key_protector.py:28`, `:127-200`). A backup without the key cannot be read.
- A `PassphraseProfileKeyProtector` that writes a key-bundle file also exists (`:209-300`), so not every custody mode is keyring-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides a cohort for each of the three packages (capture, exclude, or stays `unsupported`), including whether Personal Context key material travels with its database; the decisions are recorded and the doc rows follow them
- [x] #2 Any owner moved to "exclude" has an installed exclusion that a backup preview shows (precedents: `_DeviceStateExclusion`, `ProjectIndexCacheDeclaration`), with a test red before (N/A — no owner was moved to "exclude": Plugins and Actor_Packs are capture, and Personal Context keeps its 2026-11-06 capture ruling)
<!-- AC:END -->

## Implementation Plan

1. Record the owner's 2026-11-10 decisions verbatim (Plugins: capture all, TASK-40132; Actor_Packs: capture; Personal Context: the 2026-11-06 capture ruling stands, design in TASK-39148).
2. Verify the survey: actor_packs/<digest>/persona-runtime files, their persona_visual_assets rows, the persona.assets validator failure, group and max_bytes tables.
3. Test first: a profile whose persona_visual_assets row points into actor_packs/ previews those files as included under a named owner in the persona group and validates; red on base.
4. Install an `actor_packs/` asset owner beside persona.assets (same group, limits, core dependency edges), and make persona.assets accept rows under that folder by leaving them to the new owner.
5. Reclassify the Actor_Packs producer rows to their owners; leave Plugins (TASK-40132) and Personal Context rows unchanged; AC#2 N/A (no owner moved to exclude).

## Implementation Notes

**Owner decisions, 2026-11-10** (also in the doc's new "Plugins, Actor Packs and Personal Context cohorts" section):
- Plugins: "Capture all". The option as offered read: "Full capture: registry schema catalog, quiescing plugin processes, an execution gate. Large, and a restore still lands in recovery_required without the keyring marker." Implementing it, including the Plugins rows (its AC#6), is TASK-40132 on branch fu/r13-plugins this round; no Plugins code or rows changed here.
- Actor_Packs: "Capture (Recommended)". The option read: "Capture actor_packs/ with the persona asset group and fix the persona.assets validator so it accepts that folder (it currently fails on it). Small immutable images and JSON, no secrets." Implemented here.
- Personal Context: the 2026-11-06 ruling "Capture it" stands (quoted in TASK-39148); the capture waits for TASK-39148's design. Personal Context code and rows are unchanged.
- Key material: no decision beyond the 2026-11-06 ruling ("design key custody so a restore can read it"), which TASK-39148 carries.

**Survey check.** Verified at ae9f9e8d21: `_prepare_persona_sections` (`Actor_Packs/activation.py` :505-560) publishes `asset-NNNN.<ext>` and `manifest.json` under `<user data>/actor_packs/<digest>/persona-runtime/` and stores `actor_packs/...` locators in `persona_visual_assets`; `_Assets` resolves persona.assets locators against `persona_visual/` (`_path`, and `relative_to(self.leaf)` in the restore check); the group table is `data_groups._GROUP_OWNERS["conversations"]` and the limit `storage_admission._RAW_RECOVERY_LIMITS`. Two more places needed the owner: the ChaChaNotes core's optional-dependency queries in `DB/recovery_core.py` (discovery and validation) and the detached layout's validated owner set.

**Capture.** `Persona_Visual/recovery.py` installs `persona.actor_packs`, a second `_Assets` tree owner rooted at `actor_packs/`; the two persona owners split `persona_visual_assets` rows by first path component, so persona.assets accepts a profile with Actor Pack rows and still fails on a stray locator. The owner joins the `conversations` group, takes the 256 GiB raw limit, is a core dependency only when it has rows, and goes through the detached layout's validated discovery.

**Red/green.** `test_actor_pack_persona_images_preview_and_validate_with_persona_artwork` (in the lane-run `Tests/Architecture/test_backup_owner_inventory.py`) failed at base with `assert {'unknown'} == {'persona.actor_packs'}`; with that assertion moved first, base also showed persona.assets' `('...:persona.assets:references_unavailable', 'unsupported')`. It passes now, including dependency validation for both persona owners and the core, restore-topology validation, and a changed asset turning `unsupported`. Its core-edge assertion went red (`{'persona.actor_packs', 'persona.assets'} == {'persona.assets'}`) with the new core optional query removed, which is the bug the first draft had: 46 Backup_Recovery nodes failed with `dependency_unavailable` / `unknown_producer_dependency` until the query was added, and all 46 pass now.

**Rows.** Actor_Packs' 30 `unsupported` / `miscellaneous` rows now name owners: activation producers `qualified` under persona.actor_packs / persona.visual_identity; import staging `unsupported` under the installed `actor_packs.import_staging`; archive reads `external_input` / `external`; export publication `external_input` / `external-notes-workspace-or-export`; the archive writer `generic_boundary` (caller's stream). The three image-decode `memory` rows are unchanged.

**Limits recorded in the doc** (not fixed here): activation writes take no participant token; the owner is outside the later-rollback safety scope (as persona.visual_identity is); it has no recovery-review label ("Other saved data").

Files: `tldw_chatbook/Persona_Visual/recovery.py`, `tldw_chatbook/DB/recovery_core.py`, `tldw_chatbook/Backup_Recovery/data_groups.py`, `tldw_chatbook/Backup_Recovery/storage_admission.py`, `tldw_chatbook/Backup_Recovery/detached_layout.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md`.
