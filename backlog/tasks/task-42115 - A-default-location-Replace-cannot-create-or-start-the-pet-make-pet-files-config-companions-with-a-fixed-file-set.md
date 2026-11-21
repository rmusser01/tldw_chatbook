---
id: TASK-42115
title: "A default-location Replace cannot create or start the pet: make pet files config companions with a fixed-file set"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-20 18:48
updated_date: 2026-11-21 20:46
labels:
- follow-up
- backup
- pets
priority: high
references:
- tldw_chatbook/Backup_Recovery/config_participants.py
- tldw_chatbook/Backup_Recovery/settings_file_participants.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/publication.py
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split from TASK-42113 (round 18, coordinator decision 2026-11-20). TASK-42113 keeps the default-location reproduction (its refusal tests) and the notes on other owners. The changes below extend the guarded publication and rollback proofs, so they get their own stream and review.

Until then, the failure is fail-safe, and `test_created_owner_rollback.py::test_a_default_location_pet_replace_refuses_and_keeps_the_pets` pins it. At the default location:
- a Replace that would create a pet file refuses before publication (`replacement_destination_parent_required`);
- a Replace that retires the profile's pet file refuses at activation (`activation_profile_outside_maintenance`) and rolls back.

In both cases nothing is created, the profile keeps its pets, and the app starts.

**Still exposed:**
- **dev builds since TASK-42000:** an install whose registry already holds a dated pet copy beside `config.toml` as a root of its own stays exposed. That means a backup taken since TASK-42000 and before TASK-42114. Once the pet store rotates the copy away, every admission refuses and the app does not start. TASK-42114 stops new registrations but cannot remove old ones, so such an install needs AC#6 or the workaround.
- **main builds** (by reading main's code, TASK-42114): backups register the dated copies in the shared `~/.config/tldw_chatbook`, and main's pet store rotates them there, so a main install stays exposed for as long as it runs a main build. No TASK-42000 build rotates the shared folder, so upgrading ends the trigger. Only an install that already stopped on main needs AC#6 or the workaround.

The workaround is untested: recreate an empty file, mode 0600, at the missing path.

**2026-11-20, coordinator decision after review rev2:** deleted-file-root tolerance needs one design across all root consumers, the recovery journal included. The attempt is parked on `fu/r18-deleted-roots-wip` (tip `ee6d0ac966`, not pushed) and moves here, with TASK-42113's step A (`_bind_activation`, `bootstrap._binding`). It recorded each regular-file root with its folder's inode and let a deleted file root hold nothing, then scope its own exact path.

Three reviews in a row each found another root consumer that the per-site approach missed:
- **rev1, finalization:** `publication._finalization_session` required coverage of a root the publication roots had dropped, so a later rollback, its abort and a second Replace stuck in `recovery_required`.
- **rev1, exact-path ownership:** the abort (`replacement._abort_prepublication`) and containment could not cover the deleted root's own path.
- **rev2, crash recovery:** under a recovery journal, `Admission._groups` (and `_publication_roots`, which passes `deleted=()`) still refuses any missing root the journal did not move (`replacement._recovery_admission_aliases`: `recovery_root_unverified`). After a `[pets-none]` Replace at the default location and a second Replace killed at `finalize_candidate`, both `finish` and `rollback` failed, and startup was refused.

At the default config location (`~/.config/tldw_cli`), the config folder holds `recovery-bootstrap`, so it is never a root. Only `config.toml`'s own members and the three fixed leaves in `_CONFIG_CAPTURE_LEAVES` (`ui_state.toml`, `recent_emojis.json`, `runtime_policy.json`) have a path there through admission (`config_participants.sibling_selector`/`companion_guard`), capture (`_config_capture_sources`), publication (`_config_publication_scopes`) and later rollback (`_config_file_relations`). Pet files fall back to roots of their own. Evidence from TASK-42113's probes (scratch HOME, `RecoveryService(default_control_root())`):

- A Replace that would create `tamagotchi_pets.json` (`[pets-missing]`, `[pets-present]`, `[pets-ungrouped]`) refuses with `replacement_destination_parent_required` for the destination `~/.config/tldw_cli` (`replacement._register_publication_parents`).
- After a successful default-location Replace, the footer pet fails with `RecoveryRequired: storage_scope_not_enrolled`. The `pet` route admits the config folder itself (`raw_participants.scope`, `admission_paths` starting with the parent) and leases each dated copy on its own (`settings_file_participants.preflight`). At the default location, neither falls under a root.

Since TASK-42114, the pet store's dated copies are excluded from backups. Its review also made a restore skip the dated rows that older archives hold (`rotating_copy`, coordinator decision 2026-11-20), so neither capture nor restore handles a dated copy any more. A backup holds only two fixed pet files: `tamagotchi_pets.json` (local id `""`) and `tamagotchi_pets.seeded` (local id `seeded`). That fits the fixed-leaf mechanism once it allows an owner more than one fixed leaf.

Plan (TASK-42113's steps B to D):
- **B.** The `pet` route becomes a config companion: `sibling_selector` returns the selector for `tamagotchi_pets.json`. The preflight checks dated-copy members under the companion guard instead of leasing each one, and the config folder is no longer leased on its own.
- **C.** `_CONFIG_CAPTURE_LEAVES` generalizes from owner → leaf to owner → {local id: leaf}, with the pet owner's two leaves. Affected: `storage_admission._config_capture_*`, `publication._config_scope_candidates`/`_config_publication_scopes`, and `later_rollback._config_file_relations`, `_config_file_retirement_scopes` and `_created_config_file_targets` (whose owner map must include the pet adapter).
- **D.** The pet cases of `test_created_owner_rollback.py` at the default location: the scenario already takes `LOCATION`. They need laning in the owner suites.

Related, from TASK-42113:
- Every other config-folder owner hits the same creation refusal at the default location: `config.history` (`config.toml.bak`, `config_backup_<stamp>.toml`), `rag.definitions` (`rag_pipelines.toml`) and `notes.templates` (`note_templates.json`). This predates TASK-42000. `config.history`'s dated snapshots have the variable-name problem pets had before TASK-42114, so those owners are left to a separate task. Retiring them, and creating them later from the app, were not probed.
- `bootstrap._binding` (a bound profile's own binding check) is part of the same design. Changing it means updating the `bootstrap.py` digest that `Tests/Backup_Recovery/test_config_companion_missing_record_evidence.py` pins.
- Until the design lands, a registered single file that the user deletes still stops every admission (TASK-42114's AC#4 notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At the default config location, a Replace completes for every pet case the separate-folder suite covers ([pets-missing], [pets-present], [pets-dated-only], [pets-none], [pets-ungrouped]), with the same outcome as on a separate config folder, including the later rollback; each case is red before the change
- [x] #2 The footer pet starts and saves after a default-location Replace, with no RecoveryRequired; a test is red before the change
- [ ] #3 The fixed config leaves keep their current behaviour at both locations (the config sibling and companion suites stay green), and pet files are never registered as roots of their own at either location
- [x] #4 The default-location pet cases run on a PR or dev-push lane, with their times recorded
- [ ] #5 The notes say whether config.history, rag.definitions and notes.templates should join the fixed-leaf set, and a task is filed for them if they should
- [ ] #6 One design for a deleted (absent) file root covers every root consumer: admission grouping, capture naming, maintenance sessions, publication and finalization coverage, containment, Replace activation (step A), a bound profile's binding check, the abort, and the recovery journal (finish and rollback after a crash). It has tests, each red first, for the next Replace, a later rollback, an abort, finish and rollback after a crash, and startup, at the default location and on a separate folder. An absent folder root (an unmounted drive) still refuses
<!-- AC:END -->

## Implementation Plan

**One deleted-file-root rule (AC#6).** A registered root is a *deleted file root* when all of these hold: it was registered as a regular file with its folder's evidence (`Admission.register` records `file:<folder inode>:<root>`), it is absent, its folder is present, is not a mount point and is still the registered folder (inode, checked twice). One predicate, `effective_roots.deleted_file_roots(roots, entries)`, decides it over the whole registry. Such a root **holds nothing and scopes exactly its own path** (as a present file root scopes its object), journal or not; grouping keeps its recorded history. Everything else stays strict: a folder root, a root registered before the evidence existed, a file whose folder is gone, replaced or a mount point (an unmounted drive), and any root while a remap is pending.

Root consumers (grep of `Backup_Recovery/` for the registry, `_tokens`, `_groups`, `session._roots`, `_publication_roots`, `_contains_owned_path`, `_contains_capture_path`, `_binding`, `_bind_activation`, `_finalization_session`, `recover_replacement`, `_capture_names`, `_refresh_roots`), and the rule each gets:

| # | Consumer | Rule |
|---|---|---|
| 1 | `Admission.register` | records the folder evidence for each regular-file root |
| 2 | `Admission._groups` (every admission, `register`, `pause_requested`, `_admit`'s recheck) | a deleted root adds no fresh tokens; its history still groups. Under a recovery journal the missing roots go to the journal first (rule 5) |
| 3 | `Admission._publication_roots` (maintenance publication roots; finalization coverage reads them) | a deleted root is `(folder/name, file)`; under a journal, only one the journal did not alias |
| 4 | `replacement._recovery_staging_roots` (`_admit` under a journal, and the live finalize at ~1680) | a deleted root the journal did not alias keeps the name `folder/name` |
| 5 | `replacement._recovery_admission_aliases` (recovery journal: finish, rollback, abort after a crash) | a root this journal moved keeps the journal's move proof; a deleted root it never moved is passed over (the caller treats it as holding nothing) instead of `recovery_root_unverified` |
| 6 | `storage_admission._contains_owned_path` (so `_contains_capture_path` and every caller: `_scope`, sessions, staging, preserved groups, `_checked_originals`, `_abort_prepublication`, `_bind_activation`, later rollback) | an absent root scopes exactly its own path when it is a deleted file root; otherwise nothing (today's rule) |
| 7 | `MaintenanceSession._refresh_roots` / `_capture_bound_sources` | sessions keep deleted roots in `_roots`/`_all_roots` (rule 6 handles containment); the staging-overlap check resolves a deleted root without `strict` |
| 8 | `bootstrap._binding` (startup permission, `_scope`, Replace/rollback binding checks, `Evals/recovery.py`) | a deleted root is skipped by the strict resolve |
| 9 | `capture_service._capture_names` (backup, Replace's first binding and capture, later rollback, service rollback) | a deleted root covers nothing and is not resolved |
| 10 | `control_records._bind_activation` (Replace activation, step A) | no own rule: physical roots keep the deleted root; rule 6 contains its path in the session |
| 11 | `publication._finalization_session` (finish, rollback, abort, credentials) | no own rule: coverage reads rule 3's publication roots |
| 12 | `replacement._abort_prepublication`, `_checked_originals`, later rollback's `_contains_owned_path` checks | no own rule: rule 6 |
| 13 | `effective_roots` / `check_redundant_profiles` | unchanged: without a journal a file inside an enrolled folder root still meets the alias proof (and its profile co-ownership check) first; a journal skips that proof, so there rule 5 decides |
| 14 | string-only consumers (profile/registry mapping equality, `_overlap`, `_absence_names`, activation witnesses, generation witnesses, `register_pending`, `bind_profile` overlap) | unchanged: lexical, no filesystem read |
| 15 | `_metadata_evidence` (evidence reuse) | unchanged: an absent root disables reuse, so admission falls back to the full derivation (slower, not wrong) |
| 16 | `Admission.remap` (no production caller) | unchanged: strict |
| 17 | `preserved_groups._read_scope` (a Replace or rollback's preserved-group check) | found by the tests: a fixed config leaf created after the binding is read as a companion of a selector the session holds (`storage_admission._session_config_sources`) |
| 18 | `later_rollback._stage_read_sources` / `_builtin_stage_names` | found by the tests: a fixed config leaf is proven as a config file, not stage-read as a created file under a root |

`finish` after a crash is TASK-42154's; note whether this changes it, do not fix it.

**Pets as config companions (steps B-D).**
1. `_CONFIG_CAPTURE_LEAVES` becomes owner → {local id: leaf}; `tamagotchi.config` gets `""` → `tamagotchi_pets.json` and `seeded` → `tamagotchi_pets.seeded`. Update `storage_admission._config_capture_sources`, `publication._config_scope_candidates`/`_config_publication_scopes`, `later_rollback._config_file_relations`/`_config_file_retirement_scopes`/`_created_config_file_targets` (its owner map gains the pet adapter; discovery may list more than the one file).
2. `config_participants.sibling_selector` returns the selector for the `pet` route's `tamagotchi_pets.json`; under the companion guard, `settings_file_participants.preflight` checks the dated-copy members as companions instead of leasing each one, so the config folder is no longer leased on its own for a bound profile.
3. Tests, each red first where it is a fix, at the default location (scratch HOME, `RecoveryService(default_control_root())`) and on a separate folder: the five pet cases with their later rollback; the pet starts and saves after each; after a retired pet root, the next Replace, a later rollback, an aborted rollback, a backup, and a crash at `finalize_candidate` then `rollback` then startup; unit cases for the predicate (absent folder root, folder gone/replaced, mount point, pending remap, no evidence).
4. Lanes and times, docs (Backup-and-Recovery, architecture, owner inventory), ratchets, preflight, ruff gates.

## Implementation Notes

**Summary.** At the default config location a pet Replace now completes for all five cases, with the later rollback, and the footer pet then starts and saves. Two pieces, in two commits:
- **One deleted-file-root rule (AC#6).** `Admission.register` records each regular-file root's folder (`file:<folder inode>:<root>`). `effective_roots.deleted_file_roots` decides, over the whole registry, that a root is a deleted file root: registered with that evidence, absent, its folder present, not a mount point and still the registered folder. Such a root holds nothing and scopes exactly its own path at every consumer in the plan's table.
- **Pets as config companions (steps B-D).** `_CONFIG_CAPTURE_LEAVES` is owner → {local id: leaf}, with the pet file and its seed-once marker. The pet route's `sibling_selector` returns the selector, and dated copies are checked under the companion guard.

**Decisions and deviations from the plan.**
- *Journal first.* Under a recovery journal, a missing root goes to the journal first. A root the journal moved keeps its move proof, so tampered retained evidence still refuses (`test_same_session_retirement_capture[retained]`). `_recovery_admission_aliases` passes over only a deleted root the journal never moved (the rev2 site). "Deleted first" would have skipped that proof.
- *Contained roots.* The WIP excluded a file inside an enrolled folder root. Under a journal, `effective_roots`' alias proof is skipped, so on a separate folder a crash after a retired pet file failed with `recovery_root_unverified` (red at base too). The predicate now covers contained files. Without a journal the alias proof, with its profile co-ownership check, still runs first (`test_effective_admission_roots` unchanged apart from the standalone case).
- *Found by the tests (rows 17, 18).* `preserved_groups._read_scope` refused a leaf created after the binding (`preserved_group_native_scope_required`; the store writes the marker on start). It now derives the leaves of the selectors the session holds (`_session_config_sources`). `later_rollback._stage_read_sources` treated a created pet file as a stage-read created file under a root (`local_snapshot_builtin_scope_unverified`). It now leaves fixed leaves to the config-file proof.
- *The marker has no row while absent.* Listing it as `unused` broke seed-once: the archive's unused row made Replace retire the target's marker, and the store re-seeded the old shared pets (`[pets-none]` failed on `pets_after_restore`). Reverted. A marker a Replace creates is accepted without a target row in `_config_publication_scopes` and `_config_file_relations`.
- *A legacy dated root* (registered by a backup before TASK-42114) that still exists is still leased in the pet preflight, as before. Checking it as a companion would refuse every save on a separate folder.
- *Evidence reuse.* While a deleted root exists, `_metadata_evidence` returns None, so admission takes the full derivation (slower, not wrong).
- *Roots registered before this change* carry no folder evidence and stay strict. Such a root cannot be told apart from a folder root on an unmounted drive. An install already stopped by one still needs the untested workaround.

**Touched in replacement.py / publication.py** (TASK-42151/42154 stream): `_recovery_admission_aliases` (the pass-over), `_recovery_staging_roots`, `publication._config_scope_candidates`/`_config_publication_scopes`. Not touched: `_first_profile_container`, `recover_replacement`, `_acquired_source`, `publication._descriptor`. **TASK-42154:** not run. By reading, `finish` after a crash now passes admission with a deleted root and reaches TASK-42154's own failure (`_descriptor`'s missing staged artifact), instead of `recovery_root_unverified`.

**AC#3.** Once a profile is bound, its pet files never become roots. The scenario asserts that the only pet roots are the target's own files its first binding found. A backup of a profile that is not bound yet, and a profile's first binding, still register every captured config-folder file as a root of its own: `config.toml`, the three fixed leaves and the pets, at both locations (`_config_capture_sources` needs a binding). That predates this task. Making pets "never" roots there needs unbound companions in capture and in `_ensure_first_bindings`, which would also change the three leaves' current behaviour. Left for the coordinator; see the report.

**AC#5.** `notes.templates` (`note_templates.json`, one fixed name, `note_templates` raw route) and `rag.definitions` (`rag_pipelines.toml`, one fixed name) fit the fixed-leaf set and should join it. `config.history` should not, as it stands. Its `.bak` is already a config-route member, but its dated `config_backup_<stamp>.toml` snapshots have variable names, like the pet's dated copies before TASK-42114. They need their own design, either exclusion like TASK-42114 or a pattern rule. Proposed as a follow-up in the round report; this task does not file tasks.

**Tests.**
- `test_created_owner_rollback.py`:
  - `test_a_default_location_pet_replace_completes_and_rolls_back[5 cases]`: Replace, the pet's start and save, and the later rollback at the default location.
  - `test_the_operations_after_a_retired_pet_root[default|separate]`: after a `[pets-none]` Replace, a startup check follows each step: a backup; a second Replace killed at `finalize_candidate` (child `os._exit(17)`, a new `crash` role) then `rollback`; the next Replace; its later rollback, which must stop for review once and be aborted; then the pet starts and saves.
  - The five pet cases now start and save the pet at both locations. The scenario asserts that a bound profile's only pet roots are the target's own files, and it names any refusal behind `backup_operation_failed`.
- `test_config_participant_lifetimes.py`, unit tests of the predicate and its consumers:
  - absent folder root, folder gone, folder replaced, mount point, no evidence, pending remap;
  - exact-path scope;
  - a bound profile's binding and scope;
  - capture naming and session publication, containment and staging roots.
- Updated: `test_same_session_retirement_capture.py` (an evidence case each), `test_effective_admission_roots.py` (standalone alias registered without evidence) and the `bootstrap.py` digest pin.

**Red/green.**
- At base `dc371f0842`, all five default cases were red: `replacement_destination_parent_required` for missing, present and ungrouped; `activation_profile_outside_maintenance` for none and dated-only. The default chain was red at activation. The separate chain was red with `recovery_root_unverified` at the crash rollback. The unit cases were red on assertions or `FileNotFoundError`.
- Negative controls, one rule reverted at a time, each red on the cases named:
  - evidence ignored: `[file_without_evidence]`, `[folder_root]`, `[False]` cases;
  - mount check removed: `[file_under_mount_point]`;
  - pending check removed: the remap case;
  - containment rule: the scope, bound-profile and capture-naming cases, and default `[pets-none]` (`activation_profile_outside_maintenance`);
  - `_binding`, `_capture_names`, `_capture_bound_sources`, `_publication_roots`: their cases;
  - journal pass-over: `[unrelated_deleted_file]`, and the default chain, whose second Replace failed in its live `_recovery_staging_roots`.
  - The folder inode re-check (a swap race between two pins) has no forcing test; a replaced folder is caught by the evidence token.

**Lanes and times (AC#4).** `test_created_owner_rollback.py` runs whole in the Backup_Recovery owner-suites dev-push lane. In that lane's run (`-n 4`, load 24-45):
- the five default cases: 97-133 s each;
- the chains: 169 s (default) and 181 s (separate).

The unit cases are in `test_config_participant_lifetimes.py`, already a whole-file lane target.

**Commands.**
- The owner-suites lane's file list, `-n 4 --timeout=1500`: 1075 passed.
- `Tests/Architecture`, serially: 1488 passed, 4 failed. `test_no_new_upward_import` was mine; it was fixed in the third commit and is green. The other three failures (diagnostic inventory, two timer-path inventory tests) are red at base with the same failure list.
- Covering in-process suites (companion, sibling, lifetimes, admission, operational owners, preserved groups, same-session, config-file retirement): green. Two exceptions are red at base too: `test_retained_config::test_fresh_process_finish_and_abort_preserve_config[finish]` (TASK-42154's shape, on `ui_state.toml`) and two `test_settings_file_participant_lifetimes` cases.
- `preflight.sh`, `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py`: pass.

**Files.**
- Source: `Backup_Recovery/effective_roots.py`, `admission.py`, `bootstrap.py`, `capture_service.py`, `replacement.py`, `storage_admission.py`, `publication.py`, `later_rollback.py`, `config_participants.py`, `settings_file_participants.py`, `preserved_groups.py`.
- Tests: as listed above.
- Docs: `Docs/Backup-and-Recovery.md`, `Docs/Architecture/backup-recovery.md`, `backlog/docs/backup-recovery-owner-inventory.md`, `backlog/docs/lessons-testing-evidence.md`.

**Not met.**
- AC#3: an unbound profile's backup still registers pet files as roots (above).
- AC#5: the follow-up still needs filing.
- AC#6: `finish` after a crash is TASK-42154's (brief).
