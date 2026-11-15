---
id: TASK-40134
title: "A backup captures Personal Context's repository database"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 17:13
updated_date: '2026-11-13 17:30'
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Backup_Recovery/config_adapter.py
- tldw_chatbook/Backup_Recovery/credential_policies.py
- tldw_chatbook/Backup_Recovery/credentials.py
- tldw_chatbook/Backup_Recovery/data_groups.py
- tldw_chatbook/Sync_Interop/recovery.py
- tldw_chatbook/DB/private_sqlite.py
- backlog/docs/backup-recovery-owner-inventory.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** Today the repository is `_LocalSQLite(..., "personal_context_repository_capture_unsupported", excluded=False)` (`Backup_Recovery/config_adapter.py:1163-1176`).
- A present file is an `unsupported` row, which blocks every backup preview of a profile that has used Personal Context (`config_adapter.py:705-712`, `inventory.py:57`).
- Owner ruling, 2026-11-06: "Capture it: back up the database and design key custody so a restore can read it."

**Approved design (decisions 1, 3, 7, 8, 9).**
- **Capture the file as it is.** It is ordinary SQLite with ciphertext columns, so it can be copied and schema-validated without the key (`Personal_Context/crypto.py:39-71`, `repository.py:246-261`).
  - Copy it lock-safely like `Sync_Interop/recovery.py`'s `_SyncAdapter` (`:49-90`).
  - Its private-SQLite policy lacks `centralized_backup_allowed` / `recovery_capture_allowed` (`DB/private_sqlite.py:470-475`).
  - Its schema version lives in `personal_context_schema`, not `user_version` (`repository.py:229-234`; `Backup_Recovery/sqlite_validation.py:1150-1188`).
  - The adapter replaces the `_LocalSQLite` row. Registering both under one owner id raises `conflicting_installed_owner` (`owner_registry.py:84-86`).
- **Group.** The database goes in a new selectable "Personal Context" group (`data_groups.py`). `Tests/Backup_Recovery/test_data_groups.py:353-354` lists the owner as non-selectable today.
- **Key.** The profile key travels as a `personal_context` credential record, referenced by `(personal_context.repository, profile_meta, key_account)` from TASK-40133.
  - It is captured only in `include` and `rollback` modes, both of which force encryption (`capture.py:301-302`).
  - The key is never in a readable archive.
  - A database whose reference is still empty carries no key record.
- **The reference is not a secret.** Exclude mode is the default, and it rewrites every `SQLITE_CREDENTIAL_COLUMNS` owner's database in staging (`credentials.py:1100-1101`). It turns a citation column into `recovery:setup_required` (`credentials.py:704-706`).
  - The Personal Context reference must be kept unchanged there, never sanitized. Otherwise a later same-device restore of a key-less backup would lock for good, contrary to decision 3.
  - A backup without credentials carries the database alone, and its preview says its key is not in the backup.
- **Restore before readable restores ship.** This task does not install keys, so no restore publishes the repository yet.
  - A replace restore, including one with the default "Everything" selection, keeps the live Personal Context repository out of publication. The live database is left byte-for-byte unchanged, and the archive keeps the captured copy.
  - The restore plan carries one line saying Personal Context will not be restored.
  - Today the row is `unsupported`, so restores never touch the live database. This keeps that property.
    - Stamping a marker instead would lock a readable profile with no way back: a later rollback is itself a replace restore (`later_rollback.py`), and a locked service refuses Remove and Start fresh (`Personal_Context/service.py:184-187`).
  - Isolated restores likewise leave the repository out.
  - This keeps the task safe on its own.
- **Link keys.** The device wrapping key, staged integrity keys and dataset staging keys are never captured (decision 5). A backup that includes credentials discloses that they are left out, as it already does for skill-trust keys (`credentials.py:1052-1065`).
- **Interim sync guard.** A restored link must not resume with stale cursors.
  - The repository's adapter is `activation_required`, and every restore binds all activation-required owners in `finalize_candidate` (`publication.py:592-604`).
  - So Personal Context sync (`app.py:2625-2662`) can stay off until the restored generation is reviewed, through `activation.py:224-227`.
  - This guard holds on its own.
  - Its review approves `personal_context.repository` only after a restore that published neither `personal_context.repository` nor `runtime.sync_state`. After one that published either, sync stays off, with a named reason shown to the user, until a review that clears the link state exists.
  - Personal Context sync also needs the shared owners, which this review never approves (see the separate `runtime.sync_state` investigation). So its copy must not claim that sync resumes.
  - The review approves only `personal_context.repository`. It never approves `config` or `runtime.sync_state`, which every Sync V2 call also needs (`Sync_Interop/server_sync_service.py:85-88`; approval is per owner, `activation.py:351-384`). Those stay with their own review.
- **First link in flight.** Capture is refused while a `first_link_freeze` or `first_link_rebaseline_commit` row exists (`repository.py:214-226`, `:311-320`).
- **Device-only records** are included, as **Export recovery copy** already does (decision 8).
- **Detached imports.** A `_SQLiteDeclaration` owner is picked up by the detached layout automatically (`detached_layout.py:636-642`, `:1080-1085`). Without a guard, that would import the repository into an archive without its key. Decision 10 refuses it there by name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backup preview of a profile holding the repository database (with its `-wal`) lists it as captured in a selectable "Personal Context" group, and the preview is no longer incomplete because of it (red first)
- [x] #2 The archive holds a consistent copy of the database, including device-only records, that passes the installed exact-schema validation for each v9 layout from TASK-40133
- [x] #3 A backup that includes credentials carries the profile key as a `personal_context` credential record, and capture checks that the key opens the captured copy (its manifest authenticates) before writing the record
- [x] #4 A backup without credentials carries no key material and keeps the database's key reference exactly as it was (it is not sanitized); its preview says the key is not in the backup
- [x] #5 After a replace restore with the default "Everything" selection, the live Personal Context repository is byte-for-byte unchanged, the restore plan said beforehand that Personal Context would not be restored, and the archive still holds the captured copy; an isolated restore likewise leaves the repository out, and its plan says so (red first)
- [x] #6 Inert extraction cannot write the key to disk from any archive; a test pins it
- [x] #7 Capture is refused with a named reason, shown in the preview, while a first-link freeze or rebaseline marker exists
- [x] #8 The review of a backup that includes credentials says the device wrapping key, staged integrity keys and dataset staging keys are left out, as it does for skill-trust keys
- [x] #9 After any restore, Personal Context sync does not resume until the restored generation is reviewed; this guard holds on its own (red first)
- [x] #10 That review approves only `personal_context.repository`, never `config` or `runtime.sync_state`, and only after a restore that published neither `personal_context.repository` nor `runtime.sync_state`; its copy does not claim that Personal Context sync resumes; after one that published either, Personal Context sync stays off and the user sees a named reason saying so
- [x] #11 A detached source holding the repository is refused by name and never imported without its key
- [x] #12 The owner inventory doc's Personal Context rows and the Personal Context User Guide page's backup paragraph describe the new behaviour
<!-- AC:END -->

## Implementation Plan

1. Capture owner: a `_SQLiteDeclaration` adapter in `Personal_Context/recovery.py` (the module TASK-40133 started), validated against the six v9 catalogs with the version read from `personal_context_schema`, copied through a new backup-allowed private-SQLite policy, registered at the tail of the installed factories; the `_LocalSQLite` row goes. New selectable "Personal Context" group.
2. First-link refusal: discovery marks the repository unavailable plus a marker row while a freeze or rebaseline row exists; inventory names `personal_context_first_link_in_progress`, partial capture omits it, and the preview shows its copy (the Plugins writers-active pattern).
3. Key: a `personal_context` credential column on `profile_meta.key_account`. Include/rollback capture reads the named keyring entry and writes the record only after the key authenticates the captured copy's manifest; empty references, removed profiles and exclude mode carry no record, and exclude mode never rewrites the database. Include mode also discloses the link keys it leaves out. Restore keeps the record retained (no install).
4. Restore: archive Personal Context payloads are never planned for publication in replace or isolated mode (also not offered as a restore group); the plan carries `personal_context_not_restored`, and the live database is preserved.
5. Sync guard and review: Personal Context sync loading checks `execution_allowed` for `personal_context.repository` and `runtime.sync_state`; a review approves only `personal_context.repository`, and only for generations whose plan published neither owner, else it refuses with a named reason. Settings shows it.
6. Detached sources holding the repository are refused by name.
7. Tests (red first where asked) in the Backup_Recovery / Architecture / Personal_Context files that already cover these seams; owner inventory doc and User Guide paragraph.

## Implementation Notes

**2026-11-12 (round-13 closeout): ADR-203.** ADR-203's model-disclosure grants have not shipped: the ADR is "Accepted (design only)", and `tldw_chatbook/` has no grant code. Grants are "encrypted, peer-local and never synchronize/export" (ADR-203 :30), and "Recovery never imports local grant authority" (:85). When they ship, their storage must stay out of the captured repository database, or a restore must drop them, with a test that pins it.

**2026-11-13 (round 14, implementation).** The repository is a captured owner; no restore publishes it yet.

- **Owner.** `Personal_Context/recovery.py` adds `_Repository`, a `_SQLiteDeclaration` (activation-required, optional default) over `<user data>/tldw_chatbook_personal_context.db`. It validates against the six v9 catalogs from TASK-40133 with the version read from `personal_context_schema` (also taught to `sqlite_validation._version`), copies through a new backup-allowed private-SQLite policy `recovery.personal_context.repository` (WAL frames fold in; device-only rows included), and `relocate` refuses. Registered at the tail of `owner_registry.install_adapters` and `sqlite_validation._installed_owner`; the `_LocalSQLite` row is gone. New selectable group `personal_context` ("Personal Context").
- **First link in flight.** A freeze or rebaseline row makes discovery return the database as `unavailable` plus a `:first_link` marker; `inventory.discover` adds `personal_context_first_link_in_progress` after group selection, partial capture accepts it, capture refuses before copying, and the backup preview shows the copy (the TASK-40132 plugin-writers pattern).
- **Key.** `SQLITE_CREDENTIAL_COLUMNS` gains `(personal_context.repository, profile_meta, key_account, personal_context)`. Include/rollback read the named keyring entry and write the `personal_context` record (not remappable, so it is retained, never installed) only after `repository_keys.profile_key_opens` authenticates the captured copy's manifest with it; a non-authenticating key becomes `credential_unreadable`. An empty reference, a removed profile or a marker carries no record. Exclude mode never reads a key and skips the database rewrite entirely, so the reference stays byte-for-byte; the preview then says the key is not in the backup. `_material` accepts the kind only when its account is the one the archived database names and the value still opens it. Include mode adds `credential_personal_context_link_keys_excluded:<id>`, shown with copy in the credential review, as skill-trust keys are.
- **Restore.** `restore_plan.plan_restore` drops every Personal Context root from the destinations (and from group selections) in both modes, so the live database is preserved, and adds `personal_context_not_restored`, which the restore review renders with copy; `available_group_ids` never offers the group for restore.
- **Inert extraction.** It showed and accepted the capture writer's own dependency group for `credential-recovery.json`, so manual extraction could write the key (and every server credential) to disk. `preview_inert_extraction` now refuses any group holding credential material, and the inspection summary no longer lists one.
- **Sync guard and review.** `app._load_personal_context_sync_runtime` first calls `recovery.require_sync_allowed()`: `execution_allowed(("personal_context.repository", "runtime.sync_state"))` for the selected profile (path `None`, as Sync V2's `_sync_call` does), so no unreviewed restore resumes Personal Context sync. `recovery.approve_restore_review` approves only `personal_context.repository`, only for generations whose restore plan (read from the operation journal; unreadable counts as published) published neither owner, and otherwise raises `personal_context_restored_link_state`. Settings > Personal Context shows **Review restored profile** with copy that keeps sync off, or the named reason.
- **Detached.** `detached_layout` refuses a source holding the repository or any of its sidecars as `detached_owner_unavailable:personal_context.repository` (TASK-40138 adds the user copy and the drafts row).
- **Docs.** Owner inventory (Personal Context section at the end, registry and producer rows), SQLite private-owner inventory (C100/B42, C52 text), User Guide Privacy notes (backup and restore paragraphs).

**Keyring isolation.** Credential tests patch `credentials._credential_store` to the in-process `SecureKeyring` dict; repository fixtures use `InMemoryProfileKeyProtector` or that dict through `KeyringProfileKeyProtector`; the activation test's subprocess inherits the null keyring from `Tests/conftest.py`.

**Evidence.** The 34 new nodes were run against the TASK-40133 commit's source (tests kept): 31 fail, most on assertions (`DID NOT RAISE`, missing copy, regex mismatch on `detached_unknown_content`, `'unsupported' == 'included'`, `PC root in plan.restore`), some as errors where an API is new (`recovery.SYNC_REVIEW_REQUIRED`). The 3 that pass there pin invariants base already keeps (exclude mode leaves an untracked database alone; repository-only extraction), so they carry negative controls on the new code, each red on an assertion and then reverted: exclude mode rewriting the database (2 nodes), recording without authentication (1), loader without the gate (1), review ignoring published owners (3), first-link check removed (2), restore hold-back removed (2).

**Suites.** `Tests/Backup_Recovery` in full (7,015 nodes, `-n 4`, load 30-45): 391 failed and 37 errors. The same 428 nodes on base 26b2e677dd: 350 still fail there. Of the 78 that passed on base, a rerun on this branch at lower load passed 61; the other 17 (14 `test_created_persona_subtree_rollback` setups hitting their subprocess timeout under xdist, `test_rag_projection_capture[foreign]`, `test_generated_media_capture`, `test_storage_initialization_wait[path-drift]`) all pass serially. `test_data_groups` was updated for the new group. Also green: `Tests/Personal_Context`, the Personal Context Sync/App/Agents/Chat suites, `Tests/UI/test_backup_data_groups.py`, `Tests/UI/test_settings_personal_context.py`, `Tests/UI/test_personal_context_link_app_flow.py`, `Tests/Plugins/test_recovery.py` (its in-place replace restore timed out once under load and passes serially), `Tests/DB/test_private_sqlite_inventory.py` (pins extended for C100/B42, one more backup copy and the new recovery-capture owner; `test_backup_inventory_matches_current_sqlite_and_settings_operations` stays red as on base, which never listed the TASK-40132 plugin capture), `Tests/Architecture/test_backup_owner_inventory.py`, the module-size ratchet (`app.py`, `private_sqlite.py` and `repository.py` at their budgets), and preflight.

**Not verified / limits.** AC#5 is pinned at the plan (what publication executes), not by an executed end-to-end replace restore. `_published_owners`' journal read is fail-closed and exercised only through a stub in the activation test; no test drives a real restore journal into the review. The settings review test is a node outside the PR lane (the UI census has no shard margin this round).

**Review r14pca-rev1 fixes (2026-11-13).**
- **Restored link state.** `PersonalContextLinkService.resume` and `resume_after_local_activation`, and the Settings link flow once a stored link exists (before its staged-key reads and activation), now call `require_sync_allowed()` first. Test: `test_restored_link_resumes_nothing_before_its_review`. Negative control (gate removed from `resume`): `restored link state was read`.
- `_published_owners` no longer constructs `Journal` (which creates the operation directory) unless that directory already exists.
- **Renamed code.** The backup-preview code is now `personal_context_backup_key_excluded`; `personal_context_key_not_in_backup` stays reserved for TASK-40136's restore lock.
- **Dead code.** `_LocalSQLite`'s `excluded=False` branch and its docstring are gone.
- **Summary filter pinned.** `test_inspection_summary_never_offers_credential_groups`; negative control (filter removed): the credentials group is listed.
- **Unreadable review.** When the restore review cannot be read and the gate keeps sync off, Settings shows `personal_context_restore_review_unavailable` with copy; an ordinary profile shows nothing. Tests: `test_an_unreadable_restore_review_is_named_only_while_sync_is_off` (negative control: `None == ...unavailable`) and the Settings test's `[unavailable]` case.
- **Not changed here.** The detached-layout refusal is left to the pcc restack, as instructed.

**Review r14pca-rev2 fixes (2026-11-13).**
- `test_restored_link_resumes_nothing_before_its_review` now gives the stored link every field `_key_binding` reads and records custodian reads (`reads == []`). Without the Settings link-flow gate it fails on that assertion instead of passing through the catch-all.
- `test_reading_published_owners_never_creates_an_operation_directory` pins the no-create read on a control root that passes `qualified_for("admission", ...)`. Without the `is_dir` check, `Journal()` creates `operation-<sha256>` and the test fails.

