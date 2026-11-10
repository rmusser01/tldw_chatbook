---
id: TASK-40132
title: "Back up Plugins: capture the plugin registry, package bytes and plugin data instead of leaving them unsupported"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 16:15
updated_date: 2026-11-10 19:31
labels:
- follow-up
- backup
priority: medium
references:
- backlog/docs/backup-recovery-owner-inventory.md
- tldw_chatbook/Plugins/
- tldw_chatbook/Backup_Recovery/
dependencies:
- TASK-39116
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39116's owner decision. Checked on dev ae9f9e8d21, 2026-11-10. The paths below come from a read-only survey of dev; verify each before relying on it.

**The ruling.** Owner decision (2026-11-10), TASK-39116, Plugins: "Capture all". The option as offered: "Full capture: registry schema catalog, quiescing plugin processes, an execution gate. Large, and a restore still lands in recovery_required without the keyring marker." The recommended alternative (exclude the registry and packages, keep plugin data unsupported) was declined.

**What the plugins root holds** (`<user_data>/plugins`, `Plugins/service.py`):
- `registry.sqlite3` and its WAL: installs, revisions, selections, sources, mappings, data roots, processes and receipts (`Plugins/migrations/001-005`).
- `packages/<install>` and `packages-revisions/<install>/<digest>`: immutable package bytes. A reinstall can fetch them again, unless the source was a local path that no longer exists.
- `data/<root_id>`: per-plugin data that plugin processes write. This is the only part the user made and cannot recreate.
- `staging/` and `acquisition-custody/*.json`: in-flight acquisition state, swept after 24 h. `runtime.lock`: a lock file.
- Plugin authority lives at `skills/trust/plugins/`, which the Skills owner already captures (`Skills_Interop/recovery.py`). The generation marker sits in the OS keyring, so a restored profile without it reports `recovery_required`.
- The registry holds credential reference ids only; values are resolved at use.

Today every one of these producers sits in the backup review's conservative `unsupported` default, so a profile with a plugin installed cannot produce a complete backup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A backup preview of a profile with an installed plugin and plugin data shows the plugins root as captured, not unsupported: the registry database, the package bytes and every plugin data root
- [x] #2 Only a lock file or in-flight acquisition state is left out of the capture, each named in the notes with its reason and shown in the preview as excluded; anything else under the plugins root is captured
- [x] #3 The registry database is captured consistently, like the other SQLite owners (schema catalog, WAL folded), with a test that is red before the change
- [x] #4 While a plugin process or another plugin data writer is running, capture either waits for it to stop or is refused with a reason the preview shows, so no capture holds half-written plugin data; a test shows it
- [x] #5 A restore puts the registry, packages and data back byte for byte. The plugin system then reports its documented state for a missing keyring marker, and the user sees copy that says what to do; no plugin data is lost. Tests cover both
- [x] #6 The owner inventory doc's Plugins rows and the backup User Guide page describe the new behaviour
- [x] #7 After a restore, the plugin service refuses to run plugins until plugin authority recovery completes, which approves them; a test shows both
- [x] #8 A restore into a separate profile refuses the plugins owner with the reason owner_relocation_unverified, and a same-profile restore is unaffected; a test shows both
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checked on worktree base 42fc21939d (dev ae9f9e8d21 plus the filing commit), 2026-11-10.

**Owner shape.** One installed owner, `plugins`, modelled on `recovered.media` (`Backup_Recovery/recovered_media.py` `_RecoveredAdapter`), the only existing owner that holds a SQLite catalog and raw files in one tree. Installed from a new `Plugins/backup.py` through `owner_registry.install_adapters()` (`Plugins/recovery.py` is the plugin operation-recovery module and stays as it is). It is a `_RawDeclaration` tree over `<user_data>/plugins`:

- `registry.sqlite3` has restore role `sqlite`. Every other file has role `file`; the root and its subdirectories have role `directory`.
- The tree is in the `tools` group ("Tools and skills"), next to `skills`, which already captures the plugin authority under `skills/trust/plugins/`. `owner_label("plugins")` is "Plugins".
- `activation_required=True`, like the other durable owners. This is an open question; see Q3 below.

**Registry (AC3), joined like the other SQLite owners.**
- A new `DB/private_sqlite.py` owner policy, `recovery.plugins.registry`, with `_PRIVATE_AND_READ_ONLY`, `centralized_backup_allowed`, `recovery_capture_allowed` and `preserve_read_only_source_mode`, plus its `sqlite:` row in the owner inventory doc.
- Capture uses `_checked_capture` + `copy_private_sqlite`. That copy goes through the SQLite backup API, so committed `-wal` frames are folded into the captured file.
- Validation uses `_validate_sqlite` (`PRAGMA user_version`) against a frozen, exact schema-v5 catalog `PLUGIN_REGISTRY_SCHEMAS`. The catalog is measured from `PluginRegistry._reference_schema()`, and a test pins the frozen text to that reference, so a migration 006 has to update the catalog. Only v5 is accepted, with no migration steps. A registry still at v1 to v4 (the runtime owner upgrades it when it opens) shows as `unsupported`.
- `-wal`/`-shm` are declared as `sqlite.transient` and intentionally excluded through `inventory._sqlite_sidecars`. The two `recovered.media` name checks become one small owner-to-SQLite-member map, shared by `inventory._sqlite_sidecars` and `capture._item_validator`; `replacement.py` already reuses the latter.
- The new owner joins `sqlite_validation._installed_owner`'s factory list.

**Exclusions (AC2).** `runtime.lock` is intentionally excluded, and its capture is refused. Everything else under the root is captured (Q1 decision), including `staging/`, `acquisition-custody/`, `cache/` and unknown names.

**Quiescing (AC4): refuse, never wait.**
- Discover reads the registry through the preview copy. If any `processes.state` is not `settled`, a plugin process may be running. Also, if a non-blocking `portalocker` probe of an existing `runtime.lock` fails, a live runtime owner holds it; the probe never creates the file. In either case every `plugins` item becomes `unavailable`, plus a `plugins:writers_active` item.
- The inventory carries the issue `plugin_writers_active`, which the preview lists. A partial backup can still be made without plugins.
- At capture, each plugins item re-probes the lock and holds it while that item is copied. If the probe fails, capture raises `plugin_writers_active`.
- Known ceiling: the app's `PluginService` holds `runtime.lock` from its first plugin use until quit. After using plugins in a session, a complete in-app backup therefore needs an app restart. See Q4.

**Restore (AC5).**
- The registry goes through the SQLite route (`validate_candidate`, no migration) and everything else through the raw route.
- Tests compare package and data bytes with sha256, check that the restored registry equals the archive payload, and check that the registry rows equal the source.
- A restored profile whose keyring holds no marker reports `PluginAuthorityStore.posture() == "recovery_required"` (tested with a fake secure keyring).
- The user-facing copy is Q2.

**Docs (AC6).** Update the owner inventory Plugins rows: the `sqlite:` rows, the new `Plugins/backup.py` producer rows, and the cohort and classification of the existing `Plugins/*` rows. Update the Backup & Restore User Guide page and `Docs/Architecture/backup-recovery.md`.

**Tests.** Discover and capture tests go in `Tests/Architecture/test_backup_owner_inventory.py`, next to the Personal Context owner test; it is on the backup-recovery-qualification lane. A roundtrip test goes in `Tests/Backup_Recovery/test_complete_roundtrip.py` (same lane). Each test is shown red before its change.

**Owner decisions (2026-11-10, via the coordinator's AskUserQuestion), recorded verbatim:**

- Q1 staging/custody: "Capture both (Recommended)" — "Capture staging/ and acquisition-custody/; exclude only runtime.lock. Git-installed plugins can be recovered after a restore."
- Q2 restore copy: "Plugins screen (Recommended)" — "Rewrite the Plugins screen's recovery_required copy to name the restore case and the Reset authority action."
- Q3 execution gate: "Enforce it (Recommended)" — "The plugin service refuses to start plugins after a restore until plugin authority recovery completes, which then approves it."
- Q4 relocation: "Refuse plugins there (Recommended)" — "Restoring into a separate profile refuses the plugins owner ('owner relocation unverified') until a relocation design exists; restoring onto the same profile works."
- The refusal ceiling (runtime.lock held until quit) is accepted; the preview reason copy and the User Guide state it.

Q3 and Q4 added criteria #7 and #8.

**Questions as asked (kept for the record):**

- **Q1: staging and acquisition custody.** The description calls `staging/` and `acquisition-custody/*.json` in-flight state, but they are not only that. For every git-acquired installation, `acquisition.verify_reference` needs the terminal, `recovery_referenced` custody record and the retained `staging/<sha>/snapshot` bytes. That call is used by plugin recovery (`recovery.retained_inspections`, through `retained_inspection._verify_retained_source`) and by `source_service.validate_inspection_source`. These entries are never swept. Excluding them makes those installations unrecoverable after a restore. Options:
  - (a) Capture all of `staging/` and `acquisition-custody/`, so only `runtime.lock` is excluded. **Recommended.**
  - (b) Capture only entries whose custody record is `recovery_referenced`, and exclude unreferenced in-flight ones. This parses the custody JSON at discover.
  - (c) Exclude both, as the description suggests. Git-acquired installs then cannot be reconstructed.
- **Q2: what the user sees after a restore, and where.**
  - Replace restore on the same machine, at the same path: the keyring account (`skill_trust_account_scope` of the trust path) is unchanged, so the marker survives. The posture is ready only if plugins did not change after the backup; otherwise it is `recovery_required`.
  - Separate-profile or new-machine restore: no marker, so `recovery_required`.
  - Reduced-protection (file marker) mode: the marker is restored with the trust dir, so the posture is `ready_reduced`.
  - In every case, restored data roots fail their physical-identity binding (inode and birth time; `data_cleanup.open_root` raises `root_incarnation_changed`) until they are rebound.
  - Today Plugins ▸ Authority shows "Authority: recovery_required. Restore the protected backend or wait for the owning instance; no automatic reset or takeover." The only way out is Reset authority, then setup, then reinstall or re-review. Reset keeps the data roots on disk (`root_reset_retained_custody`) but does not re-attach them to a new install.
  - Options:
    - (a) Add a restore-specific line to the Backup & Restore post-restore summary.
    - (b) Rewrite the Plugins screen's `recovery_required` copy to name the restore case and the Reset authority action. **Recommended:** the posture and the reset live there.
    - (c) Both.
- **Q3: execution gate.** With `activation_required=True`, every restore adds `plugins` to the "Needs setup" list. No plugins review flow approves it yet, so it stays listed; most raw owners already behave this way. Options:
  - (a) Keep it as an inert requirement.
  - (b) Have `PluginService` refuse to start while `execution_allowed(("plugins",))` is false, and approve it when the user completes authority recovery. This is the owner's "execution gate"; it adds a small amount of `PluginService` work.
  - (c) Set `activation_required=False` and let the authority posture be the only gate.
  - Recommended: (b) if the gate is wanted now, otherwise (a).
- **Q4: refusal scope and relocation** (flagged, not blocking).
  - Refusal scope: is refusing while the runtime lock is held acceptable, given that it forces an app restart before a complete in-app backup? The alternative is a maintenance participant that pauses `PluginService` (the "wait" branch of AC4).
  - Relocation: the registry stores absolute paths (`data_roots.path`, revision `materialized_identity`, acquisition references), and the authenticated authority snapshot binds them. In a separate-profile restore, `data_cleanup.open_root` refuses the old paths (`root_path_unowned`). However, `recovery.retained_inspections` reads package bytes at the old absolute path, so a file-marker profile could resolve another profile's packages. Recommendation: in a separate-profile restore, refuse the plugins owner (`owner_relocation_unverified`) until a relocation design exists; allow replace restores at the same paths.
<!-- SECTION:PLAN:END -->


## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** A backup now captures the whole plugin root through a new installed `plugins` owner (`tldw_chatbook/Plugins/backup.py`). It covers the registry, package bytes, revisions, every data root, staging and acquisition custody. Only `runtime.lock` is left out. Capture refuses while plugin writers may be live. A restore goes back onto the same profile only, and restored plugins stay off until plugin authority recovery approves them. Implemented on dev ae9f9e8d21 + the filing commit, 2026-11-10. Owner decisions Q1–Q4 are recorded verbatim in the plan above.

**Approach.**
- **Owner shape.** One tree owner, shaped like `recovered.media`.
  - `inventory.sqlite_tree_member` replaces the two `recovered.media` name checks, for sidecar declaration and capture dispatch.
  - `registry.sqlite3` goes through a new SQLite owner, `recovery.plugins.registry`: `copy_private_sqlite` (backup API, so the WAL folds in) and `_validate_sqlite` against the frozen v5 catalog `REGISTRY_SCHEMAS`. A test pins that catalog to `PluginRegistry._reference_schema()`.
  - Other files go through the raw route (`_RAW_RECOVERY_LIMITS["plugins"]`). The owner joins the `tools` group and is labelled "Plugins".
- **Exclusion (AC2).** Only `runtime.lock` is left out: it is the live runtime owner's lock, holds no data, and restoring a held lock would be meaningless. It shows as `intentionally_excluded`. Registry `-wal`/`-shm` show as `sqlite.transient`, because they are folded into the copy. A stray `-journal` is `unsupported`, because it could roll back a restored copy. Staging and custody are captured, per Q1.
- **Quiescing (AC4): refuse, never wait.** Two things count as live writers: any `processes.state != 'settled'`, or a failed non-creating `portalocker` probe of `runtime.lock`.
  - Either one marks every plugin item `unavailable`, adds a `plugins:writers_active` row, and puts the inventory issue `plugin_writers_active` in the preview. The Backup screen renders that issue with its `issue_message`, which states the restart ceiling.
  - Partial capture is allowed to omit plugins. Each item's capture re-probes the lock and holds it while copying.
- **Restore.**
  - `staging._config_targets` refuses `owner_relocation_unverified:plugins` when the target selector's profile id differs from the archive's (Q4).
  - `relocate_restore` also refuses a registry whose `data_roots.path` rows name another root.
  - Staged registry copies are opened only through `open_recovery_validation`.
- **Gate (AC7, Q3).**
  - `PluginAdmission.capture` raises `plugin_restore_review_required` while a restored generation that lists `plugins` is unapproved. That check is `Plugins/backup.restore_review_required`, which fails closed.
  - Generations restored before this owner existed are ignored, so they cannot lock plugins forever.
  - `PluginService.recover()`/`unlock()` approve the generation once coordinator recovery returns no `recovery_required` receipt and the posture is ready.
- **Copy (Q2).** The Plugins screen's `recovery_required` status now names the restore case and Reset authority…. It renders in a `markup=False` Static and interpolates nothing.

**Red → green evidence** (load 30–60 throughout).
- AC1/AC2: the preview test was red on `assert None == ('plugins', 'intentionally_excluded')` — the whole root previewed as `{'.': ('unknown', 'unsupported')}`.
- AC3: red on `assert 'plugins' in owners`.
- AC4 (both params): red on `assert 'plugin_writers_active' in (...)`.
- Preview reason UI: red on the raw code `plugin_writers_active` only.
- Plugins copy UI: red on the old "Authority: recovery_required. Restore the protected backend…" text.
- AC8 negative control (the profile check removed in staging): the e2e test went red with "separate-profile plugin restore was planned".
- Registry-root defense control (the raise removed): red on "DID NOT RAISE".
- AC7 control (the admission gate disabled): red with "restored plugin ran before authority recovery", after the authority was unlocked and reconciled outside the service.
- All were green after the change. The e2e test takes about 2 minutes.

**Tests.**
- `Tests/Plugins/test_recovery.py` (plugin-qualification lane):
  - `test_backup_preview_captures_plugin_root_except_runtime_lock`
  - `test_plugin_registry_capture_folds_wal_into_its_exact_catalog`
  - `test_live_plugin_writers_refuse_preview_and_capture[process|lock]`
  - `test_restored_registry_must_name_its_own_plugin_root`
  - `test_plugin_backup_restores_in_place_and_waits_for_authority_recovery` (managed runtime only). It covers a Tools-group backup, an isolated plan refused, a same-profile replacement, byte-equal package/data/staging files and an identical registry dump (the registry is compared by its rows, since the SQLite backup route rewrites pages and folds the WAL, so the file is not the source's bytes), missing-keyring posture `recovery_required`, the gate before and after recovery, and plugin data kept.
- UI: `Tests/UI/test_backup_data_groups.py::test_backup_review_explains_refused_live_plugins` and `Tests/UI/test_plugins_browse.py::test_authority_recovery_names_restore_and_reset`.
- Census and allowlist updates:
  - `Tests/DB/test_private_sqlite_inventory.py`: C99/B41 ranges, the `copy_private_sqlite` count 23, the reviewed authority set, and the `_Plugins` fixed-validation declaration.

**Docs (AC6).**
- `backlog/docs/backup-recovery-owner-inventory.md`: a new "Plugins backup owner" section, the `sqlite:` rows, the three new producer rows, and all 64 existing `Plugins/*` rows reclassified.
- `backlog/docs/sqlite-private-owner-inventory.md` (C99, B41).
- `Docs/Backup-and-Recovery.md`: the backup user page; there is no `Docs/User_Guide/` page for Backup & Restore.
- `Docs/Plugins/operations.md` ("After restoring a backup"): the Plugins user page.
- `Docs/Architecture/backup-recovery.md`.

**Findings.**
- Capturing staging and custody is necessary for recovering git-installed plugins, but by code reading it is not sufficient. `acquisition._qualify` compares the plugin root's and staging's physical identities, and `data_cleanup.open_root` compares data-root inode and birth time. Restored copies have new identities, so a git-installed plugin may still fail authority recovery after a restore and need Reset authority. This is not exercised by a test; the e2e plugin was a local-folder install.
- Staging also holds copies of the host git/SSH configuration a fetch used. The backup page recommends encryption.

**Files.**
- New: `tldw_chatbook/Plugins/backup.py`.
- Changed:
  - `Backup_Recovery/{capture,data_groups,inventory,owner_registry,recovery_service,sqlite_validation,staging,storage_admission}.py`
  - `DB/private_sqlite.py`
  - `Plugins/{admission,service}.py`
  - `UI/Screens/{backup_restore_screen,plugins_screen}.py`
  - the tests, docs and lesson listed above.
<!-- SECTION:NOTES:END -->
