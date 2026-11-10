---
id: TASK-40132
title: "Back up Plugins: capture the plugin registry, package bytes and plugin data instead of leaving them unsupported"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 16:15
updated_date: 2026-11-10 21:51
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
- [x] #2 Only runtime.lock and, inside staging, the host git/SSH configuration copies (host-git-config, host-ssh-*), repository.git and content/ are left out of the capture, each named in the notes with its reason and shown in the preview as excluded; anything else under the plugins root, including staging snapshots and acquisition custody records, is captured
- [x] #3 The registry database is captured consistently, like the other SQLite owners (schema catalog, WAL folded), with a test that is red before the change
- [x] #4 While a plugin process or another plugin data writer is running, capture either waits for it to stop or is refused with a reason the preview shows, so no capture holds half-written plugin data; a test shows it
- [x] #5 A restore puts the registry, packages and data back byte for byte. The plugin system then reports its documented state when the keyring marker is missing or does not match the restored files, and the user sees copy that says truthfully what works (an in-place restore with plugins unchanged since the backup needs only Unlock) and what does not yet (the other cases keep plugins off, with their files and data on disk), with no step that dead-ends; no plugin data is lost. Tests cover both
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

Review r13g-rev1 (2026-11-10) owner decisions, via the coordinator's AskUserQuestion, recorded verbatim:

- D1 staging: "Only what recovery needs (Recommended)" — "Capture snapshot/ and the custody records. Exclude host-git-config, host-ssh-*, repository.git and content/, shown as excluded in the preview. Git-installed plugins stay recoverable, and no SSH, git or credential config goes into archives." (Supersedes Q1 for those staging members; AC#2 rewritten to match.)
- D2 recovery: "Truthful copy now, feature next (Recommended)" — "This PR states exactly what works (an in-place restore with plugins unchanged needs only Unlock) and what doesn't, with no false steps. I file a follow-up feature: a reset that keeps plugin data and re-attaches it, plus rebinding of restored data roots." (AC#5 rewritten to match; the coordinator files the feature.)

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
**Summary.** A backup captures the plugin root through a new installed `plugins` owner (`tldw_chatbook/Plugins/backup.py`).
- **Captured:** the registry, package bytes, revisions, every data root, and each repository fetch's `snapshot/` and custody record.
- **Left out:** `runtime.lock`, and inside each `staging/<name>/` the fetch's host git/SSH configuration copies, `repository.git` and `content/` (D1).
- **Capture** refuses while plugin writers may be live, without blocking backups that leave plugins out.
- **Restore** goes back onto the same profile and data folder only. Restored plugins stay off until the plugin service's authority recovery approves them.
- **User copy** says truthfully what works and what does not (D2).

Implemented on dev ae9f9e8d21 plus the filing commit, 2026-11-10. Owner decisions Q1–Q4 and review decisions D1–D2 are recorded verbatim in the plan above.

**Approach.**
- **Owner shape.** One tree owner, shaped like `recovered.media`.
  - `inventory.sqlite_tree_member` replaces the two `recovered.media` name checks, for sidecar declaration and capture dispatch.
  - `registry.sqlite3` goes through a new SQLite owner, `recovery.plugins.registry`: `copy_private_sqlite`, which folds the WAL in, and `_validate_sqlite` against the frozen v5 catalog `REGISTRY_SCHEMAS`. A test pins that catalog to `PluginRegistry._reference_schema()`.
  - Other files take the raw route (`_RAW_RECOVERY_LIMITS["plugins"]`). The owner is in the `tools` group and labelled "Plugins".
- **Exclusions (AC2, D1)**, each shown as `intentionally_excluded`:
  - `runtime.lock`: the live runtime owner's lock, which holds no data.
  - `staging/<name>/host-git-config` and `host-ssh-*`: copies of the host git/SSH configuration, which can hold credential helpers and SSH config. Capture refuses them with `plugin_staging_member_excluded`.
  - `staging/<name>/repository.git/` and `content/`: the fetch's object store (for a local source, the user's whole objects and refs, up to 500 MB) and working copy. Recovery reads only `snapshot/` and the custody record.
  - Registry `-wal`/`-shm` are `sqlite.transient`, because they are folded into the copy. A stray `-journal` is `unsupported`.
- **Quiescing (AC4, C1): refuse, never wait.** Two things count as live writers: an unsettled `processes` row, or a failed non-creating `portalocker` probe of `runtime.lock`.
  - Either one reduces the owner to one unavailable root (no children, so the tree stays valid) plus a `plugins:writers_active` row.
  - Inventory adds `plugin_writers_active` after group selection, from that row only. A backup without `tools` is therefore unaffected, and a partial backup omits plugins. The Backup screen explains the issue (M4 wording).
  - Capture creates `runtime.lock` if it is absent and holds it while copying each item (M3).
  - A plugin service that found the lock busy retries on its next use instead of caching `plugin_owner_busy` (M3).
- **Restore (AC8, M1, M2).**
  - `staging._config_targets` refuses `owner_relocation_unverified:plugins` when the target selector's profile id differs.
  - `relocate_restore` also refuses a registry whose data-root paths, or whose `materialized_identity` paths in `revisions.inspection_json` (installed and fetched packages), are outside the destination root. This catches a same-profile restore after `data_dir` moved.
  - Detached import refuses a plugin root by name (`detached_owner_unavailable:plugins`).
- **Gate (AC7, Q3).**
  - `PluginAdmission.capture` refuses `plugin_restore_review_required` while a restored generation that lists `plugins` is unapproved.
  - `PluginService` approves it only when coordinator recovery returns no `recovery_required` receipt and the posture is ready (M5 tests).
- **Copy (Q2 narrowed by D2).** It states what works and what does not, with no step that dead-ends:
  - What works: an in-place restore with plugins unchanged since the backup needs only Unlock.
  - On another computer or profile, the `recovery_required` copy on the Plugins screen says Chatbook cannot recover the plugins yet; they stay off with files and data on disk.
  - After plugins changed since the backup, unlock raises `FileNotFoundError`, and `authority_failure_message` names the mismatch.
  - No Reset step is suggested, because bootstrap refuses while the registry lists installs.

**Red → green evidence** (load 30–60).
- AC1/AC2 preview: red first on the root previewing as `unknown`; with D1, red on `staging/op/host-git-config` included.
- AC3: red on `'plugins' in owners`.
- AC4: red on the missing `plugin_writers_active`.
- C1 held-lock child: red on `('overlapping_owner_roots', 'plugin_writers_active', …)`.
- M2: "DID NOT RAISE" with no data roots.
- M3 capture lock: `['absent'] == ['held']`. M3 busy retry: `'plugin_owner_busy' == 'needs_setup'`.
- M4: "in Plugins" in the copy.
- M5: three cases red with both conditions removed.
- M1: detached "DID NOT RAISE".
- Unlock copy: red with its branch removed.
- Plugins copy UI: red on the old Reset text.
- Earlier controls: removing the separate-profile check went red with "separate-profile plugin restore was planned"; removing the registry-root raise went red with "DID NOT RAISE"; disabling the admission gate went red with "restored plugin ran before authority recovery".
- All green after the changes.
- E2e `test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`: 1 passed in 10 min at load ~45–50. It covers:
  - a repository install through `acquire_snapshot` + `SourceService.review_acquired`;
  - the staged host files, `repository.git` and `content/` excluded and absent from the archive, with no credential bytes in any payload;
  - an isolated plan refused;
  - a same-profile restore after `data_dir` moved refused (`owner_relocation_unverified`);
  - an in-place restore with byte-equal files and an identical registry dump;
  - posture `recovery_required` with another marker;
  - the gate refusing until the service's recovery;
  - the git-installed plugin re-verifying and running after restore;
  - then, after a later plugin change and a second restore of the same archive: posture `locked`, unlock raising `FileNotFoundError` with the truthful copy, plugins still refused, data kept.

**Tests.**
- `Tests/Plugins/test_recovery.py` (plugin-qualification lane, whose path filter now includes `tldw_chatbook/Backup_Recovery/**`):
  - the preview, catalog, live-writer, held-lock, registry-root, capture-lock, busy-retry, approval-condition, reason-copy and unlock-copy tests;
  - the e2e test.
- `Tests/Backup_Recovery/test_detached_layout.py::test_detached_import_refuses_a_plugin_root_by_name`.
- UI: `test_backup_data_groups.py::test_backup_review_explains_refused_live_plugins` and `test_plugins_browse.py::test_authority_recovery_names_restore_and_reset`. Both files are now on the UI census tail: the backup file whole (~21 s serial) and the plugins file as 15 node ids (12–25 s each serially). The floor is 1929. The shard replay was not run, so the coordinator rebalances if needed.

**Covering runs (review round, load 25–60).**
- Tests/Plugins plus 12 Backup_Recovery files, the DB SQLite inventory and 5 CI census/workflow pins, at `-n 4`: 1902 passed, 19 failed. All 19 also fail at base 42fc21939d:
  - 10 recovered-media console image specs;
  - 2 shipped76 restore-plan cases;
  - 3 `detached_layout._normalize_sqlite` SQLite-inventory census checks;
  - `test_storage_worker_retires_its_workspace_lookup_connection`, `test_actual_console_registers_selected_managed_agent`, `test_actual_native_fields_stable_in_separate_processes` and `test_existing_owned_directory_metadata_uses_verified_node_rollback`.
- `Tests/Architecture/test_backup_owner_inventory.py` (serial) and the module and screen size ratchets: 224 passed.
- `test_plugin_review.py`, `test_plugins_browse.py` and `test_backup_data_groups.py`: 101 passed, 2 failed. One is `test_refresh_growth…[size1]`, which also fails at base (1 pass and 1 fail in two base runs). The other is `test_confirmed_explicit_test_survives_close_during_child_prune[success]`, which passed twice when run alone.
- `./scripts/preflight.sh` passes, and `ruff --select F` is clean.

**Docs (AC6).**
- `backlog/docs/backup-recovery-owner-inventory.md`: the Plugins section, rows reclassified, and the I1 identity text.
- `backlog/docs/sqlite-private-owner-inventory.md` (C99, B41).
- `Docs/Backup-and-Recovery.md`: the backup user page; there is no `Docs/User_Guide/` page for it.
- `Docs/Plugins/operations.md` ("After restoring a backup"): the Plugins user page.
- `Docs/Architecture/backup-recovery.md`.

**Known limits.**
- A plugin root, `staging/` or data root deleted before a restore comes back with a new identity. Data roots then report `root_anchor_changed`, custody records fail `_qualify`, and the app has no rebind yet. The D2 feature follow-up is filed by the coordinator.
- A restored plugin snapshot containing a symlink would be `unsupported` by the tree inventory. This is not exercised.

**Files.**
- New: `tldw_chatbook/Plugins/backup.py`.
- Changed:
  - `Backup_Recovery/{capture,data_groups,detached_layout,inventory,owner_registry,recovery_service,sqlite_validation,staging,storage_admission}.py`
  - `DB/private_sqlite.py`
  - `Plugins/{admission,service}.py`
  - `UI/Screens/{backup_restore_screen,plugins_screen}.py`
  - `Widgets/plugin_review_modal.py`
  - `.github/workflows/plugin-qualification.yml`
  - `scripts/{ui_pr_gate_census.txt,check_ui_pr_gate_census.py}`
  - the tests and docs above, and `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
