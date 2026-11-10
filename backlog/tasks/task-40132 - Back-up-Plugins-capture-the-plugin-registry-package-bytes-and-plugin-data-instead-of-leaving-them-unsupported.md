---
id: TASK-40132
title: "Back up Plugins: capture the plugin registry, package bytes and plugin data instead of leaving them unsupported"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-10 16:15
updated_date: 2026-11-10 17:10
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
- [ ] #1 A backup preview of a profile with an installed plugin and plugin data shows the plugins root as captured, not unsupported: the registry database, the package bytes and every plugin data root
- [ ] #2 Only a lock file or in-flight acquisition state is left out of the capture, each named in the notes with its reason and shown in the preview as excluded; anything else under the plugins root is captured
- [ ] #3 The registry database is captured consistently, like the other SQLite owners (schema catalog, WAL folded), with a test that is red before the change
- [ ] #4 While a plugin process or another plugin data writer is running, capture either waits for it to stop or is refused with a reason the preview shows, so no capture holds half-written plugin data; a test shows it
- [ ] #5 A restore puts the registry, packages and data back byte for byte. The plugin system then reports its documented state for a missing keyring marker, and the user sees copy that says what to do; no plugin data is lost. Tests cover both
- [ ] #6 The owner inventory doc's Plugins rows and the backup User Guide page describe the new behaviour
- [ ] #7 After a restore, the plugin service refuses to run plugins until plugin authority recovery completes, which approves them; a test shows both
- [ ] #8 A restore into a separate profile refuses the plugins owner with the reason owner_relocation_unverified, and a same-profile restore is unaffected; a test shows both
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
