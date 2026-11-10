---
id: TASK-40132
title: "Back up Plugins: capture the plugin registry, package bytes and plugin data instead of leaving them unsupported"
status: To Do
created_date: 2026-11-10 16:15
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
<!-- AC:END -->
