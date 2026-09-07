# ADR-126: Complete local backup and recovery

Status: Proposed — conversational decisions approved; written specification awaiting review
Date: 2026-09-07

Task: [TASK-31978](../tasks/task-31978%20-%20Design-complete-local-backup-and-restore.md)

Design: [Complete local backup and restore](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)

Extends: ADR-004, ADR-029, ADR-036, ADR-059, ADR-060

Supersedes: None. Existing selective Chatbook export and storage-settings Save
contracts remain unchanged.

Allocation: 126 was free across 425 locally available branch/remote refs and 68
worktrees on 2026-09-07. Recheck upstream/open-branch allocations before merge.

## Context

Users need an exposed, complete recovery workflow covering Chatbook-owned local
data, both replacement and isolated-profile restoration, and explicitly selected
external files/models. Existing selective Chatbook bundles are not a lossless
installation image; the legacy bulk database backup covers only three databases.
Configuration and databases can live outside default roots. Other stores own
assets, recoverable journals, indexes, credentials, and device-local operational state.

Current profile instance locking is advisory. Individual SQLite snapshots do not
coordinate multiple stores and referenced files. A full restore spans filesystems
and cannot be one SQLite or portable filesystem transaction. Normal startup can
migrate databases, start workers, or erase temporary media before a restore is
known safe. A copied profile can alias original databases or reuse shared credentials.

Portable credential exclusion also conflicts with exact rollback: a safety copy
cannot remove the credentials needed to recreate the old installation state.
Device-bound journals and permissions contain useful recovery data but cannot grant
new filesystem or execution authority when imported.

## Decision

1. Add an explicit versioned recovery archive and app-owned recovery service,
   exposed through canonical F9 Settings, first-run setup, and a dependency-light
   pre-bootstrap recovery launcher. Keep selective content import/export separate.

2. Use a declared storage-owner inventory and explicit profile list. Default coverage
   includes identified app-owned durable data and configured custom storage. External
   folders, models, and currently available temporary media are optional. Unknown or
   unavailable required data blocks a complete result. Partial archives are labeled
   partial and cannot perform installation replacement in v1. Server data is excluded.

3. Coordinate supported persistence participants through enforced maintenance
   admission over verified shared storage identities. Drain writers to safe boundaries,
   snapshot through checked SQLite/file owners, then resume before packaging. Refuse
   unverified exclusivity. External folders receive an explicit per-file consistency
   label rather than a whole-folder point-in-time guarantee.

4. Use a ZIP64 container with a manifest, dependency groups, schema versions, sizes,
   digests, coverage, and relocation metadata. Validate pinned archive bytes and
   enforce extraction/crypto budgets. Application-owned adapters validate and migrate
   staged candidates; imported schema code, executables, or scripts never run.

5. Exclude managed credentials by default using typed owner adapters on staged
   copies, including qualified treatment of SQLite secret remnants. Explicit export
   of supported readable credentials requires encryption. Arbitrary user content is
   not promised secret-free. Environment contents and unrelated keychain entries are
   not exported; existing memory-only Sync key boundaries remain intact.

6. Encrypt the whole recovery container with age v1 passphrase encryption using a
   small bundled helper built from the official age Go library. Do not invent a
   cryptographic format or repurpose config-value encryption for large files. Secrets
   pass through anonymous pipes, never command arguments/environment or persistent
   request files. Qualify streaming, resource limits, packaging, interoperability,
   dependency licensing, and every advertised platform before shipping. There is
   no plaintext fallback when encryption is requested or required.

7. Restore through private staging, immutable target mappings, qualified publication
   primitives, and a durable operation journal outside replacement targets. Check
   pending recovery before ordinary configuration fallback, migrations, cleanup, or
   service composition. Ambiguous interruption blocks normal boot and preserves both
   generations for recovery. Cross-volume replacement is recoverable, not described
   as globally atomic, and is enabled only on qualified platforms/filesystems.

8. Replacement first creates and verifies an encrypted exact local rollback archive.
   The user supplies a rollback password before any live mutation. Portable export
   redaction does not apply to this artifact. Never overwrite shared keyring entries.
   Retain recovery copies until explicit deletion; protect unresolved evidence. A
   later rollback preserves intervening changes with another verified recovery copy.

9. Isolated restoration uses a new data namespace, dedicated config, explicit fresh
   process launch, owner-aware path remapping, and new device/credential scopes. A
   small owner-private recovery catalog makes the profile reopenable without hot
   switching or building a general profile manager. Reject writable aliases to the
   original profile. Shared settings and ambient credential references do not silently
   become active in the recovered profile.

10. Restore external folders only to newly created destinations in v1. Original
    overwrite is deferred. Retain included temporary media under a durable recovered
    asset owner. Store models inertly; restoration does not launch/download them.

11. Restore operational definitions and history without restoring active authority.
    Schedules, queues, agents, model processes, network activity, tool permissions,
    sync bindings, claims, cursors, and pending journals stay paused/quarantined until
    existing owners complete explicit review/reconciliation. This defines the explicit
    recovery exception anticipated by ADR-060 without adding those fields to ordinary
    Chatbook export or activating imported device-local journals.

12. Distinguish Archive verified, Restoration validated, Opened successfully, and
    Needs setup. Release requires real data round trips, both destinations, startup
    independence, process/crash fault injection, credential isolation, and a first
    open with no unintended execution or network activity.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Extend selective Chatbook export into total recovery | Requires every data domain to maintain a lossless logical serializer and still leaves settings, assets, device state, and exact rollback unsolved. |
| Copy the application directories | Misses custom paths and coordinated writes, can capture unrelated links, and cannot prove total coverage. |
| Restore each database from the Settings screen | Cannot safely publish a whole installation, survive broken startup, or coordinate cross-store rollback. |
| Rename a data folder to make a second profile | Config/custom paths and credentials can still alias the original installation. |
| Use credential-excluded export as the rollback copy | Cannot recreate the exact pre-restore credential/config state. |
| Restore imported permissions/journals unchanged | Transfers device-local authority and can replay writes against changed files or servers. |
| Custom chunked archive encryption | Adds unnecessary cryptographic protocol design and interoperability risk; use a maintained standard file-encryption implementation. |
| Overwrite original external folders in v1 | Requires independent conflict, metadata, multi-volume recovery, and external-writer contracts beyond app-owned replacement. |

## Consequences

This is a new recovery subsystem, not an extension of legacy Settings handlers.
It introduces storage-owner capture/relocation declarations, maintenance admission,
a recovery catalog/journal, an encryption helper packaging dependency, and a
pre-bootstrap recovery path. New schemas follow normal migration/version rules.

The main UI remains usable for inspection/progress, but coherent capture can pause
writes for the duration of snapshot work. Space is required for staging, encrypted
output, and retained rollback. Password loss prevents decrypting an encrypted
backup/rollback; passwords are not persisted automatically. Private plaintext
working files may exist during capture/restore, and no forensic-erasure claim is made.

Complete means complete for the shown profile inventory and selected content, not
proof that an unknown custom profile or remote server has been backed up. Restored
data is intentionally inactive until reviewed, so recovery is not a clone of live
execution state. The design preserves this distinction in UI and verification.

Implementation is decomposed after written-spec approval. This ADR does not grant
permission to write production code, run a full test sweep, or claim the feature
implemented. Platform qualification can restrict destructive replacement while
inspection and safe extraction remain available.

## Related contracts

- [ADR-004](004-settings-storage-defaults-restart-boundary.md): ordinary path Save is restart-required and does not relocate live stores.
- [ADR-029](029-local-private-data-boundary.md): private files, checked paths, and metadata-only diagnostics.
- [ADR-036](036-application-service-composition-lifecycle.md): one application composition root and memory-only Sync dataset keys.
- [ADR-021](021-file-backed-notes-disk-authority-and-recovery.md): File Notes filesystem authority and independent recovery ownership.
- [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md), [ADR-060](060-notes-sync-round-trip-and-interoperability-constraints.md): device-local Notes sync and explicit paused recovery boundary.
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md): imported/local context state does not grant tool authority.
- [ADR-051](051-private-tts-clone-reference-assets.md): private voice reference assets and repository-owned validation.
- [age v1 format](https://age-encryption.org/v1) and [official implementation](https://github.com/FiloSottile/age): standard encrypted-container boundary selected here.
