# ADR-126: Complete local backup and recovery

Status: Proposed — conversational decisions approved; written specification awaiting review
Date: 2026-09-07

Revision: 3 — incorporates the third user-requested design review.

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
   Valid current configuration is required for complete backup-source discovery, not
   archive inspection or isolated restore. Restore destination discovery uses the
   archive and independently verified local targets; damaged config never causes a
   silent fallback to default database paths. Unverifiable replacement targets block
   replacement while inspection and new-destination recovery remain available.

3. Coordinate protocol-aware persistence participants through enforced maintenance
   admission over stable logical namespaces outside replaced data. Verified file/path
   identity establishes shared-store aliases; replacement of an inode does not create
   a new unlocked namespace. Reserve both namespaces during remapping, acquire locks
   deterministically, and drain without circular waits. For ordinary backup, resume
   after coherent capture, before packaging. Replacement/later rollback remains fenced
   through encrypted rollback verification, publication, and installed validation.
   Known incompatible activity blocks maintenance. Arbitrary legacy/external processes
   do not honor the protocol and are outside its exclusion guarantee; PID scans are
   not proof of their absence. Native safety qualification remains required. External
   folders receive per-file consistency, not a whole-folder point-in-time guarantee.

4. Use a ZIP64 container with a manifest, dependency groups, schema versions, sizes,
   digests, coverage, and relocation metadata. Inspect/execute only a completed private
   source copy or qualified immutable snapshot bound to the preview by digest; a
   pinned handle alone does not prevent in-place modification. Enforce source-copy,
   decrypted-container, header, extraction, and crypto budgets before/during streaming.
   Authenticated bytes remain untrusted application input. Owner-specific schema
   allowlists, trusted_schema=OFF, restricted functions/authorizers, and SQL execution
   budgets apply before and during staged migration. Installed migration SQL must
   not activate unvalidated imported schema; no unrestricted-connection fallback.
   Represent selected folder topology, empty directories, and versioned supported
   directory metadata explicitly in the manifest. Directory/file names share bounded
   collision/containment checks. Restore parents before children and metadata last;
   unsupported metadata requires an explicit partial-metadata result. New archives
   and rollback copies publish atomically to new files only, never overwrite an
   existing destination, and reject protected source/control aliases or unexcluded
   output-inside-source layouts. Check-then-replace publication is insufficient.

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
   dependency licensing, and every advertised platform as an early delivery gate.
   Define the helper protocol, integrity/version checks, release/update ownership,
   and wheel/source/editable install behavior before dependent feature implementation.
   Release installs must not unexpectedly need Go or download/resolve a helper at
   backup time. Missing/unqualified helpers fail capability preflight before passwords
   or maintenance; change the ADR if the selected integration cannot qualify. There
   is no plaintext fallback when encryption is requested or required.

7. Restore through private staging, immutable target mappings, qualified publication
   primitives, and a durable operation journal outside replacement targets. Check
   pending recovery before ordinary configuration fallback, migrations, cleanup, or
   service composition. Durable associations in a fixed bootstrap admission directory
   make custom control roots discoverable by every supported launch route, independent
   of the convenience catalog and restored config. Register before publication and
   clear only after a durable verified outcome. Ambiguous interruption blocks affected
   storage and preserves both generations. Other profiles may launch only if intact
   evidence proves their namespaces disjoint. Cross-volume replacement is recoverable,
   not globally atomic, and is enabled only on qualified platforms/filesystems.
   Before clearing an operation fence, persist its installed generation's independent
   activation-required state. Completing recovery is not permission to start work.

8. Replacement first creates and verifies an encrypted local rollback archive of the
   exact stored-data snapshot and supported captured managed credentials.
   The user supplies a rollback password before any live mutation. Portable export
   redaction does not apply to this artifact. Never overwrite shared keyring entries.
   Retain recovery copies until explicit deletion; protect unresolved evidence. A
   later rollback preserves intervening changes with another verified recovery copy.
   Replacement previews restore, retire into rollback storage, and preserve-outside-
   scope sets. Archive absence alone cannot authorize deletion; unknown files or
   unsupported old-owner mappings block publication until reviewed. Owners retire
   obsolete managed objects only after verified rollback, and final inventory checks
   reject accidentally active newer objects/sidecars outside the desired generation.
   Capture readable affected Chatbook-owned keyring values and scope mappings; old
   references alone do not guarantee later credential recovery. If an existing scope
   no longer matches, restore into a new scope through the owning adapter rather
   than overwrite shared credentials. Journal those references/scopes. Disclose
   unsupported/unreadable credentials before replacement and require acknowledgement
   of the missing credential coverage if proceeding. Exact local stored-data recovery
   does not imply that revoked tokens, expired sessions, or environment credentials
   become available; no provider authentication check runs automatically.

9. Isolated restoration uses a new data namespace, dedicated config, explicit fresh
   process launch, owner-aware path remapping, and new device/credential scopes. A
   small owner-private recovery catalog makes the profile reopenable without hot
   switching or building a general profile manager. Reject writable aliases to the
   original profile. Shared settings and ambient credential references do not silently
   become active in the recovered profile.

10. Restore external folders only to newly created destinations in v1. Original
    overwrite is deferred. Retain included temporary media under a durable recovered
    asset owner with stable catalog/reference identity and transcript resolution.
    Recovered assets enter baseline subsequent backups even when temporary-media
    capture is off. Explicit deletion/cleanup rechecks references and recovery holds;
    missing bytes render a missing-media state instead of resolving a different file.
    Temporary-store TTL/startup sweeps cannot remove committed recovered assets.
    Store models inertly; restoration does not launch/download them.

11. Restore operational definitions and history without restoring active authority.
    Schedules, queues, agents, model processes, network activity, tool permissions,
    sync bindings, claims, cursors, and pending journals stay paused/quarantined until
    existing owners complete explicit review/reconciliation. This defines the explicit
    recovery exception anticipated by ADR-060 without adding those fields to ordinary
    Chatbook export or activating imported device-local journals.
    A durable per-generation activation record is authoritative for every supported
    launch route, including later ordinary/headless launches. Per-owner approvals
    do not grant other owners permission. Completed-operation fence clearing, UI
    closure, report rebuilding, or imported approval claims cannot clear this state.
    Missing/corrupt state for a known restored generation keeps capabilities inactive.
    Every restore/rollback creates a new locally reviewed generation.

12. Distinguish Archive verified, Restoration validated, Opened successfully, and
    Needs setup. Release requires real data round trips, both destinations, startup
    independence, process/crash fault injection, credential isolation, and a first
    open with no unintended execution or network activity.

Second-review evidence additionally covers lock continuity across inode changes,
known incompatible participants, restore/retire/preserve reconciliation, in-place
archive mutation, resource failure before manifest parsing, schema-trigger attacks
during installed migrations, custom-root discovery through normal launchers, scoped
startup blocking, separate maintenance intervals, packaged helper availability, and
recovered-media deletion/re-backup. These extend the existing release gate rather
than claiming those runtime checks have already been performed.

Third-review evidence adds recovery without parseable current configuration,
activation persistence after ordinary relaunch and operation-fence clearing, changed
or unavailable keyring values, publication races against an existing good backup,
and empty-directory/metadata round trips. These are contract corrections; they do
not enlarge the feature into server credential repair or general filesystem backup.

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
a recovery catalog/journal plus independent bootstrap admission records, an encryption
helper packaging dependency, a recovered-media owner, and a pre-bootstrap recovery
path. New schemas follow normal migration/version rules. Helper delivery and the
maintenance/bootstrap protocol must be qualified before dependent feature slices.

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
