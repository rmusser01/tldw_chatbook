# Complete local backup and restore

Date: 2026-09-07

Status: Draft for written user review. Conversational scope and review corrections
approved; implementation and implementation planning have not started.

Task: [TASK-31978](../../../backlog/tasks/task-31978%20-%20Design-complete-local-backup-and-restore.md)

ADR required: yes

ADR path: [ADR-126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md)

Reason: archive and credential contracts, coordinated storage lifecycle, isolated
profile launch, and interruption recovery across multiple persistence owners.

## 1. Purpose and decisions

Users can create a portable backup of Chatbook-owned local data from F9 Settings,
inspect a backup, and restore either over selected current data or into a separate
local profile. A clean installation can use the same recovery flow. Existing
Chatbook content export/import remains the selective sharing and merging feature.

Approved decisions:

- All identified Chatbook-owned local profiles and durable data are included by
  default, including configured database locations outside the default directory.
- External folders and model files are explicit options. Server-owned data has a
  separate recovery boundary and is not captured by this feature.
- Both replacement and isolated-profile restore are required.
- Managed credentials are excluded by default. Including exportable credentials
  requires a password-encrypted archive. Encryption is available without credentials.
- A maintenance pause during capture and a controlled restart for replacement are
  acceptable. Restored automation and permissions require review before activation.
- Coverage and restore previews are required. Missing required data prevents a
  complete result; an explicitly selected partial backup is labeled partial.
- Review corrections include enforced multi-process coordination, isolated launch
  configuration, startup-independent recovery, distinct rollback policy, explicit
  temporary-media handling, and observable restoration evidence.
- External-folder overwrite is deferred. External content restores into newly
  created directories without replacing existing files.

The remaining implementation choices below are concrete proposals for this written
review, particularly the encryption helper, recovery catalog, and verification
limits. They do not authorize implementation before approval of this document.

### Non-goals

No server backup or mutation, cloud upload, recurring schedules, incremental backup,
deduplication service, record merging, restoration of a running LLM process, or
installer/OS backup. No automatic downloads, external-folder overwrite, credential
replacement in a shared keyring, or replay of restored work. No general profile
manager or hot-swapping of the application service graph.

## 2. Existing boundaries and corrections

The legacy Settings bulk worker snapshots three databases through the checked
SQLite helpers. It is useful machinery but is not the complete storage inventory.
The canonical entry point remains UI/Screens/settings_screen.py, not the deprecated
UI/Tools_Settings_Window.py. New screens delegate to an app-owned recovery service.

config.py selects one effective config, a user-specific data directory, and several
custom database paths. That is not a complete catalog of all historical profiles.
Utils/instance_lock.py is explicitly advisory and cannot authorize maintenance.
DB/private_sqlite.py provides checked individual SQLite copy/restore primitives;
it does not establish a joint snapshot across every database and filesystem owner.

[ADR-004](../../../backlog/decisions/004-settings-storage-defaults-restart-boundary.md)
keeps ordinary path settings restart-bound; this feature adds an explicit recovery
workflow rather than silently changing Save semantics.
[ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md) governs
private files and diagnostics; [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
governs application service composition.

[ADR-021](../../../backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md)
keeps File Notes disk-authoritative and separates recovery stores from projections.
[ADR-059](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md)
and [ADR-060](../../../backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md)
keep Notes sync device-local. This explicit recovery format may retain their data,
but imported journals and bindings must never become live filesystem authority.

## 3. Inventory and completeness

An explicit storage inventory combines application storage-owner declarations,
effective configuration, known profile locations, and user-added profile configs.
It does not scan the entire home directory or import optional runtime engines just
to discover their files. Profile configuration must parse successfully; a fallback
to default_user is not an acceptable inventory result.

Each storage owner declares discovery, dependencies, classification, safe capture,
validation, relocation, and activation behavior. Its declaration uses canonical
path resolvers rather than filenames duplicated in a backup module. Owners sharing
a file are detected by verified identity; the file is captured once and its logical
relationships remain explicit. Unknown durable entries are reported, not swallowed
by a recursive directory copy or silently ignored.

| Category | Default and restoration contract |
| --- | --- |
| Authoritative databases | Include all discovered owners, including inactive/optional features whose data exists. Preserve records, soft deletions, relationships, and recovery content. |
| Config and app definitions | Include current settings, templates, prompts, workspace metadata, local definitions, and supported history after managed-secret processing. |
| Persistent assets | Include attachments, saved media, persona artwork, voice references, and other app-owned user content. Resolve their referencing records as one dependency group. |
| Sync, queue, permission, and recovery state | Retain recoverable data, but restore operational state into quarantine; activation rules are in section 9. |
| Derived indexes and embeddings | Include durable indexes by default when a qualified capture adapter exists. Optional omission is explicit and reports rebuild prerequisites/cost; no rebuild starts automatically. |
| Models | Off by default, with per-model selection, byte sizes, dependency details, and available-source verification. Model recipes alone do not imply model bytes were captured. |
| External folders | Off by default, selected per root; source identity, coverage, and consistency limitations are shown. Source folders are never modified by backup. |
| Temporary generated media | Offer Include currently available temporary media, off by default; count referenced available/missing items. Included bytes restore as retained recovered assets, outside expiry/startup-cleanup namespaces. |
| Disposable caches and process artifacts | Exclude query caches, thumbnails that can be rebuilt, locks, sockets, temporary downloads, and running process state. Do not classify solely by a directory named cache. |
| Diagnostics and old backup archives | Exclude logs and diagnostic bundles by default with explicit optional inclusion. Exclude existing backup archives, active staging, and rollback stores from recursive capture. |

The user reviews the exact profile list and can add another local config. Discovery
cannot prove that no unrelated custom profile exists elsewhere; UI wording is
Complete for the listed profiles, with a visible list of selected optional content
and exclusions. A location outside the default data directory is not automatically
external: a configured app-owned database remains baseline content.

Every inventory entry records included, intentionally excluded, absent because
unused, unavailable, or unsupported. Known missing required data and unknown durable
entries block a complete result. A partial backup requires explicit acknowledgement
of omissions, retains dependency failures in the manifest, and cannot later be
relabeled complete. Partial archives support extraction and isolated recovery of
validated dependency groups; they cannot replace current installation data in v1.

## 4. User experience

F9 Settings gains Backup & Restore, with Create backup, Inspect / restore backup,
Recovery copies, and Restored profiles. A command-palette entry reaches the same
surface. First-run setup offers Restore a backup. A minimal recovery launcher
exposes inspection, restore, and rollback even if normal configuration is corrupt.

### Create backup

1. Discover profiles and show categorized coverage, sizes, source issues, and the
   Add profile configuration action.
2. Select optional external folders, models, temporary media, and diagnostic history.
3. Choose output path and encryption. Including credentials forces encryption;
   the password is entered twice and never silently persisted. Show managed-secret
   exclusions and credentials that cannot be exported.
4. Show maintenance impact, required working space by volume, and final coverage.
5. Capture with progress, package after writers resume, verify the finished archive,
   then publish it. Completion shows the actual file and Archive verified status.

Phases are Discovering, Waiting for maintenance access, Capturing, Packaging,
Verifying, and Complete / Partial / Failed / Cancelled. Counters derive from real
work. Cancellation remains available before publication; a failed or interrupted
output is a staging artifact, never a completed backup in the history list.

### Inspect and restore

Inspection asks for a password when necessary and shows format/version, capture
time, profiles, coverage, selected additions, validation results, and prerequisites.
Archive metadata is untrusted text and never interpreted as markup or commands.

The destination step offers Replace current data and Restore as separate profile.
It previews old-to-new mappings, shared items affected, and bytes required. A
separate profile requires a new display name and private destination. Replacement
requires explicit confirmation naming affected data and the rollback location.
External additions always go to new directories in either mode.

The persistent result distinguishes Archive verified, Restoration validated,
Opened successfully, and Needs setup. The report links missing assets/models,
credentials to re-enter, quarantined work, external folder remapping, and rollback.
Opening succeeds without starting network activity or optional model engines.

Use existing Textual tokens, forms, pickers, and keyboard conventions. Keep screens
thin, progress app-owned, and workers exclusive. Navigation does not cancel an
operation. Closing the app either cancels safe preparation or hands off a durable
recovery operation; it never abandons an in-progress replacement silently.

## 5. Components and ownership

| Component | Responsibility and dependencies |
| --- | --- |
| Inventory and profile catalog | Discover declared storage, classify contents, track explicitly registered recovered profiles, and produce immutable coverage models. Reads configuration without booting services. |
| Maintenance coordinator | Establish exclusive access across affected persistence namespaces and drain participating writers to safe boundaries. Depends on lifecycle participants, not widgets. |
| Capture service | Snapshot SQLite through checked helpers and copy qualified files into private immutable staging. Reports dependency and consistency evidence. |
| Archive codec | Encode/decode the versioned container and encryption envelope; enforce resource limits and content integrity. Has no application-service or network dependency. |
| Restore planner and executor | Build immutable mappings, validate staged candidates, preserve rollback, journal publication, and recover interruption. Never invokes ordinary startup as a validation shortcut. |
| Minimal recovery launcher | Read control state before normal config/bootstrap/migrations/cleanup and run the same recovery services. Launch a fresh app only after successful recovery. |
| Settings / first-run views | Present service models, collect choices, and show progress/results. No database/file replacement logic. |

Introduce a focused Backup_Recovery package, not a generic lifecycle framework.
Owner adapters remain close to their existing storage modules. The app composes
the recovery service once under ADR-036. No schema upgrade occurs during inventory
or inspection; any new persistent owner schema uses normal numbered migrations.

A small versioned, owner-private recovery catalog lives outside profile data under
the default app config namespace, alongside rather than inside config.toml. It
records opaque operation/profile IDs and verified config/data locators, not secrets
or content. Atomic writes and cross-process coordination protect it. Explicit
control-root selection is available to the recovery launcher for portable/custom
installations; it is never accepted from an imported archive as authority.

Control journals, temporary working storage, and retained rollback archives are
distinct. Their parent roots are verified non-overlapping with all replacement
targets. Sensitive local locators may appear in private control records and explicit
Details views, but not routine diagnostic logs or unencrypted portable manifests.

## 6. Capture consistency and availability

Maintenance access is enforced, not inferred from the advisory instance lock.
Every supported Chatbook persistence participant, including headless workers and
new processes, honors shared normal-operation and exclusive maintenance admission
for its verified storage namespaces. Shared databases/configuration acquire the
same locks even when reached through different profiles. Locks use deterministic
ordering. An incompatible or uncooperative process means access is unverified and
capture/replacement is refused with close-other-instances guidance; PID detection
alone is not proof of exclusive access.

Close admission to new mutations, then drain existing transactions and cross-store
operations to completed or durable recoverable boundaries. Unresolved operations
whose bytes/ownership cannot be captured coherently block a complete backup. Do
not kill a worker merely to obtain a snapshot or discard an unsaved editor draft;
ask the user to save/discard a draft through its existing flow before maintenance.

Use DB/private_sqlite.py for each SQLite snapshot, including committed WAL content.
Do not copy live .db/-wal/-shm files as independent assets. Required databases and
their referenced file assets share the same capture boundary. Capture from pinned,
verified regular-file handles into staging; never hold writers paused for later
compression, encryption, or output-device transfer. The pause is proportional to
required snapshot work, not guaranteed to be brief. Show that before confirmation.

External editors and independent servers are outside this coordinator. v1 external
capture is verified per file, with before/after identity/change checks and a bounded
retry; unstable files are unavailable. Even a successful copy is labeled per-file
capture, not a folder-wide point-in-time snapshot. Missing roots never mean delete.
Directory links, nested mounts, aliases, and unsupported metadata are surfaced;
v1 does not follow them speculatively. Model-store links may only be resolved by a
qualified model adapter within an explicitly identified store, preserving selected
model dependencies without following arbitrary filesystem links.

## 7. Archive, credentials, and validation

### Container and encryption

Use a versioned ZIP64 recovery container, distinct from a Chatbook content bundle.
Use .tldw-backup.zip for plaintext and .tldw-backup.zip.age for encrypted output.
Only stored and deflated regular-file entries are allowed. Already compressed
media/model files use stored entries. All payload names are generated relative
identifiers; destination paths are chosen locally, never obeyed from archive paths.

The manifest includes format version, producer version, capture time, profile IDs,
owner/schema versions, payload sizes and SHA-256, dependency groups, capture
consistency, exclusions, credential policy, and required capabilities. The archive
also carries a versioned, sanitized human-readable recovery report. Managed absolute
storage locators belong only in a protected relocation record inside encrypted
archives. Owner adapters replace those locators in staged config/metadata with
logical IDs; plaintext backups ask for new local mappings. This does not rewrite
or promise removal of paths appearing inside arbitrary user-authored content.

Encrypt the entire container, including its manifest, using the standard age v1
passphrase format through a small bundled helper built from the maintained official
age Go library. This avoids adapting config-value encryption or inventing a custom
chunked cipher format. End users do not need a Go toolchain. The helper has no
network role; it receives secrets via anonymous pipes, never arguments, environment,
logs, or persisted request files. It streams bytes with bounded memory. Dependency
pinning, reproducible packaging, license review, and platform availability are
prerequisites to shipping the encrypted path; no plaintext fallback is permitted.
The [age format](https://age-encryption.org/v1) supplies streaming authenticated
encryption; the [official implementation](https://github.com/FiloSottile/age) is the
upstream implementation dependency.

No plugin recipient, shell extension, public-key recipient, or externally supplied
helper is activated by an imported file. Use only the standard single-passphrase
recipient mode; bound password derivation to 256 MiB working memory, one active
derivation, and a cancellable helper. Reject larger work factors before derivation;
do not choose a writer work factor exceeding the reader's limit. Reject unsupported
envelopes. Decrypt into owner-private staging and verify the complete
authenticated stream before parsing ZIP or treating metadata as trusted. Plaintext
checksums detect corruption, not authorship; password authentication also does not
identify the creator as a trusted application operator.

### Credential processing

Managed credentials means known secret fields and owned credential references,
including provider keys, token-bearing connection definitions, and supported
credential-bearing configuration history. It does not promise detection of secrets
inside user-authored prose, imported documents, or arbitrary external files.

Each relevant owner supplies a typed sanitization/export adapter. Exclude known
secret values and encrypted secret blobs by default, retaining setup hints rather
than live credential references. Unknown credential-bearing formats block a
credential-excluded complete result until excluded explicitly or safely supported.
Selecting diagnostic history or external files includes an explicit disclosure that
their arbitrary contents are not credential-sanitized.

Operate only on staged copies. Secret-bearing SQLite copies require a qualified
logical reconstruction or compaction after sanitization so deleted-page remnants
do not carry removed managed secrets. Preserve semantic identities; do not blindly
VACUUM stores relying on implicit row IDs. Sidecars and unprocessed config backups
are never added afterward. SQLite documents both [snapshot backup](https://www.sqlite.org/backup.html)
and [deleted-content/compaction behavior](https://www.sqlite.org/lang_vacuum.html).

Credential inclusion exports only explicitly supported, readable Chatbook-owned
secrets, with missing/unexportable entries listed. Do not enumerate unrelated
keychain entries, serialize the process environment, or export memory-only Sync
dataset keys contrary to ADR-036. Environment variable names remain setup hints.
Existing encrypted values require their own unlock; wrapping unreadable ciphertext
does not count as a usable credential export. Imported credentials are not tested
against providers automatically.

### Resource and archive safety

Inspect from pinned source bytes or an immutable private source copy; selection
followed by reopening a changed archive invalidates the plan. Reject duplicate
entries, normalization/case collisions, absolute paths, traversal, links, devices,
unsupported compression, overlapping ZIP structures, and unexplained entries.
Never use extractall against a live destination or execute supplied schema scripts.

Enforce member count, individual/aggregate expanded bytes, manifest size, path
length, compression expansion, and crypto-work budgets before and during processing.
v1 reader defaults: 100,000 members; 16 MiB manifest; 1 TiB expanded payload;
256 GiB per member; 1,024 UTF-8 bytes per path. Unusual high-compression entries
require explicit review of declared expanded bytes, while hard byte budgets remain
enforced. Locally increasing a size/count budget requires a renewed space preview;
an archive cannot raise its own limits. Writer preflight uses the same limits and
never produces a backup its configured reader cannot inspect.

Treat SQLite as untrusted data: use restricted inspection connections with extension
loading disabled and no application startup hooks. Validate owner/schema identity,
integrity, foreign keys where applicable, domain constraints, and referenced assets.
Migrations come only from the installed application and run on disposable staged
candidates. An unsupported newer schema blocks that dependency group. Older data
needs an explicitly supported migration chain; do not promise arbitrary backwards
compatibility or downgrade.

Space admission accounts separately for immutable capture, output, decrypted input,
staged/migrated candidates, retained rollback, journals, and filesystem overhead on
each affected volume. Check continuously; do not rely on compression ratios or
optimistic free-space estimates. Stream large blobs/files without whole-archive
buffering. Private plaintext working files are disclosed and cleaned best-effort;
the product makes no forensic-erasure or encrypted-working-disk guarantee.

## 8. Restore, publication, and rollback

### Shared preparation

Inventory -> Validate archive -> Choose destination -> Build mapping -> Stage ->
Validate candidate -> Obtain maintenance -> Revalidate targets -> Preserve rollback
(replacement only) -> Journal publication -> Publish -> Validate installed state ->
Commit -> Offer isolated first launch.

Staging never changes live data. Target changes since preview invalidate the plan
and require a renewed preview. Stage each publication artifact on its target volume
where the platform's qualified replacement primitives require it. Never fall back
from a failed atomic rename to uncontrolled copy/delete across volumes.

Keep state transitions durable and recoverable at each boundary. An operation
journal records target identities, generation, expected previous and candidate
state, verified rollback references, and per-artifact progress. A separate stable
operation pointer is checked before normal bootstrap. Recovery classifies actual
filesystem evidence as well as journal intent; interrupted rename/journal updates
cannot be treated as proof of either success or failure.

No transaction spanning arbitrary filesystems is promised. Replacement is enabled
only on platform/filesystem combinations whose lock, identity, replacement, and
durability semantics pass the release qualification suite. Elsewhere, offer archive
inspection/extraction or an isolated restore into a qualified destination, with a
specific reason replacement is unavailable.

### Replace current data

Only complete validated dependency groups are admitted. Show which current profiles
and shared settings are affected; other profiles remain untouched. If a selected
shared item cannot be replaced without affecting an unselected profile, require an
explicitly expanded selection or refuse that replacement. Do not silently leave a
mixture of old records and restored records masquerading as total replacement.

Stop affected services and prevent new participants from opening their storage.
Produce and verify an encrypted exact local rollback archive before the first live
change. Require a rollback password at this step, even when the incoming backup is
plaintext; the UI can explicitly offer to reuse the entered archive password.
Never persist that password. On restart, recovery can ask for it before rollback.

Rollback retains exact pre-restore data and managed credentials, without portable
export redaction. Raw pre-restore config is retained as bytes even if normal config
parsing is broken. If a required current store is damaged beyond the qualified
rollback capture contract, refuse in-place replacement and offer isolated restore;
do not label an unverifiable rollback copy safe. The source backup and rollback
archive are immutable. Rollback does not depend on successful schema migration.

Stage credential material separately and do not overwrite shared OS-keyring entries.
Exact rollback may preserve original credential reference bytes because this is the
same local installation; it does not need to enumerate or copy the entire keyring.
Revert only credential entries newly created by this recovery operation, after
checking their identities. Portable credential-exclusion rules remain unchanged.

After publication, validate installed artifacts while normal admission stays closed.
Failure triggers rollback when the password remains available or enters Recoverable
interruption awaiting unlock. If neither direction is provable, show Needs attention
and retain all evidence; never boot into an ambiguous mixture of generations.
Successful validation records a durable commit before normal startup can proceed.

### Separate profile

Create a private directory and dedicated config. Persist a stable opaque recovery
profile ID and display name in the small recovery catalog, then expose Open profile
through Settings and the recovery launcher. Launch a new process with an explicit
config selection; no hot service rebinding or automatic selection of the new
profile. Register it only after validation and atomic publication succeed.

Relocate every writable storage owner and its cross-record asset references through
typed owner adapters. Retain original logical record IDs where relationships need
them, but create fresh device/installation identities where required. Reject path
aliases to the original profile, including custom DBs, shared caches with writes,
external bindings, relative-path resolution, and platform-equivalent names.

The launch descriptor overrides inherited application storage-selection settings;
it does not blindly inherit another TLDW_CONFIG_PATH or custom database override.
Ambient provider credentials and shared keyring references stay unavailable until
explicit reconnect review. Generate new credential scopes or use isolated encrypted
config references; never replace another profile's keyring credential. Shared global
settings are copied into the dedicated config rather than saved over the original.

Cancellation/failure removes only provably owned unpublished artifacts. A crash
between profile publication and catalog registration is recoverable via the journal
without guessing directories by name or erasing unknown files.

### Retention and later rollback

Keep rollback archives until explicit deletion, with sizes and affected profiles
visible in Recovery copies. No automatic retention job is introduced in v1. Do not
delete unresolved-operation evidence, the source being restored, or the only copy
needed by an active operation. Rollback retention is included in space admission.

Before a later rollback, explain that post-restore changes would be replaced and
preserve those current changes in a new verified encrypted recovery copy. Apply
the same preview, fencing, journal, and verification flow. Cancellation is immediate
before publication; once publication begins, the action is Finish recovery or Roll
back at a safe boundary, not abandon work halfway through.

## 9. Activation, external data, and recovery independence

Restored content is available for local inspection without starting schedules, sync,
agents, MCP servers, skill scripts, managed model processes, downloads, updates,
model catalog refresh, remote authentication checks, or network index rebuilding.
A recovery launch policy applies before service composition, not after a first
background task has already started. The persistent Needs setup report owns review
actions; ordinary future activation uses existing capability/permission owners.

Preserve historical runs and definitions, but never resume a queued mutation or
replay a captured operation merely because its old status was running. Retain old
permissions, root bindings, leases, device claims, cursors, and pending journals as
quarantined recovery evidence, not live authorization. Reconnect sync requires fresh
identity/claim validation and a complete dry-run. Same-machine replacement also
requires reconciliation because files and remote data may have changed since backup.

Owner-specific recovery bytes remain inspectable/exportable. File Notes pairing and
Notes sync recovery are validated through their owners; old filesystem intents do
not run against relocated paths. A restore report explains inactive managed folder
memberships rather than silently converting or deleting them. Retained temporary
media uses a durable recovered-asset owner that normal startup cleanup cannot erase.

Selected external folders restore only into newly created directories whose parent
the user chooses. Preserve supported bytes and metadata, detect case/Unicode/path
collisions, and show unsupported metadata before admission. Do not rewrite path
strings inside arbitrary user documents or execute project hooks. References can
be remapped only after the restored directory exists and the user approves binding.
External source files and existing destination content remain untouched.

Model bytes restore as inert files with provenance/checksums and compatibility
status. Do not execute binaries or deserialize model payloads while validating the
archive. Missing/incompatible engines and optional dependencies are setup issues,
not a reason to discard otherwise valid user data or download replacements silently.

Recovery startup is a dependency-light path before config fallback, ordinary DB
migrations, ephemeral cleanup, profile seeding, and app composition. It can read an
explicit operation directory if the catalog is unreadable. If journal/target evidence
cannot be validated, offer safe report/export actions and preserve both generations;
do not treat corrupt control state as no pending recovery and continue normal boot.

## 10. Verification and release evidence

Run targeted checks only; a full suite needs separate user authorization. This is a
design-only change, so no application tests are claimed for this document.

Implementation release evidence must include:

- Storage inventory fixtures covering all persistent owner declarations, nondefault
  profiles/configs, custom paths, optional owners, unknown durable entries, aliases,
  shared stores, and deliberately missing data. New persistence owners cannot ship
  without a classification/capture declaration or explicit justified exclusion.
- Real SQLite/WAL data and real filesystem assets captured under concurrent writes
  from separate processes. Prove writers drain together and another process cannot
  enter during capture/publication; legacy/unverified access must fail admission.
- Round-trip comparisons of records, relationships, soft-deleted data, retained
  recovery bytes, asset digests, settings, and indexes in a fresh isolated home/config
  environment with no accidental access to developer data or ambient credentials.
- Both restore destinations, multi-profile/shared-data cases, path remapping, new
  device identities, unsupported newer schemas, staged migration failures, missing
  optional engines, and malformed current config. Prove original isolated-profile
  sources remain unchanged and restored profiles can be reopened later.
- Failure injection before/after every durable journal/publication boundary,
  including process termination, out-of-space, target changes, disconnected volumes,
  unavailable rollback password, corrupt control state, and interrupted rollback.
  Check recovered filesystem state, not just an exception or status message.
- Archive attacks and resource limits: traversal, aliases, duplicates, conflicting
  names, malicious SQLite schema, truncation, changed selected source, unsupported
  encryption, huge KDF work, oversized payloads, and misleading compression metadata.
- Credential fixtures spanning config history, supported DB locations, encrypted
  values, references, keyring scopes, and SQLite unused pages. Search emitted plaintext
  artifacts for known managed-secret sentinels; do not rely only on JSON field checks.
- Encryption interoperability against the official age implementation, bounded
  memory on large archives, packaging on each advertised platform, wrong-password
  behavior, and truncated final-stream detection. Verify secrets never enter process
  arguments/environment, logs, persistent control requests, or helper diagnostics.
- Product-level tests entering F9 Settings and first-run/recovery entry points,
  exercising actual create/inspect/restore/rollback services. Include progress,
  cancellation, separate-profile reopening, and retained temporary media after startup.
- A first-open test with network/process-spawn sentinels and restored queued work,
  proving no remote contact, code execution, synchronization, or schedule catch-up
  occurs before explicit activation.

Ship replacement only for qualified OS/filesystem combinations; the UI must derive
availability from the same capability evidence. Self-review and targeted static
checks accompany each implementation slice. Documentation must distinguish an
off-device copy from a backup still stored on the same disk, without claiming to
provide automatic remote storage.

## 11. Delivery boundaries and accepted review changes

This is one product design but more than one implementation PR. After written
approval, planning should separate storage inventory/profile admission, maintenance
coordination, archive/encryption qualification, isolated restore, replacement and
rollback/recovery startup, and final Settings/first-run integration with end-to-end
evidence. These are planning boundaries, not undeclared future task dependencies.
Do not expose Complete backup or destructive replacement before its full contract
is qualified. Existing selective exports remain available during development.

| Review issue | Incorporated resolution |
| --- | --- |
| Advisory instance locking | Enforced participant admission plus refusal of unverified exclusive access. |
| Folder-only profile isolation | Dedicated launch config, owner-aware relocation, fresh scopes, alias checks, and reopenable catalog. |
| Recovery depends on working app | Minimal pre-bootstrap recovery path with independent journal and explicit-path recovery. |
| Rollback versus portable export | Exact encrypted local rollback, no export redaction, explicit password and later-change preservation. |
| Credential remnants and shared secrets | Typed staged sanitization, qualified DB processing, separate credential scopes, no global keyring overwrite. |
| Ephemeral content disappears after restore | Explicit optional capture and retained recovered assets outside cleanup namespaces. |
| External consistency and overwrite | Honest per-file capture, newly created destinations only, original overwrite deferred. |
| Overstated completeness | Explicit profile inventory, partial labeling, dependency validation, and separate verification/open statuses. |

## 12. Written review boundary

The next step is user review of this document and ADR-126. TASK-31978 remains In
Progress until that review is received. No production changes, implementation task
completion, tests-passing claim, or implementation plan is implied by this draft.

After approval, invoke writing-plans and create atomic implementation Backlog tasks
in dependency order, with ADR links and required evidence scoped to each slice.
