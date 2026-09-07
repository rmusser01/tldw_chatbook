# Complete local backup and restore

Date: 2026-09-07

Revision: 4 — fourth design-review corrections incorporated.

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
- The second review tightens maintenance protocol boundaries, replacement retirement,
  immutable archive input, SQLite schema validation, custom-root startup discovery,
  and distinct backup/replacement downtime. Encryption packaging is an early gate;
  recovered media has a complete persistence and deletion lifecycle.
- The third review separates damaged-installation recovery from backup discovery,
  makes activation restrictions durable across launches, bounds credential rollback
  claims, prevents archive-output overwrite, and preserves explicit folder structure.
- The fourth review invalidates dependent indexes across restored source generations,
  revalidates source inventory under maintenance, and distinguishes intentionally
  deleted recovered media from unexpectedly missing required payloads.

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

[ADR-030](../../../backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md)
keeps media authoritative and indexes derived; recovery must reconcile projections
before exposing them against restored sources.

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
to discover their files. For backup-source discovery, profile configuration must
parse successfully; a fallback to default_user is not an acceptable inventory result.
This precondition does not apply to inspecting an existing archive or restoring it
into a new isolated destination.

Restore-destination discovery is a separate read-only operation. Archive inspection
uses only the archive, reader capabilities, and independent private working storage.
An isolated restore uses the validated archive plus a newly selected destination,
without opening or parsing damaged current profiles. Replacement needs independently
verified target locators from intact local admission/catalog records or explicit
user-selected targets checked against owner identity. Never guess custom database
locations from fallback defaults or trust original archive paths as destination
authority. If targets cannot be verified, replacement is unavailable while inspection
and isolated recovery remain available. Reading raw corrupt config bytes into an
encrypted rollback copy does not require treating them as parsed configuration.

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
| Derived indexes and embeddings | Include durable indexes by default when a qualified capture adapter exists. Optional omission is explicit and reports rebuild prerequisites/cost; no rebuild starts automatically. Restored or preserved indexes cannot serve a changed source generation until owner validation/reconciliation succeeds. |
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
unused, intentionally deleted with a validated owner tombstone, unavailable, or
unsupported. An intentional deletion includes its durable deletion/reference state,
not a requirement for the deleted payload. Known missing required data and unknown durable
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

Create backup publishes only to a new file in v1. Suggest a unique filename, report
an existing destination, and let the user choose another name; there is no overwrite
option in this flow. Validate the parent and reject destination/staging aliases to
source files, active input archives, bootstrap/control state, and retained recovery
artifacts. Reject output inside a selected source subtree unless its directory is
an explicit inventory exclusion. Known backup-output directories are excluded from
recursive source discovery to prevent capturing this operation's growing output.

Write and verify a private temporary artifact on the output volume, then publish
with a qualified atomic no-replace operation. An existence check followed by ordinary
replace/rename is insufficient. If another file appears after preview, leave it
untouched, retain or safely clean only this operation's temporary file, and offer
a new destination. No completed-history entry is recorded before durable publication.
Changing the destination reruns path/space checks; insufficient platform support
does not permit a silent overwrite fallback. The same rule protects newly created
rollback archives and later-rollback safety copies.

### Inspect and restore

Inspection asks for a password when necessary and shows format/version, capture
time, profiles, coverage, selected additions, validation results, and prerequisites.
Archive metadata is untrusted text and never interpreted as markup or commands.

The destination step offers Replace current data and Restore as separate profile.
It previews old-to-new mappings, shared items affected, and bytes required. A
separate profile requires a new display name and private destination. Replacement
requires explicit confirmation naming affected data and the rollback location.
External additions always go to new directories in either mode.

Replacement preview separately lists Restore, Retire into rollback storage, and
Preserve outside this replacement, including intentional exclusions. It explains
that maintenance lasts through encrypted rollback preparation, publication, and
installed-state validation. The shorter ordinary-backup pause is not advertised
for replacement or later rollback.

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
| Recovered-media owner | Retain explicitly included temporary media with stable transcript lookup, deletion, subsequent backup, and orphan handling. It is independent of temporary-media expiry. |
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

The catalog is a convenience index, not the only startup fence. A fixed bootstrap
admission directory in that default app config namespace holds durable per-operation
records associating affected logical storage namespaces/config selectors with their
control roots and operation IDs. Custom control roots must register here before
publication can begin. Registration is local authority, never restored from a
portable archive. Records remain discoverable when the selected config is corrupt,
has been replaced, or cannot yet resolve its custom data paths.

Every supported launch route, including console scripts, python -m, headless workers,
and Open profile, consults bootstrap admission before opening affected owners. A
custom-root launch option cannot disable this check. An interrupted registration
is conservatively pending until reconciled against the operation journal; commit
or verified rollback is recorded durably before its admission fence is cleared.
Before clearing that operation fence, atomically associate the installed generation
with the separate durable activation-required state described in section 9. Ending
replacement maintenance does not authorize automatic execution or reconnection.
The settings catalog may be rebuilt from intact admission records without guessing
whether a missing catalog means no operation. An unreachable custom control root
keeps its affected namespaces blocked and provides explicit-path recovery.

Fence only affected storage. Another profile may launch when intact admission
evidence and its storage mapping positively establish disjointness, including shared
config/databases. Corruption that makes scope unknowable blocks the uncertain scope;
do not promise unrelated-profile availability without that proof. Unknown or damaged
admission records cannot be silently deleted by startup cleanup. Legacy launchers
that do not implement this protocol are outside the coordination guarantee below.

Control journals, temporary working storage, and retained rollback archives are
distinct. Their parent roots are verified non-overlapping with all replacement
targets. Sensitive local locators may appear in private control records and explicit
Details views, but not routine diagnostic logs or unencrypted portable manifests.

## 6. Capture consistency and availability

Maintenance access is enforced among protocol-aware Chatbook participants, not
inferred from the advisory instance lock. Each participant takes normal-operation
admission before opening an owner; maintenance closes admission, drains writers,
releases their normal-operation holds, and acquires exclusive admission before
capture. New processes must participate before config writers or database owners
open. Failed or timed-out acquisition publishes nothing and leaves an actionable
waiting/refused result. Lock acquisition uses deterministic ordering and never waits
while retaining a lock that a draining participant needs to finish.

Locks are keyed by stable logical storage namespaces recorded outside replaceable
data, not only by the inode/file identity being replaced. Verified path/file
identities establish aliases and shared ownership during admission; the namespace
and its lock survive replacement. Shared configuration or a database reached through
two profiles has one maintenance namespace. Path remapping reserves both old and
new namespaces, and registry changes participate in the same admission protocol.
Startup checks pending generation evidence before resolving a new inode as a new,
unlocked store. Lock files are not removed/recreated during publication.

The coordination guarantee covers declared owners and protocol-aware entry points.
Older clients and external editors do not honor it. Known incompatible activity
blocks the operation with close-other-clients guidance; users must keep those
clients closed during maintenance. PID scans cannot prove all writers are absent.
Qualified native database/filesystem checks supplement coordination, but arbitrary
out-of-protocol modification by another same-user process is outside the guarantee,
not claimed detectable or preventable. Observed out-of-protocol changes abort
capture or enter recovery before further publication. If required native safety
checks or participating-owner coverage are unavailable, complete capture/replacement
is unavailable rather than silently falling back to advisory locks.

Close admission to new mutations, then drain existing transactions and cross-store
operations to completed or durable recoverable boundaries. Unresolved operations
whose bytes/ownership cannot be captured coherently block a complete backup. Do
not kill a worker merely to obtain a snapshot or discard an unsaved editor draft;
ask the user to save/discard a draft through its existing flow before maintenance.

After admission closes and writers drain, rediscover the selected owners, effective
storage mappings, aliases, dependencies, and required assets under maintenance.
Compare this inventory with the approved scope; a preview-time enumeration is not
the capture inventory. Changed roots/owners, shared-profile impact, exclusions, or
required coverage invalidate the preview. Release maintenance and obtain a renewed
preview, then reacquire and revalidate; never extend the lock set out of order while
holding existing locks. Registry admission keeps this final scope stable during capture.

Ordinary record/asset growth within already approved owners is included in the final
capture, subject to rechecked capacity and limits. If growth changes an approved
coverage choice or exceeds the budget, stop for a revised preview. Reconcile the
captured dependency groups and manifest against this final inventory before writers
resume. Record the actual capture boundary and coverage, not discovery-time counts.
External folders retain the separate per-file consistency contract below.

Use DB/private_sqlite.py for each SQLite snapshot, including committed WAL content.
Do not copy live .db/-wal/-shm files as independent assets. Required databases and
their referenced file assets share the same capture boundary. Capture from pinned,
verified regular-file handles into private immutable staging while participating
mutations are fenced. For an ordinary backup, resume writers after coherent capture;
packaging, encryption, final verification, and output-device transfer follow without
the maintenance hold. Its pause is proportional to snapshot work, not guaranteed
to be brief. Show that before confirmation.

Replacement and later rollback have a different downtime contract. Once current
state is captured for exact rollback, admission stays closed through rollback
encryption/verification, publication, and installed-state validation. Releasing it
between these stages would make the safety copy stale. Their preview and progress
show the entire maintenance interval; estimates include rollback size, encryption,
verification, target-volume performance, and retained recovery capacity.

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

Directory structure is represented explicitly in the manifest, rather than by ZIP
directory entries or inference from file names. Record selected root IDs, validated
relative directory paths (including empty roots/directories), parent relationships,
and a versioned supported-metadata record. Files map to this tree through logical
IDs. Directory records count against manifest/count/path budgets and share the file
namespace for duplicate, case/Unicode, ancestor, and file-versus-directory collision
checks. No link, mount traversal, or absolute locator becomes a directory record.

Restore directories only below an approved newly created root, create parents before
children, and apply supported final metadata after publishing their children. v1
preserves directory modification times and, on qualified POSIX destinations, ordinary
permission bits subject to the existing private-storage policy; never restore
special privilege bits or foreign ownership. ACLs, extended attributes, file flags,
alternate streams, and platform-specific metadata without a qualified round-trip
adapter are reported as unsupported. Private app-owned directories remain owner-only.
Preview metadata changes or omissions, and require explicit acceptance of a partial
metadata restore rather than silently claiming an exact filesystem round trip.

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

Qualify helper delivery before any encrypted-backup or replacement implementation
depends on it. Publish the exact supported OS/architecture/package matrix, pinned
helper/library versions, byte-stream protocol version, helper integrity verification,
and update ownership. Prove normal wheel installation and documented source/editable
installation paths in isolated environments. Release installs must not unexpectedly
require Go, fetch an executable at backup time, or substitute an arbitrary PATH tool.
Contributor source builds may explicitly require Go; an unbuilt source checkout must
report the unavailable capability before collecting passwords or entering maintenance.

The packaged helper is updated through the application's dependency/release process;
interoperability fixtures protect archives across helper upgrades. Qualify anonymous
pipe handling, cancellation/child cleanup, large streaming I/O, and supported Python
versions on each advertised platform. If this integration cannot be qualified,
revise this ADR and the dependency choice before implementation proceeds; do not
quietly ship a different encryption format or weaken encrypted rollback requirements.

No plugin recipient, shell extension, public-key recipient, or externally supplied
helper is activated by an imported file. Use only the standard single-passphrase
recipient mode; bound password derivation to 256 MiB working memory, one active
derivation, and a cancellable helper. Reject larger work factors before derivation;
do not choose a writer work factor exceeding the reader's limit. Reject unsupported
envelopes. Decrypt into owner-private staging and verify the complete
authenticated stream before parsing ZIP. Authenticated metadata remains untrusted
application input and passes all schema, path, and resource checks. Plaintext
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

Before inspection, make a completed owner-private source copy or use a platform-
qualified immutable source snapshot. An open/pinned file handle alone does not stop
in-place writes and is insufficient. Check source stability during copying, close
the staged writer, hash the completed artifact, and use only that sealed artifact
for inspection and execution. Bind the preview and restore plan to its digest and
validated manifest version. Reopening the original selected pathname or modifying
the staged artifact invalidates the plan. This does not authenticate the author.
Reject duplicate
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

Resource admission precedes source copying and decryption, not just manifest parsing.
Default encrypted/plain input-container and decrypted-container budgets are each
2 TiB, independently enforced as actual bytes stream; expanded payload remains
limited to 1 TiB by default. The extra container allowance covers ZIP metadata and
encryption overhead without trusting archive declarations. Budget storage for both
input and decrypted staging; refuse insufficient capacity before starting and stop
on any runtime limit/space failure. A user-raised budget requires renewed preflight
before retry. Bound outer-header parsing to 64 KiB and the permitted single-passphrase
recipient before KDF work. Invalid, truncated, cancelled, or oversized intermediate
streams cannot be inspected as archives or published as completed artifacts.

Treat SQLite as untrusted data throughout inspection and staged migration. Disable
extension loading, use trusted_schema=OFF, and register no side-effecting application
functions or unnecessary virtual-table modules. Bound SQL execution with progress/
interrupt handlers and owner-appropriate memory/SQL limits. Use read-only inspection
connections and authorizer rules that deny attachment, unauthorized writes, and
unapproved operations. An ordinary repository constructor with startup hooks is not
a safe inspection connection.

Validate the actual schema against the installed owner's allowlisted supported
tables, columns, indexes, triggers, views, and virtual-table definitions before
running migrations. A matching schema-version number is insufficient. Preserve
recognized FTS and required triggers through a qualified owner-specific policy;
do not disable features globally and then claim domain validation succeeded.
Unexpected executable schema blocks normal restoration. Offer isolated inert
extraction for manual recovery, without loading the schema through app services.

Migrations come only from the installed application and run on disposable staged
candidates under the same restricted connection policy, with narrowly admitted
writes for that owner's known migration. Installed migration SQL must not activate
unvalidated imported triggers or functions. Revalidate schema, integrity, foreign
keys where applicable, domain constraints, and asset references after migration.
Unsupported security primitives/SQLite capabilities block the affected operation;
never retry using unrestricted application connections. See SQLite's
[untrusted-database guidance](https://www.sqlite.org/security.html).
An unsupported newer schema blocks that dependency group. Older data needs an
explicitly supported migration chain; do not promise arbitrary backwards compatibility
or downgrade.

Space admission accounts separately for immutable capture, output, decrypted input,
staged/migrated candidates, retained rollback, journals, and filesystem overhead on
each affected volume. Check continuously; do not rely on compression ratios or
optimistic free-space estimates. Stream large blobs/files without whole-archive
buffering. Private plaintext working files are disclosed and cleaned best-effort;
the product makes no forensic-erasure or encrypted-working-disk guarantee.

## 8. Restore, publication, and rollback

### Shared preparation

Acquire bounded archive input -> Validate archive -> Choose destination -> Discover
and verify destination scope -> Build mapping -> Stage ->
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

Resolve the target inventory into three explicit sets before confirmation:

- Restore: archive objects and supported empty/default state that the selected
  generation requires, including records and referenced assets as dependency groups.
- Retire into rollback storage: current managed objects in the selected replacement
  scope that must no longer be live in that generation. Merely overwriting matching
  files is not a replacement. Preserve these objects in verified rollback before
  retiring them through their owner; never recursively delete a target directory.
- Preserve outside this replacement: unselected profiles, external source folders,
  independent optional exclusions, shared items outside the approved scope, and recovery
  control/artifacts. Optional exclusion does not keep a dependent projection active
  after its source changes. Show intentional preservation in the result.

Absence from the archive alone does not authorize retirement: distinguish explicit
producer inventory/omissions, optional exclusions, and an owner introduced after
that backup version. Unknown current files or unsupported older-owner mappings
block publication until classified and reviewed; no inferred garbage deletion.
Require explicit selection changes when shared dependencies cross the scope.
Checked SQLite owners handle old sidecars only after connections close and the
rollback snapshot includes committed WAL state. Journal every restore, retire,
and preserve decision, and validate the final live owner inventory against the
approved plan. A stale database or attachment absent from the desired
generation cannot survive as an accidentally active store.

Apply ADR-030 to every derived index, embedding projection, and query-result cache
whose authoritative source is restored. Invalidate cache entries and disable the
affected projection before the restored generation can be queried. An omitted index
is not an independent optional exclusion: retire its affected managed state into
rollback or quarantine it through its owner, with that choice shown in the preview.
Shared indexes require the same scope expansion/refusal rules as shared databases.

Both restored and retained projections require owner-verified source identity/state,
schema and embedding compatibility, and reconciliation with active source records
before serving retrieval. Presence, matching paths, or matching record IDs alone are
insufficient. Without sufficient provenance or a qualified validation adapter, leave
the projection unavailable and report explicit rebuild/reconciliation prerequisites.
No rebuild starts automatically, including a local rebuild. Isolated restore and
later rollback obey the same rules; approving a runtime capability cannot bypass
projection validation. Safe source-content inspection remains available meanwhile.

Stop affected services and prevent new participants from opening their storage.
Produce and verify an encrypted exact local rollback archive before the first live
change. Require a rollback password at this step, even when the incoming backup is
plaintext; the UI can explicitly offer to reuse the entered archive password.
Never persist that password. On restart, recovery can ask for it before rollback.

Rollback retains the exact pre-restore stored-data snapshot and supported captured
managed credentials, without portable export redaction. Exact stored-data recovery
does not promise externally usable authentication. Raw pre-restore config is retained
as bytes even if normal config parsing is broken. If a required current store is
damaged beyond the qualified
rollback capture contract, refuse in-place replacement and offer isolated restore;
do not label an unverifiable rollback copy safe. The source backup and rollback
archive are immutable. Rollback does not depend on successful schema migration.

Capture the readable supported Chatbook-owned credential values used by affected
owners in the encrypted rollback artifact, with their scope/reference mapping and
capture status. Select entries through those owners, not whole-keychain enumeration.
Retaining only a keyring reference is not evidence that its value can be recovered
after another action changes or deletes it. The immutable stored-data snapshot keeps
original references as recovery evidence; activation can require explicit remapping.

Stage credential material separately. Reuse an existing keyring entry only if its
scope and value still match the captured entry. Otherwise restore the captured value
into a new non-conflicting scope and remap through the affected owner, without
overwriting another profile's credential. Journal newly created scopes and changes
to references; rollback cleanup removes only entries still provably owned by that
operation. If an owner cannot safely remap, retain the encrypted value for its
explicit credential-recovery flow rather than silently changing a shared entry.

List unreadable/unexportable owned credentials and external dependencies before
replacement. These are excluded from the credential-recovery guarantee and require
explicit acknowledgement if replacement proceeds; they do not invalidate a verified
stored-data snapshot. Do not capture the process environment or memory-only Sync
keys. Revoked provider tokens, expired sessions, unavailable environment values, and
other externally controlled authentication may require setup even after exact local
data recovery. No provider validation occurs automatically. Portable backup credential
inclusion remains opt-in; this separate encrypted rollback policy is unchanged by
that export choice.

After publication, validate installed artifacts while normal admission stays closed.
Failure triggers rollback when the password remains available or enters Recoverable
interruption awaiting unlock. If neither direction is provable, show Needs attention
and retain all evidence; never boot into an ambiguous mixture of generations.
Successful validation records a durable commit and activation-required generation
state before the maintenance fence can clear. Normal startup then permits local
inspection under that persistent activation restriction.

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
A durable activation-required record for each restored generation applies before
service composition through every supported launch route, not only Open profile or
the recovery launcher. It is local control state, separate from the operation journal
and the convenience report. Commit associates the generation and its affected owner
namespaces with that record before releasing maintenance. Clearing a completed
operation's startup fence does not clear activation requirements.

Record review status per execution/reconnection owner. Existing capability/permission
owners clear only their own requirement after explicit review; enabling one provider
does not resume schedules or sync. Closing the UI, rebooting, normal command launch,
or rebuilding the report cannot grant activation. Missing, corrupt, or mismatched
activation state for a known restored generation keeps those capabilities inactive
while allowing safe local inspection. Imported archives cannot supply already-
approved activation records; every restore, including a later rollback, creates a
new locally controlled generation requiring review. Protocol-unaware old launchers
remain outside the guarantee defined in section 6.

The persistent Needs setup report presents these authoritative requirements rather
than owning them. Normal local content access does not require globally enabling
automatic execution. Activation does not imply replay of old queued work; existing
owners retain their normal explicit reconciliation and permission checks.

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
media uses the durable owner defined below, which normal startup cleanup cannot erase.

### Recovered-media lifecycle

A profile-scoped recovered-media catalog owns a stable asset ID, verified payload
size/digest/type, recovery provenance, and explicit message-to-asset references.
Its physical files have generated private names; no imported locator chooses a
destination. Catalog schema changes use the normal database migration rules. The
owner publishes verified bytes before finalizing references, with operation-journal
recovery for an interrupted file/catalog update.

Transcript media resolution consults the recovered reference for a restored message
and original media key before the temporary-store resolver. Mapping uses source
profile/message identity and the original slug/type, not display name alone. A
known recovered reference with unexpectedly missing bytes renders Missing recovered media rather
than falling through to an unrelated same-named temporary file. Persist mappings
across restart and re-backup; restore collisions require matching identity/content
or explicit remapping, never silent reassignment. No media is decoded or executed
merely to register it during archive validation.

Once recovered, these assets are durable app-owned content and join the baseline of
every subsequent full backup, independent of the temporary-media option. Include
catalog, references, and referenced payloads in one capture dependency group. Include
intentional-deletion tombstones with their references without requiring deleted bytes.
Restore and relocation use this same owner and keep logical asset IDs stable. Existing
live-generated media retains its original session/TTL policy unless explicitly
captured and restored through this feature.

Expose recovered assets and sizes in storage/recovery details with explicit deletion.
Deleting a message releases its reference through the owner; it does not immediately
erase a payload still referenced elsewhere or retained for recovery. Deleting a
recovered asset previews affected messages, retains references marked intentionally
deleted, and removes bytes through the checked owner lifecycle. Persist a versioned
owner tombstone tied to the stable asset ID and affected references; journal tombstone
publication and payload retirement so an interrupted deletion is recoverable. Render
these references as Deleted recovered media, with no temporary-store fallback.
Only validated owner deletion state authorizes absence: never infer a tombstone from
a missing file, failed read, or digest mismatch. A valid tombstone survives subsequent
backup/restore and does not make the backup partial; an absent required payload with
no valid tombstone still blocks completeness. Restoring the tombstone cannot revive
the deleted asset from unrelated retained files. Unreferenced durable payloads
are listed as cleanup candidates and removed only by an explicit user cleanup action
after rechecking references and active recovery holds. Automatic cleanup is limited
to provably operation-owned unpublished staging. It cannot sweep unknown files,
retained rollback, or committed assets based on age, filenames, or temporary-store TTL.

### External content and startup recovery

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
do not treat corrupt control state as no pending recovery and boot affected storage.
Bootstrap admission from section 5 applies even when the user omitted the custom
control-root option on restart. Unaffected profiles can open only after proving
their namespaces disjoint from every pending or uncertain recovery scope.

## 10. Verification and release evidence

Run targeted checks only; a full suite needs separate user authorization. This is a
design-only change, so no application tests are claimed for this document.

Implementation release evidence must include:

- Storage inventory fixtures covering all persistent owner declarations, nondefault
  profiles/configs, custom paths, optional owners, unknown durable entries, aliases,
  shared stores, and deliberately missing data. New persistence owners cannot ship
  without a classification/capture declaration or explicit justified exclusion.
- Inventory changes between preview and maintenance: create an asset, enable an owner,
  remap a database, and introduce a shared alias from another participating process.
  Assert normal in-scope growth is captured, scope/budget changes renew the preview,
  lock reacquisition preserves ordering, and the manifest matches final fenced coverage.
- Real SQLite/WAL data and real filesystem assets captured under concurrent writes
  from separate processes. Prove participating writers drain together and another
  protocol-aware process cannot enter during capture/publication. Cover shared-store
  aliases, replacement changing an inode, old/new path reservations, timeout, and
  drain-order deadlocks. Known incompatible activity must refuse maintenance; tests
  must not imply advisory locks exclude arbitrary legacy or external processes.
- Round-trip comparisons of records, relationships, soft-deleted data, retained
  recovery bytes, asset digests, settings, and indexes in a fresh isolated home/config
  environment with no accidental access to developer data or ambient credentials.
- Both restore destinations, multi-profile/shared-data cases, path remapping, new
  device identities, unsupported newer schemas, staged migration failures, missing
  optional engines, and malformed current config. Prove original isolated-profile
  sources remain unchanged and restored profiles can be reopened later.
- Damaged-installation recovery with missing/malformed/encrypted-unavailable current
  configuration and unreadable current databases. Inspection and isolated restore
  must work without loading those sources; replacement must refuse unverified target
  mappings instead of falling back to default paths. Raw-config rollback capture
  must not depend on successful parsing.
- Target-inventory reconciliation with newer managed files absent from an older
  backup, declared optional exclusions, unknown files, shared stores, and old SQLite
  sidecars. Assert restore/retire/preserve sets, rollback recovery of retired objects,
  and no active stale objects or unreviewed deletion after publication.
- Derived-index omission with newer target-only and deleted-source documents, imported
  projections with incompatible embedding/schema provenance, shared indexes, and
  rollback to an older source generation. Assert stale results are unavailable across
  restart, cache invalidation occurs, no rebuild starts automatically, and retrieval
  resumes only after qualified validation/reconciliation of the restored sources.
- Failure injection before/after every durable journal/publication boundary,
  including process termination, out-of-space, target changes, disconnected volumes,
  unavailable rollback password, corrupt control state, and interrupted rollback.
  Check recovered filesystem state, not just an exception or status message.
- Custom control roots reached by every normal/headless launch route without an
  explicit recovery option, corrupt catalog/config, unavailable control storage,
  interrupted admission registration/clearing, and profiles with disjoint versus
  shared namespaces. Assert affected startup is fenced and provably unrelated
  profiles remain usable.
- Archive attacks and resource limits: traversal, aliases, duplicates, conflicting
  names, malicious SQLite schema, truncation, changed selected source, unsupported
  encryption, huge KDF work, oversized payloads, and misleading compression metadata.
- In-place modification of the original archive after preview, mutation of staged
  input, and overflow during source copy/decryption before a manifest is available.
  Assert execution uses the exact previewed artifact or invalidates the plan, and
  incomplete intermediate bytes never become inspectable/publishable archives.
- Supported SQLite schemas with malicious extra triggers/views/virtual tables or a
  correct version number but altered definitions. Prove installed migration SQL
  cannot activate them, missing security primitives fail closed, SQL budgets cancel
  excessive work, and legitimate FTS/migrations still work under owner-specific rules.
- Credential fixtures spanning config history, supported DB locations, encrypted
  values, references, keyring scopes, and SQLite unused pages. Search emitted plaintext
  artifacts for known managed-secret sentinels; do not rely only on JSON field checks.
- Rollback credential fixtures with affected keyring values changed or deleted after
  capture, scope conflicts, unsupported/unreadable entries, and expired/revoked external
  credentials. Verify readable owned values are retained encrypted, new scopes do not
  alter another profile, remapping is journaled, and unavailable authentication is
  reported distinctly from verified stored-data recovery.
- Encryption interoperability against the official age implementation, bounded
  memory on large archives, packaging on each advertised platform, wrong-password
  behavior, and truncated final-stream detection. Verify secrets never enter process
  arguments/environment, logs, persistent control requests, or helper diagnostics.
- Early helper packaging qualification in installed wheels and documented source/
  editable builds, helper version/integrity mismatch, missing helpers, and upgrade
  interoperability. Capability failures appear before password entry/maintenance;
  no implicit download, PATH substitution, or end-user build-tool requirement.
- Product-level tests entering F9 Settings and first-run/recovery entry points,
  exercising actual create/inspect/restore/rollback services. Include progress,
  cancellation, separate-profile reopening, and retained temporary media after startup.
- Recovered-media transcript lookup, duplicate slugs across source profiles, missing
  bytes, reference deletion, shared references, explicit orphan cleanup, interrupted
  catalog/file publication, and a second backup/restore with temporary-media capture
  disabled. Durable recovered content must remain covered and resolvable.
- Intentional recovered-asset deletion versus unexpected file loss and digest failure.
  Round-trip deletion tombstones and references through another complete backup and
  both restore destinations; assert Deleted versus Missing rendering, no payload
  resurrection/fallback, and missing required bytes still prevent a complete result.
  Inject interruption between tombstone publication and payload retirement.
- Separate timing/lifecycle assertions: normal-backup writers resume before packaging,
  while replacement/rollback remains fenced through encrypted safety-copy verification
  and installed validation. Progress and cancellation describe the actual interval.
- A first-open test with network/process-spawn sentinels and restored queued work,
  proving no remote contact, code execution, synchronization, or schedule catch-up
  occurs before explicit activation.
- Repeat that first-open proof after closing recovery UI, clearing the completed
  operation fence, ordinary CLI/headless launch, reboot-equivalent process restart,
  catalog/report rebuild, and activation-record corruption. Approving one owner must
  not enable another; every restored generation, including rollback, remains gated
  until its own local review completes.
- Output publication with pre-existing filenames, a destination created after preview,
  symlink/hardlink aliases, output nested in selected source folders, and cancellation
  or failure immediately around no-replace publication. Existing source/backups/control
  files remain unchanged and failed output never appears as a completed archive.
- Directory round trips with empty roots, nested empty directories, metadata-only
  changes, mixed file/directory name collisions, case/Unicode collisions, unsupported
  metadata, and platform differences. Recreated structure and supported metadata must
  match the approved manifest; metadata omissions are explicitly reported.

Ship replacement only for qualified OS/filesystem combinations; the UI must derive
availability from the same capability evidence. Self-review and targeted static
checks accompany each implementation slice. Documentation must distinguish an
off-device copy from a backup still stored on the same disk, without claiming to
provide automatic remote storage.

## 11. Delivery boundaries and accepted review changes

This is one product design but more than one implementation PR. After written
approval, planning should first qualify encryption-helper delivery and the maintenance/
bootstrap protocol, then separate storage inventory/profile admission, maintenance
coordination, archive qualification, recovered-media ownership, isolated restore, replacement and
rollback/recovery startup, and final Settings/first-run integration with end-to-end
evidence. These are planning boundaries, not undeclared future task dependencies.
Do not expose Complete backup or destructive replacement before its full contract
is qualified. Existing selective exports remain available during development.

| Review issue | Incorporated resolution |
| --- | --- |
| Advisory instance locking | Stable namespace admission for protocol-aware participants, native qualification, and an explicit legacy/external-writer boundary. |
| Folder-only profile isolation | Dedicated launch config, owner-aware relocation, fresh scopes, alias checks, and reopenable catalog. |
| Recovery depends on working app | Minimal pre-bootstrap recovery path with independent journal and explicit-path recovery. |
| Rollback versus portable export | Exact encrypted local rollback, no export redaction, explicit password and later-change preservation. |
| Credential remnants and shared secrets | Typed staged sanitization, qualified DB processing, separate credential scopes, no global keyring overwrite. |
| Ephemeral content disappears after restore | Explicit optional capture and retained recovered assets outside cleanup namespaces. |
| External consistency and overwrite | Honest per-file capture, newly created destinations only, original overwrite deferred. |
| Overstated completeness | Explicit profile inventory, partial labeling, dependency validation, and separate verification/open statuses. |
| Locks split after replacement or overpromise legacy exclusion | Stable logical namespace locks survive inode/path publication; shared aliases and the supported participant boundary are explicit. |
| Newer files survive a replacement accidentally | Previewed restore/retire/preserve sets, owner-controlled retirement into rollback, and final inventory reconciliation. |
| Pinned handles mistaken for immutable input | Completed private source copies or qualified snapshots, digest-bound previews, and byte limits before copying/decryption. |
| Imported SQLite schema activates during migrations | Actual schema allowlists, restricted inspection/migration connections, execution budgets, and authenticated-but-untrusted metadata. |
| Custom control roots bypass normal startup checks | Fixed bootstrap admission associations consulted by every supported launch route, with scope-aware blocking. |
| Backup pause promise applied to replacement | Separate downtime contracts; replacement keeps admission closed through verified encrypted rollback and publication. |
| Encryption helper selected before packaging proof | Early platform/package/protocol qualification with no runtime download or silent implementation substitution. |
| Recovered media has no complete ownership lifecycle | Stable references, transcript lookup, baseline re-backup, explicit deletion, and guarded orphan cleanup. |
| Damaged config prevents recovery discovery | Separate backup-source and restore-destination discovery; isolated recovery never opens damaged current profiles. |
| Activation restriction disappears on ordinary relaunch | Durable per-generation, per-owner activation state independent of completed-operation fences and reports. |
| Exact rollback promises external credential availability | Capture supported owned values encrypted, remap conflicting scopes, and report external/unreadable authentication separately. |
| Backup publication overwrites an existing good archive | New-file-only flow, source/control alias checks, and atomic no-replace publication under destination races. |
| File-only archive loses empty folder structure | Explicit bounded directory manifest, parent ordering, collision checks, and supported metadata/omission policy. |
| Omitted indexes remain active against replaced sources | Dependent projections are invalidated/quarantined; source and embedding compatibility plus reconciliation gate retrieval under ADR-030. |
| Preview inventory drifts before capture | Rediscover under maintenance, renew scope/budget previews, and bind completeness to final captured inventory. |
| Deliberate media deletion appears as backup corruption | Validated owner tombstones preserve intentional absence and references; unexpected missing payloads still block completeness. |

## 12. Written review boundary

The next step is user review of this document and ADR-126. TASK-31978 remains In
Progress until that review is received. No production changes, implementation task
completion, tests-passing claim, or implementation plan is implied by this draft.

After approval, invoke writing-plans and create atomic implementation Backlog tasks
in dependency order, with ADR links and required evidence scoped to each slice.
