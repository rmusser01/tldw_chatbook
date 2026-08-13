# audio.cpp Clone Voice Bundle Portability Design

**Status:** Approved

**Date:** 2026-08-11

**Task:** TASK-13206
**Extends:** the approved guided-model setup and private clone-reference designs

## Summary

Chatbook will add an explicit, warning-gated portable container for one saved
audio.cpp clone voice while keeping ordinary profile export sanitized. The
container is a strict four-entry ZIP. Import treats it as hostile, validates it
twice across the user-review boundary, and commits only through the profile
repository's atomic profile-plus-reference operation.

The feature also closes a prerequisite provenance gap. A saved clone profile
currently retains its model selection and private reference but discards the
exact recipe ID/revision that admitted generation. The profile store therefore
advances from schema v3 to v4 and stores optional exact recipe provenance beside
the reference. New saves and imports require that provenance. Migrated v3
references retain compatibility but are never assigned invented provenance.

## Goals

- Preserve reference-free ordinary export as byte- and behavior-compatible
  wire v1.
- Make ordinary export of a reference-bearing profile explicitly sanitized.
- Require an operation-local plaintext/sensitive-data acknowledgement before
  any bundle export.
- Transfer only the exact sanitized profile selection, exact recipe/model
  requirement, canonical WAV, and canonical transcript.
- Admit hostile archives without trusting member paths, metadata, or source
  stability.
- Keep UUID/name/dependency conflicts explicit and never overwrite, assign,
  default, or retarget automatically.
- Allow a valid bundle with a missing exact dependency to be imported only as
  an explicitly acknowledged inactive profile.
- Keep private bytes, transcripts, paths, digests, and raw errors below the
  service/UI boundary and out of logs, events, and exception graphs.

## Non-goals

- Character-card embedding, character assignment, or app-default mutation.
- Model weights, model installation, recipe implementation, or automatic
  dependency substitution.
- Encrypted bundles, signatures, identity claims, legal-consent proof, or
  forensic erasure claims.
- Sending a local clone reference to External audio.cpp or user-JSON sources.
- Windows portability before owner-private ACL containment is verified under
  ADR-029.
- A general archive-import framework or a second profile-management screen.

## Governing decisions

This design amends ADR-051 and continues to follow ADR-028 and ADR-029. No new
ADR is required: ADR-051 already owns clone-reference schema, backup, privacy,
runtime admission, and portability policy.

## User journeys

### Ordinary safe export

1. The user selects a voice profile and activates **Export**.
2. A reference-free profile opens the existing JSON destination picker and
   emits exact wire v1.
3. A reference-bearing profile opens a choice modal. **Export sanitized
   profile** is the default; **Export portable voice bundle** is secondary.
4. Sanitized export emits wire v2 and never reads the reference BLOB.

### Explicit bundle export

1. The user chooses **Export portable voice bundle**.
2. Chatbook shows a fixed acknowledgement:

   > This bundle contains plaintext voice audio and transcript. Anyone with
   > the file can access them. I confirm I have permission to export and share
   > this material.

3. Continue remains disabled until the operation-local checkbox is selected.
4. The user chooses a destination. Chatbook freezes the selected repository
   generation/profile revision, reads the exact reference and recipe
   requirement, builds a deterministic private temporary file, and publishes
   it atomically.
5. Cancellation or failure publishes nothing and joins cleanup.

### Explicit bundle import

1. The user activates **Import voice bundle**.
2. Before opening the file picker, Chatbook explains that import reads
   plaintext voice audio and transcript and that a bundle declaration is not
   proof of permission or identity.
3. The user selects a candidate. File-extension filtering is only a convenience.
4. The app-owned bundle portability service creates one bounded, expiring,
   single-use inspection session. A retained worker copies and validates the
   unchanged source in an owner-private operation directory. It performs no
   adapter acquisition, registry lease, supervisor preparation, audio.cpp
   launch, health probe, HTTP request, or Settings mutation.
5. The review modal displays only validated safe facts: profile name/UUID,
   provider/model, recipe ID/revision, dependency state, UUID/name conflicts,
   the proposed copy name/UUID, and whether an exact private duplicate exists.
6. The user chooses the destination action independently from dependency
   consent:
   - **Create** when no collision exists;
   - **Reuse exact existing profile** when every relevant public and private
     value matches;
   - **Import as copy** with the displayed collision-free UUID/name; and
   - when the dependency is unavailable, a separate acknowledgement to create
     the selected new/copy destination inactive.
7. Commit consumes the opaque session once, reopens and fully revalidates its
   privately retained source authority, compares the inspection fingerprint,
   and refreshes local-only dependency evidence. Any changed visible fact
   returns a fresh session/review and requires confirmation again.
8. One serialized repository-lane transaction rechecks exact UUID/name/profile
   revision/reference identity and either confirms exact reuse or atomically
   creates the profile, recipe requirement, and reference. It creates no
   assignment and changes no default.

## Ordinary portability formats

### Wire v1

Reference-free profiles retain the current exact v1 field set, validation,
serialization, and unsupported-provider behavior. This task does not add fields
or reorder output for v1.

### Wire v2

A reference-bearing ordinary profile export contains this exact object shape,
using the existing v1 validators and serialization order:

```json
{
  "schema_version": 2,
  "profile_id": "<canonical UUID>",
  "name": "<validated display name>",
  "provider_id": "<validated provider ID>",
  "model_id": "<validated model ID>",
  "voice_id": "<validated exact voice ID or null>",
  "response_format": "<validated format>",
  "speed": 1.0,
  "options": {},
  "reference": {"status": "omitted"}
}
```

It contains no WAV, transcript, reference UUID, digest, size, duration, recipe
provenance, timestamps, path, assignment, endpoint, generated configuration,
or runtime state. An older v1-only reader returns its existing bounded
unsupported-version result. The new decoder recognizes the exact v2 omission
shape only as a bounded `reference_omitted` skip/no-mutation outcome. This task
does not add standalone sanitized-profile import or an attach-local-reference
workflow; **Import voice bundle** is a separate explicit action.

## Bundle format

### Container

Bundle schema v1 is a non-ZIP64 single-disk ZIP with exactly four regular-file
members in this order:

```text
manifest.json
profile.json
reference.wav
reference.txt
```

The application writer uses `ZIP_STORED`, extract/create version `20`, create
system `3` (Unix), general-purpose flags `0`, fixed DOS timestamp
`1980-01-01T00:00:00`, regular-file external attributes `0o100600 << 16`, zero
internal attributes, and no archive/member comments, extra fields, encryption,
or data descriptors. JSON is strict UTF-8 without a BOM, sorted keys,
`ensure_ascii=False`, separators `(",", ":")`, finite numbers only, and one
trailing LF. `reference.txt` is the canonical bounded UTF-8 transcript without
a BOM or added newline.

The writer's complete archive bytes are deterministic for the same canonical
input. Import may accept the bounded interoperable metadata below; canonical
decoded contents, not compressed bytes, define equivalence for repacking.

Accepted importer metadata is exact:

- compression method `0` (`STORED`) or `8` (`DEFLATED`);
- general-purpose flag bits either `0` or only bit 11 (UTF-8 names); encryption,
  data-descriptor, patched-data, strong-encryption, masked-header, and every
  other bit are rejected;
- version-needed `10` or `20` for `STORED`, exactly `20` for `DEFLATED`, and
  version-created `10` or `20`;
- create system `3` (Unix) with a regular-file mode and no setuid/setgid/sticky
  bits, or create system `0` (DOS) with directory and volume-label attributes
  clear; and
- no member/archive comment, extra field, multipart marker, or ZIP64 value.

Member names are the four exact ASCII strings, so the optional UTF-8-name bit
does not change their decoded value. Imported mode attributes never determine
staging permissions; Chatbook creates every staged file itself as owner-private.

### `profile.json`

This is bundle profile schema v1, distinct from ordinary wire v1 only in its
canonical compact encoding. Its exact object is:

```json
{
  "model_id": "<1..256 UTF-8-safe characters>",
  "name": "<validated profile display name>",
  "options": {},
  "profile_id": "<canonical lowercase-hyphenated UUID>",
  "provider_id": "audio_cpp",
  "response_format": "wav",
  "schema_version": 1,
  "speed": 1.0,
  "voice_id": "<validated exact voice ID or null>"
}
```

The semantic fields are:

- profile UUID hint;
- display name;
- provider ID (`audio_cpp`);
- public model ID;
- optional exact voice ID;
- response format;
- speed; and
- empty canonical options under the current audio.cpp profile contract.

Import validates the exact voice/reference combination against the identified
recipe. A recipe permitting both preserves the voice ID; export never coerces
it to null. In both JSON examples, the voice placeholder denotes either a
validated JSON string or the JSON value `null`; it is not a literal wire value.

It contains no database revision, normalized name, timestamps, reference
metadata, assignment, default, endpoint, credential, origin, path, generated
configuration, or process evidence.

### `manifest.json`

The exact manifest object is:

```json
{
  "bundle_format": "tldw_chatbook.clone_voice_bundle",
  "declaration": {
    "plaintext_sensitive_data_acknowledged": true,
    "version": 1
  },
  "dependency": {
    "model_id": "<must equal profile.json model_id>",
    "provider_id": "audio_cpp",
    "recipe_id": "<audio.cpp recipe token>",
    "recipe_revision": 1
  },
  "entries": {
    "profile.json": {"sha256": "<64 lowercase hex>", "size": 1},
    "reference.txt": {"sha256": "<64 lowercase hex>", "size": 1},
    "reference.wav": {"sha256": "<64 lowercase hex>", "size": 44}
  },
  "reference": {
    "byte_length": 44,
    "channels": 1,
    "duration_ms": 1,
    "sample_encoding": "pcm_s16le",
    "sample_rate_hz": 16000
  },
  "schema_version": 1
}
```

Every key shown is required and every unshown key is rejected. The value
placeholders are validated values, not literal defaults. `recipe_id` uses the
ASCII grammar `[a-z0-9][a-z0-9._-]{0,127}` and recipe revision is an exact
integer in `1..2147483647`. Model IDs retain the profile domain's 256-character
UTF-8/control-safe bound.

The strict manifest therefore contains only:

- format ID and bundle schema version;
- fixed declaration version and
  `plaintext_sensitive_data_acknowledged: true`;
- exact provider/model/recipe ID and positive recipe revision;
- bounded canonical WAV facts: byte length, duration, sample rate, channels,
  and sample encoding; and
- exact byte length and lowercase SHA-256 for `profile.json`,
  `reference.wav`, and `reference.txt`.

The manifest cannot checksum itself without recursion. Its integrity is
established by strict schema/content validation and the archive/source bounds.
Checksums prove byte integrity only. The UI never calls them a signature,
authenticity proof, speaker identity, or consent proof. The fixed declaration
records only that the export flow was acknowledged; import treats it as
untrusted informational data and always displays its own warning.

### Numeric limits

| Limit | Value |
| --- | ---: |
| Member count | exactly 4 |
| Source/archive bytes | 40 MiB |
| `manifest.json` uncompressed | 64 KiB |
| `profile.json` uncompressed | 16 KiB |
| `reference.wav` uncompressed | existing 32 MiB canonical-reference cap |
| `reference.txt` uncompressed | existing 16 KiB UTF-8 cap |
| Aggregate uncompressed | 33 MiB |
| Aggregate compressed | 40 MiB |
| Per-member/aggregate expansion ratio | 100:1 |
| JSON nesting | 4 container levels |

Limits are checked from central metadata before decompression and again while
streaming; declared metadata never relaxes a streaming counter.

## Schema v4 and recipe provenance

### Domain

Introduce one bounded immutable recipe requirement containing:

- exact recipe ID;
- positive recipe revision; and
- exact public model ID, cross-checked against the owning profile.

The reference summary and exact-reference projections carry the optional
requirement without exposing the private digest, transcript, or BLOB. New clone
profile saves and bundle imports require it. Repository writes reject a
half-present or model-incoherent requirement.

### Migration

Migration never upgrades the active file in place. Under exclusive repository
ownership it creates an owner-private candidate from a consistent SQLite
backup and advances that candidate through every required schema step.

For each supported starting version:

- v3: validate v3, durably prepare the exact candidate as
  `<profile-db>.pre-v4.sqlite3`, then migrate a separate candidate to v4;
- v2: durably prepare validated v2 as the existing pre-v3 backup, migrate the
  candidate to validated v3, durably prepare that exact intermediate as the
  pre-v4 backup, then migrate the candidate to v4; and
- any older supported version: advance the private candidate through the
  existing ordered migrations, preparing the applicable validated v2 and v3
  downgrade snapshots at those boundaries before reaching v4.

Restore qualification uses the same rule. A v3 restore candidate prepares a
new pre-v4 snapshot before active v4 publication; an older candidate prepares
every applicable boundary snapshot. New retained backups are not yet
authoritative: every prior retained backup remains under rollback identity
through the active-file replacement, reopen, and validation protocol.
All new candidates and backups are durably prepared under private temporary
identities. A small owner-private publication journal records the old/new
identities before the point of no return; startup recovery completes or rolls
back an interrupted multi-file publication before opening any store.

Final migration cleanup is descriptor-bound. Because macOS/Linux POSIX APIs do
not expose an exact-inode unlink-by-descriptor, cleanup atomically quarantines
each exact journal/candidate/rollback leaf without replacement and fsyncs the
pinned regular-file descriptor and parent. Cleanup never truncates: a hardlink
can race any preceding link-count check. The exact owner-only `0600` tombstone
may therefore retain private bytes as bounded cleanup evidence.
The tombstone set is bounded by the journal and the candidate/rollback leaves
for the maximum three migration slots and does not grow on replay. Candidate
and rollback authority is closed by slot to
`.profile-migration-{active|pre-v3|pre-v4}.candidate.sqlite3` and the matching
`.rollback.sqlite3` leaf. Preparation rejects any other leaf before hashing,
and journal construction and parsing enforce the same mapping. Tombstones are
never parsed as recovery authority. An already-zero tombstone is reusable only
after exact parent/inode/type/uid/mode/link/zero-length and sidecar validation.
A nonzero tombstone is never overwritten or reused and may be retried only if
safe disposal eligibility can later be proven. A foreign or substituted
occupant is never deleted; cleanup fails closed with bounded unavailability.
Race, cancellation, replay, foreign-holding, and initial-journal cleanup tests
in `Tests/TTS/test_profile_migration_recovery.py` and
`Tests/TTS/test_profile_migration_publication.py` enforce this protocol.

The v3→v4 candidate transaction adds nullable recipe ID/revision fields with a
both-null-or-both-valid invariant, preserves every existing value, and leaves
both fields null. It never infers provenance from Settings or a recipe catalog.
The candidate is fully domain-validated, checkpointed, fsynced, closed, and
reopened before publication.

Before active-file replacement, cancellation may abort and leaves the original
active database byte-for-byte authoritative. Entering atomic replacement is a
documented non-cancellable point of no return. The repository retains the old
active file under a private rollback identity, replaces with the validated v4
candidate, fsyncs the directory, reopens/validates v4, and only then releases
rollback ownership. A post-replace failure synchronously restores and fsyncs
the old active file and every prior retained backup before returning failure.
Only after the new active store is reopened and authoritative are the prepared
pre-v3/pre-v4 backups durably published and prior rollback identities released.
If storage failure prevents completion or full restoration, the repository
remains boundedly unavailable with all recovery files retained; it does not
claim that the failed restore candidate or its backups are authoritative.

The task acceptance contract therefore distinguishes pre-publication rollback
from the non-cancellable publication protocol rather than promising an
impossible rollback under total storage failure. Retained backup paths are
owner-private, reserved against normal backup destinations and hardlink
aliases, and never logged in full.

Downgrade requires closing Chatbook, restoring the retained pre-v4 database,
then opening it with a v3-capable build. This loses post-migration changes and
all recipe provenance added under v4; switching runtime setup is not a database
downgrade.

### Legacy references

Migrated references with absent provenance remain executable under the existing
guided-Managed compatibility checks. Their library row additionally shows
`Recipe provenance unavailable`, and bundle export is disabled with the
recovery **Preview/generate this voice, save it as a new profile, then reassign
or remove the legacy profile if desired**. Chatbook does not claim an in-place
regenerate operation. Legacy references can never qualify as an exact bundle
duplicate because recipe equality is unknown.

This compatibility exception applies only to migrated null provenance. Every
new or replaced v4 reference requires exact recipe evidence.

## Availability and runtime admission

The existing availability vocabulary remains `available`, `unavailable`, and
`unverified`. Add a bounded blocking dependency reason rather than another
broad state. Relevant blocking reasons include:

- `none`;
- `recipe_missing`;
- `recipe_mismatch`; and
- `recipe_pending_apply`.

Recipe provenance absence is a separate nonblocking portability advisory, not
a competing availability reason. It stays visible when another blocker owns
the primary action.

Add a separate immutable dependency-action projection so existing generic
profile recovery actions are not overloaded:

| Reason | Display | Action |
| --- | --- | --- |
| `recipe_missing` | Needs compatible model | `open_audio_cpp_settings` |
| `recipe_mismatch` | Needs compatible model | `open_audio_cpp_settings` |
| `recipe_pending_apply` | Compatible model saved; apply settings | `open_speech_lab_apply` |

The separate `recipe_provenance_unavailable` advisory displays **Recipe
provenance unavailable** and offers secondary action `generate_new_profile`
(open exact profile Preview in Speech Lab).

Primary blocker precedence is:

1. damaged or structurally invalid reference/profile;
2. provider/configuration unavailable;
3. exact recipe missing, mismatched, or pending apply; and
4. no blocker.

The provenance advisory is rendered alongside whichever primary state wins. It
never replaces a repair action with an impossible generate action.

The reason is derived from the persisted requirement and an immutable
local-only saved/applied guided-configuration plus recipe-registry snapshot.
That snapshot API is pure: it cannot acquire/materialize an adapter, take a
registry lease, prepare a supervisor, probe health, launch, contact a provider,
or write Settings. Unknown but valid recipes classify as missing. Installing or
configuring the exact dependency may change availability after explicit
refresh, but never assigns the profile or changes a default.

For a reference carrying a recipe requirement, the service-owned clone
execution snapshot includes the exact requirement. Before acquiring a provider
lease or materializing an adapter, request admission compares it with the pure
applied registry/configuration snapshot. After the exact lease is acquired, a
side-effect-free adapter configuration preflight repeats the check before
`ensure_ready`; recipe/model/config mismatch still causes no
launch/network/provider work. After deliberate readiness, admission compares
the same requirement again with adapter-issued process-generation capability.
Post-ready generation drift may be detected after launch/health work but always
blocks before reference materialization or the synthesis HTTP request. Neither
failure falls back or retargets.

An unknown but syntactically valid future recipe is a missing dependency, not a
malformed bundle, and may be stored inactive after explicit confirmation.
Invalid syntax, non-positive revisions, or cross-field disagreement is a
malformed bundle and is rejected.

### Profile edit semantics

Reference-bearing profiles may always change display name through the existing
optimistic-revision editor. Generation-field edits are read-only in that
editor. Changing model, voice, format, speed, or options requires producing a
new exact result and saving a new profile, or a future explicit atomic
reference-replacement workflow. Every repository reference set/replacement API
requires exact recipe evidence under v4. Migrated null-provenance profiles have
the same display-name-only edit boundary until regenerated as a new profile.

## Hostile archive admission

### Source and containment

- Refuse work before reading private content unless owner-private containment
  is verified for the platform.
- POSIX application-owned staging, temporary output, backup, and final bundle
  files verify type, effective-user ownership, no-follow identity, owner-only
  traversal/modes, and retained descriptor identity under an ownership lock.
- A user-selected source or destination parent (for example `Downloads`) is not
  recursively chmodded and need not be owner-only. The source must be a bounded
  no-follow regular file owned by the effective user and remain identity/content
  stable. The destination parent must be descriptor-stable and writable; the
  final sensitive bundle itself is published mode `0600` on POSIX.
- Until Windows ACL parity is verified, the UI disables bundle import/export
  with a truthful unsupported-platform explanation.
- Open the user-selected source without following symlinks, bound its size
  before copying, and verify identity/size/mtime plus a full source digest
  before and after the copy.
- Parse only the private staged copy. Re-open and repeat the complete operation
  at commit; compare a private inspection fingerprint so the modal never owns
  extracted bytes or paths.

### Structural rejection

Reject before repository mutation:

- encryption or unsupported general-purpose flags;
- duplicate names and case/Unicode-normalization collisions;
- names other than the four exact ASCII names, including absolute paths,
  traversal, alternate separators, drive/UNC forms, or directory entries;
- symlink, device, FIFO, socket, or other special-file metadata;
- central/local-header disagreement, invalid offsets, overlapping member
  ranges, data descriptors, ZIP64, multipart archives, archive/member comments,
  or unexpected extra fields;
- prefix payloads, trailing payloads, or bytes outside the exact archive layout;
- unsupported compression, per-entry/aggregate compressed and uncompressed
  limits, excessive ratios, or excessive member count;
- CRC, declared size, manifest size, or SHA-256 mismatch;
- malformed/oversized/deep JSON, invalid UTF-8/control characters, invalid
  profile fields, or inconsistent dependency fields; and
- noncanonical, malformed, unsupported, truncated, or quota-violating WAV and
  transcript content.

Never call general-purpose archive extraction. Stream each accepted member
through explicit byte counters into an application-chosen fixed staging file,
then run the existing canonical WAV/transcript validators. Cleanup follows
retained identities, never paths supplied by the archive.

## Inspection, conflict, and commit authority

An application-scoped `TTSVoiceBundlePortabilityService` owns inspection
sessions independently of Textual widgets. At most four sessions may exist;
each expires after ten minutes, is single-use, has a redacted representation,
and privately retains source path/descriptor identity, source/content
fingerprint, safe review facts, repository/config evidence, and cleanup state.
It retains no first-pass extracted staging directory while the modal is open.
The UI receives only an opaque unforgeable handle plus safe canonical facts;
the handle contains no public path/reference fields and cannot be copied.

Cancel, modal replacement/unmount, commit, expiry, and service close invalidate
the handle and join owned cleanup. Replay and foreign-service handles fail
boundedly. The service closes and joins before the profile repository closes.
On construction or first use, a nonempty owner-private operation root reports
`cleanup_failed` and remains untouched, even when an entry resembles a prior
operation. Restart does not create deletion authority. Recovery requires
exiting Chatbook, manually inspecting the app-owned portability root, removing
only confirmed residue, and retrying; the bounded error does not reveal its
runtime path.

Exact duplicate reuse requires equality of:

- profile UUID/name and all generation fields;
- recipe ID/revision/model requirement;
- canonical transcript; and
- canonical WAV bytes.

Private equality is computed below the service boundary and projected as one
boolean. A local digest is never exposed as an oracle. Reuse is a no-op and
reports `Exact voice profile already exists — no changes made`.

UUID/name conflicts that are not exact duplicates permit only an explicitly
reviewed copy. The modal shows the generated collision-free name and UUID
before confirmation. Dependency consent is orthogonal: Create or Copy may be
confirmed inactive when the exact dependency is missing; Reuse cannot mutate
the existing profile's state.

Commit consumes only an exact live session and repeats source validation plus
local dependency inspection. It then submits one repository command. Inside
the serialized repository lane and one `BEGIN IMMEDIATE` transaction, that
command re-reads UUID/name matches, expected profile revisions and exact
reference identity, validates the reviewed copy destination, and either
confirms exact reuse or inserts profile+recipe+reference atomically. Any
disagreement returns `stale_inspection` plus safe refreshed facts; no partial
write occurs. No character or default owner is in the call graph.

Dependency snapshot validation follows the producer's full cross-product,
not a state-to-pending shortcut. `exact` requires only the exact applied
requirement and may report a missing/drifted saved requirement plus queued
settings (`pending_configuration=true`); `missing` and `mismatch` may report
either pending value; and `pending` requires queued configuration plus the
exact saved requirement. The pending flag remains coherent with saved/applied
generations. Inspection and commit consume only these pure facts and perform no
adapter, provider, launch, health, network, or Settings work.

## Export publication

After acknowledgement and destination selection, export:

- freezes the exact selected profile generation/revision;
- reads the exact reference and recipe requirement under repository fences;
- builds the complete ZIP in an owner-private temporary file in the validated
  destination directory;
- requires the selected destination to remain absent;
- fsyncs the file, publishes by atomically moving the sibling with the native
  no-replace primitive, and fsyncs the directory where supported; and
- retains a randomized `0600` sibling if failure occurs before that move,
  because pathname deletion in a user-selected parent is not exact-safe; the
  bounded recovery explains that this hidden random sibling may remain and may
  be removed manually only after the user verifies its random filename.

If the selected filename already exists, the user must choose a different name
(or remove it outside this operation and reselect). A destination appearing or
parent type/identity change before publication returns `destination_changed`.
The final bundle is owner-private even when its user-selected parent is not.

The successful no-replace rename is the export point of no return. Before it,
cancellation propagates and leaves no final; its randomized `0600` sibling is
retained as harmless cleanup evidence rather than pathname-deleted.
After it, cancellation is deferred and the retained worker completes exact
final identity, mode, content, and parent-fsync verification before reporting
successful publication. Ordinary post-publication faults are retried through
that convergence path. The service never unlinks any export pathname: POSIX has
no exact-inode unlink-by-descriptor operation, and a `stat`-then-`unlink` pair
could delete a foreign substitution. If post-publication substitution or total
storage failure makes verification impossible, the service preserves the
pathname occupant and returns bounded cleanup/unavailability; it does not
claim rollback.

The acknowledgement is not persisted as a preference. Selection changes,
navigation, modal replacement, or shutdown fence late publication and UI
status.

## Lifecycle and cancellation

Archive copy, validation, decompression, atomic export publication, and cleanup
run as retained bundle-service work. Repository migration, backup, and commit
remain retained repository-owned work.

Cancellation joins a shielded blocking worker before propagating. Before bundle
publication it retains the randomized private temporary sibling; after the
export point of no return it defers cancellation and reports successful
publication only after convergence. UI unmount prevents stale presentation but
does not abandon work.

The bundle portability service closes before repository shutdown. Its close
seals new import/export/session admission, invalidates handles, and joins
sessions, copies, decompressors, output publication, cleanup, and repository
commands submitted by this service. Repository `close()`/`wait_closed()` then
owns migration, backup, and general commit joining. Only composite application
shutdown, after both services have joined in that order, claims global zero
ownership.

## Error and privacy contract

Bounded public categories and recoveries are:

| Code | Recovery |
| --- | --- |
| `bundle_invalid` | Choose another bundle |
| `bundle_limit_exceeded` | Choose a bundle within documented limits |
| `source_changed` | Reselect the source and inspect again |
| `unsupported_bundle` | Use a supported Chatbook bundle version |
| `unsupported_platform` | Use a platform with verified private containment |
| `dependency_missing` | Open audio.cpp Settings |
| `dependency_changed` | Refresh the review; then open Settings or apply saved configuration |
| `recipe_provenance_unavailable` | Preview/generate and save a new profile |
| `profile_conflict` | Review the exact current collision choices |
| `stale_inspection` | Inspect again |
| `destination_changed` | Choose a new absent destination |
| `migration_failed` | Keep/restore the prior store and retry after resolving storage health |
| `cleanup_failed` | Wait for retained cleanup; if it persists after restart, exit Chatbook, inspect the app-owned portability root manually, remove only confirmed residue, and retry |
| `operation_failed` | Retry without exposing collaborator detail |

UI copy supplies one truthful next action. Logs, notifications, metrics,
events, object representations, and exception graphs contain no full source or
destination path, raw ZIP/member name, transcript, audio, private/staging path,
checksum, credential, provider origin, generated configuration, or raw
collaborator exception. Validated bounded profile name/model/recipe/UUID facts
may appear only in the explicit review UI, not general logs.

## Accessibility and action truth

- The import warning appears before the file picker and receives initial focus.
- Export defaults to sanitized JSON; the sensitive choice is secondary.
- Continue is disabled with a visible reason until acknowledgement/required
  choices are complete.
- Modal focus order is heading, safe facts, destination choice, inactive
  acknowledgement when applicable, Confirm, Cancel.
- Conflict and inactive status use text, not color alone.
- Voice Profile library rows and every Roleplay/Personas assignment consumer
  render the same immutable dependency reason/action; inactive profiles remain
  visibly unavailable and cannot be assigned.
- Narrow layouts keep the warning, proposed copy destination, and primary
  action reachable through a focusable scrolling region.
- Busy actions expose their reason; cancellation/retry remain reachable.
- The visible label and executed operation derive from one immutable action
  projection, including after late workers or refreshed inspection facts.

## Verification

### Automated

- Exact v1 compatibility and strict sanitized v2 omission/old-reader behavior.
- Deterministic writer bytes, manifest/content checksums, and logical
  equivalence for accepted deflated input.
- Named hostile archive matrix plus bounded property-based ZIP metadata/name/
  size/flag generation.
- Source mutation at every open/copy/review/commit boundary.
- v3→v4 migration backup, domain equivalence, no guessed provenance, disk-full
  rollback, non-cancellable publication, forced post-replace restoration,
  retained-backup preservation, v1/v2/v3 multi-hop and restore candidates,
  downgrade, aliases, and newer-schema refusal.
- Conflict/dependency cross-product across Create, Reuse, Copy, and inactive
  consent.
- Exact private duplicate comparison without digest exposure.
- Passive inspection proving zero audio.cpp launches, HTTP requests, or
  Settings writes, adapter acquisitions, registry leases, supervisor work, or
  health probes.
- Inspection-session cap, expiry, single-use/replay/foreign-handle refusal,
  unmount invalidation, and close-before-repository ordering.
- Repository-lane races proving collision/reference changes cannot interleave
  between recheck and exact reuse/create.
- Runtime admission proving recipe/model/config mismatch before provider
  lease/adapter/provider work, exact adapter preflight, post-ready generation
  drift refusal before private materialization or synthesis HTTP, and no
  fallback.
- Display-name-only reference-profile edits and blocked generation-field edits
  across exact and migrated-null provenance.
- Cleanup and ownership under success, refusal, malformed input, cancellation,
  unmount, migration failure, service close, and application shutdown.
- Textual keyboard/focus/announcement/narrow-layout/late-result tests.
- Cross-surface Voice Profile and Roleplay/Personas reason/action truth and
  inactive-assignment refusal.
- Log/event/notification/repr/exception-graph canaries for every private value.

### Manual UAT

Use two separate Chatbook launches with independently isolated temporary
config, data, profile, Model Library/package, generated-config, and runtime
roots plus isolated HOME/XDG/environment overrides; never use the developer's
live stores. Verify launch B cannot observe launch A's installed dependency.

1. On launch A, create a clone profile and audibly verify generation.
2. Export ordinary v2 JSON and inspect its allowlisted structure to prove the
   reference is omitted.
3. Prove bundle export cannot continue before acknowledgement, then export.
4. On launch B without the dependency, import the bundle inactive and prove it
   is persisted, visible as **Needs compatible model**, and unassigned.
5. Restart launch B and prove the inactive state persists.
6. Configure the exact pre-provisioned recipe/model, refresh, generate with the
   imported profile, play it, and confirm audible output. Prove assignment and
   app default remain unchanged.
7. Exercise cancellation and app shutdown; record zero owned staging
   directories, workers, handles, partial profiles, or output publications.

Evidence records exact application commit, schema/bundle versions, platform,
recipe/model identifiers, sanitized WAV metadata, state transitions, cleanup
counts, and human playback confirmation. It never records the transcript,
audio, source/bundle path, reference checksum, or private staging location.

## Alternatives rejected

| Alternative | Reason |
| --- | --- |
| Put the reference in ordinary profile/card export | Surprises a safe sharing path with sensitive plaintext. |
| Reuse current Settings to infer recipe provenance | Current configuration is not historical generation authority and may have changed. |
| Retain extracted bytes in the modal/UI | Extends private-data lifetime and moves authority above the service boundary. |
| Validate once and trust the source path after review | Allows source replacement between inspection and commit. |
| Auto-retarget a missing dependency | A different recipe/model can change voice semantics and request requirements. |
| Add a permanent inactive database flag | Duplicates derivable dependency truth and can become stale. |
| Always create a duplicate | Avoids equality work but produces unnecessary private copies and ignores the approved exact-reuse choice. |
| Build a separate portability screen | Adds another profile owner/surface for one fixed workflow. |

## References

- [Guided model setup design](2026-08-09-audio-cpp-guided-model-setup-design.md)
- [ADR-028](../../../backlog/decisions/028-character-tts-generation-profile-ownership.md)
- [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md)
- [ADR-051](../../../backlog/decisions/051-private-tts-clone-reference-assets.md)
