# audio.cpp Clone Voice Bundle Portability Design

**Status:** Proposed

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
4. A retained worker copies and validates the unchanged source in an
   owner-private operation directory. It performs no audio.cpp launch, HTTP
   request, or Settings mutation.
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
7. Commit reopens and fully revalidates the source, compares the inspection
   fingerprint, rechecks repository/configuration evidence, and requires fresh
   confirmation if any visible fact changed.
8. The repository atomically creates the profile, recipe requirement, and
   reference. It creates no assignment and changes no default.

## Ordinary portability formats

### Wire v1

Reference-free profiles retain the current exact v1 field set, validation,
serialization, and unsupported-provider behavior. This task does not add fields
or reorder output for v1.

### Wire v2

A reference-bearing ordinary profile export contains exactly the v1 sanitized
selection fields with `schema_version` set to `2` and one additional field:

```json
"reference": {"status": "omitted"}
```

It contains no WAV, transcript, reference UUID, digest, size, duration, recipe
provenance, timestamps, path, assignment, endpoint, generated configuration,
or runtime state. An older v1-only reader returns its existing bounded
unsupported-version result. Import never silently constructs a broken
reference-bearing profile from v2; the user must attach a local reference,
import a separate bundle, or skip it.

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

The application writer uses `ZIP_STORED`, fixed DOS timestamps, fixed creator
and owner-private mode attributes, no archive/member comments, no extra fields,
and no data descriptors. JSON is strict UTF-8 without a BOM, canonical key
ordering, finite numbers only, and one trailing newline. `reference.txt` is the
canonical bounded UTF-8 transcript without a BOM or an added newline.

The writer's complete archive bytes are deterministic for the same canonical
input. Import may accept bounded `ZIP_STORED` or `ZIP_DEFLATED` payload entries;
canonical decoded contents, not compressed bytes, define equivalence for
third-party re-packing.

### `profile.json`

This is the exact sanitized portable profile selection:

- profile UUID hint;
- display name;
- provider ID (`audio_cpp`);
- public model ID;
- optional exact voice ID;
- response format;
- speed; and
- empty canonical options under the current audio.cpp profile contract.

It contains no database revision, normalized name, timestamps, reference
metadata, assignment, default, endpoint, credential, origin, path, generated
configuration, or process evidence.

### `manifest.json`

The strict manifest contains only:

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

Opening a v3 store performs this guarded sequence:

1. Validate the v3 schema/domain and establish exclusive migration ownership.
2. Create and durably publish an owner-private sibling
   `<profile-db>.pre-v4.sqlite3` backup without replacing a retained backup on
   failed publication.
3. In one immediate transaction, add nullable recipe ID/revision fields with a
   both-null-or-both-valid invariant.
4. Preserve every existing profile/reference value and leave both new fields
   null. Never infer provenance from current Settings or a recipe catalog.
5. Validate the v4 schema/domain, commit, fsync the database and containing
   directory, then publish the repository as open.

Insufficient space, backup failure, validation failure, cancellation, or
publication failure leaves the v3 store authoritative and openable or the
repository boundedly unavailable. The retained backup path is owner-private,
reserved against normal backup destinations and hardlink aliases, and never
logged in full. The existing retained pre-v3 backup remains independent.

Downgrade requires closing Chatbook, restoring the retained pre-v4 database,
then opening it with a v3-capable build. This loses post-migration changes and
all recipe provenance added under v4; switching runtime setup is not a database
downgrade.

### Legacy references

Migrated references with absent provenance remain executable under the existing
guided-Managed compatibility checks. Their library row additionally shows
`Recipe provenance unavailable`, and bundle export is disabled with the
recovery **Regenerate and save this voice profile**. They can never qualify as
an exact bundle duplicate because recipe equality is unknown.

This compatibility exception applies only to migrated null provenance. Every
new or replaced v4 reference requires exact recipe evidence.

## Availability and runtime admission

The existing availability vocabulary remains `available`, `unavailable`, and
`unverified`. Add a bounded reason dimension rather than another broad state.
Relevant reasons include:

- `none`;
- `recipe_provenance_unavailable`;
- `recipe_missing`; and
- `recipe_mismatch`.

`recipe_missing` and `recipe_mismatch` render **Needs compatible model** with a
Settings/Model Library recovery. The reason is derived from the persisted
requirement and passive saved/applied recipe/model evidence; it is not a
mutable database status flag. Installing/configuring the exact dependency may
change availability after explicit refresh, but never assigns the profile or
changes a default.

For a reference carrying a recipe requirement, the service-owned clone
execution snapshot includes the exact requirement. Before reference
materialization or HTTP, admission compares it with the adapter-issued recipe
ID/revision/model evidence for the applied provider/process generation. A
mismatch fails with a bounded recovery and no fallback, launch, or request.

An unknown but syntactically valid future recipe is a missing dependency, not a
malformed bundle, and may be stored inactive after explicit confirmation.
Invalid syntax, non-positive revisions, or cross-field disagreement is a
malformed bundle and is rejected.

## Hostile archive admission

### Source and containment

- Refuse work before reading private content unless owner-private containment
  is verified for the platform.
- POSIX containment verifies type, effective-user ownership, no-follow
  identity, owner-only traversal, and retained descriptor identity under an
  ownership lock.
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

The public UI inspection result contains only safe canonical facts and a
service-owned opaque fingerprint. It never contains reference bytes,
transcript, staging path, source digest, or entry digests.

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

Commit accepts only the exact fresh inspection, repeats source validation,
rechecks repository generation/profile revisions and passive dependency facts,
and recomputes conflicts. If a visible fact changed, it returns a fresh review
instead of applying the old choice. The repository performs one atomic
profile-plus-recipe-plus-reference write. No character or default owner is in
the call graph.

## Export publication

After acknowledgement and destination selection, export:

- freezes the exact selected profile generation/revision;
- reads the exact reference and recipe requirement under repository fences;
- builds the complete ZIP in an owner-private temporary file in the validated
  destination directory;
- fsyncs the file, atomically replaces only the intended non-directory target
  without following a symlink, and fsyncs the directory where supported; and
- preserves the previous destination if fresh publication fails.

The acknowledgement is not persisted as a preference. Selection changes,
navigation, modal replacement, or shutdown fence late publication and UI
status.

## Lifecycle and cancellation

Archive copy, validation, decompression, migration backup, repository commit,
atomic export publication, and cleanup run as retained service-owned work.
Cancellation joins a shielded blocking worker before propagating and cleans any
published staging/output it owns. UI unmount prevents stale presentation but
does not abandon work.

Service close seals new import/export admission, cancels or completes work as
appropriate, initiates cleanup, and retains definitive joining. `wait_closed()`
cannot report ownership zero while a backup, copy, decompressor, output
publication, repository commit, handle, or cleanup task remains owned.

## Error and privacy contract

Bounded public categories include:

- `bundle_invalid`;
- `bundle_limit_exceeded`;
- `source_changed`;
- `unsupported_bundle`;
- `unsupported_platform`;
- `dependency_missing`;
- `dependency_changed`;
- `profile_conflict`;
- `stale_inspection`; and
- `operation_failed`.

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
  rollback, retained-backup preservation, downgrade, aliases, and newer-schema
  refusal.
- Conflict/dependency cross-product across Create, Reuse, Copy, and inactive
  consent.
- Exact private duplicate comparison without digest exposure.
- Passive inspection proving zero audio.cpp launches, HTTP requests, or
  Settings writes.
- Runtime admission proving exact recipe/model match before materialization or
  provider work and no fallback on mismatch.
- Cleanup and ownership under success, refusal, malformed input, cancellation,
  unmount, migration failure, service close, and application shutdown.
- Textual keyboard/focus/announcement/narrow-layout/late-result tests.
- Log/event/notification/repr/exception-graph canaries for every private value.

### Manual UAT

Use two separate Chatbook launches with independently isolated temporary
config, data, and profile roots; never use the developer's live profile store.

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
