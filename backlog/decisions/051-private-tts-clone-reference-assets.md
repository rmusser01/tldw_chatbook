# ADR-051: Store TTS clone references as private profile-owned assets

Status: Accepted
Date: 2026-08-09
Related Tasks: TASK-13203, TASK-13204, TASK-13205, TASK-13206
Extends: ADR-028, ADR-029, ADR-040, ADR-050
Supersedes: The audio.cpp empty-options/no-reference limitation in ADR-028 only;
all other profile ownership and portability rules remain in force

## Decision

Chatbook will extend its local TTS generation profiles with one typed optional
voice-clone reference. The reference is profile-owned private data, not an
audio.cpp connection setting, generic provider option, character-card field,
server configuration field, or durable source-file path.

The profile database will advance from schema v2 to schema v3. A one-to-one
reference table keyed by profile UUID stores:

- an immutable reference UUID;
- canonical, bounded WAV bytes as a SQLite BLOB;
- the bounded reference transcript;
- SHA-256, byte length, duration, sample rate, channel count, sample encoding;
  and
- creation and update timestamps.

The original selected path is never persisted. Ingest accepts only a bounded
supported WAV shape, decodes and canonicalizes it, removes arbitrary RIFF
metadata, computes the digest from the canonical bytes, and commits the profile
change and reference atomically. Global per-reference and aggregate quotas
apply independently of narrower recipe guidance. Listing profiles reads
metadata summaries rather than loading BLOBs, and BLOB import/export uses
streaming I/O.

The SQLite file remains local plaintext private storage under ADR-029. Chatbook
must say that this protects access through filesystem ownership controls but is
not encryption. Best-effort deletion, WAL checkpointing, and private temporary
cleanup are required, but the product must not claim forensic erasure.

An admitted clone request captures an immutable profile revision, reference
UUID, and canonical digest. The exact reference is materialized into an opaque,
owner-private per-session path and sent to the native audio.cpp adapter through
typed `voice_ref` and `reference_text` request fields. It never enters generated
`server.json`, generic options, general logs, child diagnostic output, HTTP
debug logging, or public exception graphs.

Reference-bearing profile execution initially requires an exact compatible
guided-Managed recipe and the application-owned local child. Chatbook does not
send a client-local materialized path to an independently owned External server
or to an unclassified user-JSON model. Those sources retain text and exact
voice-ID behavior. A future External clone path requires a separately designed
upload or remote-asset contract; filesystem coincidence is not authority.

Materialization directories use ownership locks. Normal completion removes the
exact session directory. Startup cleanup may remove a leftover only after it
proves that no live owner holds the corresponding lock; it cannot sweep
unrecognized paths speculatively.

The first clone audition may use a transient canonical reference without
forcing profile creation. After successful synthesis, Chatbook may offer
`Save as Voice Profile`. That save uses the exact canonical bytes that produced
the current successful result; it does not reopen the original path. The
transient artifact is bounded and retained only with the current result until
save, replacement, discard, or application close.

An audio.cpp profile may contain an exact `voice_id`, a clone reference, or
both only when the exact accepted recipe declares that combination. Clone
fields are typed provider capability, not arbitrary profile options. An edit
increments the profile revision. Speech admitted against an older immutable
revision may finish; later requests observe the new revision.

Ordinary profile and character-card portability remains sanitized:

- profiles without references keep the existing wire version 1;
- a reference-bearing ordinary export uses wire version 2 and states that the
  local reference was omitted;
- it contains neither reference bytes nor transcript; and
- import never silently creates, assigns, or repairs a reference-bearing
  profile. The v2 decoder returns a bounded reference-omitted skip/no-mutation
  outcome; explicit bundle import remains a separate action.

Reference transfer is a separate explicit voice bundle, not a character-card
extension. The bundle is a strict versioned ZIP containing only a manifest,
sanitized profile projection, canonical `reference.wav`, UTF-8 transcript, and
bounded metadata/checksums. It excludes model files, character data,
assignments, credentials, endpoints, source paths, and runtime state. Export and
import display a plaintext-sensitive-data warning and a generic
consent/provenance notice. Chatbook records no claim that legal consent was
proved.

Bundle import treats the archive as hostile. It rejects encrypted entries,
duplicates, case or Unicode-normalization collisions, traversal, symlinks,
devices, unknown entries, unsupported compression, excessive counts or sizes,
decompression bombs, changed source bytes, and checksum mismatch. Checksums
prove integrity only, not authenticity. UUID/name collisions require explicit
review; import never overwrites, assigns, or changes a default automatically.
A structurally valid bundle whose exact compatible model is absent may be
stored visibly as `Needs compatible model`, but is never auto-retargeted.

The existing profile repository remains the sole lifecycle owner. SQLite online
backup and restore include the reference BLOB and validate schema, quotas, and
integrity. Normal repository open remains metadata-focused; full WAV/digest
validation occurs on import, restore, backup qualification, reference edit, and
exact request admission rather than reading every BLOB at startup. A safely
isolatable bad reference makes only its profile unavailable; structural store
corruption still fails the repository closed.

Migration is eager and guarded: validate v2, create a separately retained
owner-private v2 pre-migration backup, migrate transactionally, validate v3,
then publish the new store. Existing profiles remain domain-equivalent and have
no reference row. Failure leaves v2 usable or the repository unavailable; it
never publishes a partial migration. Downgrade is explicit: close the new app,
restore the dedicated v2 backup, then use the old build, accepting loss of
post-migration profile changes. Switching to manual audio.cpp setup inside the
new build is feature rollback, not a database downgrade.

## Amendment: exact portable recipe provenance (2026-08-11)

TASK-13206 advances the profile store from schema v3 to v4 so an explicit voice
bundle can retain the exact recipe dependency after its originating model
configuration is removed. The reference row gains optional recipe ID and
positive recipe-revision fields. They are both null or both valid and the
requirement's model ID must match the owning profile.

New clone-profile saves, reference replacements, and bundle imports require
exact recipe provenance from admitted generation or a validated bundle.
Migration never infers historical provenance from current Settings or recipe
catalog state. Existing v3 references migrate with both fields null, remain
usable under the prior guided-Managed compatibility contract, show a bounded
`Recipe provenance unavailable` advisory, cannot qualify as an exact bundle
duplicate, and cannot be exported as a bundle until regenerated and saved.

The v3→v4 migration uses a private candidate rather than upgrading the active
file in place. It validates v3 under exclusive ownership, prepares a separate
owner-private `<profile-db>.pre-v4.sqlite3` sibling, migrates and fully
validates/fsyncs the candidate, then enters one non-cancellable atomic
publication protocol with retained rollback ownership. Pre-publication failure
leaves the active store unchanged. Post-replace failure restores and fsyncs the
prior active file and every prior retained backup. Prepared backups become
authoritative only after the new active store is reopened and validated. A
total storage failure that prevents completion or full restoration leaves the
repository unavailable with all recovery files retained rather than making an
impossible authority claim.

POSIX does not provide an unlink-by-descriptor operation that can prove the
removed directory entry still names a pinned inode on both macOS and Linux.
Migration disposal therefore atomically moves each exact owned journal,
candidate, or rollback leaf without replacement to a stable private holding
leaf and fsyncs the pinned descriptor and parent. It never truncates an inode:
a hardlink can appear after any link-count check, so truncation could modify a
foreign alias. The owner-private `0600` holding leaf may retain private bytes
as bounded cleanup evidence. The finite recovery
shape bounds these tombstones to one stable leaf for the journal plus one per
candidate/rollback logical leaf (at most three slots), never one per attempt.
They are ignored as recovery authority. Only an already-zero holding leaf may
be reused after exact inode, owner, mode, type, link-count, parent, and sidecar
checks; a nonzero leaf is never overwritten or reused.
A prepared artifact is admitted only under its closed slot leaf
`.profile-migration-{active|pre-v3|pre-v4}.candidate.sqlite3`; rollback authority
uses the corresponding `.profile-migration-{slot}.rollback.sqlite3` leaf.
Noncanonical input is rejected before hashing or journal admission, so caller
names cannot expand the owned recovery namespace.
A foreign or substituted holding leaf is preserved and makes cleanup boundedly
unavailable. An owner-private nonzero leaf likewise remains a finite retained
cleanup artifact, never authority. Cleanup may retry only when safe eligibility
can be proven without risking an alias.

A v2-or-older jump advances the private candidate through each supported
migration and prepares validated v2 and v3 downgrade snapshots at their
respective boundaries. Restore qualification follows the same rule, so a v3
restore candidate prepares a fresh pre-v4 snapshot before active v4
publication. Every prior backup remains under rollback identity until the
active replacement succeeds; failed publication restores and fsyncs the prior
active store and all prior backups. Backup paths and aliases remain reserved
from ordinary backup destinations. A private durable publication journal lets
startup complete or roll back an interrupted multi-file publication before
opening the store.

Startup recovery accepts each journal-recorded profile-store artifact only up
to 576 MiB: the 512 MiB aggregate canonical-reference quota plus 64 MiB of
operational headroom for SQLite pages, schema, indexes, and free-list growth.
The total-file cap supplies the bound; those structures are not assumed to be
intrinsically bounded. Publication, journaling, and recovery reject both
recorded evidence and observed files above that limit before hashing, while
hashing accepted artifacts incrementally rather than buffering a store in
memory.

Downgrade explicitly restores the pre-v4 backup while the repository is closed
and accepts loss of all post-migration changes and recipe provenance.

Bundle schema v1 contains exactly `manifest.json`, `profile.json`,
`reference.wav`, and `reference.txt`. The deterministic Chatbook writer uses
fixed `ZIP_STORED` metadata. The manifest checksums only the other three entries
(a manifest cannot checksum itself without recursion), records exact
recipe/model dependency and bounded WAV facts, and carries a fixed
plaintext-warning acknowledgement. The acknowledgement is not proof of
consent, identity, signature, or authenticity; import always displays its own
warning.

Import never trusts archive paths or a prior inspection. An application-scoped
bounded service owns expiring, single-use inspection sessions and private
source authority while the UI receives only an opaque handle and safe facts.
It uses verified owner-private application staging, validates and copies the
unchanged source, then repeats full validation and local-only dependency
inspection at commit. Exact reuse requires equality of sanitized selection,
recipe requirement, transcript, and canonical WAV, but exposes only the
boolean result rather than a digest oracle. Missing but syntactically valid
future recipes may be imported only through an explicit inactive choice.

Exact reuse or creation is decided inside one serialized repository command
and immediate transaction that rechecks UUID/name/profile revision/reference
identity and the reviewed copy destination before returning no-op or inserting
profile, recipe, and reference atomically.

The availability state vocabulary remains unchanged. A bounded blocking reason
records missing, mismatched, or pending recipe dependency. Provenance absence
is a separate nonblocking portability advisory, so damaged/provider failures
retain their truthful primary recovery while the advisory stays visible. For
references with exact provenance, a pure local guided-config/recipe snapshot
classifies dependency state without acquiring an adapter or performing
network/runtime work. Runtime rejects recipe/model/config mismatch before
provider work and compares adapter-issued process-generation evidence again
after readiness.
Post-ready generation drift still blocks before private materialization or
synthesis HTTP. It never silently falls back or retargets. Dependency recovery
changes availability only; it never assigns a character or changes a default.

The snapshot's pending flag is generation/configuration evidence rather than
an alias for its state. Exact, missing, and mismatch states may coexist with
queued settings; exact is determined by the applied requirement even when the
saved requirement is missing or drifted, while pending state requires an exact compatible saved
requirement. Bundle inspection and commit validate this producer cross-product
without acquiring provider authority.

Reference-bearing profile editing is display-name-only. Model, voice, format,
speed, or option changes require a newly admitted result/new profile or a
future explicit atomic reference replacement; every v4 reference set/replacement
requires exact recipe evidence.

Bundle output is atomic create-only and never overwrites an existing
destination. Owner-private traversal applies to application staging, backups,
temporary outputs, and final bundle files, not recursively to user-selected
source/destination parents. Sources still require no-follow regular-file
ownership/stability checks, and final POSIX bundles are mode `0600`. Until a
platform has verified ADR-029 application-owned containment, bundle controls
fail closed with a truthful unsupported-platform explanation. Archive workers,
migration backup, atomic output publication, cleanup, and repository commit are
retained and joined across cancellation and shutdown.

For bundle export, the successful atomic no-replace namespace publication is
the explicit non-cancellable point of no return. Cancellation before that point
propagates with no final; the randomized `0600` temporary sibling is retained
rather than pathname-deleted from a user-selected parent. Cancellation after
that point is deferred while the service converges
the exact published inode through mode/content/identity verification and parent
directory fsync, then returns successful publication. Atomic no-clobber rename
consumes the temporary sibling at publication, and the service never unlinks
any export pathname. POSIX has no exact-inode
unlink-by-descriptor primitive, so a pathname `stat` followed by `unlink` could
delete a substituted foreign file; post-publication substitution is therefore
preserved and reported as bounded cleanup/unavailability rather than rolled
back. An owned final can accompany failure only when substitution or total
storage failure makes convergence unverifiable.

Bounded pre-publication recovery states that a hidden randomized sibling may
remain and directs manual removal only after the user verifies that exact
random filename; the service never guesses that a pathname still names its
former inode.

The application-owned operation root is fail-closed across restart. Any
nonempty root, including a recognized prior operation name, reports bounded
`cleanup_failed`; a new service instance does not reclaim or pathname-delete
crash residue without retained exact authority. Recovery directs the user to
exit Chatbook, manually inspect the app-owned portability root, remove only
confirmed residue, and retry. The public error never includes the runtime path.

## Context

Several audio.cpp families in the approved `release-0.5.1` compatibility scope
require or benefit from a reference recording and transcript. Chatbook's current
profile schema stores exact model/voice selections but intentionally rejects
audio.cpp options and has no owner for reference audio. Persisting a source path
would break portability and could silently change the voice when that file is
replaced. Putting raw audio in character cards would leak a private biometric-
adjacent artifact through ordinary sharing.

The feature therefore needs an explicit data owner, migration, request-admission
snapshot, temporary-materialization lifecycle, backup behavior, and separate
portable container. These are privacy, storage, schema, service-contract, and
portability decisions, so they require a canonical ADR independent of the
generated server-configuration decision.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Persist the user's source path | The bytes can move or change silently, paths leak local identity, and ordinary backup/export cannot reproduce the admitted voice. |
| Put `voice_ref` and transcript in generic profile options | Evades typed validation, makes privacy review difficult, and risks sending paths or text to providers that do not support cloning. |
| Store references in generated `server.json` voice presets | Makes a private profile asset a process-wide setting, leaks paths into child diagnostics, and prevents exact per-request/profile revision admission. |
| Send the temporary local path to any External server | A remote or independently owned server may not share the filesystem and may persist/log the path or read it outside Chatbook's cleanup contract. |
| Embed reference audio in character cards | Couples reusable profiles to characters and silently exports sensitive bytes through a common interchange format. |
| Copy the original file byte-for-byte | Retains arbitrary RIFF metadata, unsupported encodings, and mutable or excessive content that Chatbook did not validate. |
| Store one reference file beside the database | Adds cross-file transactions, orphan and replacement races, and backup/restore inconsistency without a demonstrated size need. |
| Encrypt only this BLOB with a new app key | Introduces key creation, recovery, backup, rotation, and unlock UX beyond the established local-private boundary; the product must state plaintext honestly. |
| Include reference bytes in ordinary portable profile v1 | Breaks existing sanitized portability assumptions and surprises card/profile sharing with a large sensitive payload. |
| Auto-match a missing model on bundle import | A different family or variant can change voice semantics and request requirements; compatibility must remain exact and user-reviewed. |

## Consequences

### Benefits

- Clone-capable audio.cpp families can participate in reusable per-character
  voices without persisting mutable source paths.
- Exact admitted bytes, transcript, profile revision, and runtime request remain
  auditable and generation-safe.
- Ordinary character/profile sharing remains sanitized by default.
- Explicit bundles support intentional transfer with strict archive and privacy
  boundaries.
- Backup and restore stay repository-owned and transactionally consistent.

### Accepted trade-offs

- The profile database may become materially larger and needs explicit quotas
  and streaming BLOB operations.
- Reference audio is plaintext at rest; filesystem privacy is not encryption.
- Secure deletion is best effort on SQLite, journaling files, copy-on-write
  filesystems, backups, and storage media.
- Users must keep the transcript with the reference and may need to attach or
  import the reference again on another device.
- A pre-v3 build cannot consume post-migration profile changes. Downgrade uses
  the retained v2 backup and loses newer edits.
- Bundle integrity does not establish speaker identity, provenance, consent, or
  authenticity.

## Rollback

- Disable creation and admission of reference-bearing profiles without deleting
  v3 data.
- Keep non-reference v3 profiles usable in the current build where safe; do not
  ask an older build to open v3.
- Remove transient session material through the normal locked cleanup path.
- For application downgrade, restore the dedicated pre-migration v2 backup only
  while the v3 repository is closed.
- Never synthesize from a missing, damaged, quota-violating, or unsupported
  reference and never silently fall back to an unrelated voice.

## Links

- [Guided model setup design](../../Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md)
- [ADR-028: Character TTS generation profile ownership](028-character-tts-generation-profile-ownership.md)
- [ADR-029: Local private data boundary](029-local-private-data-boundary.md)
- [ADR-040: Profile-owned state and shared asset paths](040-profile-owned-state-and-shared-asset-paths.md)
- [ADR-050: Generated audio.cpp model setup](050-audio-cpp-generated-model-setup-ownership.md)
