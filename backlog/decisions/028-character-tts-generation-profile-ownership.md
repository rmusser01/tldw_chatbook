# ADR-028: Keep character TTS generation profiles in an app-owned local store

Status: Accepted
Date: 2026-07-26
Amended: 2026-07-31 (TASK-1626, explicit sanitized portability)
Related Tasks: TASK-710, TASK-763, TASK-1626
Extends: ADR-023
Supersedes: N/A

## Decision

Chatbook will store reusable TTS generation profiles and character assignments
in a dedicated, versioned SQLite database owned through one application-scoped
`TTSProfileRepository`. The profile store is separate from character cards,
provider configuration, and the main conversation database.

A profile is a complete reusable generation selection: immutable UUID, bounded
display name and normalized uniqueness key, canonical provider ID, exact model
ID, nullable exact voice ID, response format, speed, validated provider-safe
options, optimistic revision, and timestamps. It never contains provider
origins, credentials, process paths, health, message text, or other connection
settings. The first executable profile provider is the native `audio_cpp`
adapter, whose first-release profile contract is WAV, speed exactly `1.0`, and
an empty options object.

Character assignments use the full authority-scoped identity
`(source, authority_id, character_id)`. A bare character ID, display name,
active database path, current server origin, credential, or credential
fingerprint is never assignment authority. Local and server authority
acquisition and roleplay authorship integration ship in later, separately
reviewed slices.

The repository will:

- marshal every operation through one serialized off-event-loop worker and at
  most one long-lived SQLite connection;
- expose explicit `open`, `restoring`, `unavailable`, and `closed` lifecycle
  states plus a monotonic generation carried by queued work and results;
- require every mutation derived from a caller-held result to supply that
  result's expected lifecycle generation, checked under repository operation
  admission before work is enqueued;
- enter `restoring` and advance generation atomically at restore admission,
  before enqueueing restore I/O, then reject new work and prevent older
  queued work from writing or publishing after replacement;
- hold a cooperative shared interprocess lock while open and require a bounded
  exclusive lock before replacing the store;
- use transactions for profile and assignment mutations, foreign-key delete
  restriction, normalized unique names, and optimistic profile revisions;
- use SQLite online backup for consistent backup and pre-restore recovery
  copies;
- validate schema version and integrity before replacement, replace atomically,
  close any scoped validation connection, reacquire shared ownership, and only
  then open the long-lived connection for the same admitted generation;
- leave the original store intact when quiescence, locking, validation, backup,
  or replacement fails; if post-replacement shared rebind/reopen fails, leave
  the repository unavailable with recovery evidence instead of creating a
  blank database.

The profile database participates in **Backup All** through repository-owned
online-backup semantics. Restore is explicit and runs through the repository
lifecycle boundary; an open profile database is never restored by a raw file
copy.

Profile persistence remains local-only. Ordinary character-card export never
includes profile or assignment data. An explicit local export may add one
transient versioned payload at
`data.extensions["tldw_chatbook/tts_generation_profile"]`, and standalone
profile export uses the same sanitized payload. The payload contains only the
portable UUID hint, display name, canonical provider/model/voice selection,
format, speed, and validated profile-safe options. It excludes assignment
authority, origins, credentials, process paths, health, timestamps, revisions,
message text, and other connection or local lifecycle state. Export operates
on a copy and never mutates the stored card.

Imported attachments are hostile input. Chatbook bounds and structurally
validates the exact payload, removes the reserved extension before character
persistence, and treats the portable UUID as a correlation hint rather than
local authority. Unknown versions/providers and malformed known payloads skip
only the attachment with a bounded warning. Valid attachments use explicit
UUID/name/generation-tuple collision choices and never update an existing
profile silently. Current provider availability is observed separately from
structural validity before any write and revalidated before assignment; a
structurally valid unavailable profile may be stored visibly for repair but is
not assigned.

Character and profile stores do not pretend to share a transaction. Character
persistence reports whether it created a row or reused a name conflict. Profile
creation and any assignment are atomic within the profile repository. A reused
character's assignment changes only after explicit confirmation. Cancellation
performs no profile/assignment write; profile failure leaves a new character
imported and unassigned and preserves a reused character's prior assignment.

Server synchronization, managed audio.cpp process ownership, arbitrary
provider options, legacy-adapter profile execution, implicit portability, and
standalone profile import remain excluded.

## Context

ADR-023 established one app-owned TTS service and adapter registry, with
audio.cpp as the first native adapter and complete WAV delivery over the
asynchronous response interface. TASK-710 made external audio.cpp Console
speech and runtime settings coherent.

Reusable character voices require a durable owner that is not the imported
character card and is not provider configuration. Character IDs are not
globally unique across local databases or authenticated server principals.
Shared profiles may be edited while requests are in flight, and backup or
restore may race with queued operations or a second Chatbook process. Storing
profiles in an existing character database or using caller-owned SQLite
connections would make those identity and lifecycle boundaries ambiguous.

A canonical ADR is required because this feature creates a new storage and
migration track, defines data ownership and authority identities, establishes a
cross-module repository contract, and sets backup, restore, privacy, and
fail-closed behavior for later UI and runtime slices.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add voice fields directly to character cards | Mutates imported content, couples local preferences to card formats, and cannot represent one shared profile safely. |
| Store assignments in the active character database | Makes assignments follow whichever database is currently open and cannot safely scope server characters or restored profile stores. |
| Store profiles in the main conversation database | Couples independent profile lifecycle, backup, and migration to conversation storage and still leaves server-character authority ambiguous. |
| Put provider URLs and credentials in each profile | Duplicates operational configuration and risks leaking secrets through logs, exports, or card portability. |
| Persist the portable attachment in the local character card | Makes an untrusted transport object durable, leaks it through ordinary re-export, and creates two owners for profile truth. |
| Treat the character and profile databases as one transaction | SQLite files and owners are independent; explicit ordering and compensation make partial success observable without inventing unsafe cross-store atomicity. |
| Use JSON or TOML files | Loses transactional assignment updates, normalized uniqueness, optimistic revisions, foreign keys, bounded pagination, and consistent online backup. |
| Open a new SQLite connection from each caller | Allows restore and stale queued operations to cross lifecycle boundaries and makes connection ownership difficult to audit. |
| Use only an in-process lock | Cannot prevent another Chatbook process from holding the old database inode during replacement. |
| Automatically delete assignments for unreachable characters | Treats temporary authority or network failures as permanent deletion and can silently discard user choices. |
| Fall back to global speech when an assignment is broken | Hides a voice-identity failure by speaking with an unintended provider, model, or voice. |

## Consequences

- A separate profile database and schema-version track are introduced.
- Older Chatbook builds ignore the profile database; no automatic downgrade or
  destructive cleanup occurs.
- Profile names are trimmed and compared by
  `NFKC(display_name).casefold()`. Invisible control, format-control,
  surrogate, and noncharacter code points are rejected.
- A profile update requires both the repository lifecycle generation and
  profile revision loaded by the editor. Delete and duplication from a loaded
  profile also require its lifecycle generation. Duplication intentionally
  copies the immutable version the user opened and does not require the source
  to remain otherwise unchanged. A replacement store is never mutated merely
  because it contains a coincidentally matching UUID and revision. Conflicts
  preserve both the stored row and the caller's unsaved values.
- One character has at most one assignment; a profile may serve many
  characters. Assigned profiles cannot be deleted until assignments are
  explicitly detached or replaced.
- Joined assignment/profile reads return immutable profile identity, revision,
  and generation fields for later request admission.
- Already admitted speech will use its immutable profile snapshot; edits affect
  only future requests.
- The repository, not Textual widgets or services, owns SQLite connections,
  serialization, schema transitions, backup, restore, and interprocess locking.
- Profile pages and editors retain their repository generation. Service and UI
  code recheck it before publishing availability or submitting mutations, so a
  completed pre-restore result cannot repopulate state after replacement.
- Corruption, unsupported schema versions, failed migration, and unavailable
  paths fail closed. Chatbook does not silently recreate or discard the store.
- Restore may fail while another Chatbook process holds the shared store lock;
  that failure happens before replacement.
- Backup All remains a per-database consistent collection rather than a
  cross-database atomic snapshot.
- Provider health and catalog presence remain runtime observations and are not
  persisted as profile truth.
- Profile connection details, character authority, message text, credentials,
  origins, and local paths remain excluded from logs, metrics, and portable
  payloads.
- Optional portability is explicit and local-only. Ordinary stored cards remain
  free of TTS attachments, and portable UUIDs never become assignment authority.
- Managed audio.cpp launch and supervision remains a separate deferred task and
  is not implied by profile ownership.

## Rollback plan

- Disable profile-store consumers and return speech to global preferences
  without deleting the profile database.
- Do not down-migrate, drop, or recreate profile tables automatically.
- Keep stored profiles and assignments inert when their provider is
  unavailable or their runtime integration is disabled.
- If restore support must be disabled, retain online backup and fail restore
  requests explicitly rather than falling back to raw file replacement.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md)
- [ADR-023 — TTS adapter registry and audio.cpp runtime boundary](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [TASK-710](<../tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md>)
- [TASK-763](<../tasks/task-763 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md>)
- [TASK-1626](<../tasks/task-1626 - Add-sanitized-TTS-portability-to-local-character-cards.md>)
