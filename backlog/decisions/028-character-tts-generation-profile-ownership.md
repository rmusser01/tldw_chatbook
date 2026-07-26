# ADR-028: Keep character TTS generation profiles in an app-owned local store

Status: Accepted
Date: 2026-07-26
Related Tasks: TASK-710, TASK-761
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

Profile persistence is local-only in this decision. Server synchronization,
automatic speech, managed audio.cpp process ownership, arbitrary provider
options, legacy-adapter profile execution, and implicit character-card
portability are excluded.

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
- A profile update requires the revision loaded by the editor. Conflicts
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
- Later slices own STTS profile management, authority acquisition, character
  assignment UI, roleplay resolution, and optional sanitized card portability.
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
- [TASK-761](<../tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md>)
