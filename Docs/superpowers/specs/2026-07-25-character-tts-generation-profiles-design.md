# Character TTS Generation Profiles with Native audio.cpp Console Speech — Design

**Status:** design approved by the user on 2026-07-25; independent spec review completed after three passes; review amendments incorporated and awaiting final written-spec review
**Date:** 2026-07-25
**Related design:** [audio.cpp TTS Adapter Registry](2026-07-23-audio-cpp-tts-adapter-registry-design.md)
**Existing ADR:** [ADR-023 — TTS Adapter Registry and audio.cpp Runtime Boundary](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
**Planned ADR:** `backlog/decisions/027-character-tts-generation-profile-ownership.md`

## Goal

Make an external, user-owned audio.cpp server work end to end from the Console
page, then let a user create reusable TTS generation profiles and assign one to
each roleplay character.

The first supported profile provider is the native `audio_cpp` adapter. A
profile records a complete generation selection—provider, exact model, optional
voice, format, speed, and validated provider options—without owning provider
connection settings. When a character-authored assistant message is spoken,
Chatbook resolves the character's assigned profile and sends one immutable
request through the existing app-owned `TTSService`.

The feature is delivered in independently shippable slices. The first slice
fixes native audio.cpp Console speech without waiting for character profiles.
Later slices add the local profile library, character assignment, and optional
character-card portability.

## UAT baseline and problem statement

A first-time-user UAT was run against an existing Homebrew
`audiocpp_server`, configured by its owner at `http://127.0.0.1:8080` with the
Supertonic model. Chatbook did not launch, restart, or stop that process.

The settings flow successfully selected the external audio.cpp provider, tested
the connection, and reported it ready. Saving those settings exposed three
integration defects:

1. Textual `Select.BLANK` sentinel values were serialized as empty
   `default_model` and `default_voice` strings.
2. the Console handler continued to use the legacy
   `generate_audio_stream()` route rather than the native `TTSRequest` route;
3. the running app retained stale effective settings after a successful save.

After restart, Console speech therefore failed with "The selected TTS model is
not available." A direct control request through the native service succeeded
against the same server with provider `audio_cpp`, model `supertonic-3`, voice
`M1`, and one complete WAV chunk. The response was valid mono PCM16 at
44.1 kHz, lasted approximately 6.3 seconds, played successfully with `afplay`,
and left the external server healthy.

This design treats that control result as proof that the adapter and external
server contract work. The immediate gap is application routing and settings
state, followed by durable per-character profile ownership.

Before Slice 1 UAT, Chatbook records the installed external server build and
characterizes `/health`, `/v1/models`, `/v1/audio/voices`, and a complete-WAV
`/v1/audio/speech` response against the contract pinned by ADR-023. The
[current server documentation](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md)
and [release history](https://github.com/0xShug0/audio.cpp/releases) are evidence
for selecting the installed build, not permission to silently repin the
adapter.

This is a stop/go compatibility gate:

- if the installed Homebrew/release build matches the pinned endpoint, request,
  response, MIME, and WAV semantics, Chatbook records its build identity and
  adds that build to compatibility fixtures and UAT evidence without changing
  the pin;
- if it is incompatible, Slice 1 stops before implementation continues or UAT
  proceeds. A separate reviewed change must amend ADR-023, its pinned contract,
  adapter fixtures, and compatibility policy before work resumes.

Upstream streaming capability does not expand this release. Chatbook continues
to request and consume one complete WAV through the asynchronous response
interface.

## Design principles

- Keep one app-owned TTS service and one adapter registry.
- Treat external audio.cpp as user-owned; Chatbook connects but does not manage
  the process.
- Keep provider configuration and credentials out of reusable profiles.
- Resolve character identity with source and authority, never with a bare
  character ID.
- Fail clearly when an assigned profile cannot be honored; never silently
  substitute another voice, model, provider, or legacy route.
- Preserve the asynchronous audio-response interface while initially consuming
  complete WAV responses.
- Keep character-card export unchanged unless the user explicitly includes a
  sanitized TTS profile.
- Make shared-profile edits visible and concurrency-safe.
- Add no new runtime dependency.

## Scope

### Included

- Native external audio.cpp synthesis from Console manual **Speak** actions.
- Immediate use of newly saved TTS preferences without app restart.
- Explicit global model and voice selection modes.
- A versioned, local SQLite repository for generation profiles and character
  assignments.
- A shared profile library managed from the STTS Playground.
- Profile selection and repair from the character editor.
- Manual speech for character-authored assistant messages.
- Complete-WAV playback through the existing asynchronous response contract.
- Optional, sanitized profile attachment on character-card export and import.
- Backup and explicit restore coverage for the profile database.
- Existing legacy-provider behavior when no character assignment applies and
  the global provider remains legacy.

### Excluded

- Automatic speech after every roleplay response.
- Managed audio.cpp binary launch, supervision, restart, or shutdown.
- More than one configured audio.cpp instance.
- Profiles that execute through legacy adapters.
- Server-side profile storage or synchronization.
- Server-side character-card import/export or a new server conversation
  transport built solely for TTS.
- Voice cloning or voice-reference uploads.
- Migration of audiobook `CharacterVoiceWidget` data.
- Group-roleplay speaker attribution.
- Arbitrary, unvalidated provider option dictionaries.
- Standalone profile import.
- Provider fallback, model fallback, or voice fallback after a request is
  admitted.
- Packaging or redistributing audio.cpp or its models.

## Terminology and identities

### Provider configuration

Provider configuration contains operational connection details such as an
audio.cpp base URL, credentials, timeouts, or future binary/config paths. It
continues to belong to application settings and the adapter registry. It is not
copied into a profile or export payload.

### Global TTS preferences

Global preferences select the default provider, model mode, voice mode, format,
and speed for speech that has no character assignment. They are represented in
memory by one immutable `TTSPreferencesSnapshot`.

Model and voice modes are explicit:

- model mode is `exact` or `first_available`;
- voice mode is `exact` or `server_default`;
- exact model and voice identifiers are stored in separate nullable fields.

The persisted `[app_tts]` keys are `default_model_mode` and
`default_voice_mode`, alongside the existing `default_model` and
`default_voice` exact-value keys. The mode keys are authoritative. While the
legacy compatibility reader remains, exact values are dual-written to their
existing `[tts_settings]` aliases. When a dynamic mode is saved, both its
`[app_tts]` exact key and its legacy alias are removed in the same locked,
atomic configuration mutation. Empty strings are accepted only by the
backward-compatibility reader and are never written as exact values.

An `exact` mode requires a non-empty corresponding identifier. A
`first_available` model is resolved once at request admission. A
`server_default` voice is sent as `None`.

Provider-specific constraints apply to global preferences as well as profiles.
When audio.cpp is selected, the settings UI and validator lock format to `wav`
and speed to exactly `1.0`, consistent with ADR-023.

For backward compatibility, existing blank audio.cpp model and voice values are
read as `first_available` and `server_default`. Startup does not rewrite the
configuration file. The next successful settings save persists explicit modes.

### TTS generation profile

A `TTSGenerationProfile` is a reusable, complete, exact generation selection.
It contains:

- immutable UUID profile ID;
- trimmed display name plus a Unicode-normalized, case-insensitive uniqueness
  key;
- canonical provider ID;
- exact model ID;
- nullable exact voice ID;
- requested audio format;
- speed in the inclusive range `0.25` through `4.0`;
- a provider-validated, JSON-serializable options object, encoded in at most
  16 KiB with no more than four container levels;
- positive revision;
- created and updated timestamps.

Profile model selection is always exact. A null voice means the adapter's
declared server-default behavior; it does not mean "pick any voice."

The first release executes profiles only when `provider_id == "audio_cpp"`.
The schema remains provider-neutral so a future native adapter can define its
own profile-safe option schema without a database redesign. Consistent with
ADR-023, audio.cpp accepts only `wav`, speed exactly `1.0`, and an empty options
object in this release; arbitrary options and other speeds are rejected.

### Character reference

A bare character ID is not globally unique. Assignments use the canonical
three-part identity:

```text
CharacterRef = (source, authority_id, character_id)
```

- a local character is
  `("local", <persisted-local-database-authority-id>, <local-character-id>)`;
- a server character is
  `("server", <persisted-server-authority-id>, <server-character-id>)`.

The local authority is the existing durable `local_authority_id` singleton
owned by the character database. A literal `"local"`, database path, or current
process identity is not sufficient because the profile database may survive a
character-database switch or restore.

The server authority is an opaque stable scope supplied by the source-aware
runtime. It incorporates the durable server profile and the stable
authenticated principal or tenant when the same server can expose different
character namespaces. It is not a display name, normalized origin, or whichever
server happens to be active later, and it must remain stable across ordinary
credential rotation. If the runtime cannot establish the required stable
server authority, Chatbook fails closed rather than creating or resolving that
assignment.

`authority_id` and `character_id` are non-empty canonical text identifiers of
at most 256 characters. The source-aware character layer supplies the
canonical representation; profile code does not parse, renumber, or derive it
from a display label.

Server authority acquisition follows one explicit policy:

1. use a durable Chatbook server-profile ID plus a stable authenticated
   principal or tenant returned by the server's authenticated identity
   contract;
2. use server-profile-only authority only when that server contract explicitly
   guarantees that character IDs are global across principals;
3. never use an API key, token, credential source, or credential fingerprint as
   durable authority;
4. when neither stable form is available, disable server-character assignment
   and assigned speech with the actionable reason **Server identity
   unavailable**.

The opaque encoded server authority carries an authority-schema version so a
future construction change cannot collide with an older authority. Credential
rotation must preserve the resulting authority, while different principals on
the same normalized origin must remain distinct.

A server-backed conversation records its full authority scope when the
conversation is launched. Existing conversations without that provenance
cannot receive or use a server-character assignment until they are reopened or
explicitly repaired.

A local conversation persists its complete local `CharacterRef`, not only a
bare character ID. A legacy local conversation containing only
`character_id` may be interpreted and backfilled using the
`local_authority_id` of the character database that owns that conversation.
Legacy server conversations are never backfilled from whichever server or
credential happens to be active.

Speech admission resolves a `CharacterRef` only for assistant messages authored
by the selected roleplay character. User, system, tool, persona-only, and
generic assistant messages do not inherit a character assignment.

### Console speech snapshot

Clicking **Speak** creates an immutable, app-issued
`TTSMessageSpeechSnapshot`. It contains:

- native session and message IDs;
- nullable persisted conversation and message IDs;
- the visible content snapshot and stable selected-variant token or content
  revision;
- message role and completion status;
- assistant kind and the complete source-aware `CharacterRef`, when one was
  established from trusted session or conversation state.

The Console store, not an event caller, issues this value. Changing the active
session after the click does not change its scope. Before request admission,
the resolver verifies that the message still belongs to that session, is a
completed assistant message, still selects the captured variant/content
revision, and still has matching persisted or in-memory authorship. Deletion,
variant switching, content replacement, or authority mismatch makes the
snapshot stale and fails safely rather than speaking different content or
borrowing the current session's character.

## Architecture and ownership

### Existing TTS runtime remains authoritative

The application continues to own one `TTSService` and one
`TTSAdapterRegistry`, as defined by ADR-023. This feature does not create a
second service, adapter lookup path, audio.cpp client, or playback subsystem.

Exact resolved profiles become canonical `TTSRequest` objects and go directly
to `TTSService.synthesize()`. The native audio.cpp adapter remains authoritative
for readiness, catalog refresh, request validation, upstream I/O, response
limits, and WAV validation.

The temporary legacy bridge remains available only for global, unassigned
speech whose selected global provider is legacy. An audio.cpp request never
falls back to that bridge.

### New components

#### `TTSRequestAdmissionCoordinator`

One app-owned admission coordinator makes preference/profile resolution and
provider lease acquisition a single consistency boundary. It owns a
shared/exclusive asynchronous gate:

- a request enters the shared side, verifies its
  `TTSMessageSpeechSnapshot` when present, resolves and freezes its global
  preference or exact profile selection, resolves any `first_available`
  catalog choice, and obtains a service-owned provider lease with the matching
  configuration revision before leaving the gate;
- a settings publication enters the exclusive side, preventing any request
  from resolving a selection or acquiring a provider lease during the
  transition;
- the acquired lease remains owned by the resulting service operation and
  `TTSAudioResponse` through asynchronous body consumption and close.

The coordinator exposes no concrete adapter or lease to UI code. It either
hands a coherent admitted operation to `TTSService` or returns a structured
admission failure. An expected provider revision is checked at the service and
registry boundary as a defensive invariant; a mismatch never proceeds with a
mixed selection and adapter.

#### `TTSProfileRepository`

The repository owns a dedicated, versioned SQLite database in the existing user
data directory. It provides:

- schema initialization and migrations;
- transactional profile and assignment operations;
- normalized unique-name enforcement;
- optimistic revision checks for profile updates;
- referential delete protection;
- bounded profile listing and assignment counts;
- one joined assignment/profile read for request admission;
- explicit backup and restore participation;
- a repository-owned serialized worker, connection, and lifecycle generation;
- cooperative shared/exclusive interprocess store locking.

It has no dependency on Textual, adapter instances, provider health, character
cards, or the active server.

#### `TTSProfileService`

The service owns profile-domain validation and workflows:

- create, update, rename, duplicate, list, and delete profiles;
- assign, replace, and remove character assignments;
- validate a profile against current native provider capabilities;
- calculate profile availability and repair guidance;
- prepare standalone and character-card export payloads;
- consume the optional profile attachment during character-card import;
- apply import collision policy;
- expose structured profile-domain errors.

It depends on the repository and the app-owned `TTSService` catalog APIs, but it
does not synthesize audio itself.

#### `CharacterTTSRequestResolver`

The resolver converts message authorship, assignment state, and the current
global preference snapshot into either:

- an immutable resolved request selection; or
- a structured, user-actionable resolution failure.

For a persisted message, authorship is resolved from the identifiers in the
app-issued speech snapshot and rechecked against the persisted conversation.
For an unsaved in-memory message, the Console store rechecks the immutable
snapshot against its native session and message state. `TTSRequestEvent` accepts
only that app-issued snapshot or a generic global-speech request; it is not
allowed to supply arbitrary text, session identity, or `CharacterRef` as
authority. Generic or ad-hoc callers therefore resolve through global
preferences.

After authorship resolution, the resolver performs one joined repository read
for an assigned character. It does not fetch health from the database, call a
concrete adapter, mutate assignments, or silently fall back.

### Dependency flow

```text
Console / STTS / Character editor
             |
             v
 TTSRequestAdmissionCoordinator
             |
             +----> CharacterTTSRequestResolver -----> TTSProfileService
             |                                                |
             |                                                v
             |                                      TTSProfileRepository
             v
      app-owned TTSService
             |
             v
      TTSAdapterRegistry
             |
             v
   native audio_cpp adapter
```

UI layers may use the profile service for management and submit app-issued
speech snapshots to the admission coordinator. They never access SQLite rows,
provider leases, or concrete adapters directly.

## Persistence model

### Dedicated profile database

Profiles and assignments live in a separate database from character cards.
This makes the local profile library independent of character source, prevents
voice preferences from mutating imported cards, and gives local and
server-backed characters the same assignment contract.

Logical table `tts_generation_profiles` contains:

| Field | Constraint |
| --- | --- |
| `profile_id` | UUID primary key; immutable |
| `display_name` | trimmed, 1–128 Unicode characters with forbidden control, format-control, surrogate, and noncharacter code points rejected |
| `normalized_name` | `NFKC(trimmed display_name).casefold()` uniqueness key |
| `provider_id` | canonical identifier, 1–64 characters |
| `model_id` | exact opaque identifier, 1–256 characters |
| `voice_id` | nullable exact opaque identifier, at most 256 characters |
| `response_format` | normalized supported format, 1–32 characters |
| `speed` | finite number from 0.25 through 4.0 |
| `options_json` | canonical JSON object, schema-validated, at most 16 KiB, and at most four container levels |
| `revision` | positive integer incremented on update |
| `created_at`, `updated_at` | UTC timestamps |

Logical table `character_tts_assignments` contains:

| Field | Constraint |
| --- | --- |
| `source` | `local` or `server` |
| `authority_id` | opaque canonical text identifier, 1–256 characters |
| `character_id` | opaque canonical text identifier, 1–256 characters |
| `profile_id` | foreign key to profile with delete restriction |
| `created_at`, `updated_at` | UTC timestamps |

The three identity fields form the assignment primary key. One character has
at most one assigned profile; one profile may serve many characters.

Names are trimmed and compared by the stored
`NFKC(trimmed display_name).casefold()` key. Case-only and canonically
equivalent duplicate names fail with a conflict. Unicode control,
format-control, surrogate, and noncharacter code points are rejected so an
import cannot create an invisible or unrenderable library entry. Subject to
those checks, the separately stored display spelling remains user-controlled.

### Transaction and concurrency rules

- All repository operations are marshalled onto one repository-owned serialized
  worker with at most one SQLite connection active at a time. No caller-owned
  executor or thread-local connection may outlive repository lifecycle
  transitions.
- The repository has explicit `open`, `restoring`, `unavailable`, and `closed`
  states plus a monotonic lifecycle generation. New operations are rejected
  while restoring. Every queued operation and result carries its generation;
  stale work from before a restore cannot write or repopulate service/UI caches
  afterward.
- Each Chatbook process holds a shared interprocess lock while the profile store
  is open. Restore closes its own connection, releases its shared hold, and
  acquires the exclusive lock within a bounded deadline. Another process that
  still has the database open therefore causes restore to fail safely before
  replacement rather than leaving two processes attached to different inodes.
- Profile creation and assignment mutation are transactional.
- An update supplies the revision the editor loaded. A mismatched revision
  returns an optimistic conflict and preserves both the stored row and the
  user's unsaved values.
- A request resolves assignment and profile in one joined read and copies the
  profile into an immutable snapshot containing its UUID and revision.
- Editing a profile changes future requests for every assigned character.
  Already admitted requests continue with their immutable snapshot.
- The UI shows the profile's assignment count before save and delete actions.
- Deletion is blocked while assignments exist. The user must explicitly remove
  or replace those assignments first.
- "Use global default" removes the assignment; it does not create a special
  global-profile row.
- Provider health and catalog presence are never persisted as truth. They are
  evaluated against the current native registry.

Assignment lifecycle belongs to the assignment slice rather than portability.
A restorable local or server soft deletion marks the target `inactive` and
preserves its assignment. Restoring the same authoritative character makes the
target `active` again without changing the assignment. After a confirmed
permanent deletion succeeds, the delete flow attempts assignment removal and
reports cleanup failure without undoing the character deletion.

The profile library lists assignment targets by `active`, `inactive`,
`unverified`, or `missing` status. Automatic and bulk cleanup offers
**Remove missing assignments** only for authoritatively and permanently
`missing` targets. It never removes an `inactive` or `unverified` assignment.

Separately, every listed assignment has an always-reachable
**Detach assignment** action. It identifies the exact source, authority, and
character, requires explicit confirmation, and may detach an `active`,
`inactive`, `unverified`, or `missing` target. This user-initiated operation is
not automatic cleanup and does not delete or modify the character. It works
without contacting the character authority and guarantees a permanently
unreachable server cannot lock a shared profile forever.

Database corruption, unsupported schema versions, failed migrations, and
unavailable paths produce a profile-store failure. They do not cause Chatbook
to discard or recreate the database automatically.

### Backup and restore

The profile database participates in the application's **Backup All** flow.
Its backup uses SQLite's online backup mechanism so a concurrent profile write
cannot produce a torn copy. This guarantee applies per database; Backup All is
not a cross-database atomic snapshot, and this slice does not refactor the
legacy backup implementation for unrelated stores.

Backup and restore execute through the same serialized repository lifecycle
lane as ordinary CRUD. Restore is explicit and first changes the repository to
`restoring`, rejects new work, drains or cancels queued work according to its
generation, closes its long-lived connection, and acquires the exclusive
interprocess store lock. On the same serialized lane and under that lock,
Chatbook validates the candidate database version and integrity, opens a scoped
source connection to create a pre-restore SQLite online backup of the current
store, closes it, atomically replaces the file, advances the lifecycle
generation, reopens and rebinds the repository, and invalidates
profile-related service and UI state.

Failure to quiesce or acquire the exclusive lock within the bounded deadline
leaves the current store untouched and reopens it. Validation or replacement
failure likewise leaves the current store in place. Reopen failure reports the
profile store unavailable rather than creating a fresh database. No operation
or cached result from the pre-restore generation is published afterward.

Assignments restored beside a different character database or server authority
retain their full `CharacterRef` and become `unverified` or `missing` according
to authoritative checks. They are never rebound by bare character ID.

## Global settings save and publication

TTS settings publications are serialized. When more than one provider is
affected, provider transitions are acquired in canonical provider-ID order and
released in reverse order. Saving TTS settings is one ordered operation:

1. validate the complete proposed configuration, including explicit model and
   voice modes;
2. atomically replace the configuration file, including all required exact-key
   sets and dynamic-mode deletions across `[app_tts]` and legacy aliases;
3. enter the exclusive side of `TTSRequestAdmissionCoordinator` and mark each
   affected provider slot `reconfiguring`, which detaches it from new lease
   acquisition while allowing already admitted leases to finish;
4. begin each exclusive provider handoff with the saved publication generation;
5. when old leases drain within the bounded foreground handoff deadline, close
   the old adapter, install the saved lazy configuration, and mark the slot
   ready; if they do not drain, retain the old adapter only for those leases and
   leave the new slot unavailable to requests with a latest-generation pending
   configuration;
6. publish the new immutable `TTSPreferencesSnapshot` while the admission gate
   remains exclusive. Every affected slot is now either ready with the same
   saved generation or unavailable/reconfiguring; none can admit the old
   adapter;
7. release the gates and report either **Saved** or **Saved — applying after
   current speech** without blocking the Textual event loop;
8. for a pending handoff, a background finalizer waits for the existing leases,
   closes the old adapter, and installs only the latest saved generation. It
   marks the slot ready on success or unavailable on failure and exposes
   **Retry/Reconnect**.

If validation or file replacement fails, neither the in-memory snapshot nor the
registry changes. If file replacement succeeds but provider reconfiguration
fails, the saved snapshot remains authoritative and the provider is reported
unavailable with **Retry/Reconnect** recovery. The app must not continue using
an old adapter configuration or silently restore an old selection.

The foreground handoff deadline bounds settings completion, not legitimate
speech. Reconfiguration never silently cancels an admitted synthesis or closes
its response lease. A leaked or abandoned response therefore cannot freeze the
settings UI indefinitely: its provider remains unavailable to new requests,
and bounded service-response cleanup plus retry/restart provides recovery.

A newer settings save supersedes an older pending generation. A stale
finalizer may close its retired adapter but may never publish its configuration
or mark the slot ready. Any unexpected failure after the gates are acquired
leaves affected provider slots unavailable until retry or restart; cleanup
never releases a usable slot against a mismatched preference snapshot.

For exclusive audio.cpp handoff, the pending configuration is inert data. A
replacement adapter is not constructed until all old leases have drained and
the old adapter has closed, so old and replacement audio.cpp instances never
coexist.

Requests admitted before the publication barrier continue with their already
frozen old snapshot and lease. Requests admitted after it is released see the
new coherent snapshot and either its matching provider slot or a structured
reconfiguring/unavailable failure. No request is admitted while new preferences
are paired with an old adapter or vice versa.

This sequence fixes the stale-settings UAT defect. Textual sentinel objects are
interpreted as selection modes and are never serialized as empty exact values.

## Profile creation and management UX

### First-time external audio.cpp journey

1. In Speech settings, the user selects audio.cpp external mode, enters the
   server URL, and runs **Test Connection**.
2. In STTS, Chatbook loads the native audio.cpp catalog and lets the user choose
   an exact model and optional exact voice.
3. The user enters preview text, generates one complete WAV, and plays it.
4. A successful result exposes **Save result as profile**.
5. Chatbook creates the profile from the immutable request/response artifact,
   not from whatever controls happen to contain after generation.
6. In the character editor's **Voice & Speech** section, the user assigns the
   saved profile.
7. A newly generated character-authored roleplay response exposes **Speak**.
   Clicking it uses the assigned profile and plays the complete WAV.

Each successful generated-audio artifact contains a text-free immutable
generation-selection snapshot with the requested provider, exact model,
submitted voice, requested format, speed, validated options, and provider
configuration revision used at admission. Actual response provider, model,
format, content type, sample rate, and other response metadata remain separate
artifact fields and are not silently substituted into the reusable request
profile.

STTS preview uses the same admission coordinator as Console speech. Selector
snapshot, any catalog-derived value, provider configuration revision, and
matching service lease are therefore admitted atomically rather than assembled
from independently changing settings and registry state.

**Save result as profile** reads only that immutable selection snapshot. It
does not reread mutable Playground controls. Before creation, the profile
service confirms that the provider configuration revision is still current and
revalidates the exact selection against current native profile rules. A
configuration change or newly invalid selection requires the user to
regenerate or repair rather than saving ambiguous provenance. A newly created
profile starts at revision 1; no profile revision exists in the preview
artifact.

### STTS profile library

STTS provides the shared profile-management surface:

- bounded, paginated list and search;
- create from a successful preview;
- preview;
- rename and edit;
- duplicate;
- standalone export;
- delete with assignment-count protection;
- availability and repair status.

The profile editor discovers valid values from native catalogs. New audio.cpp
profiles and new assignments can be saved only when their exact provider,
model, format, and voice semantics validate against the current native
catalog and their speed is exactly `1.0`. An imported or previously valid
profile that becomes unavailable remains visible with a repair action.

The editor clearly warns that editing a shared profile changes future speech
for all assigned characters and shows the assignment count. It uses optimistic
revision checks so two open editors cannot silently overwrite each other.

### Character editor

The character editor adds a **Voice & Speech** section with:

- **Use global default** or an assigned profile selector;
- current profile availability and assignment count;
- **Preview**, **Create**, **Edit**, and **Remove assignment** actions;
- repair guidance when an existing assignment is unavailable.

Only valid current profiles can be newly assigned. Existing broken assignments
remain selected and visible so the user can repair or remove them. The editor
does not embed provider credentials or duplicate the STTS catalog logic.

The existing audiobook `CharacterVoiceWidget` remains a separate audiobook
concept and is not consulted, migrated, or treated as profile authority.

### Manual speech only

This release adds no automatic speech. The user explicitly clicks **Speak** on
a message. The existing progress, completion, error, autoplay, and cleanup
events remain the UI contract.

## Request resolution and synthesis

### Resolution order

For a character-authored assistant message:

1. Validate the app-issued `TTSMessageSpeechSnapshot` against the persisted
   message and conversation or the owning in-memory Console store.
2. Build its authority-scoped `CharacterRef`.
3. Read assignment and profile together.
4. If a valid assignment exists, freeze the exact profile UUID, revision, and
   generation fields.
5. If no assignment exists, resolve the current global preference snapshot.
6. If an assignment references an unavailable or invalid profile, fail closed.
   Do not try global preferences.

For a generic request without a trusted `CharacterRef`, resolve only global
preferences. A stale or invalid character speech snapshot is an error; it does
not become a generic global request.

A persisted message ID, native message ID, or text value is a lookup handle,
not proof of authorship. The resolver verifies the snapshot's session
ownership, stable visible variant/content revision, completed assistant role,
assistant kind, and conversation character. `TTSRequestEvent` does not accept a
caller-chosen `CharacterRef` or raw character-speech text. For a server-backed
roleplay, this path is enabled only when the Console session persisted a
versioned durable server authority established by the authenticated identity
policy; this feature does not create a new server conversation transport.

When the profile store itself is unavailable, every message carrying a
`CharacterRef` fails closed because Chatbook cannot establish that no assignment
exists. The error UI may offer **Use global for this message**, an explicit
one-shot override. That override does not create, remove, or modify an
assignment and is recorded as `explicit_override` resolution. Messages without
a `CharacterRef` continue to use global preferences.

### Exact assigned profile

An assigned profile supplies an exact model. The resolver constructs one
canonical `TTSRequest`. While the admission coordinator's shared gate remains
held, it records the provider configuration revision and asks `TTSService` to
acquire the matching lease without a redundant profile-layer catalog preflight.
Only that admitted operation may synthesize. The adapter owns readiness and
validates the request against current upstream state.

An explicit voice is submitted exactly. Voice discovery helps users create and
repair profiles, but it is not a second request-time availability gate because
some adapters legitimately omit or lazily expose voice catalogs. A null voice
is submitted only when the model contract allows server-default omission.

If the adapter rejects the model, voice, format, speed, or options, the request
fails with its safe native error. No alternate value is chosen.

### Global preferences

An exact global model follows the same direct native request path. A global
`first_available` model is the only path that asks
`get_catalog(refresh=False)` before synthesis. Catalog resolution, selection
freeze, provider-revision capture, and matching lease acquisition all occur
inside the same shared admission gate. The first eligible model is selected and
frozen for that request. If it disappears before synthesis, the request fails;
the resolver does not choose a second model.

A global `server_default` voice is submitted as `None`. A global exact voice is
submitted unchanged.

If there is no character assignment and the global provider is a retained
legacy provider, the existing compatibility bridge remains available during
migration. Profiles themselves never execute through the legacy bridge.

### Complete WAV over the asynchronous interface

The native audio.cpp adapter continues to return `TTSAudioResponse` with an
asynchronous byte iterator. The first release expects one validated, complete
WAV response rather than true progressive audio.

The Console consumer:

1. opens the response as an async context;
2. consumes the complete bounded byte stream into a secure temporary artifact;
3. discards the artifact if reading, cancellation, or validation fails;
4. publishes completion only after the WAV is complete;
5. sends the artifact through the existing playback/autoplay path;
6. cleans it up according to the existing playback lifecycle.

Response truth comes from actual response fields: format, content type, sample
rate, provider, and model provenance. Profile UUID and revision accompany the
operation for diagnostics. Character authority, input text, credentials, URL,
and local filesystem paths are excluded from logs and exports.

### Retry and fallback

Chatbook does not automatically retry a synthesis POST because generation is
not guaranteed to be idempotent. Read-only health or catalog operations may use
their existing bounded retry policy. A user may explicitly retry generation.

No audio.cpp failure falls back to a legacy provider, another model, another
voice, or global preferences after an assigned profile has been selected.

## Availability, detach, and errors

### Assignment status

An assignment's character target has one of four states:

- `active`: the current authoritative character source confirms the character;
- `inactive`: the authoritative source confirms a restorable soft deletion;
- `unverified`: the character cannot currently be checked because its authority
  is not active, unavailable, unreachable, or does not provide an authoritative
  not-found contract;
- `missing`: the authoritative source confirms permanent deletion or a
  contractually authoritative permanent not-found result.

Only `missing` assignments are eligible for automatic or bulk cleanup.
`inactive` and `unverified` assignments are preserved by those flows. A 401,
403, timeout, network failure, or ambiguous privacy-preserving 404 is
`unverified`, never `missing`. Switching active servers does not make
assignments from another authority missing.

Target verification is lazy, cached for a bounded interval, page-bounded, and
performed only against the currently matching authority. The profile library
does not perform a background full sweep or probe characters belonging to
non-current server authorities; those targets remain `unverified`. A confirmed
**Detach assignment** remains available offline for every status.

Profile availability is separate from target status. A profile may be locally
present but unavailable because its provider is unconfigured, not native, its
model is missing, or its request fields no longer validate.

Provider reconfiguration invalidates cached profile availability. Exact IDs are
revalidated on the next display or request; cached health or catalog results
never remain authoritative across a provider configuration revision.

Authoritative character deletion cleanup and the profile-side
**Remove missing assignments** action ship with assignment support. The
restriction to `missing` applies to automatic and bulk cleanup only; a confirmed
**Detach assignment** remains available for every status. Soft delete and
restore transitions preserve the assignment. Portability does not own
assignment garbage collection.

### Structured failures

Profile-domain failures are distinct from adapter `TTSOperationError` values:

| Failure | Recovery |
| --- | --- |
| profile store unavailable | Retry store access; optionally use global for this message |
| assignment target lacks authority | Reopen or repair the server-backed conversation |
| assigned profile missing/corrupt | Repair or remove the assignment |
| provider not native or not configured | Configure the provider or edit the profile |
| profile currently unavailable | Refresh catalog, reconnect, or edit the profile |
| optimistic revision conflict | Reload, compare, and reapply the edit |
| import requires repair | Review validated fields before assigning |

Native adapter error codes remain intact and supply their existing safe,
UI-neutral recovery actions. UI messages never include raw upstream bodies,
exception representations, character text, credentials, or local paths.

## Character-card portability

Slice 4 integrates only with the application's existing local character-card
import and export surfaces. It does not add server-side card persistence,
server profile synchronization, or a cross-store server transaction. A
server-character assignment remains local Chatbook state and is not written
back to the server's character card.

### Export is explicit and non-mutating

Normal character export remains byte-for-byte governed by the existing card
flow and does not include local TTS data. If the user selects
**Include TTS profile**, export creates a transient deep copy and merges a
sanitized, versioned payload at
`data.extensions["tldw_chatbook/tts_generation_profile"]` for V2 cards. The
equivalent normalized `extensions` map is used internally for other supported
card formats. Export never updates the stored character or its local
assignment.

The portable payload contains only:

- schema version;
- portable profile UUID as a correlation hint;
- display name;
- canonical provider ID;
- exact model ID;
- nullable voice ID;
- requested format;
- speed;
- validated profile-safe options.

The version-1 wire shape is exact:

```json
{
  "schema_version": 1,
  "profile_id": "00000000-0000-4000-8000-000000000000",
  "name": "Character voice",
  "provider_id": "audio_cpp",
  "model_id": "supertonic-3",
  "voice_id": "M1",
  "response_format": "wav",
  "speed": 1.0,
  "options": {}
}
```

It excludes character authority, provider origin, credentials, binary/config
paths, health, assignment count, timestamps, input text, and local profile
revision.

If a card already contains a malformed or conflicting Chatbook TTS namespace,
explicit TTS export fails rather than overwriting or ambiguously merging it.
Unrelated extension namespaces are preserved.

Standalone profile export uses the same sanitized profile payload without a
character card. Standalone profile import is not part of this design; only an
attachment arriving through character-card import is consumed.

### Import treats the extension as hostile

Import first extracts the Chatbook TTS attachment from the transient parsed
card, validates it, and removes it before character persistence. Ordinary
re-export of that stored character therefore cannot leak the imported voice
attachment unless the user explicitly includes it again.

Validation is exact and bounded:

- known schema version only;
- exact object shape with no unknown fields;
- valid UUID syntax;
- the same identifier, name, format, speed, and options bounds used locally;
- canonical provider identifier;
- attachment encoded size no greater than 16 KiB and no more than four
  container levels;
- provider-specific allowlist;
- `wav`, speed exactly `1.0`, and empty options for first-release audio.cpp
  profiles.

An unknown attachment version or provider skips only the TTS attachment with a
warning; the character can still import. Malformed known payloads also leave
the imported character unassigned and report a clear warning rather than
executing or persisting untrusted values.

Structural validity and current runtime availability are separate. A
structurally valid audio.cpp attachment whose provider is unconfigured or
unreachable, or whose model/voice cannot currently be validated, is imported as
a visible unavailable profile after collision resolution but is not assigned.
The character import reports partial success and directs the user to repair and
validate the profile before making an explicit assignment.

### Collision policy

The portable UUID is a correlation hint, not authority. The generation tuple
used for collision comparison is provider, model, voice, format, speed, and
validated options; display name and UUID are compared separately.

- If an existing local profile has the same UUID and the same generation
  tuple, prompt for **Reuse existing** or **Import copy**. A copy receives a new
  local UUID and, if necessary, a collision-safe display name.
- If an existing local profile has the same UUID but a different generation
  tuple, import is allowed only as a confirmed copy with a new local UUID and,
  if necessary, a collision-safe display name.
- If only the normalized name collides and the generation tuple is the same,
  prompt for **Reuse existing** or **Import copy**. A copy keeps the
  collision-free portable UUID and receives a collision-safe display name.
- If only the normalized name collides and the generation tuple differs,
  require confirmation, keep the collision-free portable UUID, and create a
  collision-safe display name.
- If neither UUID nor normalized name collides, create the profile with the
  portable UUID and display name.

Import never silently changes an existing profile, character assignment, or
pre-existing character.

### Cross-database ordering

Character-card and profile data live in different stores, so import uses
compensating behavior rather than pretending to provide a cross-database
transaction:

1. parse and structurally validate the character and optional attachment
   without writes, then separately evaluate current profile availability;
2. persist the character without the TTS attachment and return a structured
   outcome distinguishing a newly created character from a reused
   duplicate/conflict;
3. if the character is new, resolve any profile collision prompt and, in one
   profile-database transaction, create/reuse the profile and:
   - create the assignment when the profile is currently valid and available;
   - leave it unassigned when the profile requires repair;
4. if persistence reused an existing character, leave its current assignment
   untouched unless the user explicitly confirms **Apply imported TTS to
   existing character**. Only after that confirmation does Chatbook resolve
   profile collisions and atomically create/reuse the profile. It replaces or
   creates the exact assignment only when the profile is currently valid and
   available; otherwise the imported profile remains visible for repair and the
   existing assignment remains untouched.

Canceling a profile collision prompt or declining the existing-character
confirmation performs no profile or assignment write. If the profile
transaction fails, a newly created character remains successfully imported but
unassigned, while a reused character retains its previous assignment. The UI
reports partial success and offers profile repair. No partial profile or
assignment survives the failed profile transaction.

## Delivery slices

This design has four ordered delivery milestones, not four omnibus PRs. After
the written design is approved, planning decomposes them into the six atomic,
independently testable PR-sized sub-slices below. Each sub-slice receives its
own Backlog task, acceptance criteria, ADR links, and implementation plan; no
task combines repository lifecycle, UI library work, authority migration, and
speech runtime integration merely to preserve a milestone number.

### Slice 1 — Native external audio.cpp Console TTS

- Amend ADR-023 before implementation with atomic request
  admission/publication, expected-revision enforcement, and bounded exclusive
  handoff behavior.
- Add typed global preference modes and backward-compatible blank-value reads.
- Atomically remove canonical and legacy stale exact values for dynamic modes.
- Publish saved preferences and targeted provider reconfiguration behind one
  shared/exclusive admission coordinator.
- Route Console audio.cpp **Speak** through native `TTSService.synthesize()`.
- Preserve the async response interface and existing complete-WAV autoplay.
- Keep legacy global speech working through the temporary bridge.
- Apply the installed-build stop/go characterization gate before changing the
  pinned audio.cpp endpoint or complete-WAV contract.
- Prove the external user-owned server lifecycle is untouched.

This slice independently fixes the first-time-user UAT failure.

### Slice 2A — Profile domain and repository lifecycle

- Create and link
  `backlog/decisions/027-character-tts-generation-profile-ownership.md`,
  extending ADR-023.
- Add domain models, normalized-name validation, the versioned serialized
  repository, optimistic concurrency, interprocess lifecycle locking, backup,
  and restore.
- Verify concurrent CRUD/restore and stale-generation exclusion before any UI
  consumes the store.

### Slice 2B — Profile service and STTS library

- Add the profile service and native-provider validation.
- Add bounded STTS list/search, save-from-preview, edit, duplicate, preview,
  export, delete, availability, and repair flows.
- Extend successful preview artifacts with immutable requested-selection and
  provider-configuration provenance.
- Support native audio.cpp execution only.

Together, Slices 2A and 2B deliver reusable local profiles before character
assignment.

### Slice 3A — Character identity, authorship, and assignment

- Persist durable local-database and stable server-profile/principal authority
  provenance, including safe legacy-local backfill and fail-closed server
  identity acquisition.
- Add canonical `CharacterRef` assignment behavior and app-issued immutable
  Console speech snapshots.
- Add character-editor assignment controls.

### Slice 3B — Roleplay resolution and speech runtime

- Add trusted persisted and in-memory authorship resolution through the
  admission coordinator.
- Apply assigned profiles to character-authored Console roleplay messages.
- Add fail-closed recovery and the explicit one-message global override.
- Preserve assignments through soft delete/restore; add permanent-delete
  cleanup attempts, bounded assignment-target verification, missing-only
  automatic/bulk cleanup, and confirmed profile-side detach for every status.

### Slice 4 — Optional character-card portability

- Add transient explicit export with a sanitized Chatbook extension.
- Add hostile import validation and collision prompts.
- Add structured created-versus-reused character outcomes and cross-database
  import compensation.
- Limit portability integration to existing local card surfaces.
- Preserve ordinary card import/export behavior.

Managed audio.cpp launch and supervision remains a separate future task and is
not part of any slice here.

## Verification strategy

### Repository and service tests

- initial schema creation and every supported migration;
- transaction rollback, database corruption, and unsupported-version behavior;
- normalized unique names, composed/decomposed canonical-equivalent conflicts,
  non-ASCII case-fold conflicts, and rejected control, format-control,
  surrogate, and noncharacter code points;
- optimistic update conflicts;
- assignment foreign key and delete restriction;
- bounded pagination and assignment counts;
- serialized worker ordering and rejection of new work while restoring;
- online backup during a concurrent write;
- restore validation, open-repository quiescence/rebind, pre-restore recovery
  copy, and character-authority mismatch behavior;
- concurrent CRUD plus restore cannot publish a queued pre-restore write or
  stale read result after the lifecycle generation advances;
- a second Chatbook process holding the shared store lock makes exclusive
  restore fail safely before file replacement;
- joined assignment/profile read returns one immutable revision snapshot.

### Resolution matrix

Cover:

- assigned valid profile;
- unassigned character using global preferences;
- broken or unavailable assignment failing closed;
- profile-store failure and explicit one-message override;
- profile-store failure for an apparently unassigned `CharacterRef`;
- generic message using global preferences;
- missing server authority;
- API-key rotation preserving one server authority;
- two authenticated principals on one normalized server origin producing
  different authorities;
- missing stable authenticated identity disabling server assignments rather
  than using a credential fingerprint;
- active, inactive, unverified, and missing targets;
- soft delete followed by restore preserves the assignment;
- permanent deletion alone triggers automatic cleanup;
- 401, 403, timeout, ambiguous 404, non-current authority, and network failure
  remain unverified;
- exact versus first-available model;
- exact versus server-default voice;
- global legacy provider with no assignment.

Use two local database authorities containing the same character ID, and two
authenticated principals on the same server profile containing the same
character ID, to prove assignments cannot collide or follow the currently
active authority.

Use controlled interleavings that change the active Console session, switch the
selected variant, replace content, and delete the message after **Speak** is
clicked but before its event is handled. Only the unchanged app-issued snapshot
may be admitted; every stale or mismatched snapshot fails safely.

### Settings and runtime tests

- old blank audio.cpp model/voice values read as explicit modes without startup
  rewrite;
- Textual sentinel values never persist as empty exact selections;
- dynamic mode saves write `default_model_mode` and `default_voice_mode` and
  atomically remove both canonical and legacy stale exact-value keys, while
  exact modes dual-write and round-trip their required model and voice
  identifiers;
- successful save updates the next request without restart;
- validation/file-replacement failure publishes nothing;
- post-save reconfiguration failure leaves saved preferences authoritative and
  reports the provider unavailable;
- a controlled pause between preference/profile resolution and lease
  acquisition cannot produce new preferences with an old adapter or old
  preferences with a new adapter;
- an expected provider-revision mismatch fails admission defensively;
- a long-running or abandoned response never blocks the settings action past
  its foreground handoff deadline, is not silently cancelled, and leaves new
  requests rejected until the latest saved generation becomes ready;
- a superseded background reconfiguration finalizer cannot publish stale
  configuration, and multi-provider transitions use deterministic ordering;
- preview profile creation preserves requested model, voice, format, speed, and
  options even after controls are changed;
- response-model provenance remains distinct from the requested profile model;
- provider configuration change after preview requires regeneration rather
  than saving stale provenance;
- profile edit after request admission does not change the in-flight request;
- exact assigned speech performs one joined profile read and zero
  profile-layer catalog preflights;
- first-available global selection freezes one model and never selects a
  second;
- exact voice is sent unchanged;
- audio.cpp rejects every speed other than exactly `1.0`;
- complete WAV validation, bounded consumption, playback, cancellation, and
  cleanup;
- a matching installed audio.cpp build passes health, model, voice, MIME, and
  complete-WAV characterization and records its build identity without silently
  changing the pin;
- an incompatible installed build stops Slice 1 and requires a separately
  reviewed ADR-023 and adapter-contract amendment;
- an older configuration reader retains exact-mode values through dual-write,
  while dynamic-mode downgrade is explicitly rejected or requires a
  pre-feature configuration backup;
- no automatic synthesis POST retry;
- no adapter, model, voice, global, or legacy fallback after assigned
  resolution;
- no event-loop blocking during store, network, or playback operations.

### UI tests

- STTS preview and save-result-as-profile;
- profile list, search, edit, duplicate, conflicts, availability, repair,
  export, and protected deletion;
- character assignment, removal, repair, and shared-profile warnings;
- soft-delete/restore preservation, permanent character deletion cleanup,
  missing-target removal, and preservation of inactive and unverified
  assignments;
- temporary server unavailability never auto-detaches an assignment, while a
  confirmed profile-library action can detach that exact unverified assignment;
- Console **Speak**, progress, autoplay, explicit override, and errors;
- settings display **Saved — applying after current speech** and structured
  unavailable/retry states without freezing the Textual event loop;
- persisted and app-issued authorship resolution rejects spoofed or mismatched
  character references, stale sessions, deleted messages, and changed variants,
  and leaves generic assistants on global preferences;
- server-character authority missing and same-ID/different-authority cases;
- paginated profile views perform bounded, cached verification only against the
  matching active authority;
- import/export confirmation and collision prompts.

### Portability and security tests

- ordinary export contains no profile payload;
- explicit export does not mutate stored cards;
- unrelated extensions survive export/import;
- malformed existing Chatbook namespace fails explicit export;
- unknown versions/providers skip the attachment while importing the character;
- a structurally valid but currently unavailable audio.cpp profile imports
  visibly but remains unassigned pending repair;
- reject unknown fields, oversized identifiers/payloads, deep JSON, invalid
  UUIDs, non-finite or non-`1.0` audio.cpp speed, unsupported formats, and
  audio.cpp options;
- reject canonically duplicate or invisible profile names after local or card
  import normalization;
- every UUID/name collision combination follows the explicit matrix, including
  the different-UUID/same-name/same-generation case;
- a collision-free import adopts the portable UUID; any required copy UUID is
  newly generated;
- collision outcomes require explicit choice and never mutate an existing
  profile;
- duplicate-name character persistence returns a reused outcome and never
  changes that character's assignment without explicit confirmation;
- a confirmed but currently unavailable imported profile leaves a reused
  character's existing assignment untouched and exposes the profile for repair;
- canceling a collision or existing-character confirmation creates no profile
  and changes no assignment;
- simulated failure after character persistence leaves it imported and
  unassigned with no partial profile transaction, while a reused character
  retains its prior assignment;
- Slice 4 performs no server-side card persistence or synchronization;
- logs, events, exports, and metrics exclude text, authority, credentials,
  origins, and filesystem paths.

### Legacy regression

Existing OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk global
speech behavior remains covered through the compatibility bridge. No new
profile may select or execute those providers until they receive native
adapters.

### End-to-end UAT

Run from isolated application config and data directories against a
user-started external audio.cpp server:

1. configure and test the external URL;
2. discover the exact Supertonic model and voice;
3. generate and play a preview;
4. save the successful result as a profile;
5. assign it to a local roleplay character;
6. use a deterministic local LLM-compatible fixture to generate a new
   character-authored roleplay response through the real Console flow;
7. click **Speak** and validate provider/model/voice provenance, a complete
   playable WAV, expected format metadata, and successful playback through the
   application's player abstraction; on the current macOS UAT host, record
   `afplay` as the selected platform player;
8. confirm server health;
9. confirm Chatbook did not launch, restart, signal, or stop the server.

Focused automated UI tests support the UAT but do not replace the real Console
flow.

## Performance, privacy, and observability

- Profile list/search is bounded and paginated.
- All profile-store work runs through the repository-owned serialized worker
  off the Textual event loop; no UI or ad-hoc executor owns a database
  connection.
- Persisted assigned request admission uses one bounded
  message/conversation-authorship lookup followed by one joined profile lookup;
  an app-issued in-memory authorship snapshot avoids the first lookup.
- The admission coordinator never holds its exclusive gate on the Textual event
  loop while waiting for synthesis leases to drain. Foreground settings
  completion is bounded and longer handoffs use the generation-checked
  finalizer.
- Assignment target verification is not part of synthesis admission and adds no
  character-server network request to the speech path.
- Exact assigned requests add no profile-layer catalog network call.
- Metrics contain only provider ID, resolution source (`assigned`, `global`, or
  `explicit_override`), safe outcome code, and latency.
- Metrics and logs never include message text, character authority, profile
  display name, voice text if provider-sensitive, credentials, URLs, raw
  upstream errors, or local paths.

## Rollback

The separate profile database makes code rollback non-destructive: older builds
ignore it. Removing the profile/assignment UI returns behavior to global
preferences without mutating character cards. Schema downgrades are not
performed automatically.

Slice 1 code and routing are reversible, but its configuration downgrade
compatibility depends on the selected modes. Exact model/voice modes remain
readable by older builds because their legacy exact keys are dual-written.
Dynamic modes intentionally remove those keys, and an older build does not
understand `default_model_mode` or `default_voice_mode`. Before downgrading from
dynamic modes, the user must either save explicit model and voice values with
the current build or restore a trusted pre-feature configuration backup. The UI
and release notes must not describe that downgrade as transparent.

Each later slice must preserve data created by earlier slices if its UI
integration is disabled.

## ADR assessment

**ADR required:** yes

**ADR paths:**

- amend
  `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
  before Slice 1 implementation;
- create
  `backlog/decisions/027-character-tts-generation-profile-ownership.md`
  before Slice 2A implementation.

**Reason:** Slice 1 strengthens the existing cross-module TTS service contract
with atomic request admission/publication, expected configuration revisions,
and a bounded exclusive handoff state machine; those decisions amend ADR-023
rather than waiting for profile persistence. The later profile feature
establishes a new versioned store, data ownership and identity rules,
fail-closed runtime semantics, and character-card import/export policy. ADR-027
will extend rather than duplicate the amended ADR-023.

No new dependency decision is required. The implementation uses the existing
SQLite, validation, TTS service, registry, Textual, backup, and playback
facilities.

## Acceptance criteria

- A first-time user can save external audio.cpp settings and immediately use
  Console **Speak** without restarting Chatbook.
- Blank legacy audio.cpp defaults have explicit compatible semantics and are
  never treated as unavailable exact identifiers.
- Preference/profile selection, catalog-derived choices, provider revision,
  and adapter lease acquisition form one admission boundary, so a successful
  settings save admits no mixed old/new request.
- Settings completion is bounded even while an existing audio.cpp response
  retains its lease; admitted speech is not silently cancelled, new requests
  fail safely during handoff, and only the latest saved generation can become
  ready.
- Console speech uses the native adapter and plays a complete validated WAV
  while preserving the asynchronous response contract.
- Console character speech uses an immutable app-issued session/message/content
  snapshot and rejects stale, deleted, variant-switched, spoofed, or mismatched
  authorship.
- A user can save a successful STTS audio.cpp preview as a named reusable
  profile and manage it in a shared library.
- Profile names use trimmed NFKC case-folded uniqueness and reject invisible or
  invalid control code points across local and imported data.
- Every first-release audio.cpp global selection and profile uses WAV, speed
  exactly `1.0`, and no arbitrary options.
- A user can assign one profile to a local or authority-scoped server character.
- Assignments remain scoped across local database changes and authenticated
  principals sharing one server, remain stable across credential rotation, and
  survive character soft delete/restore.
- Server assignments require a durable versioned server-profile/principal
  authority; credential fingerprints and the currently active server are never
  used as substitutes.
- Manual speech for a character-authored response uses the assigned immutable
  profile revision.
- An unavailable assigned profile produces an actionable error with no silent
  fallback.
- Editing a shared profile affects only future requests and warns about all
  assigned characters.
- Provider connection details and credentials never enter profile persistence,
  logs, metrics, or exports.
- Ordinary character-card export remains free of TTS profile data; explicit
  inclusion uses the sanitized, hostile-input-safe portability contract.
- Import never changes an existing character assignment without explicit
  confirmation.
- Backup uses a consistent per-profile-database SQLite snapshot, and failed
  restore validation, concurrent queued work, or failure to exclude another
  Chatbook process does not replace or split the current profile store.
- The installed audio.cpp build must pass the pinned-contract characterization
  gate; an incompatible build stops the slice until ADR-023 and adapter
  fixtures are separately amended.
- Exact-mode configuration remains readable by the legacy reader; dynamic-mode
  downgrade requires explicit values or a pre-feature configuration backup.
- Existing unassigned legacy-provider speech continues through the temporary
  bridge.
- Chatbook never manages the external audio.cpp process in this feature.

## Open questions

None. The architecture, fallback policy, identity model, settings publication,
profile lifecycle, manual-speech scope, portability rules, delivery order, and
verification requirements were explicitly resolved during design review.
