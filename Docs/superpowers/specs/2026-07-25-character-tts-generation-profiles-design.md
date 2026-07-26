# Character TTS Generation Profiles with Native audio.cpp Console Speech — Design

**Status:** approved by the user on 2026-07-25; independent spec review pending
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
- Voice cloning or voice-reference uploads.
- Migration of audiobook `CharacterVoiceWidget` data.
- Group-roleplay speaker attribution.
- Arbitrary, unvalidated provider option dictionaries.
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

An `exact` mode requires a non-empty corresponding identifier. A
`first_available` model is resolved once at request admission. A
`server_default` voice is sent as `None`.

For backward compatibility, existing blank audio.cpp model and voice values are
read as `first_available` and `server_default`. Startup does not rewrite the
configuration file. The next successful settings save persists explicit modes.

### TTS generation profile

A `TTSGenerationProfile` is a reusable, complete, exact generation selection.
It contains:

- immutable UUID profile ID;
- trimmed display name plus a case-insensitive normalized uniqueness key;
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
own profile-safe option schema without a database redesign. audio.cpp accepts
only `wav` and an empty options object in this release; arbitrary options are
rejected.

### Character reference

A bare character ID is not globally unique. Assignments use the canonical
three-part identity:

```text
CharacterRef = (source, authority_id, character_id)
```

- a local character is `("local", "local", <local-character-id>)`;
- a server character is
  `("server", <persisted-server-authority-id>, <server-character-id>)`.

`authority_id` is an opaque stable identity supplied by the existing active
server profile/runtime context. It is not a display name and is never inferred
from whichever server happens to be active later.

`authority_id` and `character_id` are non-empty canonical text identifiers of
at most 256 characters. The source-aware character layer supplies the
canonical representation; profile code does not parse, renumber, or derive it
from a display label.

A server-backed conversation records its authority when the conversation is
launched. Existing conversations without that provenance cannot receive or use
a server-character assignment until they are reopened or explicitly repaired.

Only assistant messages authored by the selected roleplay character carry a
`CharacterRef`. User, system, tool, persona-only, and generic assistant messages
do not inherit a character assignment.

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
- explicit backup and restore participation.

It has no dependency on Textual, adapter instances, provider health, character
cards, or the active server.

#### `TTSProfileService`

The service owns profile-domain validation and workflows:

- create, update, rename, duplicate, list, and delete profiles;
- assign, replace, and remove character assignments;
- validate a profile against current native provider capabilities;
- calculate profile availability and repair guidance;
- prepare and consume standalone or character-card export payloads;
- apply import collision policy;
- expose structured profile-domain errors.

It depends on the repository and the app-owned `TTSService` catalog APIs, but it
does not synthesize audio itself.

#### `CharacterTTSRequestResolver`

The resolver converts message authorship, assignment state, and the current
global preference snapshot into either:

- an immutable resolved request selection; or
- a structured, user-actionable resolution failure.

It performs one joined repository read for an assigned character. It does not
fetch health from the database, call a concrete adapter, mutate assignments, or
silently fall back.

### Dependency flow

```text
Console / STTS / Character editor
             |
             v
 CharacterTTSRequestResolver -----> TTSProfileService
             |                              |
             |                              v
             |                     TTSProfileRepository
             v
      app-owned TTSService
             |
             v
      TTSAdapterRegistry
             |
             v
   native audio_cpp adapter
```

UI layers may use the profile service for management and the resolver for
speech admission. They never access SQLite rows or concrete adapters directly.

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
| `display_name` | trimmed, 1–128 Unicode characters |
| `normalized_name` | case-folded uniqueness key |
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

Names are trimmed and compared by a stored case-folded key. Case-only duplicate
names fail with a conflict. Display spelling remains user-controlled.

### Transaction and concurrency rules

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

Database corruption, unsupported schema versions, failed migrations, and
unavailable paths produce a profile-store failure. They do not cause Chatbook
to discard or recreate the database automatically.

### Backup and restore

The profile database participates in the application's **Backup All** flow with
the same consistent-snapshot expectations as other local stores. Restore is an
explicit operation that validates the database version and integrity before
replacement. An unsupported or corrupt backup fails without replacing the
current profile store.

## Global settings save and publication

Saving TTS settings is one ordered operation:

1. validate the complete proposed configuration, including explicit model and
   voice modes;
2. atomically replace the configuration file;
3. publish a new immutable `TTSPreferencesSnapshot` to the running app;
4. perform targeted adapter-registry reconfiguration for affected providers.

If validation or file replacement fails, neither the in-memory snapshot nor the
registry changes. If file replacement succeeds but provider reconfiguration
fails, the saved snapshot remains authoritative and the provider is reported
unavailable with **Retry/Reconnect** recovery. The app must not continue using
an old adapter configuration or silently restore an old selection.

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

Saving a result captures the requested provider, exact model, submitted voice,
requested format, speed, validated options, and profile revision. Actual
response format, content type, and sample rate remain artifact metadata and are
not silently substituted into the reusable request profile.

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
catalog. An imported or previously valid profile that becomes unavailable
remains visible with a repair action.

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

1. Build its persisted `CharacterRef`.
2. Read assignment and profile together.
3. If a valid assignment exists, freeze the exact profile UUID, revision, and
   generation fields.
4. If no assignment exists, resolve the current global preference snapshot.
5. If an assignment references an unavailable or invalid profile, fail closed.
   Do not try global preferences.

For a message without `CharacterRef`, resolve only global preferences.

When the profile store itself is unavailable, assigned character speech fails
closed. The error UI may offer **Use global for this message**, an explicit
one-shot override. That override does not create, remove, or modify an
assignment and is recorded as `explicit_override` resolution. Generic messages
continue to use global preferences.

### Exact assigned profile

An assigned profile supplies an exact model. The resolver constructs one
canonical `TTSRequest` and calls `TTSService.synthesize()` without a redundant
profile-layer catalog preflight. The adapter owns readiness and validates the
request against current upstream state.

An explicit voice is submitted exactly. Voice discovery helps users create and
repair profiles, but it is not a second request-time availability gate because
some adapters legitimately omit or lazily expose voice catalogs. A null voice
is submitted only when the model contract allows server-default omission.

If the adapter rejects the model, voice, format, speed, or options, the request
fails with its safe native error. No alternate value is chosen.

### Global preferences

An exact global model follows the same direct native request path. A global
`first_available` model is the only path that asks
`get_catalog(refresh=False)` before synthesis. The first eligible model is
selected and frozen for that request. If it disappears before synthesis, the
request fails; the resolver does not choose a second model.

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

An assignment's character target has one of three states:

- `active`: the current authoritative character source confirms the character;
- `unverified`: a server character cannot currently be checked because its
  authority is unavailable or unreachable;
- `missing`: the authoritative source confirms deletion or not-found.

Only `missing` assignments are eligible for cleanup. `unverified` assignments
are preserved. Switching active servers does not make assignments from another
authority missing.

Profile availability is separate from target status. A profile may be locally
present but unavailable because its provider is unconfigured, not native, its
model is missing, or its request fields no longer validate.

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

It excludes character authority, provider origin, credentials, binary/config
paths, health, assignment count, timestamps, input text, and local profile
revision.

If a card already contains a malformed or conflicting Chatbook TTS namespace,
explicit TTS export fails rather than overwriting or ambiguously merging it.
Unrelated extension namespaces are preserved.

Standalone profile export uses the same sanitized profile payload without a
character card.

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
- empty options for first-release audio.cpp profiles.

An unknown attachment version or provider skips only the TTS attachment with a
warning; the character can still import. Malformed known payloads also leave
the imported character unassigned and report a clear warning rather than
executing or persisting untrusted values.

### Collision policy

The portable UUID is a correlation hint, not authority:

- if an existing local profile has the same UUID and identical portable
  generation fields, prompt for **Reuse existing** or **Import copy**;
- if the UUID or normalized name collides but the generation fields differ,
  create a new UUID and collision-safe display name after user confirmation;
- if no match exists, create a new profile.

Import never silently changes an existing profile or assignment.

### Cross-database ordering

Character-card and profile data live in different stores, so import uses
compensating behavior rather than pretending to provide a cross-database
transaction:

1. parse and validate the character and optional attachment without writes;
2. persist the character without the TTS attachment;
3. in one profile-database transaction, create/reuse the profile and create the
   assignment.

If step 3 fails, the character remains successfully imported but unassigned.
The UI reports partial success and offers profile repair. No partial profile or
assignment survives the failed profile transaction.

## Delivery slices

This design is implemented as four ordered, independently reviewable PR-sized
slices. Each slice receives its own Backlog task, acceptance criteria, and
implementation plan only after this written design is approved.

### Slice 1 — Native external audio.cpp Console TTS

- Add typed global preference modes and backward-compatible blank-value reads.
- Publish saved preferences immediately in the running app.
- Route Console audio.cpp **Speak** through native `TTSService.synthesize()`.
- Preserve the async response interface and existing complete-WAV autoplay.
- Keep legacy global speech working through the temporary bridge.
- Prove the external user-owned server lifecycle is untouched.

This slice independently fixes the first-time-user UAT failure.

### Slice 2 — Local profile repository and STTS library

- Create and link
  `backlog/decisions/027-character-tts-generation-profile-ownership.md`,
  extending ADR-023.
- Add the versioned profile repository, service, validation, optimistic
  concurrency, backup, and restore.
- Add bounded STTS list/search, save-from-preview, edit, duplicate, preview,
  export, delete, availability, and repair flows.
- Support native audio.cpp execution only.

This slice delivers reusable local profiles before character assignment.

### Slice 3 — Character assignment and roleplay speech

- Persist server conversation authority provenance.
- Add canonical `CharacterRef` assignment behavior.
- Add character-editor controls and resolver integration.
- Apply assigned profiles to character-authored Console roleplay messages.
- Add fail-closed recovery and the explicit one-message global override.

### Slice 4 — Optional character-card portability

- Add transient explicit export with a sanitized Chatbook extension.
- Add hostile import validation and collision prompts.
- Add cross-database compensation and detached-assignment cleanup.
- Preserve ordinary card import/export behavior.

Managed audio.cpp launch and supervision remains a separate future task and is
not part of any slice here.

## Verification strategy

### Repository and service tests

- initial schema creation and every supported migration;
- transaction rollback, database corruption, and unsupported-version behavior;
- normalized unique names and case-only conflicts;
- optimistic update conflicts;
- assignment foreign key and delete restriction;
- bounded pagination and assignment counts;
- backup and explicit restore validation;
- joined assignment/profile read returns one immutable revision snapshot.

### Resolution matrix

Cover:

- assigned valid profile;
- unassigned character using global preferences;
- broken or unavailable assignment failing closed;
- profile-store failure and explicit one-message override;
- generic message using global preferences;
- missing server authority;
- active, unverified, and missing targets;
- exact versus first-available model;
- exact versus server-default voice;
- global legacy provider with no assignment.

Use two server authorities containing the same character ID to prove assignments
cannot collide or follow the currently active server.

### Settings and runtime tests

- old blank audio.cpp model/voice values read as explicit modes without startup
  rewrite;
- Textual sentinel values never persist as empty exact selections;
- successful save updates the next request without restart;
- validation/file-replacement failure publishes nothing;
- post-save reconfiguration failure leaves saved preferences authoritative and
  reports the provider unavailable;
- profile edit after request admission does not change the in-flight request;
- exact assigned speech performs one joined profile read and zero
  profile-layer catalog preflights;
- first-available global selection freezes one model and never selects a
  second;
- exact voice is sent unchanged;
- complete WAV validation, bounded consumption, playback, cancellation, and
  cleanup;
- no automatic synthesis POST retry;
- no adapter, model, voice, global, or legacy fallback after assigned
  resolution;
- no event-loop blocking during store, network, or playback operations.

### UI tests

- STTS preview and save-result-as-profile;
- profile list, search, edit, duplicate, conflicts, availability, repair,
  export, and protected deletion;
- character assignment, removal, repair, and shared-profile warnings;
- Console **Speak**, progress, autoplay, explicit override, and errors;
- server-character authority missing and same-ID/different-authority cases;
- import/export confirmation and collision prompts.

### Portability and security tests

- ordinary export contains no profile payload;
- explicit export does not mutate stored cards;
- unrelated extensions survive export/import;
- malformed existing Chatbook namespace fails explicit export;
- unknown versions/providers skip the attachment while importing the character;
- reject unknown fields, oversized identifiers/payloads, deep JSON, invalid
  UUIDs, non-finite speed, unsupported formats, and audio.cpp options;
- UUID/name collision outcomes require explicit choice and never mutate an
  existing profile;
- simulated failure after character persistence leaves it imported and
  unassigned with no partial profile transaction;
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
5. assign it to a character;
6. use a deterministic local LLM-compatible fixture to generate a new
   character-authored roleplay response through the real Console flow;
7. click **Speak** and validate provider/model/voice provenance, a complete
   playable WAV, expected format metadata, and successful `afplay`;
8. confirm server health;
9. confirm Chatbook did not launch, restart, signal, or stop the server.

Focused automated UI tests support the UAT but do not replace the real Console
flow.

## Performance, privacy, and observability

- Profile list/search is bounded and paginated.
- Store work that may exceed the UI budget runs off the Textual event loop.
- Assigned request admission uses one joined profile lookup.
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

Slice 1 is independently reversible because it changes routing and preference
publication without creating profile data. Each later slice must preserve data
created by earlier slices if its UI integration is disabled.

## ADR assessment

**ADR required:** yes

**ADR path:** `backlog/decisions/027-character-tts-generation-profile-ownership.md`

**Reason:** the feature establishes a new versioned store, data ownership and
identity rules, a cross-module resolution interface, fail-closed runtime
semantics, and character-card import/export policy. ADR-027 will extend rather
than duplicate ADR-023 and will be created before Slice 2 implementation begins.

No new dependency decision is required. The implementation uses the existing
SQLite, validation, TTS service, registry, Textual, backup, and playback
facilities.

## Acceptance criteria

- A first-time user can save external audio.cpp settings and immediately use
  Console **Speak** without restarting Chatbook.
- Blank legacy audio.cpp defaults have explicit compatible semantics and are
  never treated as unavailable exact identifiers.
- Console speech uses the native adapter and plays a complete validated WAV
  while preserving the asynchronous response contract.
- A user can save a successful STTS audio.cpp preview as a named reusable
  profile and manage it in a shared library.
- A user can assign one profile to a local or authority-scoped server character.
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
- Existing unassigned legacy-provider speech continues through the temporary
  bridge.
- Chatbook never manages the external audio.cpp process in this feature.

## Open questions

None. The architecture, fallback policy, identity model, settings publication,
profile lifecycle, manual-speech scope, portability rules, delivery order, and
verification requirements were explicitly resolved during design review.
