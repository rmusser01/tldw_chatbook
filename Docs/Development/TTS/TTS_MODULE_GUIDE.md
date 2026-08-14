# TTS (Text-to-Speech) Module Guide

## Overview

The TTS module in tldw_chatbook provides a flexible, extensible system for generating speech from text using multiple providers. It supports both cloud-based APIs (OpenAI, ElevenLabs) and local models (Kokoro, Chatterbox, Higgs), with features like streaming audio generation, format conversion, text normalization, advanced voice cloning, and multi-speaker dialog generation.

## Architecture

### TTS adapter service

The application owns one sealed `TTSAdapterRegistry` and one `TTSService`.
Native adapters use canonical provider IDs and `TTSService.synthesize()`.
`audio_cpp` is the first native adapter. It is registered first, by the exact
canonical ID `audio_cpp`, with display label `audio.cpp` and no alias. The
adapter remains unmaterialized until its first operation.

The following six entries remain unchanged behind the temporary compatibility
bridge: `openai`, `elevenlabs`, `kokoro`, `chatterbox`, `higgs`, and `alltalk`.
Each bridge adapter lazily owns one provider-scoped `TTSBackendManager`;
application and UI code must not access that manager or any concrete adapter.
The bridge is removed only after every retained provider has a native adapter
and all legacy internal-model callers have migrated.

New providers are implemented as native adapters. See
[ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
and the approved
[audio.cpp adapter design](../../superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md).

### Module Structure

```
tldw_chatbook/TTS/
├── __init__.py              # Module exports
├── adapter_types.py         # Provider-neutral adapter contracts
├── adapter_registry.py      # Sealed app-scoped provider registry
├── adapter_bootstrap.py     # Application service construction
├── legacy_bridge.py         # Temporary provider-scoped compatibility adapters
├── audio_cpp_config.py      # Immutable active-mode audio.cpp configuration
├── audio_cpp_guided_config.py # Full guided/user-JSON/external Settings state
├── audio_cpp_recipes.py     # Sealed release-0.5.1 package recipes
├── audio_cpp_package_scanner.py # Bounded explicit-root package discovery
├── audio_cpp_guided_launch.py # Private generated POSIX launch snapshot
├── audio_cpp_managed_config.py # Managed launch validation and child environment
├── audio_cpp_supervisor.py  # One app-scoped managed audio.cpp child
├── audio_cpp_contract.py    # Pinned JSON and PCM16 WAV validation
├── preferences.py           # Immutable global defaults and config mutations
├── request_admission.py     # Atomic preference/revision/lease admission
├── profile_errors.py        # Value-independent profile/store failures
├── profile_types.py         # Immutable profiles, assignments, and receipts
├── profile_reference_types.py # Private clone-reference values and quotas
├── profile_reference_audio.py # Bounded WAV admission/canonicalization
├── profile_reference_storage.py # Metadata projections and streamed BLOB I/O
├── profile_schema.py        # Dedicated SQLite validation and codecs
├── migrations/
│   ├── v0_to_v1.py         # Initial profile-store schema migration
│   ├── v1_to_v2.py         # Version-two profile-store migration
│   └── v2_to_v3.py         # Private clone-reference table migration
├── profile_store_lock.py    # Cooperative shared/exclusive process locking
├── profile_repository.py    # Serialized CRUD, backup, and restore lifecycle
├── profile_service.py       # Native profile validation and capability overlay
├── playground_types.py      # Immutable Playground request/artifact contracts
├── adapters/
│   └── audio_cpp.py         # Native audio.cpp HTTP adapter
├── audio_schemas.py         # Pydantic schemas for requests/responses
├── TTS_Generation.py        # Main TTS service orchestration
├── TTS_Backends.py          # Legacy bridge manager and base class
├── audio_service.py         # Audio format conversion service
├── text_processing.py       # Text normalization and chunking
├── backends/                # Backend implementations
│   ├── __init__.py
│   ├── openai.py           # OpenAI TTS API
│   ├── kokoro.py           # Local Kokoro model
│   ├── elevenlabs.py       # ElevenLabs API
│   ├── chatterbox.py       # Chatterbox TTS (voice cloning)
│   ├── higgs.py            # Higgs Audio V2 (advanced voice cloning)
│   └── higgs_voice_manager.py # Voice profile management for Higgs
└── utils/                   # Utility modules
    ├── __init__.py
    ├── download_models.py   # Model download utilities
    ├── voice_utils.py       # Voice mixing utilities
    └── performance.py       # Performance tracking
```

### Core Components

#### 1. TTSService (`TTS_Generation.py`)
The main orchestration layer that:
- Routes canonical provider IDs through the sealed registry
- Exposes provider-neutral synthesis, catalog, voice, and reconfiguration
  operations
- Retains adapter resources until each audio response is closed
- Preserves the legacy byte-stream interface during migration

#### 2. TTSAdapterRegistry (`adapter_registry.py`)
The application-owned registry performs exact provider lookup, lazy adapter
materialization, operation leasing, targeted reconfiguration, and bounded
shutdown. Registration is sealed at construction time.

`TTSBackendBase`, `TTSBackendManager`, and the class-global legacy backend
registry are compatibility-bridge internals. They are not the extension point
for new providers.

### Local generation profiles (Slices 2A–2B)

Reusable generation profiles now have a dedicated, versioned SQLite ownership
boundary. `TldwCli` constructs one initially closed `TTSProfileRepository` and
opens it lazily for a profile-store consumer such as **Backup All**. The default
file is `tldw_chatbook_tts_profiles.db` in the current Chatbook user data
directory. An installation may instead set a validated path:

```toml
[database]
tts_profiles_db_path = "/absolute/path/to/tts-profiles.db"
```

The store is local-only and separate from character cards, provider
configuration, and conversation storage. Current schema version 3 retains the
complete, immutable profile snapshots and authority-scoped assignment records
introduced by earlier versions and adds an optional profile-owned clone
reference row. Profile display names are trimmed and have a unique key derived as
`NFKC(display_name).casefold()`. Creates begin at revision 1; updates require
the exact revision read by the editor and increment it atomically. A stale
revision or normalized-name collision reports a conflict without overwriting
the stored row. Assignment identity is the complete
`(source, authority_id, character_id)` tuple, and a foreign-key restriction
prevents deletion of an assigned profile.

Every repository operation runs through one serialized off-event-loop worker,
which owns at most one long-lived SQLite connection. An open repository keeps a
cooperative shared lock next to the database, so multiple Chatbook processes
may read and write through SQLite while each retains shared ownership. Restore
must first quiesce admitted work and acquire a bounded exclusive lock. A second
process that still holds a shared lock therefore prevents replacement and
causes restore to fail before the live file is changed.

Normal operations and results carry a monotonic lifecycle generation. Restore
advances that generation when admitted, rejects new normal work while
`restoring`, cancels queued older work, and prevents an already-running older
result from being published. The public states are `open`, `restoring`,
`unavailable`, and `closed`; definitive close is terminal.

`TTSProfileRepository.backup_to()` uses SQLite's online-backup API, validates
the completed standalone snapshot, and publishes it atomically at its
destination. **Backup All** reaches the profile database only through this
repository method; it never copies the open profile file. The databases in one
Backup All directory are individually consistent snapshots taken during the
run, not one cross-database atomic snapshot.

Restore is an explicit, bounded repository operation. It validates a private
snapshot of the candidate, stages it through SQLite online backup, performs
schema, full-integrity, foreign-key, and domain-row checks, and creates a
durable pre-restore recovery database before atomic replacement. Quiescence,
candidate validation, exclusive-lock, recovery-backup, or replacement failure
leaves the current store authoritative and rebinds it when safe. If replacement
succeeds but shared-lock reacquisition or authoritative reopen fails, the
repository reports `unavailable`, retains recovery evidence, and does not
create a blank database. Corrupt, partial, unsupported-version, or missing
established stores likewise fail closed instead of being recreated.

The restore timeout is one absolute cooperative budget. SQLite copies run in
bounded page batches; structural, quick-check, foreign-key, integrity, and
count queries use SQLite VM progress interruption; and schema-owned rows and
private candidate-copy chunks check the same deadline. Checkpoint busy waiting
is capped to the remaining budget. Checks also surround staging, sidecar
handling, recovery, replacement, final publication, and durable flush
boundaries so the exclusive lease is released promptly on expiry.
An individual kernel call such as `fsync`, `replace`, `stat`, `read`, or
`write` cannot be interrupted after it starts, so one such in-flight call may
finish just beyond the requested timeout before cleanup releases ownership.

#### Profile v3 private clone-reference storage

Schema v3 can store one canonical clone reference per generation profile: a
bounded PCM16 WAV BLOB, bounded reference transcript, digest, validated audio
metadata, immutable reference UUID, and timestamps. The source path is never
persisted. Ordinary profile reads project only the reference metadata summary.
The WAV BLOB is chunk-streamed, while the bounded transcript and digest are
selected as scalar fields; all three are revalidated for an exact reference
read, mutation, backup qualification, or restore qualification.

Reference audio and transcript are local plaintext. Owner-only filesystem
controls protect the profile database, retained migration backup, recovery
copies, and temporary backup files, but that protection is not encryption.
Profile database backups contain the same sensitive reference audio and
transcript. Deletion and SQLite/WAL cleanup are best-effort deletion, not
forensic erasure, especially on copy-on-write filesystems, backups, and storage
media. The Windows privacy posture remains unverified until TASK-13208; do not
infer a Windows ACL guarantee from the POSIX owner-private implementation.

The repository schema is version 4. Opening any supported older store advances
a private candidate through each migration step and retains validated downgrade
siblings at the applicable boundaries: `<profile-db>.pre-v3.sqlite3` and
`<profile-db>.pre-v4.sqlite3`. Version 3 reference rows receive null recipe
provenance; migration never infers it from current Settings. Publication is a
journaled, non-cancellable complete-or-restore protocol after replacement. To
downgrade to a version-3-capable build, close Chatbook completely, restore the
stable pre-v4 sibling as the configured database, and only then start the older
build. This loses every post-migration change and all v4 provenance.
For a version-2-capable build, restore the retained v2 pre-migration backup
instead; that downgrade causes loss of post-migration profile changes.

Ordinary portability remains sanitized: reference-free profiles retain exact
wire version 1; reference-bearing profiles use exact wire version 2 with only
`{"reference":{"status":"omitted"}}` added. The v2 decoder returns a bounded
skip/no-mutation result. Explicit clone transfer uses a separate warning-gated
voice-bundle portability path containing only `manifest.json`, `profile.json`,
canonical `reference.wav`, and canonical `reference.txt`. The deterministic
writer uses
fixed metadata; the hostile reader validates archive layout before bounded
streaming decompression and never invokes general extraction.

`TTSVoiceBundlePortabilityService` owns source/destination authority, at most
four expiring single-use inspection sessions, deterministic no-overwrite
publication, and retained cleanup. UI receives only an opaque handle plus safe
review facts. Commit repeats source validation and pure dependency inspection,
then delegates Create/Reuse/Copy to one serialized repository transaction.
Missing exact dependencies may be stored only as explicitly accepted inactive
profiles. Import never assigns or changes a default. The app closes and joins
the portability service before the profile repository and TTS service.

#### Guided clone execution and materialization

Character and default-profile resolution read the exact reference under the
same repository generation and profile revision fences as the profile. A
reference is eligible only when that one profile owns both the effective
audio.cpp provider and exact model. Admission then freezes the registry's
configuration revision and applied generation separately from the adapter's
accepted recipe revision and managed process generation. A later saved or
staged configuration cannot contaminate the admitted operation.

Local reference paths are authorized only for a reviewed Guided Managed recipe
running in the application-owned audio.cpp child. The source-authority
preflight runs before readiness, catalog HTTP, or launch work, so External mode
and Managed user-provided `server.json` fail closed without receiving or
materializing the reference. After readiness, the adapter admits the exact
model, optional native voice, reference policy, recipe, and process generation.
There is no model, voice, provider, or non-clone fallback.

Only after that admission does the service create an opaque owner-private
operation directory below the Chatbook user-data
`tts_clone_materializations` child. The internal adapter request carries a
typed `voice_ref` path and `reference_text`; those fields never enter public
`TTSRequest`, selection provenance, profile options, catalog state, or
`server.json`. The response owns the materialization until underlying
stream/adapter cleanup completes, after which the exact directory is removed
before the provider lease and operation capacity are released.

The materializer is lazy and POSIX-only. It retains a no-follow descriptor and
nonblocking ownership lock, serializes publication and startup sweep with a
root-scoped cross-instance lock, verifies effective-user ownership and
owner-only modes, and removes only exact recognized unlocked directories.
Unknown entries, links, substituted objects, and live owners are preserved.
Service shutdown seals new materializations and
`wait_closed()` retains all creation and cleanup work to a terminal result.
The adapter repeats those retained-versus-lexical identity checks immediately
before sending the path. Because audio.cpp opens a pathname rather than a file
descriptor, this boundary does not claim to defend against a malicious process
running as the same OS user racing the child after that check; owner-only
runtime-root access is the operating-system isolation boundary.

Reference audio, transcript, and the short-lived operation file are local
plaintext, not encrypted. Once a clone request is admitted, child-output
content is suppressed for that exact managed process generation so chunked or
delayed echoes cannot retain the path or transcript in diagnostics. Public
errors and response metadata remain value-independent. The next process
generation starts with normal bounded sanitized diagnostics.

#### Speech Lab clone setup, save, and Roleplay assignment

When the selected ready Guided model exposes a reviewed reference-required
recipe, Speech Lab replaces ordinary generation with one **Create Voice &
Generate** path. The pane accepts a bounded PCM16 WAV and required bounded
transcript, canonicalizes them off the UI loop, and retains that canonical
value only as the current setup draft. Picker cancellation is inert; validation
focuses the invalid field without discarding the other field. The interface
states that the reference, transcript, profile database, and short-lived
materialization are local plaintext rather than encrypted data.

Starting clone generation captures the provider/model/applied-generation and
draft revision. The admitted service owns the reference below the public
request boundary, materializes it only after the exact Guided capability is
accepted, and returns clone evidence only with a structurally valid complete
WAV. A failed or stale operation leaves the setup draft available for retry and
cannot replace the last playable result. A successful matching operation
transfers exact canonical authority to the handler-owned current result and
clears the pane draft; the pane receives only a playback-safe projection.

**Save as Voice Profile** is offered only for that handler-owned successful
result. The profile service atomically creates the generation profile and its
reference from the admitted evidence, without reopening the source WAV or
reading mutable selectors. The review can save unassigned or navigate to
Roleplay with a non-authoritative profile identity suggestion. It never changes
the app default or a character assignment. Roleplay validates that suggestion
against a fresh repository generation/profile revision, marks it as suggested,
and persists an assignment only when the user explicitly selects it.

Console **Speak** then resolves the exact character assignment and stored
reference before provider readiness work. The first Speak lazily starts or
joins the one compatible Guided Managed child and uses the assigned model and
reference; global defaults cannot override it. Profile/character browsing is
passive. Message, assignment, profile/reference, configuration, recipe, or
process-generation changes at an admission boundary fail stale rather than
switching identity. Result replacement, discard, pane/app close, and response
cleanup release their respective canonical, artifact, materialization, and
lease ownership in order.

Profiles persist generation selections, not connection or process
configuration. Provider origins, credentials, API keys, custom headers, binary
paths, `server.json` paths, health observations, message text, and raw local
paths are excluded from profile data and safe repository diagnostics.

`TTSProfileService` is a native-only boundary over the application-owned
repository and `TTSService`. It creates profiles only from immutable successful
audio.cpp Playground artifacts and copies the artifact's admitted provider,
exact model, optional exact voice, WAV format, speed `1.0`, and empty options.
Before saving, it verifies that the artifact's admitted
configuration revision is still current; the revision itself is not persisted.
It never derives persisted values from mutable UI selectors. The service also
provides bounded 50-row pages, optimistic
edit/delete/duplicate operations, assignment-aware deletion, and exact
capability observation without mutating stored rows.

The **Voice profiles** view renders repository rows before capability
enrichment, then marks each exact selection `available`, `unavailable`, or
`unverified`. Unavailable is an authoritative incompatibility and recovers by
editing; unverified is a transient or stale observation and recovers by
refreshing. Preview copies the stored values into a one-shot exact Playground
preset without synthesizing. Search, paging, refresh, edit, duplicate, preview,
and delete operate on the exact loaded repository generation and revision.
Repository failures remain isolated: the library shows bounded recovery copy
while ordinary Playground and Console speech continue to work.

Slice 2B does not provide character-assignment UI or authority acquisition,
roleplay routing, profile/card portability or synchronization, legacy-provider
profile execution, provider connection details, or managed audio.cpp process
behavior. See
[ADR-028](../../../backlog/decisions/028-character-tts-generation-profile-ownership.md).

### Slice 3A assignment mutation service

Character assignment identity is the exact
`(source, authority_id, character_id)` tuple. Set or replace requires the
caller-held repository generation, selected profile revision, expected current
assignment (including explicitly unassigned), and a fresh authoritative
capability check. Detach is idempotent when the assignment is already absent,
but refuses to remove a different replacement. The repository's final
transaction checks remain authoritative.

This slice adds no assignment UI, speech resolver, automatic speech, Persona
inheritance, profile portability, Sync behavior, or managed audio.cpp behavior.
See
[ADR-037](../../../backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md).

### Global defaults and Console request admission

TASK-710 represents global TTS defaults as one immutable
`TTSPreferencesSnapshot`. audio.cpp supports explicit selection modes in
`[app_tts]`:

```toml
[app_tts]
default_provider = "audio_cpp"
default_model_mode = "first_available" # or "exact"
default_voice_mode = "server_default"  # or "exact"
default_format = "wav"
default_speed = 1.0
```

`exact` mode also requires the corresponding non-empty `default_model` or
`default_voice`. `first_available` resolves the first model from one admitted
catalog snapshot, and `server_default` omits `voice` from the request. Existing
audio.cpp configurations that have no mode keys and contain blank model or
voice values read as `first_available` and `server_default` without a startup
write.

The settings UI translates its local Select sentinels before persistence, so a
sentinel cannot become an empty exact identifier. One atomic configuration
mutation always writes the authoritative mode keys. Exact values are
dual-written to the canonical and legacy exact keys; dynamic modes remove stale
exact values from both locations. Exact-mode configurations therefore remain
readable by older builds. Dynamic-mode downgrade is not transparent: save
explicit model and voice values before downgrading, or restore a trusted
pre-feature configuration backup.

`TTSRequestAdmissionCoordinator` freezes the complete preference selection,
resolves any dynamic model, reads the provider revision, and acquires the
matching registry lease under one writer-preferred admission gate. Settings
publication persists off the Textual event loop in one service-retained task,
then uses the exclusive side of that gate for a bounded handoff. A foreground
save may report **Saved — applying after current speech**; the admitted speech
continues, only the latest pending generation may become active, and the old
audio.cpp adapter closes before a replacement can be created.

Console **Speak** does not post caller-supplied message text. The Console store
issues an ephemeral immutable `TTSMessageSpeechSnapshot` and binds it to the
store validator. Before the cooldown clock, normalization, or provider work,
the TTS handler revalidates the active session and branch, native and persisted
message identity, selected text variant and exact content, process-local speech
revision, durable row version when present, completed assistant role/status,
and trusted assistant authorship. A stale, edited, deleted, incomplete,
non-assistant, or authorship-mismatched snapshot fails closed with bounded retry
copy asking the user to select **Speak** again.

The snapshot is process-local, is not persisted, and is not a voice-profile
selection. Once admitted, Console still uses the saved global TTS defaults.
Direct `TTSRequestEvent` callers outside Console retain their explicit trusted
global path.

Console **Speak** calls `TTSService.synthesize_default()`. An `audio_cpp`
selection uses the native adapter with locked WAV, speed `1.0`, and empty
options. The six retained providers continue through `LegacyTTSAdapter`. The
native complete WAV is still consumed through `TTSAudioResponse`'s asynchronous
iterator and closed through the existing artifact/playback lifecycle. Snapshot
admission neither persists the snapshot nor performs message writes, and it
does not change ownership of the external audio.cpp process.

### Native audio.cpp adapter (external mode)

Slice 2 connects to one existing `audiocpp_server`; it does not launch or
supervise a process. Configuration comes only from `[app_tts.audio_cpp]`:

```toml
[app_tts.audio_cpp]
mode = "external"
base_url = "http://127.0.0.1:8080"
connect_timeout_seconds = 5
synthesis_timeout_seconds = 600
max_input_characters = 10000
max_response_bytes = 134217728
max_metadata_bytes = 1048576
max_catalog_models = 1000
max_voices_per_model = 1000
max_identifier_characters = 256
```

`base_url` must be a canonical absolute HTTP or HTTPS origin. Credentials,
non-root paths, query strings, fragments, and invalid ports are rejected. The
configuration has no environment override, authentication or custom-header
field, binary path, `server.json` path, or other process field. HTTPS keeps
certificate verification enabled. Invalid configuration is rejected during
local projection or adapter materialization with a safe, value-independent
`ValueError`, before any provider operation; the external adapter does not emit
`configuration_invalid`.

`connect_timeout_seconds` configures HTTP connection establishment and also
bounds the complete required health-plus-models discovery sequence, including
an eligible safe-GET retry. The same value independently bounds each optional
voice-discovery operation. `synthesis_timeout_seconds` bounds the speech request
through complete response consumption; the HTTP connect timeout still applies
inside it. There is no read-inactivity timer.

The adapter implements the pinned `audio_cpp_http_v1` structure from
audio.cpp commit
[`d3d748179e5ace353386fbf17bcaedfacf482d75`](https://github.com/0xShug0/audio.cpp/tree/d3d748179e5ace353386fbf17bcaedfacf482d75):

- Required readiness surfaces: `GET /health` and `GET /v1/models`.
- Optional lazy voice metadata:
  `GET /v1/audio/voices?model=<id>`.
- Complete speech response: `POST /v1/audio/speech`.

Readiness retains only bounded TTS model metadata. Voice discovery is lazy,
bounded, per model, and cached by provider configuration and catalog revision.
A missing or invalid optional voices endpoint produces no discovered voices;
it does not make an otherwise compatible provider unavailable. Callers
represent the server-selected voice as `None`: the UI-facing “Server default”
sentinel is not sent in the speech payload.

Requests accept a known model, non-empty bounded text, an optional safe voice,
WAV output, speed exactly `1.0`, and no adapter options. Synthesis sends one
non-retried POST containing only `model`, `input`, `response_format: "wav"`,
and an optional `voice`. Safe GET operations may receive one bounded retry.
All requests disable redirects and request identity encoding.

The adapter bounds metadata and audio reads before parsing. It rejects
compressed, oversized, malformed, or incompatible responses and validates the
entire response as structurally complete, uncompressed PCM16 RIFF/WAV.
Validated bytes are then yielded as one asynchronous chunk. The asynchronous
stream contract is preserved, but Slice 2 does not provide incremental audio
streaming.

`TTSOperationError` exposes only a stable code, safe message, retryability,
local operation ID, and optional recovery action. Connectivity and
required-contract failures make cached health stale; invalid requests, optional
voice failures, busy responses, generation failures, invalid audio, and
cancellation do not. There is no automatic fallback to another model or a
legacy provider.
Successful response metadata contains only safe scalar provenance, sample, and
bounded timing values. Logs exclude submitted text, configured origins and
values, response bodies, and rejected identifiers.

The registry admits only one active audio.cpp adapter. An unchanged normalized
configuration is a no-op. A changed configuration blocks new operations,
drains active leases, closes the old adapter, and only then installs the new
configuration; the replacement remains lazy, so old and new instances never
overlap.

Normal tests use fake HTTP transport and fixtures pinned to the reviewed
upstream commit. They require neither an audio.cpp binary nor model downloads.

The installed Homebrew package `audio-cpp 0.4` was characterized on
2026-07-25 as compatible with the pinned health, model, voice, and speech
endpoints and complete PCM16 `audio/wav` response contract. This is
compatible-build evidence only: it does not move the ADR-023 upstream pin or
grant Chatbook ownership of the external server process.

An isolated clean-config Textual Console UAT subsequently selected
`audio_cpp` at `http://127.0.0.1:8080` with `first_available` model and
`server_default` voice, generated a deterministic Mira response, and exercised
one native adapter. Console produced one owner-only (`0600`) complete WAV of
594,604 bytes: mono PCM16 at 44.1 kHz, 297,280 frames, and 6.741 seconds.
Observed lifecycle counts were complete `1`, playback `1`, progress `4`, and
streaming `0`; `/usr/bin/afplay` exited `0`. The same external listener identity
and healthy response were present before and after the run, and application
shutdown took no action on that user-owned process.

After the implementation was rebased, all 23 patches were range-diff
identical. Fresh focused and broad automated suites passed, but a second live
run was unavailable because the installed `audio-cpp 0.4` binary had no
running process, listener, or healthy endpoint. Chatbook intentionally did not
launch it; external-process ownership remains with the user.

### Managed audio.cpp runtime and UI (Slices 4–5)

External mode remains the default and owns no server process. Managed mode is
an explicit alternative in canonical Global Settings. Its manual source uses a
trusted prebuilt executable and existing `server.json`; its Guided source uses
a separately installed executable plus explicitly reviewed package identities
and defers private configuration materialization to a deliberate runtime
operation. Settings preserves dormant fields across External, manual, and
Guided sources but validates and projects only the selected source. Save is
passive and performs no launch, probe, discovery, synthesis, or generated-
artifact creation.

The application constructs one provider-specific `AudioCppSupervisor` beside
the sealed registry and service. A Managed configuration carries one absolute
user-supplied executable path, one absolute user-supplied `server.json` path,
and bounded startup, health-interval, and termination-grace values. Validation
requires an executable file, a readable strict UTF-8 JSON object no larger than
1 MiB, no duplicate keys or non-finite constants, `host` exactly
`127.0.0.1`, and an integer port from 1 through 65,535. Validation and passive
snapshots do not launch, probe, discover, or adopt a process.

Only a deliberate audio.cpp service operation may start Managed mode. Native
synthesis admission, an explicit catalog or voice refresh, and
`start_and_test_audio_cpp()` all pass through the same preparation seam.
`restart_audio_cpp()` drains and stops the current generation, applies the
latest eligible staged settings, and starts only when the resulting mode is
Managed. `shutdown_audio_cpp()` drains and stops without itself launching a
replacement. Passive process/capability snapshots, descriptor reads, and
non-refresh cache reads never launch a child.

Launch is always the exact argv
`<user executable> --config <user server.json>`, without a shell, with the JSON
directory as the working directory. A fail-closed loopback-port preflight runs
immediately before spawn. The child receives a fixed runtime/platform
environment allowlist after known provider credential names and secret-like
names are removed. Chatbook never scans for an existing server, attaches to an
unowned PID, appends arbitrary arguments, or runs an automatic restart loop.

Concurrent first use shares one shielded startup. Publication and admission
gates order configuration saves and applications against startup; lifecycle
epochs and process generations fence stop, restart, shutdown, and stale work.
One exit monitor is the sole reaper for each exact child. Process generations
bind the child, health work, output drains, and generation-local adapter HTTP
client together. Only one health probe may run for a generation; periodic and
on-demand callers share it. Adapter retirement closes that generation's HTTP
resources before a replacement generation can be published.

Configuration and runtime identity use separate monotonic values:

- **Saved generation** identifies the latest durably published settings.
- **Applied generation** identifies the settings currently owned by the
  registry slot; a save may be staged while speech is still using the prior
  generation, and each newer save atomically supersedes or clears the older
  stage.
- **Process generation** identifies one exact app-owned child. An observation
  version may advance as that same child changes state or health.

All deliberate operations acquire the registry/admission side before entering
the supervisor. Explicit transitions publish Draining, reject new leases, wait
for admitted work, stop and reap only the owned child, and then retire the
generation-local adapter before promoting a replacement. Applying External
uses the same transition and cannot allow an older staged Managed mapping to
reappear. Terminal service shutdown uses one outer deadline: registry
admission seals immediately, leases/adapters drain first, and the running
child's terminate/kill grace is capped by that shared deadline. `close()` is
bounded; definitive cleanup remains retained after the foreground budget when
necessary, and `wait_closed()` cannot succeed while a child, startup, health,
reaper, output-drain, or generation-client task remains.

The immutable process snapshot reports `stopped`, `starting`, `running`,
`unhealthy`, `draining`, `stopping`, or `unavailable`, along with safe failure
metadata. Stable managed failure codes are `configuration_invalid`,
`port_in_use`, `process_spawn_failed`, `process_startup_timeout`,
`process_exited`, `contract_incompatible`, `runtime_unhealthy`, and
`cleanup_failed`. Health or exit failure never triggers a restart. Cleanup
uncertainty seals further launches when exact generated-artifact retirement
cannot be proved. A later deliberate operation may retry from the latest
eligible saved mode after ordinary runtime failure, and a successful
replacement clears the sealed Unavailable failure.

Stdout and stderr are continuously drained into a memory-only ring bounded by
line count, retained bytes, and bytes per line. ANSI/control sequences and
recognizable credential assignments are best-effort sanitized, home paths are
abbreviated, Rich markup is escaped, and an eviction count makes truncation
truthful. Raw output is not copied into general logs, configuration, or
persistence. Cleanup has bounded drain joining and closes only the parent's
pipe transports when a descendant retains inherited descriptors; it never
signals or adopts that descendant.

Managed mode reuses the existing native HTTP adapter and multi-model catalog.
It still validates and returns one complete PCM16 WAV item through the
asynchronous response interface; Managed mode does not add incremental
streaming or change playback behavior.

Speech Lab observes this state through one coherent, passive
`AudioCppRuntimeObservation`. Its single runtime card shows saved, applied, and
process generations; process/capability/endpoint state; catalog freshness; and
pending configuration. It links back to Global Settings rather than duplicating
durable fields. Start/Test, Restart/Apply, External Apply/Stop, and Shutdown are
retained asynchronous service actions; incompatible catalog/generation actions
are disabled while a transition is active, while an existing complete WAV
remains playable. Playback Stop and managed-process Shutdown are separate
controls.

The runtime card's details and recent diagnostics disclosures are collapsed by
default. Full configured paths appear only in the explicit read-only details
disclosure and are excluded from observation/projection reprs. Diagnostics use
only the supervisor's bounded, sanitized, memory-only snapshot, identify the
process generation and stream, and warn that output can still be sensitive.
Expanding either disclosure is inert: it cannot persist settings, acquire a
registry lease, or launch a child.

Managed mode must not be exposed on Windows until native Windows CI proves
direct execution of a user-supplied binary, graceful termination and
force-kill, sole child reaping, and bounded parent-pipe closure. External
audio.cpp remains available on Windows, and injected supervisor coverage
remains mandatory on every platform.

### Guided Managed audio.cpp launch (POSIX)

Curated Model Library packages are joined to recipes at the exact reviewed
audio.cpp inventory commit
`597048d9a920592808d7d4e2acd7b9c4596a143a`. The join keeps three states
distinct: downloadable, local-only, and unsupported. Provisioning always uses
the shared artifact service with `activate=False`; installing bytes does not
select a model, publish Settings, install the server binary, or launch a child.
The Settings handoff leases the exact inactive root while it rescans and merges
one non-stale result into the detached draft.

Lease ownership follows the artifact, not the screen: the shared service owns
installed roots, the unsaved Settings draft holds exact roots while reviewing
and saving, a staged generation holds them until cancellation/transfer, and an
owned child retains its immutable runtime handle until the child has stopped.
Removal goes through the public artifact-service deletion boundary only after
an ordered dependency preview and a final fingerprint recheck. Contention,
interruption, or changed source state leaves the registry and package
recoverable for a fresh preview; private clone-reference assets are separate
owners and are never deleted implicitly.

Guided setup is a structured Managed source alongside the existing advanced
user-provided `server.json` source. The pinned release accounting covers all 21
families and 67 package variants: 45 reviewed variants are downloadable in
Model Library, 8 approved variants are local-only, and 14 are explicitly
unsupported. Model Library shows only the downloadable set; local-only
variants enter through the same bounded **Add local package…** scanner and do
not become a parallel installer path. Each accepted package freezes its recipe
revision, canonical root and file identities, safe model projection, public
model ID, speech capabilities, and backend posture. A scan that finds multiple
exact candidates never chooses one silently; explicitly reviewed candidates
remain individually identifiable even when they share one selected root.

The exact recipe, not the family task label alone, determines first-sample
readiness. Supertonic is text-ready. PocketTTS standalone GGUF recipes are
revision 2 with `Reference: Required`: release-0.5.1 registers them, but the
GGUF file does not contain the separate voice embedding required by real
synthesis. The PocketTTS Safetensors layout includes reviewed embeddings and
remains `Reference: Optional`. A voice-required default is still registered in
the one child, but Settings hands off to **Test Connection** rather than
promising **Hear a Sample**.

Inflect Micro v2 also depends on eSpeak-ng and its English data. The pinned
upstream 0.5.1 guide documents explicit `inflect_v2.espeak_library_path` and
`inflect_v2.espeak_data_path` session options only when eSpeak-ng is outside
the dynamic-loader/data search locations. An installed library or data package
is not sufficient evidence that the server process can resolve its default
names. Guided configuration intentionally does not discover or persist private
host paths. Verify loader/data discoverability before testing Inflect; when the
defaults are not discoverable, provide the explicit options through the
advanced user-owned `server.json` flow.

Saving Guided Settings remains passive. The first deliberate Test, Start,
Restart & Apply, catalog refresh, voice refresh, or synthesis revalidates the
exact accepted package identities off the event loop, validates the selected
binary, resolves only a backend tuple carried by every recipe, selects a
bounded private loopback port, and creates one owner-private generation-local
`server.json`. `Auto` currently resolves to the reviewed CPU baseline on POSIX;
an accelerated backend is not inferred from installation or hardware alone.

Generated JSON is an immutable launch artifact, not another settings file. It
contains only the reviewed top-level and per-model fields: loopback host and
selected port, backend/device/thread limits, lazy loading, disabled request-body
logging, bounded body/busy limits, and absolute model paths plus recipe-owned
options. It omits CORS and arbitrary extensions. The supervisor launches the
same direct no-shell `audiocpp_server --config <generated server.json>` argv and
owns the artifact with the exact process generation.

All accepted models share that one child. The native catalog admits only exact
lowercase upstream `tts` and `clone` tasks, cross-checks the complete returned
model set against the generation snapshot, and preserves typed capabilities
such as PocketTTS `("tts", "clone")`. Unrelated ASR, voice-conversion, music,
and other task types cannot enter the TTS catalog. The Running generation keeps
its immutable launch snapshot if source files later change; the next deliberate
replacement must revalidate before stopping or replacing it.

Pre-spawn failure, failed startup, unexpected exit, replacement, explicit
shutdown, and app close all retire the exact generated artifact after owned
generation clients and tasks. Artifact identity or cleanup uncertainty fails
closed with sanitized, path-independent errors. The user-provided JSON source
retains its existing JSON-parent working directory and ownership semantics.
Windows guided launch remains out of scope until native handle, ACL, lifecycle,
and real-process parity are implemented and evidenced.

The canonical Settings panel exposes Guided as a first-class source rather
than another connection mode. Its bounded asynchronous scanner projects exact
recipe family/variant, speech tasks, evidence state, public model ID, path-safe
package identity, and lazy/resident-memory truth. The draft owns its accepted
candidate identities; a newer scan, source switch, or unmount fences late
results. Save revalidates those identities off the Textual message loop before
publishing settings. A changed or deleted package blocks persistence and asks
the user to scan again, while a successful save announces
`Configuration saved — ready to test` and offers a no-work navigation handoff
to Speech Lab's current primary action. It does not mutate separate Studio
preferences.

For a saved Guided configuration, the immutable runtime observation carries
only the path-safe facts needed to project the one primary action. A first-use
state yields **Start & Generate Sample**; pending live settings yield the
existing exact apply/restart action; a failed combined sample yields **Retry
Sample**. The click retains that displayed projection, so provider switches or
late observations cannot turn the visible action into another lifecycle
operation. The combined action composes existing service seams: prepare/start,
refresh and verify the exact saved default catalog entry, recheck provider,
configuration, catalog, and process fences, then issue the ordinary complete-
WAV synthesis request. It does not introduce a second adapter, player, or
streaming protocol.

The current-result region retains the complete validated WAV independently of
later discovery failures. It reports duration and safe provider/model/voice,
configuration-revision, and process-generation provenance, and exposes the
existing Play/Pause/Stop behavior plus **Generate again** and **Save WAV**.
Optional autoplay is read only from the persisted Studio preference and never
changed by Global Settings or Guided setup. All result and lifecycle controls
derive their disabled reason and tooltip from the same current state; live
announcements cover meaningful state transitions rather than progress ticks.

Accepted Guided models stay registered in one lazy multi-model child. Exact
model changes reuse its process generation, and the UI warns that audio.cpp may
retain a loaded model until explicit shutdown. Console continues to capture
the exact global selection and Roleplay the exact character-profile selection;
passive browsing and observation do not launch Guided mode.

### Catalog-driven STTS Playground (Slice 3)

TASK-569 implements the external audio.cpp Playground vertical. Opening the
Playground reads sealed registry descriptors through `TTSService`; descriptor
discovery does not resolve provider factories or materialize adapters. Only the
selected provider is resolved. Selecting `audio_cpp` for the first time performs
bounded readiness and model discovery against the saved external server.

Catalog and voice discovery use independent Textual worker groups. Their result
tokens include the canonical provider ID and configuration revision, plus the
catalog revision and model ID where applicable. Results from an old selection,
configuration, catalog, or model are discarded. Catalog refresh, generation,
and playback cannot cancel one another, and a second generation cannot replace
the active generation operation.

One catalog-control projection drives provider, model, voice, format, and speed
controls. For audio.cpp, the local **Server default** voice sentinel is initially
selected and becomes `voice=None`; it is never sent as an identifier. Format is
locked to WAV and speed to `1.0`. Switching to one of the six legacy providers
restores that provider's prior model, voice, format, speed, and provider-specific
control state. If refreshed metadata removes a selection, the Playground
announces and selects a valid fallback. A stale catalog remains visible but
disables new generation until readiness recovers.

Generation captures an immutable provider-neutral request. `audio_cpp` is the
native path and calls `TTSService.synthesize(TTSRequest)`; the six existing
providers remain on the temporary `generate_audio_stream()` compatibility path.
The validated complete WAV is stored as an immutable artifact containing its
provider, model, optional voice, source-text snapshot, operation ID, actual
format/content type, and safe response metadata. Playback and export use that
artifact, so later selector changes cannot relabel the result or its filename.

Stable adapter failures map to safe, actionable Playground messages and
recovery actions. Cancellation remains cancellation, existing artifacts remain
playable and exportable after discovery failures, and an audio.cpp generation
never automatically falls back to another model or provider. The UI and logs
do not expose submitted text, configured origins or values, credentials, raw
remote bodies, or unsafe remote identifiers.

Slice 3 connects only to an existing externally managed `audiocpp_server`.
Slice 4 adds the managed runtime core described above. Slice 5 exposes its
passive configuration in global Settings and its deliberate lifecycle,
capability, catalog, diagnostics, generation, and playback controls in Speech
Lab.

#### 3. Audio Service (`audio_service.py`)
Handles audio format conversion with:
- `StreamingAudioWriter`: Real-time encoding for streaming
- Support for MP3, Opus, AAC, FLAC, WAV, PCM
- Async and sync conversion methods

#### 4. Text Processing (`text_processing.py`)
Provides text preparation for TTS:
- `TextNormalizer`: Handles URLs, emails, phone numbers, units
- `TextChunker`: Splits long texts respecting sentence boundaries
- Language detection based on voice selection

## Backend Implementations

### OpenAI Backend

**Features:**
- Supports tts-1 and tts-1-hd models
- Multiple voices: alloy, echo, fable, onyx, nova, shimmer
- Streaming response support
- All OpenAI audio formats

**Configuration:**
```toml
[app_tts]
OPENAI_API_KEY_fallback = "sk-your-api-key"
```

### Kokoro Backend

**Features:**
- Local text-to-speech using Kokoro-82M model
- ONNX runtime support (PyTorch planned)
- Multiple voice packs
- Voice mixing capabilities (planned)
- No internet connection required

**Configuration:**
```toml
[app_tts]
KOKORO_ONNX_MODEL_PATH_DEFAULT = "models/kokoro-v0_19.onnx"
KOKORO_ONNX_VOICES_JSON_DEFAULT = "models/voices.json"
KOKORO_DEVICE_DEFAULT = "cpu"  # or "cuda" for GPU
KOKORO_MAX_TOKENS = 500
```

**Voices:**
- Female: af_bella, af_nicole, af_sarah, af_sky, bf_emma, bf_isabella
- Male: am_adam, am_michael, bm_george, bm_lewis

### Chatterbox Backend

**Features:**
- Zero-shot voice cloning with 7-20 seconds of reference audio
- Emotion exaggeration control
- Ultra-low latency streaming (< 200ms)
- Advanced text preprocessing (dot-letter correction, reference removal)
- Multi-candidate generation with Whisper validation
- Audio normalization and post-processing
- Voice library with metadata tracking
- Fallback strategies for robust generation
- MIT licensed open-source model

**Configuration:**
```toml
[app_tts]
CHATTERBOX_DEVICE = "cuda"  # or "cpu"
CHATTERBOX_EXAGGERATION = 0.5  # Emotion control (0.0-1.0)
CHATTERBOX_CFG_WEIGHT = 0.5    # Pace/style control
CHATTERBOX_TEMPERATURE = 0.5   # Voice variation (0.0-2.0)
CHATTERBOX_NUM_CANDIDATES = 1  # Number of candidates (1-5)
CHATTERBOX_VALIDATE_WHISPER = false  # Enable validation
CHATTERBOX_PREPROCESS_TEXT = true    # Text preprocessing
CHATTERBOX_NORMALIZE_AUDIO = true    # Audio normalization
CHATTERBOX_TARGET_DB = -20.0         # Target volume (dB)
CHATTERBOX_RANDOM_SEED = null        # For reproducibility
CHATTERBOX_MAX_CHUNK_SIZE = 500      # Max text chunk size
CHATTERBOX_VOICE_DIR = "~/.config/tldw_cli/chatterbox_voices"
```

**Advanced Features:**
- **Text Preprocessing**: Automatically converts "J.R.R." to "J R R", removes [1] references, URLs
- **Multi-Candidate Generation**: Generate multiple versions and select the best using Whisper
- **Voice Cloning**: Upload any 7-20 second audio clip for instant voice cloning
- **Metadata Tracking**: Save voices with creation time, duration, sample rate
- **Fallback Strategies**: Three-tier system (high_quality → balanced → safe)

### ElevenLabs Backend

**Features:**
- High-quality voice synthesis
- Advanced voice settings (stability, similarity boost, style)
- Multiple languages and accents
- Speaker boost for enhanced clarity
- Multiple output formats

**Configuration:**
```toml
[app_tts]
ELEVENLABS_API_KEY_fallback = "your-api-key"
ELEVENLABS_DEFAULT_VOICE = "voice-id"
ELEVENLABS_DEFAULT_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_192"
ELEVENLABS_VOICE_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.8
ELEVENLABS_STYLE = 0.0
ELEVENLABS_USE_SPEAKER_BOOST = true
```

### Higgs Audio Backend

**Features:**
- State-of-the-art voice cloning from 15-30 second samples
- Multi-speaker dialog generation
- 15 built-in high-quality voices (professional, energetic, calm, etc.)
- Real-time streaming audio generation
- Voice profile management for custom voices
- Support for mixed cloned and built-in voices in dialogs
- Cross-lingual voice transfer

**Configuration:**
```toml
[app_tts]
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_DEVICE = "cuda"  # or "cpu", "mps" for Apple Silicon
HIGGS_ENABLE_FLASH_ATTN = true
HIGGS_MAX_NEW_TOKENS = 2048
HIGGS_TEMPERATURE = 0.8
HIGGS_TOP_P = 0.95
HIGGS_REPETITION_PENALTY = 1.05
HIGGS_GUIDANCE_SCALE = 1.0
HIGGS_VOICE_SAMPLES_DIR = "~/.config/tldw_cli/higgs_voices"
```

**Voice Cloning:**
```python
# Create a voice profile
success = await backend.create_voice_profile(
    profile_name="custom_voice",
    reference_audio_path="/path/to/sample.wav",
    display_name="My Custom Voice"
)

# Use the cloned voice
request = OpenAISpeechRequest(
    input="Hello from my cloned voice!",
    voice="custom_voice"
)
```

**Multi-Speaker Dialog:**
```python
# Format text with speaker tags
dialog_text = """[Speaker: professional_female]
Welcome to our presentation.

[Speaker: energetic_male]
We're excited to share our findings!

[Speaker: custom_voice]
Let me add my perspective..."""

request = OpenAISpeechRequest(
    input=dialog_text,
    voice="multi"  # Special voice for multi-speaker
)
```

## Installation

### Basic Installation
The core TTS functionality (OpenAI, ElevenLabs) is included with the base installation:
```bash
pip install tldw_chatbook
```

### Local TTS Support
For local TTS models like Kokoro, install the optional dependencies:
```bash
pip install tldw_chatbook[local_tts]
```

### Chatterbox Support
For Chatterbox voice cloning capabilities:
```bash
pip install tldw_chatbook[chatterbox]
```

This installs:
- chatterbox-tts: Core Chatterbox model
- torchaudio: Audio processing
- torch: PyTorch runtime
- faster-whisper: For validation (optional)

### Higgs Audio Support
For state-of-the-art voice cloning and multi-speaker generation:
```bash
pip install tldw_chatbook[higgs_tts]
```

This installs:
- boson-multimodal: Higgs Audio V2 model
- torch: PyTorch runtime
- torchaudio: Audio processing and voice cloning
- numpy/scipy: Audio manipulation
- librosa: Advanced audio features
- soundfile: Audio I/O
- transformers: Text processing

Local TTS installs:
- kokoro-onnx: ONNX runtime for Kokoro
- scipy: Audio processing
- nltk: Text tokenization
- pyaudio/pydub: Audio playback
- transformers: Advanced tokenization
- torch: PyTorch support (for future backends)
- onnxruntime: ONNX model inference

### Kokoro Model Setup
1. Download the model files:
   - Model: `kokoro-v0_19.onnx` (~300MB)
   - Voices: `voices.json`

2. Place them in your configured paths or use the download utility:
   ```python
   from tldw_chatbook.TTS.utils.download_models import download_kokoro_model
   await download_kokoro_model()
   ```

## Usage

### Basic Usage in the App

1. Click the speak button (🔊) on any chat message
2. The TTS service will:
   - Use the configured default provider
   - Generate audio with the default voice
   - Play the audio automatically

### TTS Playground (S/TT/S Tab)

The S/TT/S tab provides a comprehensive TTS testing environment:

1. **Text Input**: Enter any text to synthesize
2. **Provider Selection**: Choose `audio_cpp` or one of the six legacy
   providers from registry descriptors
3. **Voice Selection**: Discovered audio.cpp voices with Server default, or
   legacy provider-specific voices including custom uploads
4. **Advanced Settings**:
   - **audio.cpp**: Catalog-selected model, complete WAV, and speed `1.0`
   - **Chatterbox**: Exaggeration, CFG weight, temperature, candidates, validation
   - **ElevenLabs**: Stability, similarity boost, style, speaker boost
   - **Kokoro**: Language selection
5. **Audio Controls**: Play, pause, stop, and export generated audio
6. **Generation Log**: Real-time feedback on TTS processing

### Programmatic Usage

```python
from tldw_chatbook.TTS import OpenAISpeechRequest, get_tts_service

# The application binds the service before callers request it.
tts_service = await get_tts_service()

request = OpenAISpeechRequest(
    model="tts-1",
    input="Hello, world!",
    voice="alloy",
    response_format="mp3",
    speed=1.0
)

internal_model_id = "openai_official_tts-1"
async for chunk in tts_service.generate_audio_stream(request, internal_model_id):
    audio_file.write(chunk)
```

`TTSService.synthesize(TTSRequest)` is the native-adapter API. Use it directly
for `audio_cpp`. Its complete validated WAV is exposed as one chunk through the
response's asynchronous iterator, and callers must close the response. The six
legacy registry entries require private bridge metadata that
`generate_audio_stream()` supplies; do not call `synthesize()` directly for
those entries.

### Event System Integration

The TTS module integrates with Textual's event system:

```python
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent

# Request TTS generation
app.post_message(TTSRequestEvent(
    text="Text to speak",
    message_id="msg_123",
    voice="alloy"  # Optional voice override
))

# Handle completion
@on(TTSCompleteEvent)
async def handle_tts_complete(self, event: TTSCompleteEvent):
    if event.error:
        self.notify(f"TTS failed: {event.error}")
    else:
        # Audio file available at event.audio_file
        play_audio_file(event.audio_file)
```

## Configuration Reference

### Global Settings
```toml
[app_tts]
default_provider = "openai"  # openai, kokoro, elevenlabs, chatterbox
default_voice = "alloy"      # Provider-specific voice
default_model = "tts-1"      # Provider-specific model
default_format = "mp3"       # Audio output format
default_speed = 1.0          # Speech speed (0.25-4.0)
```

### Exact legacy route allowlist

The compatibility generator accepts only these internal model IDs:

- `openai_official_tts-1` → `openai`
- `openai_official_tts-1-hd` → `openai`
- `openai_official_tts1` → `openai`
- `openai_official_tts1hd` → `openai`
- `elevenlabs_eleven_monolingual_v1` → `elevenlabs`
- `elevenlabs_eleven_multilingual_v1` → `elevenlabs`
- `elevenlabs_eleven_multilingual_v2` → `elevenlabs`
- `elevenlabs_eleven_turbo_v2` → `elevenlabs`
- `elevenlabs_eleven_turbo_v2_5` → `elevenlabs`
- `elevenlabs_eleven_flash_v2` → `elevenlabs`
- `elevenlabs_eleven_flash_v2_5` → `elevenlabs`
- `elevenlabs_english_v1` → `elevenlabs`
- `elevenlabs_elevenlabs` → `elevenlabs`
- `local_kokoro_default_onnx` → `kokoro`
- `local_kokoro_default_pytorch` → `kokoro`
- `local_chatterbox_default` → `chatterbox`
- `local_higgs_default` → `higgs`
- `local_higgs_v2` → `higgs`
- `alltalk_default` → `alltalk`
- `alltalk_alltalk` → `alltalk`

These IDs are temporary bridge inputs, not native provider/model identities.
New native-adapter code selects a canonical provider and opaque model ID
explicitly with `TTSRequest`.

### Audio Formats
Supported formats vary by provider:
- **All providers**: mp3, wav, pcm
- **OpenAI**: opus, aac, flac
- **ElevenLabs**: Various bitrate/quality options
- **Kokoro**: Best with wav/pcm for quality
- **Chatterbox**: All formats via audio service conversion

## Advanced Features

### Text Normalization
Configure text preprocessing:
```python
normalization_options = NormalizationOptions(
    normalize=True,
    unit_normalization=True,      # 10KB → 10 kilobytes
    url_normalization=True,       # https://example.com → example dot com
    email_normalization=True,     # user@example.com → user at example dot com
    phone_normalization=True,     # 555-1234 → 5 5 5 1 2 3 4
)
```

### Voice Mixing (Kokoro)
Combine multiple voice characteristics:
```python
# Future enhancement
voice = "af_bella:0.6,af_sarah:0.4"  # 60% bella, 40% sarah
```

### Voice Cloning (Chatterbox)
Clone any voice with a reference audio:
```python
# Use custom voice
voice = "custom:/path/to/reference.wav"

# Or save for reuse
await backend.save_reference_voice_with_metadata(
    name="my_voice",
    audio_path="/path/to/reference.wav",
    metadata={"speaker": "John Doe", "emotion": "neutral"}
)
```

### Advanced Text Processing (Chatterbox)
```python
# Automatic preprocessing handles:
# - "Dr. Smith" → "Doctor Smith"
# - "J.R.R. Tolkien" → "J R R Tolkien"
# - "See reference [1]" → "See reference"
# - URLs and email addresses are normalized
```

### Multi-Candidate Generation (Chatterbox)
```python
# Generate multiple candidates and select best
extra_params = {
    "num_candidates": 3,
    "validate_with_whisper": True,
    "temperature": 0.7
}
```

### Legacy Streaming with Chunk Processing

Concrete backend streaming is retained only inside the temporary bridge:

```python
async for chunk in backend.generate_speech_stream(request):
    # Process chunks in real-time
    await websocket.send(chunk)
    
    # Or accumulate for post-processing
    chunks.append(chunk)
```

## Performance Considerations

### Kokoro Performance
- **CPU**: ~3.5s latency for first token
- **GPU**: ~0.3s latency for first token
- **Generation speed**: 35-100x realtime
- **Token rate**: ~140 tokens/second

### Chatterbox Performance
- **CPU**: ~2.0s latency for first generation
- **GPU**: <200ms latency (ultra-low)
- **Generation speed**: Real-time to 50x realtime
- **Multi-candidate overhead**: ~1.5x per additional candidate
- **Whisper validation**: +0.5-1.0s per candidate
- **Voice cloning**: 7-20 second reference audio required
- **Model size**: ~500MB (0.5B parameters)

### Optimization Tips
1. **Use streaming** for better perceived performance
2. **Pre-download models** for Kokoro to avoid first-run delays
3. **Cache frequently used phrases** (future enhancement)
4. **Adjust chunk size** based on network conditions
5. **Use appropriate format**:
   - PCM for lowest latency
   - MP3 for compatibility
   - Opus for best compression

### Memory Usage
- Kokoro model: ~300MB when loaded
- Chatterbox model: ~500MB when loaded
- Audio buffers: Minimal with streaming
- Text processing: Negligible
- Voice library: ~10MB per saved voice

## Troubleshooting

### Common Issues

#### "TTS service not available"
- Check if TTS was initialized successfully
- Verify API keys are configured
- Check logs for initialization errors

#### "No audio output"
- Verify audio playback system is working
- Check file permissions for temp directory
- Ensure audio format is supported by system

#### "Kokoro model not found"
- Download model files to configured paths
- Check file permissions
- Verify ONNX runtime is installed

#### "Chatterbox voice cloning fails"
- Ensure reference audio is 7-20 seconds
- Check audio format (WAV recommended)
- Verify PyTorch/CUDA installation
- Check available GPU memory

#### "API key errors"
- Check key format and validity
- Verify key has required permissions
- Check API quotas/limits

#### "Multi-candidate generation slow"
- Reduce number of candidates
- Disable Whisper validation
- Use GPU acceleration
- Check system resources

### Debug Logging
Enable debug logging for detailed information:
```toml
[logging]
level = "DEBUG"
```

Check logs at: `~/.share/tldw_cli/logs/`

### Performance Issues
1. **Slow generation**:
   - Use streaming for better UX
   - Consider using faster models (tts-1 vs tts-1-hd)
   - Check network latency for API calls
   - For Chatterbox: reduce candidates, disable validation

2. **High memory usage**:
   - Unload models when not in use
   - Use streaming instead of full generation
   - Monitor with Stats tab
   - Clear voice library cache periodically

3. **Voice quality issues**:
   - Adjust exaggeration/CFG parameters
   - Try different reference audio
   - Enable text preprocessing
   - Use multi-candidate generation

## API Reference

### Schemas

#### OpenAISpeechRequest
```python
class OpenAISpeechRequest(BaseModel):
    model: str                    # Model identifier
    input: str                    # Text to synthesize
    voice: str                    # Voice selection
    response_format: str          # Audio format
    speed: float = 1.0           # Speed adjustment (0.25-4.0)
    stream: bool = True          # Enable streaming
    lang_code: Optional[str]     # Language hint
    normalization_options: Optional[NormalizationOptions]
    extra_params: Optional[Dict[str, Any]]  # Provider-specific parameters
```

### Native Adapter Methods

#### ensure_ready()
Initialize or connect to provider resources lazily. The service synthesis path
invokes this as its prerequisite.

#### get_catalog()
Own readiness and return provider health, models, formats, voices, and
supported controls. Callers do not pre-resolve a concrete adapter.

#### get_voices(model_id, refresh=False)
Own readiness and lazily return bounded voices for one model. A refresh bypasses
the adapter's current voice result without exposing the adapter to callers.

#### synthesize()
Return a provider-neutral `TTSAudioResponse` with an asynchronous byte stream.

#### close()
Release provider resources. The registry controls when adapter shutdown occurs.

## Future Enhancements

### Planned Features
1. **SSML Support**: Advanced speech markup
2. **Caching System**: Reduce repeated generations
3. **Batch Processing**: Multiple texts in one request
4. **Real-time Streaming**: WebSocket-based streaming
5. **More Backends**: Edge-TTS, Coqui, Piper
6. **Cross-provider Voice Transfer**: Use one provider's voice with another

### Experimental Features
- **Emotion Control**: Adjust emotional tone
- **Prosody Tuning**: Fine-tune speech characteristics
- **Multi-speaker**: Different voices in one text
- **Audio Effects**: Post-processing effects

## Contributing

### Adding a Native Adapter

1. Implement the asynchronous adapter contract (`ensure_ready`,
   `get_catalog`, `get_voices`, `synthesize`, and `close`) using the
   provider-neutral request, response, catalog, health, and progress types.
   `get_catalog()` and `get_voices()` own their readiness step;
   `ensure_ready()` remains the service synthesis prerequisite.
2. Add one explicit provider specification to application service
   construction.
3. Add configuration validation, contract tests, and provider documentation.

Do not register new providers in `TTS_Backends.py` or subclass
`TTSBackendBase`; those APIs exist only for the six-provider temporary bridge.

### Testing
Run TTS-specific tests:
```bash
pytest Tests/TTS/
```

## Security Considerations

1. **API Keys**: Never log or display API keys
2. **Input Validation**: All text inputs are sanitized
3. **File Paths**: Temporary files use secure generation
4. **Network**: External audio.cpp accepts an explicit HTTP or HTTPS origin;
   synthesis text is sent to that configured origin, HTTPS certificate
   verification remains enabled, and redirects are disabled
5. **Local Models**: Verify model file integrity
6. **Voice Cloning**: Be aware of ethical implications
   - Only clone voices with permission
   - Chatterbox adds watermarks to generated audio
   - Store voice metadata securely
7. **Reference Audio**: Validate file formats and sizes
8. **Clone Voice Bundles**: Ordinary export is sanitized. Explicit bundle
   import/export is POSIX-only until Windows ACL parity is verified, requires
   plaintext warnings, treats archives as hostile, and never describes SHA-256
   as authenticity, identity, signature, or consent proof.

## License

The TTS module follows the main project's AGPL-3.0+ license. Individual model licenses:
- Kokoro: Apache 2.0
- Chatterbox: MIT License
- API providers: Subject to their respective terms of service
