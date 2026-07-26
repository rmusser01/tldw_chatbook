# ADR-023: Adopt an app-scoped TTS adapter registry with audio.cpp as the first native adapter

Status: Accepted
Date: 2026-07-23
Related Tasks: TASK-561, TASK-560, TASK-569, TASK-710
Supersedes: N/A

## Decision

Chatbook will make an app-scoped, sealed adapter registry the authoritative TTS
service boundary, using exact provider identities, provider-neutral
request/response/catalog contracts, lazy adapter instances, operation leases,
targeted configuration invalidation, and bounded shutdown. The existing
class-global, wildcard-style registry remains temporarily only inside the
legacy bridge and is removed with that bridge.

audio.cpp will be the first native adapter and will support one active instance
in either of two modes:

- Connect to an existing `audiocpp_server`.
- Lazily launch and supervise a user-provided binary with a user-provided
  `server.json`.

Managed mode is loopback-only and, for the pinned contract, accepts only
audio.cpp's default or explicit `127.0.0.1` IPv4 bind. Chatbook will not
download or build audio.cpp, generate or modify its configuration, adopt an
existing process, expose arbitrary server-side voice paths, or provide true
client streaming in the first milestone.

The STTS Playground is the first user-facing vertical slice. It will discover
TTS models and voices, request complete WAV output, and use registry metadata to
control model, voice, format, and speed UI.

Existing providers remain separate registry entries backed by provider-specific
hosts around the existing manager. The compatibility bridge is temporary and
may be removed only after every retained provider has a native adapter, every
caller supplies explicit provider/model IDs, wildcard internal IDs are absent,
and compatibility tests prove the legacy accessor, internal-model resolver, and
generation method are unused.

Delivery is split into five ordered, atomic implementation slices: registry
authority and legacy containment, the external audio.cpp adapter, the external
STTS vertical, the managed supervisor, and managed STTS integration. Each slice
receives its own single-PR Backlog task and plan rather than being combined into
an omnibus task.

This ADR supersedes the registration direction in the non-canonical historical
Higgs backend-registration document. That material remains historical context
but no longer governs new TTS provider integration.

## Implementation status

Slices 1–3 are implemented. The sealed registry registers the exact native
provider `audio_cpp` first, with no provider alias and exclusive lazy
reconfiguration, followed by the six unchanged legacy bridge entries.

TASK-560 implements external connection mode only. It reads
`[app_tts.audio_cpp]` with `mode = "external"`, a canonical HTTP(S) origin,
five-second connect and 600-second overall synthesis timeouts, and positive
bounds for input characters, response bytes, metadata bytes, catalog models,
voices per model, and identifier characters. It has no environment override,
authentication, path, binary, `server.json`, or process-management field.
Invalid configuration fails locally with a safe, value-independent
`ValueError` before a provider operation; the external adapter does not emit
the reserved provider-neutral `configuration_invalid` code.

The external native adapter has five operations: `ensure_ready()`,
`get_catalog(refresh=False)`, `get_voices(model_id, refresh=False)`,
`synthesize(...)`, and `close()`. Catalog and voice discovery own their
readiness; `ensure_ready()` remains the service synthesis prerequisite.
Callers use only service/registry APIs and never retrieve the concrete adapter.

The `audio_cpp_http_v1` fixtures and parsers are pinned to upstream commit
[`d3d748179e5ace353386fbf17bcaedfacf482d75`](https://github.com/0xShug0/audio.cpp/tree/d3d748179e5ace353386fbf17bcaedfacf482d75).
They require `GET /health`, `GET /v1/models`, and complete-WAV
`POST /v1/audio/speech`; per-model
`GET /v1/audio/voices?model=<id>` remains optional.

TASK-569 implements the catalog-driven external STTS Playground vertical.
Descriptor discovery does not materialize adapters; only a selected provider is
resolved. Independent catalog and voice workers carry configuration, catalog,
provider, and model revisions so stale results are discarded. audio.cpp
generation captures an immutable provider-neutral request and uses native
`TTSService.synthesize()`, while the six existing providers retain the
temporary compatibility generation path.

TASK-710 extends the accepted service boundary to Console defaults. Global
provider, model mode/value, voice mode/value, format, and speed are published
as one immutable snapshot. Request selection and revision-matched lease
acquisition share one admission gate with settings publication. audio.cpp
Console speech uses the native service; the six retained providers remain
inside `LegacyTTSAdapter`.

The supported audio.cpp modes are exact model or `first_available`, and exact
voice or `server_default`. Missing mode keys plus blank legacy audio.cpp values
read as the dynamic modes without a startup write. A settings save persists
authoritative mode keys in one atomic canonical/legacy mutation: exact values
are dual-written, while dynamic modes remove stale exact aliases. Publication
runs off the Textual event loop, permits a bounded foreground pending result,
does not cancel an admitted response, and allows only the latest pending
generation to complete the exclusive non-overlapping adapter handoff.

The Playground maps Server default to an omitted voice, locks audio.cpp to WAV
and speed `1.0`, and restores each legacy provider's prior controls when
switching away. A successful complete-WAV artifact retains immutable provider,
model, optional voice, source-text, operation, actual-format, content-type, and
safe response provenance for playback and export. Safe failures expose bounded
recovery actions; stale discovery disables new generation without invalidating
an existing artifact; audio.cpp never automatically falls back.

Slices 4–5 remain user-provided prebuilt binary plus user-provided
`server.json` launch/supervision and managed UI. Slices 1–3 do not launch,
monitor, restart, or stop audio.cpp and expose no managed process settings or
actions.

## Context

Chatbook's existing `BackendRegistry` maps wildcard-like internal model IDs to
backend classes through class-global state. `TTSBackendManager` owns
provider-specific configuration branches, while several event handlers,
services, generators, and UI widgets independently translate provider and model
names. The STTS Playground hard-codes provider models, voices, formats, and
controls.

The module-level service accessor also retains the first configuration used to
initialize it. Runtime settings changes can therefore fail to replace backend
configuration reliably.

audio.cpp provides a native, reusable audio runtime and HTTP server with health,
model discovery, voice discovery, and OpenAI-shaped speech endpoints. It is a
good forcing function for defining a real adapter boundary because one server
may expose multiple TTS model families and may be either independently managed
or launched as a local sidecar.

The repository requires a canonical ADR because this work changes provider and
runtime boundaries, process ownership, configuration lifecycle, security
policy, and a cross-module interface.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add audio.cpp to the existing wildcard registry | Fast, but preserves scattered routing, conditional configuration, class-global state, and hard-coded UI capabilities. |
| Port the full `tldw_server` TTS registry | Imports server-scale factory, retry, resource, and provider machinery not required by a single-user TUI. |
| Rewrite all current providers as native adapters immediately | Creates a large regression surface before the shared contract and migration seam have been proven. |
| Maintain separate old and new registries | Establishes two routing authorities and makes eventual migration harder. |
| Invoke `audiocpp_cli` for each request | Loses long-lived model/session reuse and produces worse latency and lifecycle behavior. |
| Download or build audio.cpp automatically | Adds platform detection, compiler/toolchain, GPU backend, checksum, update, trust, and redistribution responsibilities. |
| Generate audio.cpp server configuration | Duplicates an evolving upstream schema and makes Chatbook responsible for model provisioning. |
| Require true SSE/PCM streaming initially | Adds buffering, partial-failure, sample-rate, cancellation, and playback concerns before the adapter architecture is established. |
| Support multiple audio.cpp instances | Expands the first adapter into provider-instance routing, load balancing, and failover. |

## Consequences

- `TTSService` and `TTSAdapterRegistry` become application-owned lifecycle
  objects.
- Global TTS preferences are published together as one immutable snapshot.
  Under one application-owned shared admission gate, each request freezes its
  complete selection and acquires a provider lease carrying the same
  configuration revision.
- Settings publication holds the exclusive side of that admission gate.
  Requests observe either the old coherent preference-and-lease pair, the new
  coherent pair, or a structured reconfiguring/unavailable state; they never
  combine selection from one configuration revision with a lease from another.
- Registration at the app boundary is explicit and sealed. Legacy wildcard
  matching is quarantined inside `LegacyBackendHost`, reset deterministically
  in tests, closed to new providers, and removed with the bridge.
- Request routing uses canonical provider IDs and opaque model IDs. The native
  ID is `audio_cpp`; `audio.cpp` is a display label. The initial provider alias
  map is empty.
- Response lifetime extends through async byte consumption, allowing registry
  retirement without closing in-flight resources.
- Settings updates can replace one provider without restarting the application
  or disturbing unrelated providers. STTS settings saves reload the effective
  configuration once after all writes succeed, map recognized
  adapter-affecting event keys to exact provider IDs, and invoke targeted
  reconfiguration. An unmaterialized provider receives updated lazy factory
  input without being constructed; an unchanged effective configuration is a
  no-op; and the compatibility accessor never consumes replacement
  configuration.
- Service shutdown seals admission before waiting on adapter cleanup, wakes
  concurrency waiters, uses the registry timeout as its single drain deadline,
  initiates close on every abandoned service-wrapped response after the bounded
  drain, and independently releases service-owned leases and concurrency slots.
  Definitive service shutdown leaves no admission waiter blocked and does not
  wait indefinitely for a provider finalizer that ignores cancellation.
- audio.cpp reconfiguration is an exclusive handoff. The foreground settings
  wait is finite, and an admitted request keeps its old provider lease until it
  completes; that lease is never silently cancelled. Configuration pending
  handoff is inert, superseded pending generations cannot become active, and
  only the latest pending generation is eligible for activation. New
  operations are blocked while active leases drain, the old adapter closes
  before a replacement adapter can be created, and the replacement remains
  lazy. Future managed mode applies the same rule to an owned child.
- Provider-scoped legacy hosts preserve current implementations while isolating
  configuration replacement and backend caches. The quarantined class registry
  is their only shared legacy state.
- Per-internal-backend operation locks prevent double initialization and safely
  serialize mutable legacy progress callbacks.
- Existing callers retain their generation signature through the bridge.
  Provider-neutral, operation-scoped progress prevents UI access to concrete
  backends; progress-sink failures never fail synthesis.
- Slice 3 makes the Playground catalog-driven, while legacy catalogs remain
  marked as approximate until migrated. Descriptor reads remain non-
  materializing; independent catalog, voice, generation, and playback
  ownership prevents one operation from cancelling another.
- Future managed mode (Slices 4–5) launches only a user-provided executable and
  configuration using the pinned server's default or explicit `127.0.0.1`
  bind; `localhost` and `::1` are not accepted by that server version.
- Future managed process ownership is explicit: Chatbook stops only children it
  started and never silently adopts an existing listener.
- A future managed failure before first readiness rolls back the owned child
  and joins its monitor and log drains. A live child that becomes unhealthy
  after reaching Ready remains available for explicit restart.
- The first audio.cpp contract supports complete WAV output and default speed
  only. Upstream streaming metadata does not imply client streaming support.
- The complete response is fully buffered, bounded, and structurally validated
  as uncompressed PCM16 WAV before it is exposed as one asynchronous response
  chunk. This preserves the async-stream interface without claiming
  incremental streaming.
- `connect_timeout_seconds` configures HTTP connection establishment and also
  provides one overall deadline around required health-plus-models discovery,
  including an eligible safe-GET retry, plus one independent overall deadline
  for each optional voice-discovery operation.
- Complete-response synthesis uses that HTTP connection timeout and its own
  overall synthesis deadline, but no read-inactivity deadline that could abort
  quiet native inference before the WAV response begins.
- Default safety bounds are 10,000 input characters and 128 MiB of response
  data; both remain configurable.
- Server default is the initial voice selection because audio.cpp's configured
  default is not identified by the voices endpoint. Discovered voices remain
  explicit alternatives; omitting `voice` is the only server-default request
  representation.
- Slice 3 retains complete-WAV results as immutable artifacts with the request
  and actual response provenance required for playback and export, independent
  of later selector changes.
- Readiness probes health and model discovery without generating hidden audio;
  speech-endpoint compatibility is established by the first user-requested
  generation or the future opt-in live smoke test.
- External mode sends synthesis text to the configured server. Slice 2
  documentation states that privacy boundary, and Slice 3 communicates it in
  the UI. HTTP redirects are disabled.
- Metadata bodies, model and voice counts, and identifiers are bounded.
  Requests require identity content encoding so decompression cannot bypass
  response limits.
- Redirects are disabled, TLS verification remains enabled for HTTPS, safe GET
  operations may receive one bounded retry, and speech POST is never retried.
- Bounded responses receive structural uncompressed 16-bit PCM WAV validation;
  a RIFF/WAVE signature alone is insufficient.
- The adapter trusts the pinned structured `server_busy` response but does not
  parse free-form `server_error` text. After a speech `500`, it refreshes model
  discovery once to distinguish a vanished model from a generation failure and
  never retries the POST.
- External failures use stable safe operation codes. Connectivity and required
  contract failures mark cached health stale; invalid requests, optional voice
  failures, busy responses, generation failures, invalid audio, and
  cancellation do not. Cancellation propagates normally, and an audio.cpp
  request never falls back to another model or provider.
- Successful authoritative catalog refreshes invalidate voice caches through a
  new catalog revision, even when the model list is unchanged.
- Chatbook logs setting names and outcomes, never values or API keys. Managed
  child output is treated as potentially sensitive, retained only in a bounded
  in-memory diagnostic ring, and never copied into general logs or persisted.
- External synthesis sends submitted text to the configured origin. Logs and
  operation errors do not expose that text, configured origins or values, raw
  response bodies, or rejected identifiers. Response metadata is limited to
  safe immutable scalar provenance, sample, and bounded timing values.
- Normal CI uses fakes and contract fixtures. audio.cpp and model downloads are
  not test dependencies.
- Fixture provenance is pinned to audio.cpp commit
  `d3d748179e5ace353386fbf17bcaedfacf482d75`, reviewed on 2026-07-23.
- audio.cpp remains user-supplied and is not redistributed. Any future bundling
  requires a separate Apache-2.0 attribution and packaging review.
- Each ordered implementation slice requires its own atomic Backlog task,
  linked before that slice's implementation planning begins.

## Rollback plan

- Leave audio.cpp unconfigured or remove its active configuration; legacy
  provider entries continue through the compatibility bridge.
- Do not silently fall back during an audio.cpp request. Users explicitly
  select another provider after a reported failure.
- During implementation rollout, retain the provider-scoped legacy hosts and
  accessor until the bridge deletion criteria are met.
- If the future Playground routing must be reverted, restore its legacy
  provider selection path while leaving the native registry code unselected;
  no data or schema migration is involved.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md)
- [TASK-560](<../tasks/task-560 - Add-external-audio.cpp-native-TTS-adapter.md>)
- [TASK-569](<../tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md>)
- [TASK-710](<../tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md>)
- [Pinned audio.cpp server guide](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/README.md)
- [Pinned audio.cpp server runtime](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/runtime.cpp)
- [Pinned audio.cpp busy guard](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/busy_guard.h)
- [Pinned audio.cpp license](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/LICENSE)
- [Historical Higgs backend architecture](../../Docs/Development/TTS/Higgs-ADR-001-Backend-Architecture.md)
- [Historical Higgs backend registration](../../Docs/Development/TTS/Higgs-ADR-002-Backend-Registration.md)
