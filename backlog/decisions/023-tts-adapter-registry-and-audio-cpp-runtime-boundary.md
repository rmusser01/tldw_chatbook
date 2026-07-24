# ADR-023: Adopt an app-scoped TTS adapter registry with audio.cpp as the first native adapter

Status: Accepted
Date: 2026-07-23
Related Task: N/A — create or select a Backlog task before implementation planning
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

Managed mode is loopback-only. Chatbook will not download or build audio.cpp,
generate or modify its configuration, adopt an existing process, expose
arbitrary server-side voice paths, or provide true client streaming in the
first milestone.

The STTS Playground is the first user-facing vertical slice. It will discover
TTS models and voices, request complete WAV output, and use registry metadata to
control model, voice, format, and speed UI.

Existing providers remain separate registry entries backed by provider-specific
views over one shared legacy host. The compatibility bridge is temporary and
may be removed only after every retained provider has a native adapter, every
caller supplies explicit provider/model IDs, wildcard internal IDs are absent,
and compatibility tests prove the legacy accessor is unused.

Delivery is split into four ordered, atomic implementation slices: registry
authority and legacy containment, the external audio.cpp adapter, the managed
supervisor, and catalog-driven STTS integration. Each slice receives its own
single-PR Backlog task and plan rather than being combined into an omnibus
task.

This ADR supersedes the registration direction in the non-canonical historical
Higgs backend-registration document. That material remains historical context
but no longer governs new TTS provider integration.

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
- Registration at the app boundary is explicit and sealed. Legacy wildcard
  matching is quarantined inside `LegacyBackendHost`, reset deterministically
  in tests, closed to new providers, and removed with the bridge.
- Request routing uses canonical provider IDs and opaque model IDs.
- Response lifetime extends through async byte consumption, allowing registry
  retirement without closing in-flight resources.
- Settings updates can replace one provider without restarting the application
  or disturbing unrelated providers.
- One shared legacy host preserves current implementations while exposing
  provider-specific compatibility entries.
- The Playground becomes catalog-driven, while legacy catalogs remain marked as
  approximate until migrated.
- audio.cpp managed mode launches only a user-provided executable and
  configuration on a loopback bind.
- Managed process ownership is explicit: Chatbook stops only children it
  started and never silently adopts an existing listener.
- The first audio.cpp contract supports complete WAV output and default speed
  only. Upstream streaming metadata does not imply client streaming support.
- Complete-response synthesis uses a connection deadline and an overall
  synthesis deadline, but no read-inactivity deadline that could abort quiet
  native inference before the WAV response begins.
- Default safety bounds are 10,000 input characters and 128 MiB of response
  data; both remain configurable.
- When voice discovery returns IDs, the first discovered voice is the initial
  UI default. Server default remains explicit and is selected automatically
  only when discovery returns no voices.
- Readiness probes health and model discovery without generating hidden audio;
  speech-endpoint compatibility is established by the first user-requested
  generation or the opt-in live smoke test.
- External mode sends synthesis text to the configured server and communicates
  that privacy boundary in the UI.
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
- During implementation rollout, retain the legacy host and accessor until the
  bridge deletion criteria are met.
- If the new Playground routing must be reverted, restore its legacy provider
  selection path while leaving the native registry code unselected; no data or
  schema migration is involved.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md)
- [audio.cpp server guide](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md)
- [audio.cpp server runtime](https://github.com/0xShug0/audio.cpp/blob/main/app/server/runtime.cpp)
- [audio.cpp license](https://github.com/0xShug0/audio.cpp/blob/main/LICENSE)
- [Historical Higgs backend architecture](../../Docs/Development/TTS/Higgs-ADR-001-Backend-Architecture.md)
- [Historical Higgs backend registration](../../Docs/Development/TTS/Higgs-ADR-002-Backend-Registration.md)
