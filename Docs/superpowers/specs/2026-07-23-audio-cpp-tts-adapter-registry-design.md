# audio.cpp TTS Adapter Registry — Design

**Status:** approved by the user on 2026-07-23; review amendments approved on 2026-07-23
**Date:** 2026-07-23
**Related tasks:** [TASK-402](<../../../backlog/tasks/task-402 - Establish-TTS-adapter-registry-authority-and-legacy-bridge.md>) and [TASK-551](<../../../backlog/tasks/task-551 - Add-external-audio.cpp-native-TTS-adapter.md>)
**Canonical ADR:** [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
**Upstream contract reviewed:** `0xShug0/audio.cpp` commit `d3d748179e5ace353386fbf17bcaedfacf482d75`

## Goal

Replace Chatbook's partial TTS backend registry with an app-scoped adapter
architecture inspired by `tldw_server`, while making audio.cpp the first native
adapter and preserving existing providers through a temporary compatibility
bridge.

The five-slice delivery proves the architecture through the STTS Playground. A
user will be able to connect to an existing `audiocpp_server` or let Chatbook
launch and supervise a user-provided binary with a user-provided `server.json`,
discover TTS models and voices, generate a complete WAV response, and play or
save it.

## Implementation status

- Slice 1 established the authoritative registry and placed the six existing
  providers behind the temporary legacy bridge.
- Slice 2, tracked by TASK-551, implements the external-only `audio_cpp` native
  adapter, bounded discovery, lazy voices, and complete-WAV synthesis. Its
  exact, alias-free, lazy, exclusive provider spec is registered before the six
  unchanged legacy entries.
- Slice 3 remains the catalog-driven external STTS Playground vertical.
- Slices 4–5 remain user-provided binary plus user-provided `server.json`
  launch/supervision and its managed Playground UI.

No process launching, supervision, binary handling, `server.json` ownership,
or STTS UI behavior is part of the Slice 2 implementation.

## Pre-implementation baseline

Before Slice 1, Chatbook had a `BackendRegistry`, but it was a class-global map
from wildcard-like internal model IDs to backend classes. Provider
configuration was assembled through backend-specific conditionals in
`TTSBackendManager`, while provider, model, voice, and format routing was
repeated across STTS events, legacy TTS events, media reading, audio-service
interop, audiobook generation, and the Playground.

`get_tts_service()` also retained the first configuration used to construct its
module singleton, making runtime configuration replacement unreliable. The UI
could not discover provider capabilities through the service and instead
contained static provider-specific model and voice lists.

Slice 1 made an app-scoped registry authoritative without rewriting all legacy
backend implementations in one milestone. The existing class-global registry
remains temporarily, but only as an implementation detail contained inside the
legacy bridge.

## Feature-level scope

### Included

- One active audio.cpp provider instance.
- External mode connecting to an existing server.
- Managed mode launching a user-provided `audiocpp_server` binary with a
  user-provided `server.json`.
- Lazy managed-server startup on first audio.cpp use.
- App-scoped registry and service ownership.
- Exact provider registration with no provider aliases in the first milestone.
- Native request, response, catalog, health, and error contracts.
- A temporary bridge for OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and
  AllTalk.
- STTS Playground configuration, discovery, generation, playback, save, and
  recovery flows.
- Complete WAV responses exposed through an asynchronous result interface.
- Deterministic unit, contract, process-lifecycle, and Textual coverage.
- Optional live smoke testing with user-supplied runtime paths.

### Excluded

- Downloading, updating, packaging, or building audio.cpp.
- Generating, modifying, or fully validating audio.cpp's `server.json`.
- More than one configured audio.cpp instance.
- Provider routing, load balancing, or failover across servers.
- True SSE or raw-PCM playback streaming.
- Client-side voice-reference uploads or arbitrary server-side voice paths.
- External-server authentication or custom request headers.
- Immediate native rewrites of all existing TTS providers.
- New audio.cpp selection controls in chat, media reading, or audiobook UI.
- Dynamic Python plugins, package entry points, or third-party adapter loading.

## Architecture

### Application ownership

The application constructs one `TTSService` and one `TTSAdapterRegistry` before
screens or event handlers can request TTS. Application shutdown closes that
service. Settings changes call an explicit `reconfigure_provider()` operation.

The transitional `get_tts_service()` accessor returns the application-bound
service. It does not construct an untracked singleton or silently retain the
first configuration it receives. Tests and standalone utilities instantiate
`TTSService` directly. Shutdown and test teardown explicitly clear the
compatibility binding.

The handler submits all recognized STTS destinations, including compatibility
pairs, to one atomic `save_settings_to_cli_config()` call. That writer
invalidates and force-reloads the effective-settings cache once. After it
succeeds, the handler derives candidate providers only from recognized
adapter-affecting event keys, reads the resulting effective configuration once
through cached `load_settings()`, and calls `reconfigure_provider()` once for
each candidate provider:

| Provider | Exact STTS event keys |
|---|---|
| `openai` | `openai_api_key` |
| `elevenlabs` | `elevenlabs_api_key`, `ELEVENLABS_DEFAULT_MODEL`, `ELEVENLABS_OUTPUT_FORMAT`, `ELEVENLABS_VOICE_STABILITY`, `ELEVENLABS_SIMILARITY_BOOST`, `ELEVENLABS_STYLE`, `ELEVENLABS_USE_SPEAKER_BOOST` |
| `kokoro` | `KOKORO_DEVICE_DEFAULT`, `KOKORO_USE_ONNX`, `KOKORO_ONNX_MODEL_PATH_DEFAULT`, `KOKORO_ONNX_VOICES_JSON_DEFAULT`, `KOKORO_MAX_TOKENS`, `KOKORO_ENABLE_VOICE_MIXING`, `KOKORO_TRACK_PERFORMANCE` |
| `higgs` | `HIGGS_MODEL_PATH`, `HIGGS_VOICE_SAMPLES_DIR`, `HIGGS_DEVICE`, `HIGGS_ENABLE_FLASH_ATTN`, `HIGGS_DTYPE`, `HIGGS_MAX_REFERENCE_DURATION`, `HIGGS_DEFAULT_LANGUAGE`, `HIGGS_ENABLE_VOICE_CLONING`, `HIGGS_ENABLE_MULTI_SPEAKER`, `HIGGS_SPEAKER_DELIMITER`, `HIGGS_TRACK_PERFORMANCE`, `HIGGS_MAX_NEW_TOKENS`, `HIGGS_TEMPERATURE`, `HIGGS_TOP_P`, `HIGGS_REPETITION_PENALTY` |

The candidate set is based on recognized keys in the successfully persisted
atomic batch, not on a comparison of submitted UI values. A failed batch
performs no effective-config read or provider reconfiguration. The registry
compares the normalized effective provider configuration with its stored
configuration: equality is an `UNCHANGED` no-op; inequality updates the slot
configuration. For an unmaterialized provider this updates its future lazy
factory input without creating or retiring an adapter. For a materialized
provider it retires only that provider's adapter. Every candidate is attempted
even if another reconfiguration fails; diagnostics identify failed provider
IDs without including configuration values or raw exception text.

`default_provider`, `default_voice`, `default_model`, `default_format`, and
`default_speed` are selection defaults and never add a provider to the
candidate set. Initial bootstrap and replacement both use the same
provider-scoped effective projection. That projection contains only legacy
manager inputs relevant to the selected provider and resolves environment
overrides before registry comparison, so an unrelated setting or a persisted
value shadowed by the environment remains an `UNCHANGED` no-op. The
compatibility accessor remains configuration-blind.

The service owns an instance-scoped, configurable concurrency semaphore. The
initial default remains four concurrent TTS operations for compatibility with
the existing service, while individual adapters or upstream runtimes may
enforce narrower limits.

### Registry

`TTSAdapterRegistry` is constructed from explicit built-in provider
specifications and sealed before application use. It provides:

- Exact canonical provider lookup.
- An optional explicit compatibility alias map, empty in the first milestone.
- Duplicate provider and alias rejection during construction.
- One lazily materialized adapter instance per provider.
- A per-provider materialization lock.
- Provider catalog lookup and refresh.
- Operation leases.
- Effective-configuration comparison and targeted invalidation.
- Ordered, idempotent shutdown.

The registry does not use prefix or wildcard matching. Runtime code cannot add
or replace provider definitions after construction.

Canonical provider IDs are `audio_cpp`, `openai`, `elevenlabs`, `kokoro`,
`chatterbox`, `higgs`, and `alltalk`. UI labels such as `audio.cpp` and
`Higgs Audio (Local)` are presentation only and are never parsed for identity.
Historical internal-model spellings belong exclusively to the legacy resolver,
so the initial provider alias map is empty.

When effective provider configuration changes, the registry increments that
provider's configuration revision and retires its current adapter. Providers
normally allow new operations to use a replacement while the retired adapter
finishes its active leases. Saving normalized configuration that is identical
to the current value reports no change and leaves a healthy adapter running.

audio.cpp uses an exclusive handoff instead of overlapping retirement. A
configuration change places the provider in a retryable `reconfiguring` state
and stops admitting new audio.cpp operations. Existing leases finish without
being cancelled; the old adapter closes before the new configuration becomes
active. The replacement remains lazy and is created only by a later use. Old
and new audio.cpp adapters never coexist; the future managed supervisor applies
the same rule to owned children.

### Native contracts

The service boundary uses the following provider-neutral contracts:

| Contract | Purpose |
| --- | --- |
| `TTSRequest` | Canonical provider ID, opaque model ID, text, optional voice, requested format, speed, and adapter-validated options. |
| `TTSAudioResponse` | Actual format, content type, optional sample rate, asynchronous byte iterator, provenance, and a generic async cleanup callback. |
| `TTSProgress` | Provider-neutral status, optional normalized completion fraction, and safe optional metrics delivered to an operation-scoped progress sink. |
| `TTSProviderCatalog` | Provider identity plus its cached model catalog and catalog revision. |
| `TTSModelInfo` | Opaque model ID, family, upstream mode, formats, voices, control support, and voice-omission policy. |
| `ProviderHealth` | Availability state, freshness, safe diagnostics, retryability, and UI-neutral recovery action. |

The response cleanup callback is attached by `TTSService`; the DTO does not
know about registry internals. The callback releases both adapter resources and
the registry operation lease. Consumers use the response as an async context
manager or call `aclose()`. The compatibility byte generator closes it in a
`finally` block, including cancellation and partial-consumption paths.

Adapter-specific options are validated by the selected adapter. Unknown options
fail closed instead of passing arbitrary values downstream.

### Adapter interface

Native adapters implement five asynchronous operations:

- `ensure_ready()` initializes, connects, or starts lazily.
- `get_catalog(refresh=False)` returns health and model metadata.
- `get_voices(model_id, refresh=False)` lazily returns voices for one model.
- `synthesize(request, progress_sink=None)` returns an adapter audio response.
- `close()` releases HTTP, runtime, or owned-process resources.

Application and UI code do not retrieve or call concrete adapters. They use
`TTSService` and registry catalog/voice operations. `get_catalog()` and
`get_voices()` own readiness because the adapter coordinates its single
readiness and discovery state. The service synthesis path continues to call
`ensure_ready()` as its prerequisite. The progress sink is optional and
operation-scoped; audio.cpp reports indeterminate work until the complete WAV
arrives rather than inventing percentage progress.

`ProgressSink` is an asynchronous
`Callable[[TTSProgress], Awaitable[None]]`. A sink must schedule its UI update
and return promptly. The service isolates sink exceptions, records only a safe
diagnostic, and continues synthesis; display failures never become provider
failures.

### Legacy bridge

Each of the six provider-specific `LegacyTTSAdapter` entries owns a
provider-scoped `LegacyBackendHost`, which in turn lazily owns one existing
`TTSBackendManager`. The entries retain the canonical provider identities
`openai`, `elevenlabs`, `kokoro`, `chatterbox`, `higgs`, and `alltalk`.
Provider-scoped hosts keep configuration replacement and cached backend
lifecycle isolated; changing one legacy provider does not rebuild or close
another provider's host.

The app-scoped registry is the only routing authority visible to new service,
application, and UI code. The existing class-global `BackendRegistry` remains
the only shared legacy state and remains reachable only from the provider
hosts; it is not converted into a second public registry. Tests reset its state
deterministically, no new provider may register with it, and it is deleted with
the bridge.

The bridge moves today's static model, voice, format, and control metadata out
of the Playground and into compatibility catalogs. Those catalogs are marked
as approximate until their provider receives a native adapter.

One centralized legacy resolver replaces repeated internal-model routing. It
accepts only enumerated legacy forms and maps them to an exact provider and
internal model ID; it does not reproduce wildcard prefix matching.

During migration, `TTSService` retains the existing
`generate_audio_stream(request: OpenAISpeechRequest, internal_model_id: str)`
call shape, with an optional progress sink. It resolves the enumerated legacy ID
and delegates through the registry, yielding the native response bytes and
closing the response in `finally`. Existing media reading, audio interop, TTS
events, STTS, and audiobook callers continue through this compatibility method
until their own migration tasks. New code uses canonical `TTSRequest` and
`synthesize()`.

The bridge never exposes `TTSBackendManager` or a concrete backend to UI code.
Each host has a per-internal-backend operation lock covering lazy construction,
`initialize()`, mutable progress-callback installation, complete stream
consumption, and callback clearing in `finally`. This prevents double
initialization and cross-request progress delivery. Different provider hosts
may still run concurrently.

Each provider host is closed exactly once with its owning bridge adapter.

The bridge may be removed only when:

1. Every retained legacy provider has a native adapter.
2. All callers supply explicit provider and model IDs.
3. No wildcard-style internal model IDs remain.
4. Compatibility tests prove the old accessor, internal-model resolver, and
   legacy `generate_audio_stream()` method are unused.

## audio.cpp configuration

### Landed external configuration (Slice 2)

The external adapter reads only the following nested `[app_tts.audio_cpp]`
configuration:

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

The origin is canonicalized and must be absolute HTTP or HTTPS. Credentials,
non-root paths, query strings, fragments, invalid ports, non-positive limits,
and non-finite or non-positive timeouts are rejected without echoing submitted
values. TLS verification remains enabled. Redirect following is disabled so
synthesis text cannot move to another origin. There are no environment
overrides, authentication/custom-header fields, binary paths, `server.json`
paths, or process-management fields in Slice 2.

The numeric defaults are safety guards rather than statements about upstream
model limits. The 10,000-character input limit and 128 MiB response limit are
configurable. Metadata JSON, catalog counts, voice counts, and identifier
lengths are bounded separately so discovery cannot allocate unbounded memory.
Longer content remains outside the external Playground slice and belongs in a
chunked audiobook workflow after that flow adopts the adapter service.

Because ordinary audio.cpp synthesis returns no response bytes until the WAV is
complete, there is no read-inactivity timer. The five-second connect deadline
covers connection establishment, while the 600-second overall synthesis
deadline covers the request through complete bounded response consumption. An
inactivity timer may be added only with true incremental streaming.

### Future managed configuration (Slices 4–5)

Managed mode will add a user-provided executable path, a user-provided
`server.json` path, startup/shutdown limits, and a bounded diagnostic ring.
That future supervisor will require an existing executable binary and readable
JSON configuration. Chatbook will read only `host` and `port`; audio.cpp will
remain the authority for every other setting and resolve its own relative model
paths. Chatbook will never edit the file.

For the pinned `audio_cpp_http_v1` contract, a managed configuration may omit
`host` and `port`, which Chatbook interprets using audio.cpp's defaults of
`127.0.0.1` and `8080`. An explicit host must be exactly `127.0.0.1`; the
pinned server accepts only an IPv4 literal and cannot bind `localhost` or
`::1`. Other IPv4 addresses, wildcard binds, hostnames, and IPv6 are rejected.
Users who intentionally expose audio.cpp over a network run it themselves and
use external mode.

## Future managed process supervisor (Slice 4)

`AudioCppSupervisor` is an optional dependency used only in managed mode. Its
states are Stopped, Starting, Ready, Unhealthy, and Stopping.

On first catalog access, Start & Test, or synthesis:

1. Validate the executable and configuration paths.
2. Read and validate the configured loopback host and port.
3. Perform an advisory port preflight.
4. Launch `[binary_path, "--config", server_config_path]` without a shell.
5. Start bounded, sanitized stdout and stderr drain tasks.
6. Start a background `process.wait()` monitor.
7. Poll readiness until `/health` succeeds, the child exits, or the startup
   deadline expires.
8. Validate the audio.cpp HTTP contract.

The child process's bind result is authoritative; the preflight does not claim
to eliminate port races. Early exit and safe recent logs are reported as a
managed-process error.

Any failure or cancellation before the first Ready state—including a startup
deadline, invalid required endpoint, or incompatible health or model
response—rolls back the owned launch. The supervisor requests termination,
force-stops the exact child after the shutdown deadline if necessary, and joins
the monitor and log-drain tasks before returning the failure. The recent
in-memory diagnostic ring remains available until the next start or application
shutdown. A structurally compatible server with zero TTS models reaches
`not_configured` and is not treated as a failed launch.

The supervisor never adopts a process already listening on the target port. It
never restarts during a failed synthesis. An unexpected child exit marks the
provider unavailable and wakes readiness waiters. A later operation may start
a replacement. After the server has reached Ready once, a live but unhealthy
child is not killed automatically; the Playground offers an ownership-aware
Restart & Rediscover action after active operations drain.

Shutdown first asks the exact child to terminate, waits for the configured
deadline, and then force-stops only that child if required. Process-group
killing is excluded unless a future audio.cpp process tree can be proven to be
owned.

## Landed external HTTP compatibility and discovery (Slice 2)

The supported structural contract is named `audio_cpp_http_v1`. Fixture
provenance is audio.cpp commit
[`d3d748179e5ace353386fbf17bcaedfacf482d75`](https://github.com/0xShug0/audio.cpp/tree/d3d748179e5ace353386fbf17bcaedfacf482d75),
reviewed on 2026-07-23.

Compatibility requires:

- `GET /health` with the expected status, backend, and model-count shape.
- `GET /v1/models` with a list of entries containing `id`, `family`, `task`,
  and `mode`.
- `POST /v1/audio/speech` for complete WAV synthesis.

`GET /v1/audio/voices?model=<id>` is optional metadata. A missing or failed
voices endpoint does not make the server incompatible.

Slice 2 readiness validates the two required GET surfaces without synthesizing
hidden probe audio. Future Test Connection and Start & Test actions use the
same structural validation. The speech portion of the contract is validated by
the first real generation or a future explicitly enabled live smoke test. A
missing speech endpoint is then reported as an incompatible contract rather
than a generic generation failure.

Initial discovery loads health and models, normalizes `task`, and retains only
TTS entries. A healthy compatible server with zero TTS models is
`not_configured`, with guidance to add a `task: "tts"` model. It is not
classified as incompatible.

Voice discovery is lazy per selected model and cached by provider configuration
revision, catalog revision, and model ID. Every successful authoritative model
refresh increments the catalog revision and invalidates prior voice results,
even when the model list is unchanged. Returned identifiers are used as opaque
values and bounded, stripped of controls, and markup-escaped for display.

The server does not fully describe model controls or whether an unnamed default
voice exists. audio.cpp models therefore use an `unknown` voice-omission policy.
The UI offers a local "Server default" sentinel, translated to `voice=None`,
plus discovered IDs. The sentinel is never sent as a literal voice value.

The provider catalog records upstream `mode`, but first-milestone client
capabilities remain:

- WAV only.
- Complete-response synthesis only.
- Default speed `1.0` only.
- No arbitrary request options.

Upstream streaming mode is metadata, not a promise of client streaming.

Catalog data is cached in memory. Connectivity loss or an incompatible contract
marks provider health unavailable and the prior catalog stale; a future managed
child exit has the same result. Invalid voices, invalid requests, busy
responses, generation errors, and cancellation do not mark the catalog stale.

## Landed external synthesis (Slice 2)

Before sending a request, the adapter validates:

- Non-empty text within the configured character bound.
- Model membership in the latest TTS catalog.
- Voice length and absence of control characters.
- WAV output.
- Speed exactly `1.0`.
- Absence of unknown adapter options.

If a model is missing, discovery refreshes once before returning an invalid
model error.

The initial audio.cpp payload contains only:

- `model`
- `input`
- `response_format: "wav"`
- Optional `voice`

Speech POST requests are never automatically retried. Health, model, and voice
GET requests may receive one bounded transient retry. Redirects are disabled
for every request. Requests send `Accept-Encoding: identity`, and any response
with a non-identity `Content-Encoding` is rejected before parsing.

Health, model, voice, and error JSON bodies are limited by
`max_metadata_bytes` before decoding. Model and voice arrays are limited by
their configured counts, and identifiers plus required model metadata fields
are limited by `max_identifier_characters`. An oversized or malformed required
health/model response is an incompatible contract. Oversized or malformed
optional voice metadata makes voices unavailable without making the provider
incompatible.

The HTTP body is read incrementally at the network layer. `Content-Length`, when
present, is checked before reading; accumulated bytes are capped throughout.
`audio/wav` and generic binary content types are accepted only when the bounded
body parses as uncompressed 16-bit PCM WAV. Validation covers the RIFF/WAVE
container, `fmt` and `data` chunks, declared sizes, positive channel and sample
rate values, and complete declared frame data. A signature alone is
insufficient. The parsed bytes are authoritative.

A successful result is exposed as one asynchronous WAV chunk with actual
format, content type, safe scalar provenance/sample metadata, and any safely
parsed bounded timing values. This preserves the asynchronous response boundary
without pretending Slice 2 provides incremental synthesis.

Cancellation closes Chatbook's HTTP operation but may not cancel native
inference already running on the server. Cancellation does not restart the
server or mark it unhealthy.

## Future STTS Playground (Slices 3 and 5)

### Provider and settings UI

Opening STTS lists registry provider descriptors without initializing every
provider. Only the selected provider is resolved. Selecting audio.cpp is a
first use and may connect or lazily launch managed mode.

The audio.cpp settings panel contains:

- External or Managed mode.
- External base URL.
- Managed binary path.
- Managed `server.json` path.
- Advanced timeout, input, audio-response, metadata, catalog, identifier, and
  log bounds.
- Test Connection and Refresh Models for external mode.
- Start & Test and Restart & Rediscover for managed mode.

Save performs local validation and effective-configuration comparison. It does
not connect or start a process. If configuration changed, the old adapter is
retired safely. An active audio.cpp provider becomes `reconfiguring`; Generate,
Test, Refresh, Start, and Restart report a retryable state until existing leases
finish and the old adapter closes. A managed old child stops during that
handoff. Save never launches the replacement or kills an in-flight operation.

### Catalog-driven controls

Provider and model option values are canonical IDs; labels are never parsed to
recover identity.

For audio.cpp:

1. Readiness populates TTS models.
2. Model selection lazily loads voices.
3. Voice options include Server default and discovered IDs.
4. Server default is initially selected because audio.cpp's configured default
   is not identified by the voices endpoint.
5. When no voices are discovered, Server default is the only option.
6. Format is reset to WAV and disabled.
7. Speed is reset to `1.0` and disabled with an explanation.
8. Generate is enabled only when readiness, text, and model validation pass.

A single catalog-application function updates all shared controls. Switching to
a legacy provider restores that provider's speed, format, model, and voice
state, preventing sticky audio.cpp restrictions.

If refreshed metadata removes a selection, the UI picks a valid fallback and
announces the change.

### Workers and result state

Catalog discovery, voice discovery, synthesis, and playback use independent
worker groups. A refresh cannot cancel generation, and repeated Generate input
cannot replace an active generation worker.

Progress reaches the Playground only through the service-level progress sink.
Legacy adapters may provide real progress; audio.cpp remains indeterminate until
its complete response is validated.

Catalog and voice results carry the selected IDs and configuration revision.
They also carry the catalog revision used for voice discovery. Stale results are
discarded. Generation captures a request snapshot.

The generated result retains provider, model, voice, actual format, source text
snapshot, and local operation ID. Switching providers does not relabel it.
Playback and Save use response metadata, including the `.wav` extension, rather
than current selector state.

If a refresh fails, prior catalog data may remain visible and marked stale, but
Generate is disabled until readiness returns. Existing generated audio remains
playable and saveable.

## Errors and recovery

Core failures contain a stable code, safe message, retryability, a local
non-sensitive operation ID, and an optional UI-neutral recovery action such as
`open_settings`, `retry`, or `restart_managed`.

The landed external adapter uses these stable operation codes:

- `configuration_invalid`
- `connection_unavailable`
- `contract_incompatible`
- `not_configured`
- `request_invalid`
- `model_invalid`
- `server_busy`
- `generation_failed`
- `audio_response_invalid`
- `generation_timeout`

Cancellation propagates as cancellation, closes the HTTP operation, and is not
rewritten or logged as a provider failure. Exclusive handoff reports the
registry's retryable reconfiguring state. Future managed slices add
ownership-aware process recovery without changing the external codes.

Endpoint interpretation is deliberate:

- Missing voices endpoint means voices are unavailable.
- Missing speech endpoint means the server is incompatible.
- HTTP `503` with `server_busy` is retryable and does not affect health.
- The pinned server reports most other speech failures as HTTP `500` with
  `server_error`. The adapter does not classify these by matching error text.
  It refreshes models once without retrying the POST; if the requested model
  vanished, the result is invalid-model, otherwise it is generation-failed.
- Cancellation is a normal terminal state and is not logged as a provider
  failure.
- Malformed health or models data is an incompatible contract.
- Connection loss makes the external provider unavailable; a future managed
  child exit has the same result.

There is no automatic fallback to a legacy provider or another audio.cpp model.
Existing providers remain independently selectable and continue through their
bridge entries.

Chatbook-generated logs never include synthesis text, settings values, API
keys, raw error bodies, full server configuration, process environments, URL
credentials, or unsanitized remote strings. Settings-save logs record setting
names and outcomes only.

In external mode, submitted text crosses the privacy boundary to the configured
origin. Slice 2 logs neither that text nor the origin, config values, raw
response bodies, or invalid identifiers.

Managed child stdout and stderr are opaque and potentially sensitive because a
user-provided binary may print arbitrary content. They are never copied into
the general application log or persisted. The supervisor retains only the
bounded in-memory ring for explicit diagnostics, neutralizes ANSI, controls,
and Textual/Rich markup before display, and clears the ring on the next child
start or application shutdown.

## Lifecycle and shutdown

An operation lease lasts through response consumption, not merely until
`synthesize()` returns. Configuration invalidation therefore cannot close an
HTTP client while a caller is reading its result.

Application shutdown uses the registry's configured shutdown timeout as its
single drain deadline:

1. `TTSService.close()` seals service admission and wakes semaphore waiters.
2. It immediately begins `TTSAdapterRegistry.close()`, which seals registry
   lease admission and drains existing leases until that deadline.
3. At the deadline the registry begins closing active and retired adapters even
   if leases remain; there is no second drain period.
4. When the bounded registry close returns, the service concurrently invokes
   idempotent `aclose()` on every still-tracked service-wrapped response.
   Response cleanup may overlap adapter cleanup after the deadline.
5. Every response-close attempt runs even if another fails. Each wrapper
   independently releases its service-owned lease and concurrency slot after
   the registry deadline, even when its underlying response close raises or
   its provider finalizer ignores cancellation.
6. `TTSService.wait_closed()` joins those service-owned releases and the
   registry's definitive adapter cleanup, which closes external HTTP clients,
   terminates only owned managed children, and joins supervisor monitor and
   log-drain tasks. It reports response-close failures already available at
   that terminal boundary, but does not add a second timeout or wait
   indefinitely for a still-pending provider finalizer. A later finalizer
   result is observed without exposing raw provider detail.
7. Application teardown clears the compatibility service binding.

A “service-wrapped response” means every `_ManagedAudioResponse` returned by
`TTSService.synthesize()`, across native audio.cpp, legacy, and external
providers; it does not mean only a managed-mode audio.cpp response. The service
tracks each wrapper until its idempotent `aclose()` completes, including after
`wait_closed()` has returned from the terminal service-owned cleanup boundary.

A service-level close signal races concurrency admission. If shutdown wins,
the semaphore acquire task is cancelled; if both finish concurrently, the
newly acquired slot is released before the caller receives
`TTSRegistryClosedError`. `wait_closed()` cannot complete while a synthesis
waiter remains blocked on the service semaphore.

Shutdown is idempotent and never waits indefinitely for native inference.

## Testing

### Registry and bridge tests

- Exact registration, alias collision, and sealing.
- The initial provider alias map is empty and canonical IDs are never derived
  from display labels.
- Per-provider single materialization under concurrent first use.
- Operation leases, retirement, identical-config no-op, and shutdown.
- audio.cpp exclusive reconfiguration blocks new operations, preserves active
  leases, and never overlaps old and new adapters or managed children.
- Response cleanup after success, partial consumption, cancellation, and
  consumer failure.
- Provider-scoped legacy-host configuration isolation and exactly-once close.
- Enumerated routing for every legacy internal-model form used by current call
  sites.
- The retained legacy generation signature resolves through the bridge and
  always closes native responses.
- Per-backend legacy operation locks prevent double initialization and serialize
  mutable progress callbacks through complete stream consumption.
- Progress-sink exceptions do not fail synthesis.
- Application binding, explicit reset, and configuration replacement without
  first-config retention.
- Saving one provider's STTS settings retires only that materialized adapter
  and leaves unrelated materialized adapters running.
- An abandoned returned response plus a semaphore-blocked synthesis request
  converges during shutdown: the response closes, the waiter fails closed, and
  `wait_closed()` leaves no blocked synthesis admission.
- Instance-scoped concurrency across independent event loops.

### audio.cpp contract tests

- Health and model structural validation.
- TTS-only filtering and zero-TTS-model recovery.
- Lazy voice discovery, Server default as the initial selection, missing voice
  endpoint, catalog-revision invalidation, and stale voice-result rejection.
- Correct payload, omitted server-default voice, unsupported speed, and
  unknown-option rejection.
- Model refresh before a locally invalid model request and after an upstream
  `server_error`, without retrying the speech POST.
- Bounded metadata and audio reading, catalog-count and identifier limits,
  identity content encoding, early `Content-Length` rejection, redirects
  disabled, full PCM WAV validation, generic MIME acceptance, and malformed
  response rejection.
- Quiet complete-response synthesis may run until the overall deadline without
  a read-inactivity failure.
- Busy, invalid request, unavailable, incompatible, generation, timeout, and
  cancellation mapping.
- Catalog freshness changes only for health-affecting failures.
- GET retry and no POST retry.

Contract fixtures record audio.cpp commit
`d3d748179e5ace353386fbf17bcaedfacf482d75` and the review date.

### Future supervisor tests (Slice 4)

Process launching, HTTP transport, clocks, and retry timing are injectable.
Unit tests use fakes rather than sleeps or real ports and cover:

- Concurrent startup.
- Advisory preflight races.
- Early exit and safe log capture.
- Readiness and shutdown deadlines, including complete pre-Ready rollback.
- Contract failure and startup cancellation stop the exact owned child and join
  monitor and log-drain tasks.
- A post-Ready unhealthy child remains available for explicit restart.
- Unexpected exit monitoring.
- Restart on later use.
- Refusal to adopt an existing listener.
- Omitted host and port defaults, exact `127.0.0.1` acceptance, and rejection of
  `localhost`, `::1`, wildcard, hostname, and other-address binds.
- Owned-child-only termination.

A separate small subprocess integration test verifies actual launch, readiness,
and termination behavior without audio models.

### Future Textual tests (Slices 3 and 5)

- Save does not launch.
- Selecting audio.cpp and Start & Test trigger lazy readiness.
- External and managed actions differ.
- Stale provider, model, voice, and configuration-revision results are ignored.
- Server default remains the initial selection when discovered voices exist.
- An explicitly selected Server-default sentinel becomes `None`.
- Reconfiguring disables audio.cpp actions without interrupting active audio.
- Progress arrives through the service sink without direct backend access.
- Progress display failures do not fail generation.
- Format and speed restrictions restore on provider changes.
- Discovered metadata and logs render safely.
- Settings-save diagnostics never contain values or API keys.
- Generated result provenance and filename remain accurate after selection
  changes.
- Stale catalogs remain visible while generation is disabled.
- Recovery actions map from core action codes without core Textual imports.

### Future optional live smoke test (Slice 4)

The live test is explicitly enabled with environment-provided binary and
`server.json` paths. It performs port preflight, starts and stops only its own
process, discovers at least one TTS model, synthesizes a short WAV, validates
the result, and never downloads binaries or models. It is not part of normal
CI.

## Documentation and licensing

Slice 2 documentation covers the external text boundary, Server default
omission, WAV-only complete responses, configuration recovery, and safe
failure behavior. Later user documentation will add the loopback-only managed
boundary and the potentially sensitive nature of explicitly displayed managed
child logs.

audio.cpp is Apache-2.0 licensed. This milestone links to and interoperates with
a user-provided binary; it does not redistribute audio.cpp. Automatic download,
bundling, or packaging requires a separate attribution and distribution review.

## Rollout and rollback

The registry is the TTS service boundary, with legacy providers available
through bridge entries throughout the milestone. audio.cpp is registered
lazily even when no server is listening at the configured origin; its state
does not affect other providers.

If audio.cpp is unavailable or disabled by incomplete configuration, users
continue selecting existing providers. An audio.cpp request never silently
falls back.

This feature-level design is delivered as five ordered, single-PR slices rather
than one omnibus implementation task:

1. Establish registry authority and contain existing providers in the legacy
   bridge, including the compatibility generation and progress paths.
2. Add the external audio.cpp contract and native adapter, proving discovery
   and complete WAV synthesis through the service boundary.
3. Make the STTS Playground catalog-driven and complete the external audio.cpp
   end-to-end flow.
4. Add the managed audio.cpp supervisor for lazy launch, monitoring, restart,
   and owned-child shutdown.
5. Add managed audio.cpp settings and actions to the STTS Playground, completing
   the managed end-to-end flow.

Each slice receives its own atomic Backlog task and implementation plan in
dependency order. Before planning a slice, the project must create or select
that task, move it to In Progress, and link it to this specification and
ADR-023. Later slices are not folded into the active task's acceptance
criteria.

## Feature-level success criteria

- The STTS Playground can configure and use one external audio.cpp server.
- The Playground can lazily launch, monitor, restart, and stop one managed
  user-provided audio.cpp server.
- Only TTS models are offered, voices load lazily, Server default is initially
  selected, and that sentinel is omitted from the request.
- Successful generation produces a validated WAV that can be played and saved.
- Existing providers retain their visible Playground behavior through the
  legacy bridge.
- Runtime configuration replacement does not require application restart and
  never closes an in-flight response or overlaps audio.cpp instances.
- Failure and recovery states are specific, safe, and ownership-aware.
- Normal CI requires neither audio.cpp nor model downloads.

## Alternatives considered

| Option | Reason rejected |
| --- | --- |
| Extend the existing wildcard `BackendRegistry` | Leaves configuration, discovery, and UI routing scattered and preserves first-config and wildcard behavior. |
| Port `tldw_server`'s complete TTS registry | Carries server-scale provider, resource, retry, and factory machinery that the single-user TUI does not need. |
| Convert every legacy backend immediately | Greatly expands regression scope before the native contract has been proven. |
| Keep old and new registries in parallel | Creates two routing authorities and prolongs duplicated provider logic. |
| Invoke `audiocpp_cli` per request | Gives poor model reuse and latency and complicates cancellation and output handling. |
| Automatically download or build audio.cpp | Adds platform, trust, checksum, compiler, GPU backend, update, and licensing scope unrelated to proving the adapter. |
| Generate `server.json` from Chatbook | Duplicates an evolving upstream configuration schema and makes Chatbook responsible for model provisioning. |
| Require true streaming in the first milestone | Adds SSE/PCM buffering, sample-rate metadata, partial-response failures, and playback complexity before the registry is proven. |
| Support several audio.cpp instances | Turns the first adapter into provider-instance routing and failover work. |

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`

Reason: the design changes the provider contract, runtime and process ownership
boundary, configuration lifecycle, cross-module TTS interface, and long-lived
migration structure.

## References

- [audio.cpp repository](https://github.com/0xShug0/audio.cpp)
- [Pinned audio.cpp server guide](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/README.md)
- [Pinned audio.cpp server runtime](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/runtime.cpp)
- [Pinned audio.cpp busy guard](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/app/server/busy_guard.h)
- [Pinned audio.cpp license](https://github.com/0xShug0/audio.cpp/blob/d3d748179e5ace353386fbf17bcaedfacf482d75/LICENSE)
- [Historical Higgs backend architecture](../../Development/TTS/Higgs-ADR-001-Backend-Architecture.md)
- [Historical Higgs backend registration](../../Development/TTS/Higgs-ADR-002-Backend-Registration.md)
