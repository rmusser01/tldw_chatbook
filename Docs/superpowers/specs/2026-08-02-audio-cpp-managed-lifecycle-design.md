# audio.cpp Managed Lifecycle — Product Requirements and Design

Status: Approved by the user on 2026-08-02 after independent specification review

Partially superseded: 2026-08-09 by the
[guided model setup design](2026-08-09-audio-cpp-guided-model-setup-design.md),
only for the additional generated-configuration source, auto-selected loopback
port, recipe-driven model setup, and Windows target. Its user-provided
`server.json` lifecycle remains normative.

Date: 2026-08-02

Target branch: `dev`
Supersedes: The deferred managed-mode details in the 2026-07-23 audio.cpp adapter design; the external adapter remains unchanged

## Document purpose

This document defines the separately deferred managed-lifecycle workstream for
audio.cpp. Chatbook already connects to an independently started
`audiocpp_server`, discovers multiple TTS models and voices, generates bounded
complete WAV responses through the native async-stream adapter contract, and
exposes global configuration plus operational Speech Lab workflows.

The remaining work lets a user explicitly select a prebuilt
`audiocpp_server` binary and an existing `server.json`, then lets Chatbook
lazily launch, supervise, test, restart, and shut down that one owned process.
It does not make Chatbook an audio.cpp installer, configuration generator,
model manager, or general-purpose process supervisor.

This specification is the design authority for the two managed-lifecycle
implementation slices. The existing external adapter, TTS registry, Settings
ownership, Studio preference, character-profile, and complete-WAV contracts
remain authoritative except where this document explicitly extends them.

## Current-state baseline

The following behavior is already implemented and must not be rebuilt:

- The application owns one sealed `TTSAdapterRegistry` and `TTSService`.
- `audio_cpp` is the first native adapter and has an exclusive-reconfiguration
  policy.
- External audio.cpp uses one canonical HTTP(S) origin and performs no process
  management.
- `/health`, `/v1/models`, optional per-model voices, and
  `/v1/audio/speech` are represented by bounded native contracts.
- Synthesis buffers and validates one complete PCM16 WAV, then yields it as one
  item through the asynchronous byte-stream response interface.
- Console, Roleplay, and Speech Lab use the native service path without an
  automatic fallback to another model or provider.
- Global Settings owns durable provider connection/runtime-initialization
  fields. Speech Lab owns checks, discovery, generation, playback, and runtime
  diagnostics.
- Saved, applied, runtime-observation, catalog, and artifact revisions are
  already modeled separately.
- Existing providers remain behind the temporary legacy bridge and are outside
  this workstream.

The baseline for this design was verified against current `origin/dev` at
commit `656a304a3132798f27da0c1d072c3a603cd17d79`. The existing `Tests/TTS`
baseline passed with 2,154 tests and 14 expected skips before this document was
written.

## Upstream compatibility target

Managed launch compatibility is pinned to audio.cpp release
[`release-0.5`](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5),
commit
[`3178daf4028fa8f48ef63299aa1524ee2d3a4bb7`](https://github.com/0xShug0/audio.cpp/tree/3178daf4028fa8f48ef63299aa1524ee2d3a4bb7),
published 2026-07-31.

That release documents:

- `audiocpp_server --config server.json`;
- top-level `host`, `port`, `backend`, `lazy_load`, and `models` fields;
- multiple model entries in one process;
- `GET /health` and `GET /v1/models`;
- complete-WAV `POST /v1/audio/speech`; and
- retention of lazily loaded models until the server exits.

The existing HTTP adapter fixtures retain their own pinned provenance. Managed
process tests do not silently update those fixtures. A real-binary UAT against
the supported release proves that the launch and existing HTTP contracts remain
compatible.

## Goals

- Let a first-time user select a prebuilt binary and existing `server.json`
  without editing raw application TOML.
- Preserve explicit External and Managed modes with no automatic fallback,
  process discovery, or process adoption.
- Launch the managed process only on a deliberate audio.cpp operation.
- Guarantee at most one owned audio.cpp child per running Chatbook process.
- Share one concurrent first-start attempt and prevent stale lifecycle tasks
  from mutating a replacement generation.
- Provide truthful health, capability, saved-versus-active, and process state.
- Preserve admitted synthesis during an explicit manual shutdown or restart.
- Bound application shutdown and remove every owned child and lifecycle task.
- Keep captured child output bounded, memory-only, and outside general logs.
- Preserve external audio.cpp behavior and the complete-WAV async-stream
  adapter interface.
- Deliver the work in two focused, sequential PR-sized Backlog tasks.

## Non-goals

- Downloading, installing, updating, verifying, or building audio.cpp
- Generating, rewriting, or offering an editor for `server.json`
- Downloading, configuring, moving, or deleting model and voice files
- Supporting multiple managed audio.cpp children
- Enforcing a system-wide singleton across independent Chatbook processes
- Discovering, adopting, sharing, or terminating another process's server
- Managing remote or non-loopback servers in Managed mode
- Automatically restarting after a crash or failed health check
- Creating a generic subprocess-supervision framework
- Adding process lifecycle controls to Global Settings
- Redesigning legacy TTS providers or removing the legacy bridge
- Adding incremental SSE/PCM playback; complete WAV remains authoritative
- Persisting child logs, catalogs, process identifiers, or runtime diagnostics

## Product decisions

### ML-DEC-001 — Explicit mode

Global Settings exposes exactly two modes:

- **External server** — connect to a server the user starts and owns.
- **Managed local server** — launch and own the configured local child.

There is no automatic fallback between them. Managed mode never adopts a
listener found at the configured endpoint.

### ML-DEC-002 — User-provided artifacts

The user provides both the executable and `server.json`. Chatbook may help find
`audiocpp_server` on `PATH`, but detection only fills an unsaved draft after a
user action. Detection never saves, trusts, starts, or updates the binary.

### ML-DEC-003 — Lazy first use

The following deliberate operations may launch a stopped managed server:

- **Start & Test Connection**;
- **Refresh Models**;
- **Generate** in Speech Lab;
- an actual Console TTS request; or
- an actual Roleplay TTS request.

Constructing services, saving Settings, opening Speech Lab, mounting widgets,
rendering status, and passive refresh do not launch.

### ML-DEC-004 — One application-owned process

One running Chatbook process owns at most one managed audio.cpp child. Another
Chatbook process is independent. A port collision between them fails closed;
there is no cross-process lock or ownership handoff.

### ML-DEC-005 — Manual recovery

An unexpected exit produces **Unavailable** and retains bounded diagnostics.
There is no immediate retry loop. A later deliberate operation may start one
replacement. A live unhealthy child is not killed; the user explicitly
chooses **Restart**.

### ML-DEC-006 — Complete WAV

Managed mode uses the existing async-stream adapter interface but returns the
same bounded, structurally validated complete WAV as External mode. Managed
lifecycle work does not expose upstream incremental audio streaming.

## Ownership and component architecture

### ML-ARCH-001 — Supervisor boundary

`AudioCppSupervisor` is narrowly scoped to one audio.cpp process. It owns:

- the exact child handle;
- startup and exit-monitor tasks;
- periodic health scheduling;
- stdout and stderr drains;
- the bounded diagnostic ring;
- the active immutable launch snapshot;
- the process generation; and
- shutdown/restart state.

It does not parse synthesis payloads, discover voices, validate WAVs, resolve
TTS preferences, edit configuration files, or manage any other provider.

The application/service bootstrap is the sole construction path. Screens,
widgets, event handlers, and adapters cannot independently create supervisors.
The object is lightweight and causes no launch or network work during
construction.

### ML-ARCH-002 — Adapter boundary

`AudioCppAdapter` remains responsible for the audio.cpp HTTP contract:

- health and catalog contract parsing;
- TTS-model filtering;
- voice discovery;
- request validation;
- complete-WAV synthesis and structural validation;
- existing safe provider errors; and
- response/client cleanup.

Managed mode injects the app-owned supervisor. Before an HTTP operation, the
adapter obtains a ready immutable endpoint/process-generation snapshot. The
HTTP client is bound to that generation and is never reused after restart,
port change, or unexpected exit. External mode bypasses process supervision.

### ML-ARCH-003 — Registry and service authority

The registry remains provider admission and lease authority. The managed
implementation must reuse its exclusive transition boundary rather than add a
second independent synthesis quota.

- Normal discovery/synthesis obtains the existing provider lease, then calls
  the adapter/supervisor.
- Restart, shutdown, and pending-configuration application obtain the
  provider's exclusive transition, reject new leases, and drain admitted
  leases before mutating the child.
- A lifecycle command never holds a normal provider lease while waiting for
  the exclusive transition.
- Supervisor locks are not held while waiting for provider leases or HTTP
  response consumption.

This ordering prevents registry/supervisor deadlocks and preserves the current
rule that a lease lasts through response consumption.

### ML-ARCH-004 — Desired, applied, and process generations

Three identities remain distinct:

- **Saved configuration generation** — latest durably persisted audio.cpp
  configuration.
- **Applied provider generation** — configuration currently used by the
  active adapter runtime.
- **Process generation** — monotonically increasing identifier for one child
  launch.

Global Settings may advance the saved generation without advancing the applied
generation when a managed child is live. That is a truthful staged result, not
an unavailable or failed reconfiguration.

The current settings-publication path must support this narrow staged outcome:

- the active managed adapter and child remain usable;
- the latest valid saved configuration replaces any older staged value;
- new operations against the live child record the applied provider
  generation as well as the saved settings generation;
- exact model/voice integrity may reject a saved selection only against a
  complete catalog observation for that same candidate configuration
  generation; a catalog from the active generation cannot reject a different
  staged generation, so the selection remains Unverified until explicit apply
  validates it, without fallback;
- **Restart & Apply Settings** drains the active generation, applies only the
  latest staged generation, and launches/tests the replacement;
- **Shut down server** drains and stops the child, then promotes the latest
  staged configuration without materializing or launching a replacement;
- after a crash, the next deliberate operation applies the latest saved
  configuration before starting a replacement; and
- application restart naturally uses the latest persisted configuration.

External-mode reconfiguration keeps its existing behavior when no managed
child is live. A switch away from a live managed child remains staged until the
user explicitly applies it in Speech Lab.

This requirement extends the existing saved-versus-applied revision contract;
it does not introduce a second configuration store.

## Persistent configuration

### ML-CFG-001 — Active-mode projection

The canonical section remains `[app_tts.audio_cpp]`. Missing `mode` continues
to mean `external`, preserving every existing installation.

An illustrative managed configuration is:

```toml
[app_tts.audio_cpp]
mode = "managed"
managed_binary_path = "/opt/homebrew/bin/audiocpp_server"
managed_server_json_path = "/path/to/server.json"
managed_startup_timeout_seconds = 30.0
managed_health_check_interval_seconds = 10.0
managed_termination_grace_seconds = 5.0

# Existing shared adapter limits remain authoritative.
connect_timeout_seconds = 5.0
synthesis_timeout_seconds = 600.0
max_input_characters = 10000
max_response_bytes = 134217728
max_metadata_bytes = 1048576
max_catalog_models = 1000
max_voices_per_model = 1000
max_identifier_characters = 256
```

`base_url` remains persisted as the External-mode value when Managed mode is
selected, allowing an explicit later switch back without discarding the user's
external origin.

Configuration projection validates only the selected mode's connection or
lifecycle fields plus common adapter safety limits. Dormant malformed managed
fields cannot break an External user, and a dormant malformed External origin
cannot prevent a valid Managed launch. Selecting the dormant mode makes its
fields active and subject to validation before save or use.

### ML-CFG-002 — Managed timing defaults and bounds

The initial managed values are:

| Field | Default | Accepted range | Meaning |
| --- | ---: | ---: | --- |
| Startup timeout | 30 s | 1–300 s | One monotonic deadline through process readiness and required contract validation |
| Health interval | 10 s | 2–300 s | Delay after one completed periodic probe before scheduling the next |
| Process termination grace | 5 s | 0.1–60 s | Time between terminate and force-kill after admitted work has drained |

Values must be finite real numbers and cannot be booleans. The process
termination grace is not an additional registry shutdown period. During
application shutdown, its effective value is capped by the remaining
application-owned TTS shutdown deadline.

Active synthesis remains governed by the existing synthesis timeout or owner
cancellation; the termination grace begins only after admitted work is no
longer using the generation.

### ML-CFG-003 — Backward compatibility

- Existing external mappings and defaults round-trip unchanged.
- Existing environment and raw/normalized precedence remains unchanged.
- No managed-mode environment override is added.
- Unknown provider IDs and unknown managed fields fail or are ignored only
  according to the existing safe configuration-projection boundary; they are
  never passed as command-line arguments.
- No migration rewrites existing files merely because the application starts.
- The first Settings save writes only the user's explicit mode and fields
  through the existing atomic configuration owner.

## Local validation

### ML-VAL-001 — Binary path

The active managed binary path must:

- be nonblank and absolute after user-home expansion;
- name an existing regular file or a symlink whose target is a regular file;
- be executable by the current user; and
- remain executable when revalidated immediately before spawn.

The selected path, rather than its resolved target, remains persisted so a
user-approved Homebrew symlink can advance normally. Spawn failure remains
authoritative because no validation removes filesystem time-of-check/time-of-
use races.

### ML-VAL-002 — `server.json`

The active configuration path must:

- be nonblank and absolute after user-home expansion;
- name an existing readable regular file;
- remain below a fixed safe read bound;
- decode as UTF-8 JSON;
- contain one top-level JSON object;
- contain no duplicate object keys at any depth;
- contain `host` exactly equal to `127.0.0.1`; and
- contain `port` as an integer from 1 through 65,535, excluding booleans.

Both `host` and `port` are required so Chatbook can derive the one endpoint it
will supervise. `localhost`, `::1`, wildcard binds, hostnames, other IPv4
addresses, and omitted values are rejected in Managed mode. Users who need a
different bind run the server themselves and select External mode.

Chatbook does not validate or reinterpret backend, device, thread, model,
voice, request-body, CORS, lazy-load, model-spec, or model-path fields. The
server remains authoritative for its evolving schema.

The child working directory is the directory containing `server.json`. The
Settings help text states that relative paths in the server file therefore
resolve from that directory. Chatbook never rewrites those paths or the file.

### ML-VAL-003 — Side-effect-free save

Settings may read and locally validate the selected binary and JSON file. It
does not open the configured TCP port, execute the binary, initialize models,
refresh a catalog, or synthesize audio.

The same validation runs again at launch. A file modified after save can still
fail safely at first use. `server.json` is not watched; a running child keeps
its launch snapshot until explicit restart.

## Launch contract

### ML-LAUNCH-001 — Port preflight

Before spawn, the supervisor performs a bounded advisory check of
`127.0.0.1:<port>`. Any listener causes a fail-closed **Port already in use**
result with guidance to change `server.json` or choose External mode.

The preflight does not claim exclusive reservation. The child's bind result,
early exit, and readiness behavior remain authoritative for races after the
check. Chatbook never probes a listener to decide whether to adopt it.

### ML-LAUNCH-002 — Child creation

Launch uses a direct argument vector and no shell:

```text
[managed_binary_path, "--config", managed_server_json_path]
```

The process receives:

- `stdin` disconnected/disabled;
- stdout and stderr pipes drained continuously;
- the `server.json` parent as its working directory; and
- a sanitized child environment.

No arbitrary arguments field is exposed. Chatbook does not append `--log`, a
backend override, bind flags, model flags, CORS flags, or values copied from
unknown configuration keys.

### ML-LAUNCH-003 — Environment isolation

The child environment starts empty. Chatbook copies only an explicit allowlist
of ordinary OS, executable-discovery, locale, temporary-directory,
dynamic-library, CPU-threading, and GPU/runtime variables needed by local
binaries. Tests cover representative CUDA, Metal, Vulkan, OpenMP, BLAS, and
platform variables.

As defense in depth, allowlisted names are still rejected when they collide
with the repository's provider-credential inventory or conservative key,
token, secret, password, credential, and authentication patterns. An
application variable that is not explicitly allowlisted never reaches the
child even when its name does not look sensitive.

Environment names and values are never included in command displays,
diagnostics, notifications, or general logs. Settings warns that selecting a
binary authorizes Chatbook to execute that file with the current user's local
permissions; Chatbook does not claim to sandbox or verify it.

### ML-LAUNCH-004 — Startup sequence

One startup generation performs:

1. load the latest eligible immutable managed configuration;
2. validate binary and JSON paths/content;
3. derive the exact loopback endpoint;
4. perform the advisory port preflight;
5. spawn the child;
6. start stdout/stderr drains and the exit monitor;
7. poll bounded `/health` readiness while the child remains alive;
8. validate the existing `/health` plus `/v1/models` adapter contract; and
9. publish Running plus TTS capability for that generation.

One monotonic startup deadline covers polling and contract validation. A
process that is alive but cannot satisfy readiness before the deadline fails
startup. Model inference and the first lazy model load are not part of startup.

Any error or cancellation before first Running rolls back the exact owned child
and joins every task started for that generation before returning. A compatible
server with zero usable TTS models reaches **Running · TTS not configured** and
is not rolled back.

## Lifecycle state model

### ML-STATE-001 — Process states

The managed process uses:

| State | Meaning |
| --- | --- |
| Stopped | No owned child after intentional shutdown or before first use |
| Starting | One shared startup generation is in progress |
| Running | The owned child passed startup readiness |
| Unhealthy | The child is alive but bounded periodic health probes are failing |
| Draining | Restart/shutdown accepted; no new provider work is admitted |
| Stopping | The exact child is receiving terminate/kill cleanup |
| Unavailable | Startup failed or a previously running child exited unexpectedly |

TTS capability is separately **Available**, **Not configured**, or **Unknown**.
The existing provider runtime vocabulary remains a separate projection:
Not checked, Checking, Ready, Stale, Unavailable, or Reconfiguring. Process
state is not overloaded into an unrelated local-dependency status.

### ML-STATE-002 — Serialized immutable snapshots

One supervisor lifecycle lock protects process state, generation, active launch
snapshot, health counters, expected-exit intent, and lifecycle task references.
Observers receive immutable status snapshots. They cannot mutate supervisor
fields or retain the child handle.

Expensive waits, HTTP response consumption, provider-lease draining, and UI
notification work occur outside that lock. State publication includes a
monotonic observation/version token so stale worker results cannot overwrite a
newer generation.

### ML-STATE-003 — Startup sharing and cancellation

Concurrent first-use callers share one retained startup task. Each caller
awaits it through cancellation shielding:

- cancelling one waiter does not cancel startup for another waiter;
- a completed launch remains valid if all initiating UI waiters detach;
- application shutdown cancels and joins the retained startup; and
- a newer process generation prevents every prior task from publishing state.

There is no automatic retry inside the task. Repeated user operations after a
completed failure may initiate a later generation one at a time.

### ML-STATE-004 — Health supervision

After Running:

- health probes use a dedicated generation-bound direct loopback client;
- proxy environment variables are ignored and redirects are rejected;
- probes do not overlap;
- one probe has a short deadline no greater than the configured connect timeout
  or health interval;
- two consecutive failures change Running to Unhealthy;
- one later success restores the process state to Running; and
- an exit-monitor result changes the state immediately without waiting for a
  health threshold.

Periodic health checks do not refresh the complete model/voice catalog. A
health failure makes provider/catalog evidence stale. After health recovery,
the next Test, Refresh, or synthesis operation revalidates the adapter contract
before relying on the catalog.

A request that encounters Unhealthy performs one immediate bounded probe. If
that probe fails, it returns an actionable managed-unhealthy error. It does not
kill or replace the live child.

### ML-STATE-005 — Exit monitoring

The exit monitor records only bounded structured facts such as generation,
expected/unexpected classification, and exit status. It then:

- invalidates the process generation;
- invalidates/closes the generation-bound HTTP clients;
- wakes startup/readiness waiters;
- causes affected operations to receive one normalized unavailable result; and
- prevents stale health/drain tasks from restoring Running.

An exit after supervisor-requested termination completes the requested
shutdown/restart transition and is not reported as a crash. A child that exits
and daemonizes is treated as unavailable; Chatbook never searches for or adopts
its descendant.

## Restart and shutdown

### ML-LIFE-001 — Explicit shutdown

**Shut down server** is a retained, asynchronous exclusive lifecycle command:

1. transition to Draining and reject new provider work;
2. wait for admitted audio.cpp operations to finish or be cancelled by their
   owners;
3. transition to Stopping;
4. request termination of the exact child;
5. force-kill that child if the process termination grace expires;
6. join exit, health, and output-drain tasks;
7. close generation-bound HTTP clients;
8. promote the latest staged configuration without materializing it; and
9. publish Stopped.

Cancelling the UI waiter does not abandon a shutdown already accepted by the
runtime. A later deliberate operation may lazily start again; manual shutdown
does not disable Managed mode.

Chatbook does not kill a process group. A future process-tree policy requires a
separate ownership decision and evidence that every descendant is owned.

### ML-LIFE-002 — Explicit restart

**Restart** uses the same drain and exact-child cleanup, applies only the latest
saved configuration generation, then performs one normal startup. Saving
again while restart is in progress cannot make an older configuration win. If
a newer save arrives after the restart snapshot was chosen, the completed
replacement truthfully remains **Restart required** for that newer value.

**Restart & Apply Settings** performs startup contract validation after launch.
It never retries a failed synthesis and never chooses another model/provider.

### ML-LIFE-003 — Unexpected exit recovery

An unexpected exit produces Unavailable and no child. A later deliberate
operation applies the latest saved configuration and may start one replacement.
There is no timer, backoff loop, crash loop, or hidden recovery initiated by
status rendering.

### ML-LIFE-004 — Application shutdown

Application shutdown reuses the existing `TTSService`/registry shutdown
boundary:

- seal new service and provider admission;
- cancel or close service-owned admitted responses under the existing bounded
  shutdown policy;
- stop only the exact owned managed child;
- cap terminate/kill grace by the remaining registry deadline;
- attempt every response, adapter, monitor, and drain cleanup even when another
  cleanup fails; and
- clear diagnostics and application bindings at the terminal boundary.

Shutdown is idempotent. `wait_closed()` cannot report terminal completion while
an owned child, startup task, health task, exit monitor, or output drain remains
owned by the service.

## Diagnostics and privacy

### ML-DIAG-001 — Bounded capture

stdout and stderr are drained incrementally so a child cannot block on a full
pipe. Capture is bounded independently of line endings:

- at most 200 recent display lines;
- at most 64 KiB of sanitized display text per process generation; and
- at most 4 KiB per displayed line before truncation.

Eviction records a visible truncation marker and dropped-line count. Invalid
UTF-8 is replacement-decoded. ANSI escapes, unsafe control characters, and
Textual/Rich markup controls are neutralized before display.

### ML-DIAG-002 — Best-effort sanitization

Child output is arbitrary and therefore always labeled **potentially
sensitive**. Best-effort sanitization redacts recognized credentials and
normalizes the current user's home prefix where practical, but the UI never
claims that arbitrary process text is safe.

Captured lines:

- remain memory-only;
- never enter normal application logs, metrics, config, artifacts, or crash
  persistence;
- are cleared when the next process generation starts and at application
  shutdown; and
- are not populated with environment values, `server.json` contents,
  synthesis text, or WAV bytes by Chatbook code.

General logs receive only stable lifecycle codes and bounded non-sensitive
summaries. Speech Lab warns that Restart clears current diagnostics.

## Global Settings UX

### ML-UX-001 — Information architecture

`Settings → Speech & TTS → Configure Provider: audio.cpp` remains the only
durable configuration editor. The audio.cpp form adds an explicit mode
selector and shows only the selected mode's primary fields.

Managed mode shows:

- a trust notice explaining that Chatbook will execute the selected file;
- binary path input plus picker;
- **Use detected `audiocpp_server`**;
- `server.json` path input plus picker;
- loopback-only and working-directory explanations; and
- startup, health interval, and termination grace under Advanced.

Existing common safety limits remain under Advanced. The form links to Speech
Lab for operational testing and process control.

### ML-UX-002 — Detection and local validation

**Use detected** calls the platform's executable lookup only after the user
presses it. If found, the exact path fills the current draft and is announced;
if not found, the existing draft remains untouched and the UI explains how to
use the picker. Detection neither saves nor launches.

Validation errors attach to the binary, JSON, host, port, or timing field with
safe corrective text. Raw JSON bodies and exception traces are not primary UI
errors.

### ML-UX-003 — Save versus active runtime

Saving remains local and side-effect-free. If a managed child uses an older
applied generation, the form reports:

- Saved successfully;
- active managed configuration remains unchanged; and
- **Open Speech Lab to restart and apply**.

If External mode is saved while managed is running, the UI states that the
switch is pending. It does not claim the external endpoint is active or hide
the owned process.

## Speech Lab UX

### ML-UX-010 — Runtime card

Speech Lab adds one audio.cpp runtime card without duplicating durable fields.
It shows, as applicable:

- saved mode and saved configuration generation;
- active mode and applied provider generation;
- process state and process generation;
- active loopback endpoint;
- TTS capability and catalog freshness;
- Restart-required or pending-mode relation; and
- a link back to global audio.cpp Settings.

Status is communicated with text and non-color cues. Full binary/config paths
are available only in an explicit details area; compact status uses safe
basenames or labels.

### ML-UX-011 — State-specific actions

| Situation | Primary action | Other managed actions |
| --- | --- | --- |
| Managed, stopped | Start & Test Connection | Refresh/Generate may also start lazily |
| Managed, running | Test Connection | Restart, Shut down server |
| Managed settings changed | Restart & Apply Settings | Shut down server |
| External saved, managed active | Apply Settings & Stop Managed Server | Shut down server |
| Managed unhealthy | Restart | Test probe, diagnostics, shutdown |
| Managed unavailable | Start & Test Connection | Diagnostics |
| External active, no owned child | Test Connection | No process controls |

Starting, Draining, and Stopping retain visible busy state and disable
incompatible actions. Buttons do not disappear while focused. Lifecycle work
runs asynchronously and completion/failure uses the application's existing
notification mechanism.

Playback **Stop** and process **Shut down server** remain visibly and verbally
distinct.

### ML-UX-012 — Test and generation semantics

**Start & Test Connection** and **Test Connection**:

1. start only when managed and stopped/unavailable;
2. check process health;
3. refresh/validate the TTS model catalog; and
4. report runtime and TTS-capability state.

They never synthesize hidden audio. **Generate** remains the operation that
produces the current playable WAV. Existing current-result playback,
save/export, and optional auto-play behavior remain unchanged.

### ML-UX-013 — Saved versus active presentation

A live managed child remains visible even after External mode is saved. Until
the user applies the switch, TTS operations use the clearly labeled active
managed generation.

If External is active and Managed is saved, no child needs preservation. The
next deliberate audio.cpp operation applies Managed mode and launches lazily.
If no child exists, a staged managed configuration is eligible for the next
operation without a separate restart.

### ML-UX-014 — Diagnostics interaction

Recent diagnostics are collapsed by default and show:

- potential-sensitivity warning;
- process generation;
- stdout/stderr origin;
- bounded sanitized lines;
- dropped-output marker/count; and
- notice that the next start clears the buffer.

Opening, closing, scrolling, or copying from the panel cannot change lifecycle
state. Captured lines are never automatically copied to the clipboard or a
persistent file.

### ML-UX-015 — Interaction quality

- Passive screen mount and status projection perform zero process launches.
- Focus remains stable during status refresh and responsive recomposition.
- Disabled controls expose a concise reason.
- Long operations never block the Textual event loop.
- Stale worker results are rejected by saved/applied/process generation.
- Narrow layouts preserve the status, primary action, and current-result player
  before secondary diagnostics.

## User journeys

### Journey 1 — First-time managed setup

1. User opens global Speech & TTS Settings and configures audio.cpp.
2. User selects **Managed local server**.
3. User presses **Use detected** or picks the prebuilt binary.
4. User picks an existing `server.json`.
5. Local validation explains any file, host, port, or permission problem.
6. User saves; no process starts.
7. User follows **Open Speech Lab**.
8. User presses **Start & Test Connection**.
9. Speech Lab reports Running and discovered TTS models.
10. User generates a character-roleplay response, receives a current WAV
    result, and plays it.

### Journey 2 — Lazy Console/Roleplay launch

1. Managed settings are valid and saved; no child exists.
2. The user requests speech for an actual response.
3. The provider lease admits one request and concurrent callers share startup.
4. The supervisor starts and validates one child.
5. The adapter performs the existing complete-WAV request.
6. The owned child remains running for reuse and periodic health supervision.

### Journey 3 — Save while running

1. A managed child is Running on applied generation A.
2. User saves valid desired generation B in Settings.
3. Save completes without stopping, restarting, or reconnecting.
4. Settings and Speech Lab show B saved, A active, Restart required.
5. Existing compatible requests may continue against A.
6. User chooses **Restart & Apply Settings**.
7. A drains and stops; only latest B becomes eligible; the replacement is
   started and tested.

### Journey 4 — Unexpected exit

1. Exit monitor observes that current process generation ended unexpectedly.
2. HTTP clients and health/catalog evidence for that generation become stale or
   unavailable.
3. Speech Lab reports Unavailable and retains bounded diagnostics.
4. No retry occurs in the background.
5. A later deliberate Test, Refresh, Generate, Console, or Roleplay operation
   applies the latest saved configuration. It starts one replacement only when
   the latest eligible saved mode is Managed; when External is latest, it
   applies External and starts no child.

### Journey 5 — Switch to External

1. A managed child is Running.
2. User saves External mode and origin.
3. The child continues and the UI shows **Switch to External pending**.
4. User chooses **Apply Settings & Stop Managed Server**.
5. Admitted managed work drains, the exact child stops, and External becomes
   applied without an automatic connection test.
6. A later explicit Test or synthesis uses only the external origin and never
   relaunches managed mode.

## Error and recovery contract

### ML-ERR-001 — Stable managed failures

Managed lifecycle adds or uses stable safe outcomes for:

- configuration invalid;
- binary missing or not executable;
- `server.json` unreadable or invalid;
- managed bind not loopback;
- port already in use;
- spawn failed;
- startup timed out;
- child exited before readiness;
- HTTP contract incompatible;
- no TTS models configured;
- live child unhealthy;
- child exited unexpectedly; and
- lifecycle transition in progress.

Core failures contain a stable code, safe message, retryability, local
non-sensitive operation ID, and optional UI-neutral recovery action such as
`open_settings`, `retry`, `restart_managed`, or `open_diagnostics`.

### ML-ERR-002 — No unsafe echo

Provider errors, notifications, and general logs do not echo:

- synthesis text;
- environment values;
- JSON contents;
- arbitrary child output;
- raw HTTP response bodies;
- full configured paths; or
- arbitrary upstream error strings.

The editable field may naturally display the value the user entered. That does
not authorize copying it into logs or unrelated status messages.

### ML-ERR-003 — Recovery policy

- Port occupied: change the server config or choose External mode.
- Invalid binary/JSON: return to Settings.
- Startup timeout/early exit: inspect diagnostics, then retry explicitly.
- No TTS models: correct `server.json` outside Chatbook and restart.
- Unhealthy live child: immediate Test probe or explicit Restart.
- Unexpected exit: inspect diagnostics or initiate one later lazy replacement.
- Contract mismatch: use a supported audio.cpp server; do not fall back.
- Draining/Stopping: return a retryable transition-in-progress result.

## Testing strategy

### ML-TEST-001 — Configuration and validation

Deterministic tests cover:

- absent mode and legacy mappings remain External;
- invalid dormant managed fields do not break External;
- invalid dormant external fields do not break Managed;
- managed timing defaults, finite bounds, and boolean rejection;
- absolute paths, regular files, executable permission, and symlinks;
- bounded UTF-8 JSON reading;
- duplicate keys at every nesting depth;
- exact `127.0.0.1` and valid integer ports;
- rejection of omitted host/port, booleans, wildcard, hostname, IPv6, and
  non-loopback values;
- stable working-directory projection; and
- side-effect-free save/mount behavior.

### ML-TEST-002 — Deterministic supervisor tests

Process launching, port preflight, HTTP probes, clocks, and task creation are
injectable. Controlled events/fake clocks—not timing sleeps—cover:

- single launch under concurrent first use;
- one waiter cancellation without shared-start cancellation;
- advisory preflight race followed by authoritative bind/exit failure;
- startup deadline and complete pre-Running rollback;
- expected versus unexpected exit;
- two-failure Unhealthy threshold and one-success recovery;
- non-overlapping bounded health probes;
- stale process generation suppression;
- no automatic restart;
- later lazy replacement;
- manual drain versus application cancellation;
- terminate then force-kill;
- outer shutdown deadline capping process grace;
- idempotent close; and
- zero retained child/task after terminal cleanup.

### ML-TEST-003 — Environment and diagnostics

Tests prove:

- known credential/secret variables are removed;
- a non-allowlisted application variable is omitted even when its name does not
  match a secret pattern;
- PATH, locale, temporary directory, dynamic-library, CPU, and representative
  GPU/runtime variables remain;
- environment contents never enter diagnostics or logs;
- stdout/stderr drains cannot grow without a newline;
- invalid UTF-8 and terminal/Rich controls are neutralized;
- per-line, byte, and line-count bounds hold;
- eviction count/marker is truthful;
- restart and app shutdown clear capture; and
- captured lines never reach general application logs.

### ML-TEST-004 — Registry, adapter, and revision integration

Tests cover:

- app-scoped sole supervisor construction;
- generation-bound HTTP clients and invalidation;
- provider lease duration through WAV consumption;
- staged save advances saved but not applied generation;
- latest-wins staged configuration;
- active compatible operations continue while restart is required;
- explicit apply drains before replacement;
- shutdown promotes staged config without launching;
- crash recovery applies latest saved config;
- a crash while a switch to External is pending applies External and starts no
  replacement child;
- a staged exact model/voice is not rejected by catalog evidence from a
  different applied configuration generation;
- switch to External stops the child and never relaunches managed;
- no overlap between old and new children;
- external adapter behavior remains unchanged; and
- managed synthesis still yields exactly one validated complete-WAV stream
  item.

### ML-TEST-005 — Real subprocess fixture

A small controlled test executable, not audio.cpp, exercises the actual
subprocess boundary on supported platforms:

- exact argument vector and working directory;
- environment projection;
- readiness;
- stdout/stderr drain behavior;
- early exit;
- terminate;
- unresponsive-child force-kill; and
- complete monitor/drain cleanup.

Platform-specific fixture setup may differ, but production semantics remain
exact-child ownership without a shell. A skipped platform requires an explicit
reason and equivalent injected-boundary coverage.

### ML-TEST-006 — Textual UX

Pilot tests verify:

- mode-specific Settings fields and Advanced controls;
- PATH detection fills only the draft;
- Settings save and panel mount launch nothing;
- field-specific validation and trust/loopback/working-directory copy;
- saved-versus-active and pending-mode presentation;
- state-specific actions and disabled reasons;
- Start/Test/Refresh/Generate triggers;
- no passive launch;
- stale worker-result rejection;
- diagnostic truncation and potential-sensitivity labels;
- stable focus and non-color status; and
- playback Stop remains distinct from server Shutdown.

Normal CI requires no audio.cpp binary, model, network service, or download.

## Real-binary UAT

The opt-in UAT uses user-provided paths. It never changes the user's binary,
original `server.json`, or model files.

1. Begin with no Chatbook-owned audio.cpp process.
2. Follow the first-time managed Settings flow, including detected or picked
   binary and picked server file.
3. Confirm Save starts no process.
4. Start/Test in Speech Lab and confirm exactly one owned PID.
5. Confirm the catalog exposes the user's configured multi-model TTS server.
6. Generate speech for a character-roleplay response.
7. Confirm a current WAV result appears and audibly plays.
8. Save a changed managed path/setting using only application Settings or a
   temporary copied configuration; confirm the current child continues and
   Restart required appears.
9. Restart/Apply and confirm a new process generation with no overlap.
10. Terminate the owned child externally; confirm Unavailable and no background
    retry.
11. Initiate one deliberate operation and confirm one lazy replacement.
12. Shut down explicitly; confirm no child; initiate another operation and
    confirm lazy start.
13. Save and explicitly apply External mode; confirm the managed child stops,
    External requests use only the configured origin, and Managed does not
    relaunch.
14. Re-enter Managed mode if needed, then exit Chatbook and confirm no owned
    process remains.

UAT evidence records application commit, audio.cpp binary/version evidence,
sanitized server capabilities, process-generation observations, generated WAV
metadata, and the user's audible-playback confirmation. It records no model
paths, synthesis text, environment values, or child output.

## Delivery decomposition seams

These are approved delivery seams, not pre-created Backlog tasks.

### Documentation prerequisite

The accepted amendments to ADR-023 and ADR-039 must land on `dev` before Slice
4 implementation planning or Backlog task implementation begins. The
amendments are design prerequisites, not implementation deliverables hidden
inside Slice 4.

### Slice 4 — Managed runtime core

One PR delivers:

- backward-compatible managed configuration projection;
- app-owned `AudioCppSupervisor`;
- strict JSON/loopback/path/environment validation;
- launch, health, generation, diagnostics, restart, and shutdown core;
- generation-bound adapter integration;
- saved-versus-applied staging semantics;
- stable managed errors;
- deterministic unit/integration coverage; and
- no user-visible Managed selector or lifecycle controls.

The dormant core is independently verifiable through APIs/tests and preserves
External behavior.

### Slice 5 — Managed Settings, Speech Lab, documentation, and UAT

One later PR delivers:

- explicit mode and managed fields in Global Settings;
- binary detection/pickers and local validation UX;
- Speech Lab runtime card, actions, pending-state behavior, and diagnostics;
- Textual accessibility/responsive coverage;
- user setup and recovery documentation; and
- the real-binary, audible character-roleplay UAT evidence.

Managed mode first becomes user-visible in this slice. Slice 5 depends only on
the already-created and completed Slice 4 task; task references must follow
Backlog's dependency-order rules.

## Rollout and rollback

- The default remains External and existing configs require no write.
- Slice 4 is dormant until Slice 5 exposes Managed selection.
- No data migration deletes or renames existing audio.cpp values.
- Disabling/removing Managed UI returns users to External selection without
  affecting the external adapter.
- If a staged runtime application fails, persisted settings remain saved and
  the active/Unavailable state remains truthful; there is no silent rollback
  or fallback.
- A failed managed launch leaves no owned child and does not affect legacy
  providers.
- Rollback stops any owned child through normal service cleanup before removing
  the managed adapter path.
- Stored managed fields may remain inert when older code reads the file.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Port preflight races | Treat child bind/exit as authoritative and roll back exact owned child |
| Duplicate launches | One retained startup task plus app-scoped supervisor and process generation |
| Settings save kills a live child | Stage saved generation; apply only through explicit lifecycle transition |
| Stale task corrupts replacement | Every task and HTTP client is generation-bound |
| Deadlock between leases and supervisor | Registry transition/lease order is explicit; supervisor locks never wait on provider leases |
| Child blocks on output pipe | Continuous bounded stdout/stderr drains |
| Child output leaks sensitive data | Memory-only bounded ring, best-effort sanitization, explicit warning, no general logging |
| Child receives application credentials | Sanitized environment with credential removal and runtime-variable tests |
| Relative model paths behave unpredictably | Stable `server.json` parent working directory and visible help text |
| Long inference delays manual shutdown | Draining remains visible; admitted request timeout/owner cancellation remains authoritative |
| App exit leaves a child | Outer TTS deadline caps grace; exact-child terminate/kill and joined tasks are release gates |
| Multiple Chatbook processes compete | No adoption; occupied port fails closed with External-mode guidance |
| Upstream schema evolves | Parse only strict host/port locally; delegate all other fields to audio.cpp; pin compatibility evidence |
| UI confuses playback and process lifecycle | Distinct labels, placement, status, and disabled reasons |

## Functional acceptance criteria

- [ ] ML-AC-001: Existing external audio.cpp configurations and requests behave
  unchanged, including complete-WAV output and no managed process side effects.
- [ ] ML-AC-002: A user can save a user-provided executable and `server.json`
  in explicit Managed mode without launching a process.
- [ ] ML-AC-003: Managed launch accepts only an executable file, strict bounded
  JSON with explicit `127.0.0.1` and valid port, and a sanitized environment.
- [ ] ML-AC-004: Start/Test, Refresh, Generate, Console, or Roleplay first use
  launches exactly one owned child; passive UI and Settings never do.
- [ ] ML-AC-005: Concurrent first use shares startup and caller cancellation
  cannot orphan or multiply children.
- [ ] ML-AC-006: Startup failure rolls back the exact child and joins all
  generation tasks while preserving bounded recent diagnostics.
- [ ] ML-AC-007: Periodic bounded health supervision detects failure and
  recovery without automatic restart or catalog overclaiming.
- [ ] ML-AC-008: Unexpected exit invalidates the generation and reports
  Unavailable; a later deliberate operation starts exactly one replacement
  only when the latest eligible saved mode is Managed, while saved External
  applies without launching a child.
- [ ] ML-AC-009: Saving while managed is live preserves the child, advances the
  saved generation, and exposes Restart required until explicit apply.
- [ ] ML-AC-010: Manual restart/shutdown drain admitted work, reject new work,
  and terminate/kill only the exact owned child.
- [ ] ML-AC-011: Application shutdown remains bounded and leaves no owned
  child, HTTP client, startup task, health task, monitor, or drain task.
- [ ] ML-AC-012: Diagnostics are bounded, best-effort sanitized, memory-only,
  visibly truncated when needed, and excluded from general logs.
- [ ] ML-AC-013: Speech Lab exposes truthful saved/active/process/capability
  states and state-specific lifecycle actions without duplicating Settings.
- [ ] ML-AC-014: Managed synthesis discovers multiple configured TTS models and
  yields one validated complete WAV through the existing native adapter.
- [ ] ML-AC-015: A first-time-user UAT produces and audibly plays a
  character-roleplay response, verifies crash recovery and shutdown, and does
  not alter user-provided artifacts.
- [ ] ML-AC-016: Applying External mode stops the owned managed child and
  subsequent requests cannot relaunch it unless Managed mode is explicitly
  selected again.
- [ ] ML-AC-017: Normal CI passes without audio.cpp, models, server downloads,
  or external network access.

## ADR check

ADR required: yes

ADR paths:

- `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`

Reason: ADR-023 already selected managed audio.cpp as a deferred provider
runtime boundary but its process states, configuration semantics, and
supervision details must be finalized. ADR-039 made the prior program
explicitly External-only; it must be amended to admit globally owned managed
initialization fields and Speech-Lab-owned operational controls while
preserving its four-owner model. No third ADR is needed.

Both accepted amendments must land before Slice 4 implementation planning
begins. Each Backlog task and implementation plan links this specification and
the applicable ADRs.

## References

- [audio.cpp repository](https://github.com/0xShug0/audio.cpp)
- [audio.cpp release 0.5](https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5)
- [Pinned release 0.5 server guide](https://github.com/0xShug0/audio.cpp/blob/3178daf4028fa8f48ef63299aa1524ee2d3a4bb7/app/server/README.md)
- [Existing audio.cpp adapter-registry design](2026-07-23-audio-cpp-tts-adapter-registry-design.md)
- [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [Speech & TTS Settings ownership design](2026-07-31-speech-tts-settings-ownership-design.md)
- [ADR-039](../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md)
- [Speech Lab current-result design](2026-08-02-speech-lab-current-result-ux-design.md)
- [ADR-040](../../../backlog/decisions/040-speech-lab-current-result-and-auto-play.md)
