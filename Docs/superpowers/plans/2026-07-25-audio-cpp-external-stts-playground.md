# External audio.cpp STTS Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing STTS Playground catalog-driven and complete the end-to-end flow for one externally managed `audiocpp_server`, from configuration and discovery through validated complete-WAV generation, playback, and save.

**Architecture:** Extend the app-owned `TTSService` with read-only provider descriptor and configuration-revision access, then place a small pure catalog-selection layer between provider metadata and Textual controls. The Playground resolves only the selected provider, uses independent catalog and voice workers with revision tokens, and sends immutable native requests to `audio_cpp`; the six existing providers retain their compatibility generation path. Generated audio is represented by an immutable provenance-bearing artifact so playback and export never depend on the current selectors.

**Tech Stack:** Python 3.11+, Textual workers/widgets, frozen dataclasses, existing `TTSService`/`TTSAdapterRegistry`, Loguru, pytest/pytest-asyncio, Ruff, mypy.

---

## Global constraints

- Implement only delivery Slice 3 from
  `Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md`.
- Do not add binary paths, `server.json` paths, subprocess use, launch,
  supervision, restart, managed logs, process ownership, or managed-mode UI.
- `audio_cpp` remains the exact canonical ID. Labels are display-only and are
  never parsed to recover identity.
- Opening STTS may inspect descriptors but must not materialize every adapter.
  Only selecting or explicitly testing a provider may resolve it.
- The `audio_cpp` native path uses `TTSService.synthesize(TTSRequest)`. Legacy
  providers continue through `generate_audio_stream()` and the private bridge
  metadata it owns.
- The local Server-default sentinel is never sent to the adapter. It becomes
  `voice=None`.
- audio.cpp remains WAV-only, complete-response-only, and speed `1.0` only.
- Catalog, voice, generation, and playback workers have independent groups.
  Repeated generation cannot replace active generation.
- Provider, configuration, catalog, and model revision tokens reject stale
  asynchronous results.
- Generated audio retains its own provider/model/voice/text/format/operation
  metadata. Later selector changes cannot relabel it.
- UI and application logs never include submitted text, configured origins or
  values, credentials, raw remote bodies, or unescaped remote identifiers.
- No new runtime dependency is added.

## File map

- Create `tldw_chatbook/TTS/playground_types.py`: immutable request snapshot and
  generated-artifact contracts shared by the widget and event handler.
- Create `tldw_chatbook/UI/stts_playground_catalog.py`: pure catalog selection,
  Server-default, control restrictions, and stale-result token helpers.
- Modify `tldw_chatbook/TTS/TTS_Generation.py`: read-only provider descriptor and
  configuration-revision service methods.
- Modify `tldw_chatbook/TTS/__init__.py`: export the provider-neutral Playground
  contracts needed outside the TTS package.
- Modify `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`: nested
  audio.cpp settings persistence/reconfiguration, immutable request handling,
  native complete-WAV generation, safe errors, and artifact delivery.
- Modify `tldw_chatbook/UI/STTS_Window.py`: catalog-driven Playground controls,
  external audio.cpp settings/actions, worker groups, safe status rendering,
  and artifact-based playback/export.
- Modify `Tests/TTS/test_tts_registry_service.py`: descriptor/revision service
  seam coverage.
- Create `Tests/TTS/test_stts_playground_types.py`: immutable request and
  generated-artifact contract coverage.
- Create `Tests/UI/test_stts_playground_catalog.py`: pure catalog-state tests.
- Create `Tests/UI/test_stts_playground_audio_cpp.py`: Textual external
  discovery/control/stale-result tests.
- Create `Tests/TTS/test_stts_audio_cpp_generation.py`: event-handler native
  generation, provenance, errors, and cancellation tests.
- Modify `Tests/UI/test_stts_settings_widget.py`: external settings validation
  and explicit action tests.
- Modify `Tests/TTS/test_stts_settings_reconfiguration.py`: nested persistence
  and exclusive audio.cpp reconfiguration tests.
- Modify `Tests/TTS/test_stts_export_security.py`: generated-artifact export and
  extension/provenance tests.
- Modify `Docs/Development/TTS/TTS_MODULE_GUIDE.md`: landed Slice 3 service/UI
  boundary and worker ownership.
- Modify `Docs/Features/Speech-Services-Guide.md`: user-facing external server
  setup, privacy boundary, discovery, generation, playback, and save.
- Modify
  `Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md`
  and
  `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`:
  record Slice 3 as implemented without changing the managed-mode decision.
- Modify
  `backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md`:
  plan, evidence, acceptance completion, and implementation notes.

### Task 1: Add provider-neutral Playground contracts and service metadata access

**Files:**

- Create: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Create: `Tests/TTS/test_stts_playground_types.py`

- [ ] **Step 1: Write failing service metadata tests**

Add tests proving:

```python
descriptors = service.provider_descriptors()

assert tuple(item.provider_id for item in descriptors) == (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
assert factories.total_calls == 0
assert service.configuration_revision("audio_cpp") == 1
```

Also prove the descriptor tuple is immutable, canonical, ordered, and does not
acquire or materialize an adapter.

- [ ] **Step 2: Write failing immutable request/artifact tests**

Define the expected contracts in tests:

```python
snapshot = STTSPlaygroundRequest(
    operation_id="local-op",
    provider_id="audio_cpp",
    model_id="kokoro",
    text="hello",
    voice_id=None,
    response_format="wav",
    speed=1.0,
)
artifact = STTSGeneratedAudio(
    path=tmp_path / "result.wav",
    provider_id="audio_cpp",
    model_id="kokoro",
    voice_id=None,
    source_text="hello",
    operation_id="local-op",
    audio_format="wav",
    content_type="audio/wav",
    metadata={"delivery": "complete_wav"},
)
```

Assert frozen mutation fails, options/metadata are defensive immutable copies,
required identifiers are non-empty, and the artifact suffix is derived from
`audio_format`, not from a current UI selection.

- [ ] **Step 3: Run the focused tests and observe the missing APIs**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_playground_types.py -q
```

Expected: FAIL because the service methods and Playground contracts do not yet
exist.

- [ ] **Step 4: Implement the minimal contracts**

In `TTSService` add synchronous read-only forwarding methods:

```python
def provider_descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
    """Return ordered provider metadata without materializing adapters."""
    return self.registry.descriptors()

def configuration_revision(self, provider_id: str) -> int:
    """Return the selected provider's current registry revision."""
    return self.registry.configuration_revision(provider_id)
```

Create frozen, slotted dataclasses in `playground_types.py`. Copy `options` and
`metadata` into `MappingProxyType` in `__post_init__`, validate only structural
invariants, and do not put Textual or adapter implementations in this module.
Export the types through `tldw_chatbook.TTS`.

- [ ] **Step 5: Run the focused tests**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 6: Commit the provider-neutral seam**

```bash
git add \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_playground_types.py
git commit -m "feat(tts): add playground request and result contracts"
```

### Task 2: Implement pure catalog selection and restriction state

**Files:**

- Create: `tldw_chatbook/UI/stts_playground_catalog.py`
- Create: `Tests/UI/test_stts_playground_catalog.py`

- [ ] **Step 1: Write failing catalog-state tests**

Cover:

- canonical provider options preserve descriptor order;
- model fallback chooses the first valid model and reports when a removed
  selection changed;
- audio.cpp voices always begin with a local `Server default` sentinel;
- the initial audio.cpp voice is Server default even when discovered voices
  exist;
- a retained explicitly selected voice remains selected while still valid;
- Server default converts to `None` and is never returned as a literal request
  voice;
- audio.cpp format is exactly WAV and locked;
- audio.cpp speed is exactly `1.0` and locked;
- legacy model formats, voices, and speed support remain available;
- unavailable or stale health keeps prior models visible but sets
  `generation_allowed=False`;
- request tokens compare provider ID, configuration revision, catalog revision,
  and model ID;
- model and voice labels containing Rich/Textual markup remain plain text when
  converted at the rendering boundary.

Use small real `TTSProviderCatalog` fixtures. Do not instantiate Textual apps or
adapters in these tests.

- [ ] **Step 2: Run tests and confirm the module is absent**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_stts_playground_catalog.py -q
```

Expected: FAIL because `stts_playground_catalog` does not exist.

- [ ] **Step 3: Implement the pure state module**

Use focused contracts such as:

```python
SERVER_DEFAULT_VOICE_ID = "__server_default__"

@dataclass(frozen=True, slots=True)
class CatalogRequestToken:
    provider_id: str
    configuration_revision: int
    catalog_revision: int | None = None
    model_id: str | None = None

@dataclass(frozen=True, slots=True)
class PlaygroundControls:
    provider_id: str
    model_options: tuple[tuple[str, str], ...]
    selected_model_id: str | None
    voice_options: tuple[tuple[str, str], ...]
    selected_voice_id: str | None
    format_options: tuple[str, ...]
    selected_format: str | None
    format_locked: bool
    speed: float
    speed_locked: bool
    generation_allowed: bool
    selection_changed: bool
```

Keep selection, fallback, and restriction rules pure. Return strings from this
module; construct Rich `Text` labels with markup disabled only in the UI.

- [ ] **Step 4: Run the pure tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit catalog state**

```bash
git add \
  tldw_chatbook/UI/stts_playground_catalog.py \
  Tests/UI/test_stts_playground_catalog.py
git commit -m "feat(stts): add catalog driven control state"
```

### Task 3: Add external audio.cpp settings persistence and explicit discovery actions

**Files:**

- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `Tests/UI/test_stts_settings_widget.py`
- Modify: `Tests/TTS/test_stts_settings_reconfiguration.py`

- [ ] **Step 1: Write failing settings composition and validation tests**

Mount `TTSSettingsWidget` and assert the external-only panel contains:

- fixed mode `external`;
- base URL;
- connect and synthesis timeouts;
- input, response, metadata, model, voice, and identifier bounds;
- privacy copy stating submitted text is sent to the configured server;
- Save, Test Connection, and Refresh Models actions;
- no binary path, `server.json`, Start, Restart, managed log, or process control.

Set all inputs to valid non-default values, call `_save_settings()`, and assert
one defensively isolated ordinary `dict` appears under
`settings["audio_cpp"]`. Mutating the original form/candidate after posting
must not change the event snapshot.

Parametrize invalid origins, credentials, paths, query/fragment components,
non-positive numbers, booleans, non-integral limits, and oversized numeric
strings. Assert no event is posted and neither the invalid value nor origin is
logged or notified.

- [ ] **Step 2: Write failing persistence/reconfiguration tests**

Extend binding-table coverage with:

```python
"audio_cpp": _SettingBinding(
    (("app_tts", "audio_cpp"),),
    provider_id="audio_cpp",
)
```

Prove one atomic save writes `[app_tts.audio_cpp]`, reloads effective settings,
projects them with `project_audio_cpp_config()`, and calls:

```python
await service.reconfigure_provider(
    "audio_cpp",
    projected.to_mapping(),
)
```

Assert the persisted value is a defensively copied ordinary nested `dict`, not
a `MappingProxyType`, and inspect the written TOML/reloaded mapping to prove it
has the `[app_tts.audio_cpp]` shape rather than an array of keys.

Assert Save does not call `get_catalog`, `get_voices`, `synthesize`, or the
adapter factory. An unchanged canonical config is a no-op. A changed config
retires only audio.cpp and leaves all materialized legacy adapters untouched.
Capture the `ReconfigureResult`: `UNCHANGED` emits no provider-state change;
`CHANGED` emits one canonical `audio_cpp` configuration-changed notification
containing the new service revision.

- [ ] **Step 3: Run focused settings tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_stts_settings_widget.py \
  Tests/TTS/test_stts_settings_reconfiguration.py -q
```

Expected: FAIL for the missing panel, binding, and native config projection.

- [ ] **Step 4: Implement nested settings and provider config dispatch**

Add `audio_cpp` first in `_TTS_PROVIDER_ORDER`. Introduce a private helper:

```python
def _effective_provider_config(
    provider_id: str,
    effective_settings: Mapping[str, Any],
) -> Mapping[str, Any]:
    if provider_id == "audio_cpp":
        return project_audio_cpp_config(effective_settings).to_mapping()
    return legacy_provider_config(provider_id, effective_settings)
```

Use `AudioCppConfig.from_mapping()` in `TTSSettingsWidget` to validate the
complete form locally. The frozen `AudioCppConfig` is the validation snapshot;
at the persistence boundary, call `to_mapping()` and defensively copy it into a
plain nested `dict` accepted by the TOML writer. Never persist a mapping proxy.
Error notifications are fixed safe copy and logs contain only setting
names/outcomes.

After successful reconfiguration, inspect each `ReconfigureResult`. For
`CHANGED`, publish a provider-configuration-changed message with only the
canonical provider ID and new integer revision. A mounted Playground handles
that message by marking only that provider's catalog/voices stale, cancelling
its discovery workers, and disabling Generate without reconnecting. Existing
generated artifacts remain playable/exportable. `UNCHANGED` publishes nothing
and leaves readiness state untouched.

- [ ] **Step 5: Implement explicit Test Connection and Refresh Models workers**

Use separate Textual worker group `stts-audio-cpp-settings-discovery`. Both
actions operate on the currently saved/reconfigured app-owned service:

```python
service = await get_tts_service()
catalog = await service.get_catalog("audio_cpp", refresh=True)
```

Test Connection reports compatible availability and model count; Refresh Models
reports the refreshed count. Neither action displays the configured origin,
remote diagnostic, raw exception, model identifiers, or values. Explain in the
panel that form changes must be saved before testing. Capture
`service.configuration_revision("audio_cpp")` before I/O and compare it again
before applying either result; discard the result when the revision changed.

- [ ] **Step 6: Run focused settings tests**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 7: Commit external settings**

```bash
git add \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/TTS/test_stts_settings_reconfiguration.py
git commit -m "feat(stts): add external audio cpp settings"
```

### Task 4: Make Playground provider, model, voice, format, and speed controls catalog-driven

**Files:**

- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Create: `Tests/UI/test_stts_playground_audio_cpp.py`
- Modify: `Tests/UI/test_stts_capability_state.py`

- [ ] **Step 1: Build a deterministic Textual host and fake service**

The fake service records descriptor, catalog, voice, revision, and synthesis
calls. Its descriptors list `audio_cpp` first and the six legacy providers. Its
audio.cpp catalog includes multiple models, unsafe-looking opaque identifiers,
WAV-only metadata, and `omit_voice_uses_server_default=True`.

Mount `TTSPlaygroundWidget` without an app server or model download.

- [ ] **Step 2: Write failing descriptor and lazy-materialization tests**

Assert opening the widget:

- calls only `provider_descriptors()`;
- populates options as `(safe display label, canonical provider ID)`;
- makes no catalog/voice/synthesis calls until the selected provider is
  resolved;
- accepts `audio_cpp` directly without label parsing;
- leaves the six legacy provider options present and ordered.

- [ ] **Step 3: Write failing catalog, voice, and stale-result tests**

Cover:

- selecting audio.cpp starts only the catalog worker;
- selected-provider readiness populates TTS models;
- model selection starts only the voice worker;
- Server default is selected and becomes `None`;
- missing voices leaves only Server default;
- a removed model/voice selects a valid fallback and announces the change;
- old provider, configuration-revision, catalog-revision, or model results are
  discarded;
- a successful `CHANGED` settings notification immediately marks the selected
  audio.cpp catalog and voices stale, cancels only discovery workers, disables
  Generate without connecting, and preserves the current generated artifact;
- an `UNCHANGED` save emits no notification and leaves fresh state unchanged;
- unsafe identifier text renders literally;
- stale models remain visible but Generate is disabled;
- unavailable, incompatible, not-configured, and reconfiguring states expose
  fixed safe status/recovery copy.

- [ ] **Step 4: Write failing control-restoration tests**

Select a legacy provider, choose non-default model/voice/format/speed, switch to
audio.cpp, then switch back. Assert:

- audio.cpp forces WAV and disables Format;
- audio.cpp forces `1.0` and disables Speed with an explanation;
- legacy selection state is restored;
- provider-specific legacy panels keep their existing visibility behavior;
- no audio.cpp restriction remains sticky after switching.

- [ ] **Step 5: Run the Textual tests and observe the static branches**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_capability_state.py -q
```

Expected: FAIL because the Playground still hard-codes and parses provider
labels, models, and voices.

- [ ] **Step 6: Implement descriptor initialization and independent workers**

Replace the Playground's static provider option list with an initially empty
Select. On mount:

1. retrieve the app-owned service;
2. read descriptors synchronously without adapter acquisition;
3. set safe Rich `Text` labels and canonical values;
4. select the configured default if registered, otherwise the first descriptor;
5. start `stts-catalog-discovery` only for that provider.

Use `@work(exclusive=True, group="stts-catalog-discovery")` for catalogs and
`@work(exclusive=True, group="stts-voice-discovery")` for voices. Capture
`CatalogRequestToken` before I/O and verify all token fields before applying a
result. Read the service configuration revision both before and after each
await; a result applies only when both revisions and the current widget token
match.

- [ ] **Step 7: Implement one catalog-application path**

Replace shared hard-coded model/voice/format/speed updates with one method that
consumes `PlaygroundControls`. Keep small provider-specific legacy extensions
(saved Chatterbox/Higgs profiles and Kokoro blends) layered on the catalog
without changing canonical base options.

Store per-provider control snapshots before switching. Disable Generate unless
the selected catalog is fresh/available, the model is valid, text is non-empty,
the cached configuration revision still equals the service revision, and no
generation is active. Handle the canonical configuration-changed message by
invalidating only matching catalog/voice state and workers; do not call the
adapter from the message handler.

- [ ] **Step 8: Run the Textual tests**

Run the Step 5 command.

Expected: PASS.

- [ ] **Step 9: Commit catalog-driven controls**

```bash
git add \
  tldw_chatbook/UI/STTS_Window.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_capability_state.py
git commit -m "feat(stts): drive playground controls from catalogs"
```

### Task 5: Route audio.cpp generation through the native service and retain provenance

**Files:**

- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Create: `Tests/TTS/test_stts_audio_cpp_generation.py`
- Modify: `Tests/TTS/test_stts_export_security.py`

- [ ] **Step 1: Write failing immutable request-snapshot tests**

Assert `_generate_tts()` captures a single `STTSPlaygroundRequest` containing:

- a local UUID operation ID;
- canonical provider/model IDs;
- source text snapshot;
- `voice_id=None` for Server default;
- WAV and speed `1.0` for audio.cpp;
- an immutable copy of options.

Change every selector and the TextArea after posting the event and prove the
worker still receives the original snapshot.

- [ ] **Step 2: Write failing native generation tests**

With a fake `TTSService.synthesize()` response, assert the handler sends:

```python
TTSRequest(
    provider_id="audio_cpp",
    model_id=snapshot.model_id,
    text=snapshot.text,
    voice=snapshot.voice_id,
    response_format="wav",
    speed=1.0,
    options={},
)
```

Assert it consumes the async response exactly once, always calls `aclose()`,
writes an owner-only `.wav` temporary file, and returns
`STTSGeneratedAudio` with response provider/model/format/content type/metadata
plus `voice_id=snapshot.voice_id`, the local source text, and operation ID.

Prove audio.cpp never calls `generate_audio_stream()` or format conversion.
Prove a legacy snapshot still calls the unchanged compatibility stream path and
retains its requested conversion behavior.

- [ ] **Step 3: Write failing error, cancellation, and privacy tests**

Cover every stable audio.cpp operation code, retryability, recovery action,
reconfiguring, registry closed, local configuration `ValueError`, and unknown
exception fallback.

Assert:

- `str(TTSOperationError)` is treated as the established safe message contract
  and escaped before Rich/Textual display;
- unknown exception strings are not displayed or logged;
- text, origin, config values, raw identifiers, and credentials do not appear
  in logs or notices;
- `CancelledError` propagates and closes the response without a failure notice;
- repeated Generate while active is rejected without replacing the task.

- [ ] **Step 4: Run focused generation tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_stts_export_security.py -q
```

Expected: FAIL because native request and artifact paths are absent.

- [ ] **Step 5: Split native and legacy generation paths**

Keep `handle_playground_generate()` as the retained owner. Refactor the worker
into small helpers:

```python
async def _generate_audio_cpp(
    self,
    snapshot: STTSPlaygroundRequest,
    progress_sink: ProgressSink,
) -> STTSGeneratedAudio: ...

async def _generate_legacy(
    self,
    snapshot: STTSPlaygroundRequest,
    progress_sink: ProgressSink,
) -> STTSGeneratedAudio: ...
```

Use `create_secure_temp_file()` for final bytes and track the file in
`_playground_audio_files`. Preserve primary exceptions while closing native
responses. Map only stable safe operation fields into the UI.

- [ ] **Step 6: Deliver and store the immutable artifact**

Change `_generation_complete()` to accept `STTSGeneratedAudio`. Store the
artifact separately from selectors and set `current_audio_file=artifact.path`
only as a compatibility convenience.

Playback and export derive the path, extension, and status text from the
artifact. Use the artifact's actual format even after provider/model/format
selectors change.

- [ ] **Step 7: Run focused generation tests**

Run the Step 4 command.

Expected: PASS.

- [ ] **Step 8: Commit native Playground generation**

```bash
git add \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/UI/STTS_Window.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_stts_export_security.py
git commit -m "feat(stts): generate audio cpp wav through native service"
```

### Task 6: Harden worker ownership, recovery state, playback, and cleanup

**Files:**

- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `Tests/UI/test_stts_playground_audio_cpp.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`

- [ ] **Step 1: Write failing worker-isolation tests**

Prove:

- catalog refresh does not cancel generation;
- voice discovery does not cancel catalog refresh for a different token until
  its stale result is discarded;
- repeated Generate cannot replace the active handler task;
- playback uses its own `stts-playback` group;
- leaving the Playground cancels widget-owned discovery/playback workers but
  detaches, rather than cancels, the single app-handler-owned generation task;
- generation completion after unmount never calls the removed widget, remains
  available in handler-owned state, and is rehydrated by a newly mounted
  Playground;
- unmount after artifact creation preserves only the current delivered
  artifact and never invokes global STTS cleanup or cancels settings/audiobook
  work;
- failed/cancelled generation deletes its partial files, successful replacement
  securely deletes the superseded artifact and intermediates, and app shutdown
  deletes the final retained artifact;
- generation completion after a selector switch keeps original provenance;
- progress callback/display failure cannot fail synthesis.

- [ ] **Step 2: Write failing recovery-state tests**

For stale/unavailable catalog state, assert:

- prior models remain visible;
- Generate is disabled;
- Refresh remains available;
- existing generated audio remains playable/exportable;
- recovery action copy is local and fixed;
- audio.cpp failure never auto-selects or invokes a legacy provider.

- [ ] **Step 3: Run focused lifecycle tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/TTS/test_stts_audio_cpp_generation.py -q
```

Expected: FAIL for the missing worker groups and recovery state.

- [ ] **Step 4: Implement bounded worker and cleanup ownership**

Use distinct worker groups:

- `stts-catalog-discovery`
- `stts-voice-discovery`
- `stts-audio-cpp-settings-discovery`
- retained event-handler generation task
- `stts-playback`

Make cleanup idempotent. Do not add a second generation worker inside the
event-handler task. The retained generation task is app-handler-owned: widget
unmount detaches it and cancels only widget-owned discovery/playback. Completion
looks up a currently mounted Playground and checks the operation ID before any
UI call; it never retains or calls the removed widget.

Keep one read-only handler snapshot of active operation ID plus current
`STTSGeneratedAudio`. A newly mounted Playground rehydrates generation/artifact
state from that snapshot. Track in-flight files per operation. On failure or
cancellation, securely delete that operation's partial files. On successful
replacement, store the new artifact before securely deleting the superseded
artifact and all intermediates. Preserve the current artifact after export and
widget unmount so it remains playable; delete it only when replaced or during
app-level STTS cleanup. App cleanup cancels and joins retained generation before
deleting files. Never call global STTS cleanup merely because the Playground
unmounted, and never cancel settings or audiobook tasks.

- [ ] **Step 5: Implement safe recovery display**

Map core recovery actions to fixed UI actions/copy without importing Textual
into TTS core. Escape safe messages at the Rich boundary. Keep generated audio
controls independent from current provider readiness.

- [ ] **Step 6: Run focused lifecycle tests**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 7: Commit lifecycle hardening**

```bash
git add \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/TTS/test_stts_audio_cpp_generation.py
git commit -m "test(stts): harden audio cpp playground lifecycle"
```

### Task 7: Document the landed external Playground vertical

**Files:**

- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify:
  `Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md`
- Modify:
  `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- Modify:
  `backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md`

- [ ] **Step 1: Update architecture and implementation status**

Document:

- Slice 3 is implemented and tracked by TASK-569;
- descriptors do not materialize adapters;
- catalog/voice workers and revision tokens;
- native audio.cpp request boundary versus retained legacy compatibility path;
- complete-WAV artifact provenance;
- Server default omission;
- WAV/speed restrictions and legacy restoration;
- safe error/recovery behavior;
- no automatic fallback.

- [ ] **Step 2: Add user-facing external setup and privacy guidance**

Explain:

1. start the user's own compatible `audiocpp_server`;
2. open STTS Settings and save the external origin/bounds;
3. use Test Connection or Refresh Models;
4. select audio.cpp in the Playground;
5. select a discovered model and optional voice;
6. generate, play, and export the complete WAV.

State clearly that submitted text is sent to the configured server. Do not
suggest Chatbook downloads, launches, or manages the server in Slice 3.

- [ ] **Step 3: Preserve later-slice deferrals**

Keep user-provided binary plus user-provided `server.json` launch/supervision in
Slices 4–5. Do not document managed settings or actions as available.

- [ ] **Step 4: Run documentation reference checks**

Run:

```bash
rg -n "TASK-569|Slice 3|audio_cpp|Server default|complete WAV" \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md \
  backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md \
  "backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md"
```

Expected: all governing and user-facing files reference the landed behavior.

- [ ] **Step 5: Commit documentation**

```bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md \
  backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md \
  "backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md"
git commit -m "docs(stts): document external audio cpp playground"
```

### Task 8: Run full verification and finish TASK-569

**Files:**

- Modify:
  `backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md`

- [ ] **Step 1: Run focused Slice 3 tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_stts_export_security.py \
  Tests/UI/test_stts_playground_catalog.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_capability_state.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broad regressions**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS \
  Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/Audio_Services/test_local_audio_services_service.py \
  Tests/Media/test_local_media_reading_service.py -q
```

Expected: PASS with only documented optional skips/warnings.

- [ ] **Step 3: Run static, compilation, boundary, and diff checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_stts_export_security.py \
  Tests/UI/test_stts_playground_catalog.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_capability_state.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_stts_export_security.py \
  Tests/UI/test_stts_playground_catalog.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_capability_state.py

../../.venv/bin/python -m compileall -q \
  tldw_chatbook/TTS \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py

../../.venv/bin/python -m mypy \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py

git diff --unified=0 origin/dev -- \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  | rg '^\+[^+].*(subprocess|create_subprocess|binary_path|server_config_path|server\.json|restart_managed|managed_log)'

git diff --check
```

Expected: static checks pass; the added-production-line boundary pipeline prints
nothing and exits `1` because `rg` found no prohibited additions (the existing
audiobook subprocess code is outside the added lines); `git diff --check`
passes.

- [ ] **Step 4: Perform security, privacy, and scope self-review**

Confirm:

- provider values are canonical IDs;
- remote identifiers render as plain text;
- only selected providers materialize;
- stale async results are discarded;
- Server default becomes `None`;
- audio.cpp uses native `synthesize()` and one complete WAV;
- legacy generation remains on the bridge;
- no native POST retry or UI fallback exists;
- logs/notices exclude text, config/origin/credential values, raw exceptions,
  and unsafe identifiers;
- generated artifacts retain immutable provenance;
- no managed process or managed UI work entered the diff.

- [ ] **Step 5: Complete Backlog evidence**

Add exact verification counts and any documented warnings to TASK-569.
Check every acceptance criterion and Definition-of-Done item only after its
evidence passes, then move the task to Done.

- [ ] **Step 6: Final commit**

```bash
git add \
  "backlog/tasks/task-569 - Complete-external-audio.cpp-STTS-Playground-vertical.md"
git commit -m "chore(stts): complete external audio cpp playground task"
```

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`

Reason: ADR-023 already governs the catalog-driven Playground, external privacy
boundary, complete-WAV interface, no-fallback policy, lifecycle, and ordered
slice delivery. Slice 3 implements that accepted decision and updates its
implementation status; no new ADR is required.
