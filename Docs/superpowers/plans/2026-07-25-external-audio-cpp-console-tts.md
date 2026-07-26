# External audio.cpp Console TTS Settings Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make newly saved external audio.cpp defaults immediately usable by Console **Speak** through the native TTS adapter without mixed settings/adapter generations, while preserving complete-WAV delivery, legacy compatibility, and external process ownership.

**Architecture:** Add an immutable global-preferences model and an app-owned request-admission coordinator beside the existing `TTSService`. Request admission freezes preferences and acquires a revision-matched registry lease under a shared gate; settings publication uses the exclusive gate, atomically persists mode/value mutations, and starts a bounded exclusive audio.cpp handoff whose latest generation wins. Console speech asks this runtime for one admitted default request, so audio.cpp uses the native adapter and retained providers still execute inside `LegacyTTSAdapter`.

**Tech Stack:** Python 3.11+, `asyncio`, frozen dataclasses and `Literal`, Textual messages/widgets, existing `TTSService`/`TTSAdapterRegistry`, TOML configuration helpers, pytest/pytest-asyncio, Ruff, mypy.

---

## Scope and governing decisions

- Implement only TASK-710 / Slice 1 from
  `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`.
- Amend ADR-023 before changing the runtime boundary.
- Connect only to a user-owned external `audiocpp_server`.
- Do not add binary paths, `server.json` paths, subprocess calls, launch,
  supervision, restart, signals, managed logs, process adoption, downloads, or
  model provisioning.
- Keep exactly one configured audio.cpp provider and never construct its
  replacement until the old adapter has closed.
- Preserve one complete, structurally validated WAV exposed through
  `TTSAudioResponse.byte_stream`; do not add client streaming.
- Preserve the six existing providers inside `LegacyTTSAdapter`. Do not create
  native adapters for them or remove the bridge.
- Do not add profile persistence, character assignment, `CharacterRef`, or
  `TTSMessageSpeechSnapshot`; those belong to later approved slices.
- Do not lower-case or otherwise rewrite opaque audio.cpp model and voice IDs.
- Do not automatically retry a synthesis POST or fall back from audio.cpp.
- Add no runtime dependency.

## ADR assessment

ADR required: yes

ADR path:
`backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`

Reason: TASK-710 changes the existing cross-module service contract by adding
atomic preference/lease admission, expected provider revisions, and a bounded,
generation-aware exclusive reconfiguration handoff. These extend ADR-023's
accepted registry and configuration-lifecycle decision; they do not justify a
new ADR.

## File map

### New focused modules

- Create `tldw_chatbook/TTS/preferences.py`: immutable global preference modes,
  compatibility reads, validation, and exact set/delete configuration mutation.
- Create `tldw_chatbook/TTS/request_admission.py`: writer-preferred async
  shared/exclusive gate, default-request resolution, atomic lease admission,
  and settings publication result contracts.

### Existing runtime and configuration files

- Modify `tldw_chatbook/config.py`: support one locked atomic mutation that sets
  and deletes keys across sections and distinguishes pre-replacement failure
  from post-replacement cache-refresh failure.
- Modify `tldw_chatbook/TTS/adapter_types.py`: safe revision-mismatch and
  unavailable/reconfiguration contracts.
- Modify `tldw_chatbook/TTS/adapter_registry.py`: revision-checked acquisition,
  retained generation-aware reconfiguration tickets, latest-generation wins,
  and shutdown joining.
- Modify `tldw_chatbook/TTS/TTS_Generation.py`: split resource admission from
  execution, retain the existing `synthesize()` convenience API, and expose
  default speech/publication through the coordinator.
- Modify `tldw_chatbook/TTS/adapter_bootstrap.py`: construct the coordinator
  from the startup preference snapshot without materializing adapters.
- Modify `tldw_chatbook/TTS/__init__.py`: export only provider-neutral contracts
  needed by handlers.

### Existing UI and event files

- Modify `tldw_chatbook/UI/STTS_Window.py`: translate local Select sentinels
  into explicit modes and mount from authoritative persisted modes.
- Modify
  `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`: carry the immutable
  proposed preference snapshot, atomically persist sets/deletes, publish through
  the coordinator, observe service-owned pending handoffs, and render bounded
  status.
- Modify
  `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`: remove the stale
  handler-owned defaults, request one admitted global operation, consume and
  close its response, and preserve current progress/playback/cleanup events.

### Tests

- Create `Tests/TTS/test_tts_preferences.py`.
- Create `Tests/TTS/test_tts_request_admission.py`.
- Create `Tests/TTS/test_console_audio_cpp_native.py`.
- Modify `Tests/TTS/fixtures/audio_cpp_http_v1/provenance.json`.
- Modify `Tests/TTS/test_audio_cpp_contract.py`.
- Modify `Tests/test_config_delete_settings.py`.
- Modify `Tests/TTS/test_adapter_registry.py`.
- Modify `Tests/TTS/test_tts_registry_service.py`.
- Modify `Tests/TTS/test_tts_app_ownership.py`.
- Modify `Tests/TTS/test_stts_settings_reconfiguration.py`.
- Modify `Tests/TTS/test_console_speak_autoplay.py`.
- Modify `Tests/TTS/test_tts_improvements.py`.
- Modify `Tests/TTS/test_tts_logging_privacy.py`.
- Modify `Tests/UI/test_stts_settings_widget.py`.

### Documentation and task record

- Modify
  `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`.
- Modify `Docs/Development/TTS/TTS_MODULE_GUIDE.md`.
- Modify `Docs/Features/Speech-Services-Guide.md`.
- Modify
  `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`
  only to record the implemented Slice 1 status and final evidence; do not
  redesign later slices during execution.
- Modify
  `backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md`.

## Task 1: Amend ADR-023 before runtime implementation

**Files:**

- Modify:
  `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- Modify:
  `backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md`

- [ ] **Step 1: Add TASK-710 to ADR-023 traceability**

Add TASK-710 to `Related Tasks` and the Links section. Keep the existing pinned
audio.cpp commit and managed-mode deferral unchanged.

- [ ] **Step 2: Record the strengthened request boundary**

Add the following accepted consequences:

```text
- Global TTS preferences are one immutable snapshot.
- A request freezes its selection and acquires a provider lease with the same
  configuration revision under one app-owned shared admission gate.
- Settings publication holds the exclusive side of that gate; requests see
  either the old coherent pair, the new coherent pair, or a structured
  reconfiguring/unavailable state.
```

- [ ] **Step 3: Record bounded exclusive handoff**

Specify that the foreground settings wait is finite, the old lease is never
silently cancelled, pending configuration is inert, only the latest pending
generation can become active, and the replacement audio.cpp adapter is created
only after the old adapter closes.

- [ ] **Step 4: Record the pre-existing static-analysis baseline**

Before changing production Python, run:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/config.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/config.py
../../.venv/bin/python -m mypy \
  tldw_chatbook/config.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/UI/STTS_Window.py
```

Record the exact output in TASK-710. The accepted starting baseline is:

- two Ruff `F841` findings in untouched `config.py` code;
- one Ruff-format “would reformat `config.py`” result;
- twelve mypy errors in pre-existing `config.py`, `tts_events.py`, and
  `STTS_Window.py` code.

These non-zero commands document baseline debt; they do not authorize unrelated
cleanup or permit any new diagnostic.

- [ ] **Step 5: Verify ADR links and scope language**

Run:

```bash
rg -n "TASK-710|admission|revision|pending|external|managed" \
  backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
git diff --check
```

Expected: TASK-710 and all four runtime invariants are present; managed mode
remains explicitly deferred; `git diff --check` exits 0.

- [ ] **Step 6: Commit the governing decision**

```bash
git add \
  backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md \
  "backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md"
git commit -m "docs: define atomic TTS admission contract"
```

## Task 2: Run the installed-build stop/go gate

**Files:**

- Modify: `Tests/TTS/fixtures/audio_cpp_http_v1/provenance.json`
- Modify: `Tests/TTS/test_audio_cpp_contract.py`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify:
  `backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md`

- [ ] **Step 1: Capture build and process identity without changing it**

Against the already user-started server, record:

```bash
command -v audiocpp_server
brew list --versions audio-cpp
curl -fsS http://127.0.0.1:8080/health
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

Record the PID from the `lsof` row. Do not launch, restart, signal, reconfigure,
adopt, or stop the server, and do not copy its `server.json` or model path into
repository fixtures.

Expected on the current UAT host: the binary resolves from Homebrew, the
installed formula identity is recorded verbatim, health is `ok`, and one
pre-existing PID owns the listening server.

- [ ] **Step 2: Characterize all four pinned HTTP surfaces**

Use a temporary directory outside the repository and issue exactly:

```text
GET  http://127.0.0.1:8080/health
GET  http://127.0.0.1:8080/v1/models
GET  http://127.0.0.1:8080/v1/audio/voices?model=supertonic-3
POST http://127.0.0.1:8080/v1/audio/speech
```

The POST body is:

```json
{
  "model": "supertonic-3",
  "input": "Chatbook audio.cpp compatibility check.",
  "voice": "M1",
  "response_format": "wav"
}
```

Capture response headers and the complete bounded body. Do not add a streaming
request, retry, fallback, or server mutation.

- [ ] **Step 3: Validate observations with the pinned contract**

Feed the captured JSON bodies to `parse_health_response()`,
`parse_models_response()`, and `parse_voices_response()`. Require the speech
response MIME to be `audio/wav`, then call `validate_pcm16_wav()` on the
complete body and assert it is non-empty PCM16.

This is a hard stop:

- if every surface matches, continue;
- if any endpoint, shape, MIME, bound, or WAV invariant differs, stop TASK-710
  before touching runtime code and open a separate reviewed ADR-023/contract
  amendment.

- [ ] **Step 4: Record compatible-build evidence without repinning**

Add a `compatible_builds` entry to `provenance.json` containing only the
package/distribution identity, characterization date, and compatible endpoint
contract. Keep the existing pinned repository commit unchanged. Do not record
URLs, credentials, local paths, process IDs, user text, or raw server output.

Update `test_pinned_fixtures_capture_reviewed_upstream_contract()` to assert the
new bounded structure, and document that compatibility evidence does not move
the ADR pin.

- [ ] **Step 5: Run the contract gate tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_contract.py \
  Tests/TTS/test_audio_cpp_adapter.py -q
git diff --check
```

Expected: PASS. A failure blocks every later task in this plan.

- [ ] **Step 6: Commit the compatibility evidence**

```bash
git add \
  Tests/TTS/fixtures/audio_cpp_http_v1/provenance.json \
  Tests/TTS/test_audio_cpp_contract.py \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  "backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md"
git commit -m "test: characterize installed audio cpp build"
```

## Task 3: Add typed preferences and atomic set/delete persistence

**Files:**

- Create: `tldw_chatbook/TTS/preferences.py`
- Create: `Tests/TTS/test_tts_preferences.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `Tests/test_config_delete_settings.py`
- Modify: `tldw_chatbook/TTS/__init__.py`

- [ ] **Step 1: Write failing compatibility-read tests**

Cover mode-authoritative reads, exact reads, and the legacy audio.cpp blank
compatibility rule:

```python
snapshot = TTSPreferencesSnapshot.from_settings(
    {
        "app_tts": {
            "default_provider": "audio_cpp",
            "default_model": "",
            "default_voice": "",
            "default_format": "wav",
            "default_speed": 1.0,
        }
    }
)

assert snapshot.provider_id == "audio_cpp"
assert snapshot.model_mode == "first_available"
assert snapshot.model_id is None
assert snapshot.voice_mode == "server_default"
assert snapshot.voice_id is None
assert snapshot.response_format == "wav"
assert snapshot.speed == 1.0
```

Also prove:

- explicit `default_model_mode`/`default_voice_mode` override legacy values;
- exact mode requires a non-empty corresponding ID;
- audio.cpp rejects non-WAV, speed other than exactly `1.0`, and arbitrary
  options;
- opaque exact IDs preserve case and punctuation;
- reading old blanks performs no write.

- [ ] **Step 2: Write failing configuration-mutation tests**

Define the expected provider-neutral contracts:

```python
@dataclass(frozen=True, slots=True)
class TTSPreferencesSnapshot:
    provider_id: str
    model_mode: Literal["exact", "first_available"]
    model_id: str | None
    voice_mode: Literal["exact", "server_default"]
    voice_id: str | None
    response_format: str
    speed: float

@dataclass(frozen=True, slots=True)
class TTSConfigMutation:
    sets: Mapping[str, Mapping[str, object]]
    deletes: Mapping[str, tuple[str, ...]]
```

For exact mode, assert the mutation sets authoritative mode keys and dual-writes
the exact values to the current aliases. For dynamic mode, assert it sets the
mode keys and deletes exactly:

```python
{
    "app_tts": ("default_model", "default_voice"),
    "tts_settings": (
        "default_openai_tts_model",
        "default_tts_voice",
    ),
}
```

Add both mixed-mode cases:

- exact model plus server-default voice dual-writes the model ID and deletes
  only the two voice aliases;
- first-available model plus exact voice deletes only the two model aliases and
  dual-writes the voice ID.

Do not use an empty string or `None` as a deletion sentinel.

- [ ] **Step 3: Write a failing atomic file-mutation test**

Start with a TOML file containing stale exact values and unrelated keys. Call
the new structured primitive:

```python
result = apply_settings_mutation_to_cli_config(
    {
        "app_tts": {
            "default_provider": "audio_cpp",
            "default_model_mode": "first_available",
            "default_voice_mode": "server_default",
            "default_format": "wav",
            "default_speed": 1.0,
        }
    },
    delete_keys={
        "app_tts": ("default_model", "default_voice"),
        "tts_settings": (
            "default_openai_tts_model",
            "default_tts_voice",
        ),
    },
)

assert result == ConfigMutationResult(
    file_replaced=True,
    caches_reloaded=True,
    failure_phase=None,
)
```

Assert one atomic write removes only those keys, preserves unrelated sections
and `0600` permissions, and reloads configuration caches once. Add a failure
test proving overlapping set/delete targets fail before writing.

Inject a cache-reload exception after `atomic_write_text()` returns and assert:

```python
assert result.file_replaced is True
assert result.caches_reloaded is False
assert result.failure_phase == "cache_reload"
```

The file must contain the new values despite the reload failure. Also inject a
pre-replacement failure and assert `file_replaced` is false and disk is
unchanged.

- [ ] **Step 4: Run the focused tests and observe failure**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_preferences.py \
  Tests/test_config_delete_settings.py -q
```

Expected: FAIL because the preference types, structured mutation result, and
atomic set/delete primitive do not exist.

- [ ] **Step 5: Implement `TTSPreferencesSnapshot`**

Keep parsing and validation Textual-free. Resolve the raw `[app_tts]` table from
either `COMPREHENSIVE_CONFIG_RAW`, a direct raw mapping, or
`APP_TTS_CONFIG`. Use `Literal` aliases and explicit validation; do not encode
dynamic modes as blank exact IDs.

- [ ] **Step 6: Implement exact config mutations**

Return defensive immutable set/delete mappings. Keep the legacy aliases exactly
as they exist in `_TTS_SETTING_BINDINGS`; do not delete similarly named keys or
provider-specific defaults.

- [ ] **Step 7: Add a structured atomic config-mutation result**

Add:

```python
@dataclass(frozen=True, slots=True)
class ConfigMutationResult:
    file_replaced: bool
    caches_reloaded: bool
    failure_phase: Literal["before_replace", "cache_reload"] | None

    @property
    def fully_applied(self) -> bool:
        return self.file_replaced and self.caches_reloaded


def apply_settings_mutation_to_cli_config(
    section_values: Mapping[str, Mapping[Any, Any]],
    *,
    delete_keys: Mapping[str, Collection[str]] | None = None,
) -> ConfigMutationResult:
```

Under the existing `_config_file_lock`, validate target overlap, read once,
delete exact keys, set exact values, write once with `atomic_write_text`, and
set `file_replaced=True` immediately after that replacement returns. Cache
invalidation/reload is a separately caught phase: a reload failure returns
`file_replaced=True, caches_reloaded=False` and never misreports the disk as
unchanged.

Keep `save_settings_to_cli_config(..., delete_keys=None) -> bool` and
`delete_settings_from_cli_config()` as compatibility wrappers over the
structured primitive. Their boolean remains `result.fully_applied`, preserving
existing callers without performing a second write.

- [ ] **Step 8: Run focused and existing config tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_preferences.py \
  Tests/test_config_delete_settings.py \
  Tests/test_config_console_defaults.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit typed preferences and persistence**

```bash
git add \
  tldw_chatbook/TTS/preferences.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/config.py \
  Tests/TTS/test_tts_preferences.py \
  Tests/test_config_delete_settings.py
git commit -m "feat: add typed TTS preference modes"
```

## Task 4: Make the settings widget emit explicit modes

**Files:**

- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify:
  `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `Tests/UI/test_stts_settings_widget.py`
- Modify: `Tests/TTS/test_stts_settings_reconfiguration.py`

- [ ] **Step 1: Replace the sentinel-to-empty regression expectation**

Change the existing failing regression so selecting
`FIRST_AVAILABLE_MODEL_ID` and `SERVER_DEFAULT_VOICE_ID` yields:

```python
event = app.saved_events[-1]
assert event.preferences.model_mode == "first_available"
assert event.preferences.model_id is None
assert event.preferences.voice_mode == "server_default"
assert event.preferences.voice_id is None
assert "default_model" not in event.settings
assert "default_voice" not in event.settings
```

Add exact-mode coverage proving remote IDs that resemble sentinel strings remain
exact opaque strings.

- [ ] **Step 2: Add failing mount tests for authoritative modes**

Cover:

- mode keys present with stale exact aliases: the mode wins;
- old blank audio.cpp values with no modes: first-available/server-default;
- old non-blank audio.cpp values with no modes: exact/exact;
- mounting performs no configuration write;
- audio.cpp keeps WAV and speed `1.0`.

- [ ] **Step 3: Make `STTSSettingsSaveEvent` carry the proposal**

Use an immutable optional field for compatibility with provider-only tests:

```python
class STTSSettingsSaveEvent(Message):
    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        preferences: TTSPreferencesSnapshot | None = None,
    ) -> None:
        super().__init__()
        self.settings = deepcopy(dict(settings))
        self.preferences = preferences
```

`None` means “this event changes provider settings only”; it must not synthesize
a new preference snapshot from partial data.

- [ ] **Step 4: Translate UI sentinels before posting**

In `_save_settings()`, construct a `TTSPreferencesSnapshot` from the selected
provider, model/voice sentinel identity, exact strings, format, and speed.
Remove `default_*` persistence values from the untyped `settings` payload and
post:

```python
self.app.post_message(
    STTSSettingsSaveEvent(settings, preferences=preferences)
)
```

Never stringify `Select.BLANK`, `Select.NULL`, or the local enum sentinels.

- [ ] **Step 5: Mount from one parsed snapshot**

Have `_set_initial_values()` parse preferences once and set provider, model,
voice, format, and speed from that snapshot. Keep provider catalog discovery
lazy; mounting must not connect to audio.cpp or materialize its adapter.

- [ ] **Step 6: Run widget and settings-event tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_stts_settings_widget.py \
  Tests/TTS/test_stts_settings_reconfiguration.py -q
```

Expected: PASS, including old blank reads and sentinel non-serialization.

- [ ] **Step 7: Commit explicit settings modes**

```bash
git add \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/TTS/test_stts_settings_reconfiguration.py
git commit -m "fix: persist explicit TTS selection modes"
```

## Task 5: Add revision-checked service admission

**Files:**

- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/adapter_registry.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `Tests/TTS/test_adapter_registry.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Modify: `Tests/TTS/test_tts_logging_privacy.py`

- [ ] **Step 1: Write a failing registry race test**

Pause after reading revision 1, reconfigure to revision 2, then attempt:

```python
with pytest.raises(TTSConfigurationRevisionError):
    await registry.acquire("audio_cpp", expected_revision=1)
```

Assert no replacement adapter is materialized by the rejected acquisition and
the error exposes no configuration value, URL, or credential.

Also seal a slot as unavailable and assert acquisition raises
`TTSProviderUnavailableError`, while a pending exclusive handoff raises the
existing `TTSProviderReconfiguringError`. These three exception types must be
distinguishable without inspecting message strings, and their fixed messages
may contain only the canonical provider ID.

- [ ] **Step 2: Write failing admitted-operation lifetime tests**

Specify a service-internal operation:

```python
operation = await service.admit(
    request,
    expected_configuration_revision=1,
)
response = await operation.synthesize(progress_sink)
```

Cover:

- admission owns the service semaphore and registry lease before returning;
- reconfiguration sees the lease and cannot close its adapter;
- execution failure releases both resources;
- caller cancellation during admission or execution releases them once;
- successful response retains them until `response.aclose()`;
- abandoning an admitted operation and closing the service cannot leak the
  semaphore or lease.

- [ ] **Step 3: Run the tests and observe missing APIs**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_logging_privacy.py -q
```

Expected: FAIL on `expected_revision`, the safe mismatch/unavailable errors,
and `admit()`.

- [ ] **Step 4: Add safe admission errors and revision checking**

Retain `TTSProviderReconfiguringError` and add:

```python
class TTSConfigurationRevisionError(RuntimeError):
    """Raised when request selection and provider revision no longer match."""


class TTSProviderUnavailableError(RuntimeError):
    """Raised when a failed handoff has sealed a provider slot."""
```

Neither error stores or chains a raw provider/configuration exception.
Represent the failed slot state explicitly so a new reviewed settings
publication can transition it to reconfiguring, while ordinary acquisition
cannot materialize an adapter from a sealed configuration.

Extend acquisition without weakening existing callers:

```python
async def acquire(
    self,
    provider_id: str,
    *,
    expected_revision: int | None = None,
) -> TTSAdapterLease:
    ...
    async with slot.lock:
        if slot.reconfiguring:
            raise TTSProviderReconfiguringError(
                f"TTS provider is reconfiguring: {canonical_id}"
            )
        if slot.unavailable:
            raise TTSProviderUnavailableError(
                f"TTS provider is unavailable: {canonical_id}"
            )
        if (
            expected_revision is not None
            and slot.revision != expected_revision
        ):
            raise TTSConfigurationRevisionError(
                f"TTS provider configuration changed: {canonical_id}"
            )
```

Check all three states before factory materialization and before incrementing
leases. Add an explicit registry operation used only by the retained
publication path to seal affected slots unavailable after a failed handoff;
ordinary callers never clear that state.

- [ ] **Step 5: Split service admission from execution**

Move the semaphore/lease acquisition portion of `TTSService.synthesize()` into
`admit()`. Add a private, single-use `_AdmittedTTSOperation` that owns
`_OperationResources`, executes `ensure_ready()` plus adapter synthesis, and
transfers resource ownership to `_ManagedAudioResponse`.

Keep the public convenience API behavior:

```python
async def synthesize(self, request, progress_sink=None):
    operation = await self.admit(request)
    return await operation.synthesize(progress_sink)
```

No existing Playground or legacy caller should need to change in this task.

- [ ] **Step 6: Preserve shutdown and cleanup guarantees**

Track admitted-but-not-yet-executed operations alongside managed responses.
Service shutdown must start cleanup for both collections, preserve primary
errors, and retain the current bounded join behavior.

- [ ] **Step 7: Run focused service/registry tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_async_lifecycle.py \
  Tests/TTS/test_tts_logging_privacy.py -q
```

Expected: PASS with all existing synthesis, cancellation, and shutdown tests.

- [ ] **Step 8: Commit atomic resource admission**

```bash
git add \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_logging_privacy.py
git commit -m "feat: add revision-checked TTS admission"
```

## Task 6: Add the app-owned request-admission coordinator

**Files:**

- Create: `tldw_chatbook/TTS/request_admission.py`
- Create: `Tests/TTS/test_tts_request_admission.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Write failing writer-preference gate tests**

Use controlled events to prove:

- concurrent readers may enter;
- a waiting writer prevents later readers from starving it;
- cancellation removes a waiting reader/writer cleanly;
- releasing a cancelled context never corrupts counts.

Keep the gate private to `request_admission.py`; do not add a dependency.

- [ ] **Step 2: Write the mixed-generation regression**

Pause a default request after it freezes preferences but before registry
acquisition. Start settings publication, then release the request. Assert the
only allowed outcomes are:

- old snapshot + old revision lease; or
- new snapshot + new revision lease; or
- a structured reconfiguring/unavailable error.

Explicitly assert old snapshot + new revision and new snapshot + old revision
never occur.

- [ ] **Step 3: Write failing default-resolution tests**

For audio.cpp:

```python
response = await service.synthesize_default(
    text="Character response",
    voice_override=None,
)
```

Assert:

- exact model is passed byte-for-byte;
- `first_available` performs one catalog lookup, freezes the first eligible TTS
  model, and does not select a second model if synthesis fails;
- `server_default` sends `voice=None`;
- response format is `wav`, speed is `1.0`, and options are empty;
- the lease revision matches the frozen preference generation.

For each retained provider, assert the coordinator creates the same
`OpenAISpeechRequest` and enumerated internal model ID as the current Console
handler, then admits the corresponding `LegacyTTSAdapter` registry entry. The
bridge remains the only legacy implementation boundary.

- [ ] **Step 4: Implement the writer-preferred async gate**

Use `asyncio.Condition` with reader count, active writer, and waiting-writer
count. Expose only:

```python
@asynccontextmanager
async def read(self) -> AsyncIterator[None]: ...

@asynccontextmanager
async def write(self) -> AsyncIterator[None]: ...
```

Do not hold its condition lock while performing catalog, registry, or adapter
I/O.

- [ ] **Step 5: Implement coherent default request admission**

`TTSRequestAdmissionCoordinator.synthesize_default()` must:

1. enter the shared gate;
2. freeze the immutable preference snapshot;
3. resolve `first_available` once when required;
4. read the provider configuration revision;
5. build the provider-neutral native or legacy-bridge request;
6. call `TTSService.admit(... expected_configuration_revision=revision)`;
7. leave the shared gate;
8. execute the admitted operation and return its managed response.

The coordinator never exposes the concrete adapter or lease.

- [ ] **Step 6: Construct one coordinator at bootstrap**

`build_default_tts_service(app_config)` parses one initial
`TTSPreferencesSnapshot` and passes it to `TTSService`. Direct test construction
may use a safe default snapshot, but app bootstrap must use the supplied
configuration and remain adapter-lazy.

- [ ] **Step 7: Expose only narrow service methods**

Add:

```python
def preferences_snapshot(self) -> TTSPreferencesSnapshot: ...

async def synthesize_default(
    self,
    *,
    text: str,
    voice_override: str | None = None,
    progress_sink: ProgressSink | None = None,
) -> TTSAudioResponse: ...
```

Do not add a second registry accessor or global singleton.

- [ ] **Step 8: Run admission and ownership tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_app_ownership.py -q
```

Expected: PASS; app construction still creates one service and zero adapters.

- [ ] **Step 9: Commit the admission coordinator**

```bash
git add \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_app_ownership.py
git commit -m "feat: coordinate coherent TTS request admission"
```

## Task 7: Add bounded, latest-generation settings publication

**Files:**

- Modify: `tldw_chatbook/TTS/adapter_registry.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify:
  `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `Tests/TTS/test_adapter_registry.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_stts_settings_reconfiguration.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Write failing retained-handoff tests**

With an active audio.cpp lease, begin configuration generation 2 and assert:

- new acquisition is rejected as reconfiguring;
- the foreground publication returns `pending` by an injectable finite timeout;
- generation 1 speech is not cancelled;
- no replacement adapter exists.

Then release the lease and assert the old adapter closes before the replacement
can be lazily constructed.

For a save affecting multiple providers, assert transition start order is
canonical provider ID order and cleanup/unwind order is the reverse.

- [ ] **Step 2: Write the superseding-generation race**

While generation 2 waits on the old lease, submit generation 3. Release the
lease and assert:

- the old adapter closes once;
- generation 2 completes as superseded;
- only generation 3 configuration becomes current;
- the registry revision increments for the applied configuration only;
- no old and replacement audio.cpp adapters coexist.

- [ ] **Step 3: Write failure and shutdown tests**

Cover:

- old adapter close failure leaves the provider sealed/unavailable;
- cancellation of the foreground waiter does not cancel retained handoff;
- while an injected config mutation blocks on a `threading.Event`, an asyncio
  heartbeat continues to advance, proving no TOML/encryption/cache work runs on
  the Textual event loop;
- cancellation of the initiating Textual task while that thread is blocked
  does not cancel the retained persistence/publication task;
- a pre-replacement persistence failure changes neither preferences nor
  providers;
- an injected cache-reload failure after atomic replacement still publishes
  the saved immutable snapshot and leaves every affected slot either on the
  matching generation or sealed unavailable;
- the post-replacement cache warning is safe, structured, and recommends
  restart without exposing the raw failure;
- concurrent save tasks remain serialized by the retained coordinator task,
  even if an initiating handler task is cancelled;
- service shutdown joins or safely owns the pending worker;
- a leaked response cannot make the settings event await indefinitely;
- retry/restart copy contains no raw exception or configuration value.

- [ ] **Step 4: Add a generation-aware reconfiguration ticket**

Keep `reconfigure_provider()` compatible for callers that await definitive
completion. Add an internal/service-facing begin API returning a ticket:

```python
@dataclass(frozen=True, slots=True)
class TTSReconfigurationTicket:
    provider_id: str
    generation: int
    completion: asyncio.Task[ReconfigureResult]
```

The exclusive slot retains:

- latest pending config and generation;
- one old exclusive record;
- one retained handoff task;
- applied generation;
- sealed failure state.

Submitting a new config while handoff is pending replaces only the inert pending
data; it never starts a second close or adapter.

- [ ] **Step 5: Make the retained worker apply only the latest generation**

The worker waits for old leases, closes the old adapter, takes the latest
pending config under `slot.lock`, increments revision once, clears
`reconfiguring`, and leaves replacement construction lazy. A ticket whose
generation was not applied reports superseded.

Do not cancel the worker when a foreground waiter times out.

- [ ] **Step 6: Retain off-loop persistence and publication as one operation**

Add a provider-neutral persistence outcome and service-owned ticket:

```python
_TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class TTSSettingsPersistenceOutcome:
    file_replaced: bool
    caches_reloaded: bool
    failure_phase: Literal["before_replace", "cache_reload"] | None


@dataclass(frozen=True, slots=True)
class TTSSettingsPublicationTicket:
    generation: int
    foreground: asyncio.Future[TTSSettingsPublication]
    completion: asyncio.Task[TTSSettingsPublication]


def begin_preferences_publication(
    self,
    preferences: TTSPreferencesSnapshot,
    provider_configs: Mapping[str, Mapping[str, Any]],
    persistence: Callable[[], TTSSettingsPersistenceOutcome],
    *,
    foreground_timeout_seconds: float = _TTS_SETTINGS_FOREGROUND_TIMEOUT_SECONDS,
) -> TTSSettingsPublicationTicket:
    ...
```

`begin_preferences_publication()` validates all in-memory inputs, creates and
retains the completion task before returning, and must not await. The service,
not the Textual handler, owns that task through shutdown. Awaiters use
`asyncio.shield`; their timeout or cancellation never propagates into the
operation.

Production uses exactly two seconds for the provider-handoff foreground
deadline. Tests inject a shorter deadline and controlled events; they do not
sleep for two wall-clock seconds.

The coordinator owns a separate publication lock. The retained task:

1. acquires that lock so file replacement, admission publication, and
   foreground completion from separate saves cannot interleave;
2. runs the supplied persistence operation with `asyncio.to_thread()`;
3. if `file_replaced` is false, changes no in-memory preference or provider
   state and resolves a safe persistence failure;
4. if `file_replaced` is true, enters the exclusive admission gate even when
   `caches_reloaded` is false;
5. begins affected providers in canonical provider order;
6. waits no more than the injectable two-second provider-handoff deadline;
7. on any unexpected transition failure, seals every uncertain affected slot
   unavailable with no raw exception attached;
8. publishes the saved immutable snapshot exactly once;
9. resolves `foreground` with the persistence outcome and each provider as
   applied, pending, unchanged, superseded, or unavailable;
10. releases the admission and publication gates;
11. continues to own/observe pending handoffs until `completion` resolves.

The structured config primitive guarantees that every ordinary exception after
atomic replacement becomes `file_replaced=True`, never an exception with
unknown persistence state. A cache-reload failure therefore cannot abort the
matching runtime publication.

A pending provider remains sealed, so post-publication requests fail safely
until its retained task applies the matching saved generation.

- [ ] **Step 7: Persist and publish one settings proposal**

In `_persist_settings()`:

1. validate the complete proposal, merge provider setting sets with
   `event.preferences.config_mutation().sets`, build the complete prospective
   effective settings, project affected provider configs in
   `_TTS_PROVIDER_ORDER`;
2. build a no-argument persistence callable around
   `apply_settings_mutation_to_cli_config(...)` that converts its structured
   result to `TTSSettingsPersistenceOutcome`;
3. call the synchronous
   `service.begin_preferences_publication(...)` before any file work starts;
4. await only `asyncio.shield(ticket.foreground)`;
5. post `STTSProviderConfigurationChanged` only for an applied latest
   generation;
6. attach UI notification observers without taking lifecycle ownership away
   from the service;
7. notify **Saved — applying after current speech** for pending audio.cpp;
8. for `file_replaced=True, caches_reloaded=False`, notify that settings were
   saved and the TTS runtime was updated but a restart is recommended;
9. report safe unavailable plus **Retry/Reconnect** after provider failure.

If persistence does not replace the file, publication and reconfiguration do
not begin. If it does replace the file, initiating-task cancellation and cache
reload failure cannot interrupt the retained path that publishes the saved
snapshot and either applies or seals every affected provider.

- [ ] **Step 8: Run handoff and settings tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_tts_app_ownership.py -q
```

Expected: PASS with deterministic interleavings and no leaked tasks.
Assertions must distinguish revision mismatch, pending/reconfiguring, and
failed/unavailable acquisition by exception type rather than message parsing.

- [ ] **Step 9: Commit bounded settings publication**

```bash
git add \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_tts_app_ownership.py
git commit -m "fix: bound TTS settings publication handoff"
```

## Task 8: Route Console audio.cpp speech through the native service

**Files:**

- Create: `Tests/TTS/test_console_audio_cpp_native.py`
- Modify:
  `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
- Modify: `Tests/TTS/test_console_speak_autoplay.py`
- Modify: `Tests/TTS/test_tts_improvements.py`
- Modify: `Tests/TTS/test_tts_logging_privacy.py`

- [ ] **Step 1: Write a failing native Console speech test**

Configure exact external audio.cpp preferences and a fake native response.
Dispatch `TTSRequestEvent` and assert:

```python
assert captured_request == TTSRequest(
    provider_id="audio_cpp",
    model_id="<Opaque:Model>",
    text="Character response",
    voice="[Voice]",
    response_format="wav",
    speed=1.0,
    options={},
)
```

Assert `generate_audio_stream()` is not called for audio.cpp, the response is
closed exactly once, and opaque IDs are unchanged.

- [ ] **Step 2: Write complete-WAV and cleanup tests**

Prove the handler:

- consumes the bounded async iterator completely before publishing completion;
- creates a secure `.wav` artifact;
- publishes current progress and `TTSCompleteEvent`;
- follows the existing autoplay path;
- securely deletes partial artifacts on stream failure or cancellation;
- releases the service response on every exit;
- does not emit incremental-playback claims.

- [ ] **Step 3: Write legacy bridge regressions**

Parameterize all six retained providers. Assert each default request still
reaches its `LegacyTTSAdapter` and enumerated internal model route with existing
model, voice, format, speed, progress, and playback behavior.

- [ ] **Step 4: Remove the stale handler-owned preferences**

`TTSEventHandler.initialize_tts()` should retrieve the one bound service only.
Delete `_tts_config` as a runtime authority; do not call `get_cli_setting()` for
generation defaults after initialization.

- [ ] **Step 5: Use one admitted default operation**

Replace the model/provider fallback tree in `_generate_tts()` with:

```python
response = await self._tts_service.synthesize_default(
    text=text,
    voice_override=voice,
    progress_sink=progress_sink,
)
```

Consume `response.byte_stream` under `try/finally`, use actual response format
and provider/model metadata, and call `response.aclose()` exactly once. Keep the
existing cooldown, completion, playback, and handler task ownership.

- [ ] **Step 6: Keep metrics and errors safe**

Metrics use the design allowlist only: canonical provider ID, resolution source
(`global` or `explicit_override` in Slice 1), safe outcome code, and latency.
They never contain model or voice IDs, text, URLs, credentials, configuration
values, local paths, character authority, or raw upstream errors. Map
reconfiguring, unavailable, and revision mismatch by exception type to bounded
actionable UI copy, and add assertions for the allowlist and every prohibited
field.

- [ ] **Step 7: Run Console and privacy tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_tts_improvements.py \
  Tests/TTS/test_tts_logging_privacy.py -q
```

Expected: PASS.

- [ ] **Step 8: Run the STTS native-generation regression**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_stts_playground_audio_cpp.py -q
```

Expected: PASS; Playground behavior remains unchanged except that it shares the
new admission invariant.

- [ ] **Step 9: Commit native Console routing**

```bash
git add \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_tts_improvements.py \
  Tests/TTS/test_tts_logging_privacy.py
git commit -m "feat: route Console speech through native audio cpp"
```

## Task 9: Run UAT and record TASK-710 evidence

**Files:**

- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify:
  `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`
- Modify:
  `backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md`

- [x] **Step 1: Prove external process non-ownership**

Capture the external server PID and health before UAT. Run Chatbook setup,
speech, and shutdown. Capture them again and prove Chatbook did not launch,
restart, signal, supervise, adopt, or stop the process.

Run before and after:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8080/health
```

Search the changed range:

```bash
git diff origin/dev...HEAD -- \
  tldw_chatbook/TTS \
  tldw_chatbook/Event_Handlers/TTS_Events \
  tldw_chatbook/Event_Handlers/STTS_Events \
  tldw_chatbook/UI/STTS_Window.py |
rg -n "subprocess|Popen|server\\.json|binary_path|restart|terminate|kill|signal"
```

Expected: no added managed-process implementation.

- [x] **Step 2: Run first-time-user Console UAT**

Create isolated config and data directories without changing `HOME`:

```bash
TLDW_UAT_ROOT="$(mktemp -d /tmp/tldw-task710-uat.XXXXXX)"
TLDW_UAT_CONFIG="$TLDW_UAT_ROOT/config.toml"
TLDW_UAT_DATA="$TLDW_UAT_ROOT/data"
TLDW_CONFIG_PATH="$TLDW_UAT_CONFIG" \
TLDW_UAT_DATA="$TLDW_UAT_DATA" \
../../.venv/bin/python -c 'import os; from tldw_chatbook.config import save_setting_to_cli_config; raise SystemExit(0 if save_setting_to_cli_config("paths", "data_dir", os.environ["TLDW_UAT_DATA"]) else 1)'
TLDW_CONFIG_PATH="$TLDW_UAT_CONFIG" \
../../.venv/bin/python -m tldw_chatbook.app
```

Keep those task-specific variables for evidence collection and do not point any
command at the user's normal config or data directory. In that isolated app:

1. select external audio.cpp;
2. enter the existing server URL;
3. choose first-available/server-default or exact Supertonic model/voice;
4. save once;
5. do not restart Chatbook;
6. generate a deterministic character-roleplay assistant response;
7. click **Speak**;
8. verify provider/model/voice provenance and one complete playable WAV;
9. verify playback through the app player (`afplay` on the current macOS host);
10. verify the external server remains healthy with the same PID.

- [x] **Step 3: Run focused tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_tts_preferences.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_playground_audio_cpp.py -q
```

Expected: PASS.

- [x] **Step 4: Run broad TTS/STTS regressions**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS Tests/UI/test_stts_*.py -q
```

Expected: PASS, with only documented optional skips/warnings.

- [ ] **Step 5: Run the repository-wide test gate**

Run the same repository-wide suite required by the project DoD:

```bash
../../.venv/bin/python -m pytest Tests -q
```

Expected: PASS, with only documented optional skips/warnings. Do not mark
TASK-710 Done from focused tests alone.

Actual: the pre-rebase repository-wide run recorded 42 failed, 16,355 passed,
187 skipped, and 2 errors. An external rerun reduced the result to 37 failures,
and an untouched latest `origin/dev` control produced the identical exact 37
failures. The feature-only regression delta is zero, but the repository suite
is not green, so this step remains unchecked and TASK-710 remains In Progress.

- [x] **Step 6: Run static, typing, and diff verification**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS \
  tldw_chatbook/Event_Handlers/TTS_Events \
  tldw_chatbook/Event_Handlers/STTS_Events \
  tldw_chatbook/UI/STTS_Window.py \
  Tests/TTS \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/test_config_delete_settings.py

../../.venv/bin/python -m ruff check --ignore F841 \
  tldw_chatbook/config.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS \
  tldw_chatbook/Event_Handlers/TTS_Events \
  tldw_chatbook/Event_Handlers/STTS_Events \
  tldw_chatbook/UI/STTS_Window.py \
  Tests/TTS \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/test_config_delete_settings.py

../../.venv/bin/python -m ruff format --diff \
  tldw_chatbook/config.py

../../.venv/bin/python -m compileall -q \
  tldw_chatbook/config.py \
  tldw_chatbook/TTS \
  tldw_chatbook/Event_Handlers/TTS_Events \
  tldw_chatbook/Event_Handlers/STTS_Events \
  tldw_chatbook/UI/STTS_Window.py

../../.venv/bin/python -m mypy \
  tldw_chatbook/TTS/preferences.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py

git diff --check
```

Expected: the primary Ruff, Ruff-with-known-`F841`-exclusion, Ruff-format,
compileall, focused mypy, and `git diff --check` gates exit 0. The
`ruff format --diff config.py` audit remains non-zero only for the exact
whole-file baseline recorded in Task 1; inspect its hunks and confirm none was
introduced or expanded by TASK-710.

Then re-run the full baseline mypy command from Task 1. It may exit 1 only with
the same twelve recorded errors (or fewer) in the same pre-existing symbols.
There must be no new path, error code, symbol, or diagnostic on a TASK-710
changed line. Record the comparison in the task notes rather than cleaning up or
suppressing unrelated debt.

- [x] **Step 7: Update documentation and task evidence**

Document:

- explicit model/voice modes and old blank compatibility;
- atomic request admission and pending save copy;
- native Console audio.cpp versus retained legacy bridge;
- complete-WAV async-response behavior;
- dynamic-mode downgrade limitation;
- compatible installed build and UAT evidence;
- external process non-ownership.

In TASK-710, check every acceptance criterion only after its evidence exists,
add concise implementation notes and exact verification counts, confirm ADR-023
is linked, and set Done only when every repository DoD item is satisfied.

### Recorded execution evidence (2026-07-26)

- Before rebase, isolated clean-config Textual Console UAT passed against the
  user-owned listener at `127.0.0.1:8080` with provider `audio_cpp`, model mode
  `first_available`, voice mode `server_default`, and a deterministic Mira
  response.
- One native adapter produced one complete owner-only (`0600`) 594,604-byte
  mono PCM16 WAV at 44.1 kHz: 297,280 frames and 6.741 seconds. Lifecycle counts
  were complete `1`, playback `1`, progress `4`, and streaming `0`;
  `/usr/bin/afplay` exited `0`.
- The same listener identity and healthy response were present before and
  after UAT. Chatbook shut down without launching, restarting, signaling,
  supervising, adopting, or stopping the external process.
- All 25 pre-evidence commits were range-diff `=` patch-identical after the
  final rebase onto `origin/dev` `892011407`. Post-rebase, the focused suite
  passed 300 tests with 1 warning in 76.44 seconds; the broad suite passed
  1,008 tests with 14 skipped and 1 warning in 332.03 seconds.
- Primary Ruff, config Ruff with only the known `F841` baseline ignored,
  task-scoped Ruff format across 73 files, compileall, focused mypy across seven
  files, and `git diff --check` passed. Full baseline mypy retained exactly the
  same 12 errors in the same three files and symbols, and the `config.py`
  format audit retained the exact baseline hunks.
- Added-line process-keyword review found only restart-recommendation copy and
  an in-process `asyncio.Event` close signal; it found no process launch or
  control API. No character-profile production file changed.
- Final spec review identified a stale pre-admission provider comparison in
  Console. The reviewed fix makes the successful admitted response
  authoritative for metrics and moves adapter response-provider validation to
  `TTSService`, where it is checked against the canonical admitted lease before
  stream consumption. Red/green coverage proves both coherent provider
  switching and safe private-provider rejection with complete cleanup.
- Final quality review identified that a noncanonical config-derived provider
  could reach a failure metric before admission. The reviewed fix quarantines
  that initial selection as recoverably unconfigured, emits only fixed safe
  failure copy with no provider metric, and permits a later canonical settings
  publication to recover the service without restart.
- A post-rebase live rerun was unavailable: the installed
  `/opt/homebrew/bin/audiocpp_server` from `audio-cpp 0.4` existed, but there was
  no process, listener, or healthy endpoint. Chatbook did not launch it. This
  is recorded as a live-evidence limitation, not a second UAT result.

- [ ] **Step 8: Commit final evidence**

```bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md \
  "backlog/tasks/task-710 - Make-external-audio.cpp-Console-TTS-settings-coherent.md"
git commit -m "docs: record external audio cpp Console UAT"
```

## Final implementation review checklist

- [x] ADR-023 was amended before runtime code.
- [x] The installed build passed the pinned-contract gate before runtime code.
- [x] Old blank audio.cpp values read compatibly without startup writes.
- [x] Dynamic modes delete exact canonical and legacy aliases in one file
  mutation; exact modes dual-write.
- [x] Textual sentinels never become empty exact values.
- [x] Request selection and revision-matched lease acquisition are atomic.
- [x] Foreground settings completion is bounded.
- [x] One service-owned retained operation performs config mutation off-loop
  and publishes every successful replacement, so caller cancellation or
  post-replacement cache failure cannot strand disk, preferences, and registry
  state.
- [x] Config persistence, encryption, and cache reload do not block the Textual
  event loop.
- [x] Existing speech is not silently cancelled.
- [x] Only the latest pending provider generation becomes active.
- [x] Old and replacement audio.cpp adapters never coexist.
- [x] Revision mismatch, reconfiguring, and unavailable states are distinct
  safe exception types with actionable UI mappings.
- [x] Console audio.cpp uses native `TTSService` and complete WAV.
- [x] All six retained providers remain behind `LegacyTTSAdapter`.
- [x] No synthesis POST retry or fallback was added.
- [x] No managed process behavior was added.
- [x] Focused, broad, repository-wide, static, typing, privacy, and UAT evidence
  is recorded.
