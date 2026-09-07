# TTS Configuration and Voice Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish canonical chat/TTS configuration, explicit OpenAI-compatible authentication, truthful endpoint testing, simplified Settings, and one compact Voice step shared by Quick and Full Setup.

**Architecture:** Extend the existing `TTSEffectiveSettingsResolver` and settings publication revisions instead of creating parallel state. Extract OpenAI-compatible endpoint and authentication rules into a pure TTS module used by Settings, onboarding, probes, and the backend. The onboarding Voice step writes through the existing global TTS save event and updates defaults only after the saved revision is active.

**Tech Stack:** Python 3.11+, Textual, httpx, pytest, pytest-asyncio, TOML settings, existing TTS adapter registry and publication events.

## Global Constraints

- `chat_defaults.provider` and `chat_defaults.model` outrank legacy provider-specific model values.
- TTS authentication is exactly `api_key` or `none`; missing/invalid values fail closed to `api_key`.
- The normalized Official OpenAI origin always requires `api_key`.
- Authentication `none` performs no credential lookup and sends no Authorization header.
- Userinfo and query strings are rejected from configured TTS endpoints.
- Endpoint probes stay on the validated origin, disable redirects, and never guess an unknown path.
- A successful bounded speech sample is authoritative for speech-only services.
- Valid offline configuration is saveable and visibly Needs test.
- Saved, applied, and active runtime revisions remain distinct.
- Use as default never copies endpoints or credentials into a profile.

---

## File Structure

- Create `tldw_chatbook/TTS/openai_compatible_config.py`: pure endpoint/authentication normalization and destination fingerprints.
- Modify `tldw_chatbook/TTS/backends/openai.py`: consume explicit configuration; remove implicit auth behavior when mode is `none`.
- Modify `tldw_chatbook/TTS/adapter_bootstrap.py` and `tldw_chatbook/TTS/legacy_bridge.py`: pass normalized auth and endpoint fields.
- Modify `tldw_chatbook/Chat/console_session_settings.py`: canonical chat resolver and corrected model precedence.
- Modify `tldw_chatbook/TTS/effective_settings.py`: expose saved/applied/active revision provenance consistently.
- Modify `tldw_chatbook/UI/Screens/settings_speech_tts.py`: Settings state, validation, persistence proposal, and test fingerprints.
- Modify `tldw_chatbook/UI/Screens/settings_endpoint_probe.py`: structured endpoint plans and safe probing.
- Modify `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`: auth control, status hierarchy, details disclosure, and focused-input command guard.
- Create `tldw_chatbook/UI/Wizards/first_run_voice_step_state.py`: pure compact Voice-step draft and readiness projection.
- Modify `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`: shared Voice step in both tracks.

### Task 1: Correct canonical chat provider/model precedence

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py:143-220,373-445`
- Modify: `tests/Chat/test_console_session_settings.py`
- Modify: `tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `EffectiveChatConfiguration(provider: str, model: str | None, base_url: str | None, model_source: str)`.
- Produces: `resolve_effective_chat_configuration(app_config, *, provider=None, model=None) -> EffectiveChatConfiguration`.
- Produces: `build_canonical_chat_defaults_mutation(effective) -> dict[str, dict[str, str]]` for explicit Save only.
- Preserves: `build_default_console_session_settings(...) -> ConsoleSessionSettings`, now delegating provider/model/base URL resolution.

- [ ] **Step 1: Write precedence and provenance tests**

```python
def test_chat_defaults_model_outranks_provider_fallback():
    config = {
        "chat_defaults": {"provider": "openai", "model": "chosen-model"},
        "api_settings": {"openai": {"model": "legacy-model"}},
    }
    effective = resolve_effective_chat_configuration(config)
    assert effective.model == "chosen-model"
    assert effective.model_source == "chat_defaults"


def test_explicit_model_outranks_profile_and_global_default():
    config = {
        "chat_defaults": {"provider": "openai", "model": "global-model"},
        "api_settings": {"openai": {"model": "legacy-model"}},
    }
    effective = resolve_effective_chat_configuration(config, model="session-model")
    assert effective.model == "session-model"
    assert effective.model_source == "session"


def test_legacy_provider_alias_reads_without_rewrite_and_save_is_canonical():
    config = {"chat_defaults": {"provider": "OpenAI-Compatible", "model": "pocket-tts"}}
    effective = resolve_effective_chat_configuration(config)
    assert effective.provider == "openai"
    assert config["chat_defaults"]["provider"] == "OpenAI-Compatible"
    mutation = build_canonical_chat_defaults_mutation(effective)
    assert mutation["chat_defaults"]["provider"] == "openai"
```

- [ ] **Step 2: Run the tests and observe the legacy provider value winning**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_session_settings.py tests/UI/test_console_session_settings.py -k "chat_defaults_model or explicit_model" -v`

Expected: FAIL because `provider_settings.model` currently precedes `chat_defaults.model`.

- [ ] **Step 3: Implement the pure resolver and delegate the builder**

```python
@dataclass(frozen=True, slots=True)
class EffectiveChatConfiguration:
    provider: str
    model: str | None
    base_url: str | None
    model_source: str


def resolve_effective_chat_configuration(app_config, *, provider=None, model=None):
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    provider_id = provider_config_key(
        _string_value(provider) or _string_setting(chat_defaults, "provider")
    )
    provider_settings = _provider_settings(app_config, provider_id)
    candidates = (
        ("session", model),
        ("chat_defaults", chat_defaults.get("model")),
        ("provider_fallback", provider_settings.get("model")),
        ("provider_fallback", provider_settings.get("api_model")),
        ("provider_fallback", provider_settings.get("default_model")),
    )
    source, resolved_model = next(
        ((source, value.strip()) for source, value in candidates if isinstance(value, str) and value.strip()),
        ("none", None),
    )
    return EffectiveChatConfiguration(
        provider=provider_id,
        model=resolved_model,
        base_url=_default_base_url(provider_id, provider_settings),
        model_source=source,
    )
```

Keep sampling precedence unchanged: model profile, Console saved defaults, `chat_defaults`, provider fallback. Normalize legacy aliases only in the read model; opening a screen performs no write, and the next explicit Save writes the canonical provider ID.

`build_canonical_chat_defaults_mutation` returns only canonical provider/model
keys and is called by the existing explicit save path. It must not be called by
load, mount, summary, readiness, or handoff code.

- [ ] **Step 4: Run all session-settings tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_session_settings.py tests/UI/test_console_session_settings.py -v`

Expected: PASS with compatibility aliases still readable.

- [ ] **Step 5: Commit the chat resolver**

```bash
git add tldw_chatbook/Chat/console_session_settings.py tests/Chat/test_console_session_settings.py tests/UI/test_console_session_settings.py
git commit -m "fix: honor canonical chat model defaults"
```

### Task 2: Centralize OpenAI-compatible endpoint and authentication rules

**Files:**
- Create: `tldw_chatbook/TTS/openai_compatible_config.py`
- Modify: `tldw_chatbook/TTS/backends/openai.py:18-175`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `tldw_chatbook/TTS/legacy_bridge.py`
- Modify: `tests/TTS/test_openai_compatible_endpoint.py`
- Modify: `tests/TTS/test_legacy_bridge.py`

**Interfaces:**
- Produces: `OpenAIAuthenticationMode(StrEnum)` with `API_KEY="api_key"`, `NONE="none"`.
- Produces: `OpenAICompatibleEndpoint(speech_url: str, origin: str, catalog_url: str | None, official: bool)`.
- Produces: `normalize_openai_compatible_endpoint(raw: str) -> OpenAICompatibleEndpoint`.
- Produces: `normalize_openai_authentication_mode(raw, *, endpoint) -> OpenAIAuthenticationMode`.
- Produces: `openai_destination_fingerprint(provider_id, endpoint) -> str` using SHA-256 over provider and normalized origin only.

- [ ] **Step 1: Write endpoint/authentication and transport tests**

```python
@pytest.mark.parametrize(
    ("raw", "speech", "catalog"),
    [
        ("http://127.0.0.1:8765", "http://127.0.0.1:8765/v1/audio/speech", "http://127.0.0.1:8765/v1/models"),
        ("http://127.0.0.1:8765/v1/audio/speech", "http://127.0.0.1:8765/v1/audio/speech", "http://127.0.0.1:8765/v1/models"),
        ("http://127.0.0.1:8765/custom/speech", "http://127.0.0.1:8765/custom/speech", None),
    ],
)
def test_endpoint_plan(raw, speech, catalog):
    plan = normalize_openai_compatible_endpoint(raw)
    assert (plan.speech_url, plan.catalog_url) == (speech, catalog)


def test_official_origin_rejects_none_authentication():
    endpoint = normalize_openai_compatible_endpoint("https://api.openai.com/v1/audio/speech")
    with pytest.raises(ValueError, match="API key"):
        normalize_openai_authentication_mode("none", endpoint=endpoint)


async def test_none_auth_does_not_read_or_send_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
            "OPENAI_AUTH_MODE": "none",
        },
        requests,
    )
    await _close_replaced_client(backend)
    await _generate(backend, model="pocket-tts", voice="alba")
    assert len(requests) == 1
    assert "Authorization" not in requests[0].headers
```

- [ ] **Step 2: Run endpoint tests and verify implicit credential fallback fails**

Run: `.venv/bin/python -m pytest tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_legacy_bridge.py -v`

Expected: FAIL because the backend currently consults environment/config fallbacks and sends a key to custom endpoints when one exists.

- [ ] **Step 3: Implement exact endpoint/auth contracts**

```python
class OpenAIAuthenticationMode(StrEnum):
    API_KEY = "api_key"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEndpoint:
    speech_url: str
    origin: str
    catalog_url: str | None
    official: bool


def normalize_openai_authentication_mode(raw, *, endpoint):
    mode = OpenAIAuthenticationMode.NONE if raw == "none" else OpenAIAuthenticationMode.API_KEY
    if endpoint.official and mode is OpenAIAuthenticationMode.NONE:
        raise ValueError("Official OpenAI requires an API key")
    return mode
```

Use `urlsplit`/`urlunsplit`, reject userinfo, query, fragment, control characters, and concatenated schemes. Map root, `/v1`, `/v1/models`, and `/v1/chat/completions` to `/v1/audio/speech` plus `/v1/models`; map `/chat/completions` to `/audio/speech` plus `/models`; keep `/v1/audio/speech` unchanged plus `/v1/models`. Preserve an otherwise valid unknown path as the speech URL with no catalog URL. In `OpenAITTSBackend`, resolve credentials only when mode is `API_KEY`; never add Authorization in `NONE` mode. Pass auth mode through adapter bootstrap and legacy bridge.

- [ ] **Step 4: Run backend, bridge, privacy, and request tests**

Run: `.venv/bin/python -m pytest tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_legacy_bridge.py tests/TTS/test_tts_logging_privacy.py -v`

Expected: PASS; captured PocketTTS requests contain exact model/voice/format and no auth header in `none` mode.

- [ ] **Step 5: Commit OpenAI-compatible configuration contracts**

```bash
git add tldw_chatbook/TTS/openai_compatible_config.py tldw_chatbook/TTS/backends/openai.py tldw_chatbook/TTS/adapter_bootstrap.py tldw_chatbook/TTS/legacy_bridge.py tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_legacy_bridge.py
git commit -m "feat: add explicit OpenAI-compatible TTS authentication"
```

### Task 3: Persist authentication mode through global TTS Settings

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py:110-230,430-850,1782-1940`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py:1013-1065,2280-2320,2480-2585`
- Modify: `tests/UI/test_settings_speech_tts_model.py`
- Modify: `tests/UI/test_settings_speech_tts_panel.py`
- Modify: `tests/TTS/test_global_tts_settings_events.py`

**Interfaces:**
- Extends: `GLOBAL_TTS_PROVIDER_FIELD_IDS["openai"]` with `"authentication_mode"`.
- Extends: `GlobalSpeechTTSState.providers["openai"]` with normalized `authentication_mode`.
- Produces: `OpenAIPlaintextConfirmation(origin_fingerprint: str)` stored under `app_tts.OPENAI_NONE_HTTP_CONFIRMATION`.
- Preserves: credential values remain outside ordinary save proposals.

- [ ] **Step 1: Write failing load/save and preset-transition tests**

```python
def test_missing_auth_mode_loads_fail_closed():
    state = load_global_speech_tts_state({"app_tts": {}}, environment={})
    assert state.providers["openai"]["authentication_mode"] == "api_key"


def test_none_auth_does_not_require_credential_for_custom_endpoint():
    state = _state(openai={
        "base_url": "http://127.0.0.1:8765/v1/audio/speech",
        "authentication_mode": "none",
    })
    proposal = build_global_speech_tts_save_proposal(state)
    assert proposal.settings["OPENAI_AUTH_MODE"] == "none"


def test_official_preset_resets_none_before_save():
    state = _state(openai={"base_url": OFFICIAL_URL, "authentication_mode": "none"})
    with pytest.raises(GlobalSpeechTTSValidationError):
        build_global_speech_tts_save_proposal(state)
```

- [ ] **Step 2: Run Settings model tests and verify missing field ownership**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_speech_tts_model.py tests/TTS/test_global_tts_settings_events.py -k "auth or credential" -v`

Expected: FAIL because auth mode is absent and missing credentials universally mark OpenAI incomplete.

- [ ] **Step 3: Add auth mode to state, validation, save proposals, and UI**

```python
GLOBAL_TTS_PROVIDER_FIELD_IDS = MappingProxyType({
    **dict(GLOBAL_TTS_PROVIDER_FIELD_IDS),
    "openai": ("credential", "authentication_mode", "base_url", "organization_id"),
})
```

Render a two-option `Select` labeled Authentication with `API key` and `None`. The Official OpenAI preset sets API key before updating the endpoint. Require origin-bound plaintext confirmation only for `none` plus non-loopback HTTP; changing origin/auth invalidates it. The confirmation stores only `openai_destination_fingerprint("openai", endpoint)`.

- [ ] **Step 4: Run Settings model, panel, ownership, and save-event tests**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_speech_tts_model.py tests/UI/test_settings_speech_tts_panel.py tests/TTS/test_global_tts_settings_events.py tests/TTS/test_speech_tts_settings_ownership_hardening.py -v`

Expected: PASS; ordinary Save still cannot mutate credential values.

- [ ] **Step 5: Commit Settings authentication**

```bash
git add tldw_chatbook/UI/Screens/settings_speech_tts.py tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tests/UI/test_settings_speech_tts_model.py tests/UI/test_settings_speech_tts_panel.py tests/TTS/test_global_tts_settings_events.py
git commit -m "feat: expose TTS authentication mode in Settings"
```

### Task 4: Make endpoint readiness and testing truthful

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_endpoint_probe.py`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_contracts.py:25-190`
- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py`
- Create: `tests/UI/test_settings_endpoint_probe.py`
- Modify: `tests/UI/test_settings_speech_tts_model.py`
- Modify: `tests/TTS/test_openai_compatible_endpoint.py`

**Interfaces:**
- Produces: `SpeechTTSConnectionState` values `REACHABLE`, `UNREACHABLE`, `NOT_TESTED`, `UNSUPPORTED`.
- Extends: `SettingsEndpointProbeOutcome` with `state`, `operation`, and optional `model_count`.
- Produces: `ProviderTestFingerprint(provider_id, normalized_fields, saved_revision)`; SHA-256 contains no credential.
- Produces: `combine_tts_readiness(configuration, catalog, sample) -> SpeechTTSReadinessProjection`, where successful sample evidence outranks an unsupported catalog operation.

- [ ] **Step 1: Write failing malformed-path, redirect, and speech-only tests**

```python
async def test_unknown_path_is_not_extended_or_requested():
    requests: list[httpx.Request] = []
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, request=request)
    transport = httpx.MockTransport(unexpected_request)
    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await probe_settings_endpoint(
            "http://127.0.0.1:8765/custom/speech", http_client=client
        )
    assert outcome.state is SpeechTTSConnectionState.NOT_TESTED
    assert requests == []


async def test_cross_origin_redirect_is_not_followed():
    requests: list[httpx.Request] = []
    def redirect(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302,
            headers={"Location": "http://127.0.0.2:9876/v1/models"},
            request=request,
        )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(redirect), follow_redirects=False
    ) as client:
        outcome = await probe_settings_endpoint(
            "http://127.0.0.1:8765", http_client=client
        )
    assert outcome.state is SpeechTTSConnectionState.UNREACHABLE
    assert len(requests) == 1


def test_successful_sample_outranks_unsupported_catalog():
    readiness = combine_tts_readiness(configuration="valid", catalog="unsupported", sample="success")
    assert readiness.connection is SpeechTTSConnectionState.REACHABLE
```

- [ ] **Step 2: Run probe tests and reproduce blind `/v1/models` behavior**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_endpoint_probe.py tests/TTS/test_openai_compatible_endpoint.py -v`

Expected: FAIL because the current probe always appends `/v1/models` and returns a boolean reachability claim.

- [ ] **Step 3: Implement structured readiness and process-scoped evidence**

```python
class SpeechTTSConnectionState(StrEnum):
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    NOT_TESTED = "not_tested"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ProviderTestFingerprint:
    provider_id: str
    normalized_fields: tuple[tuple[str, str], ...]
    saved_revision: int
```

Probe only a declared `catalog_url`, construct `httpx.AsyncClient(follow_redirects=False)`, apply shared URL validation, and compare response request origin with the validated origin. Return Not tested for unknown paths. Record successful sample evidence in memory against the fingerprint; unchanged saves preserve it within the process and changed fields/revisions invalidate it. Do not restore evidence after restart.

- [ ] **Step 4: Run readiness, status, and endpoint tests**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_endpoint_probe.py tests/UI/test_settings_speech_tts_model.py tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_tts_settings_capability_observations.py -v`

Expected: PASS with connection and local configuration reported independently.

- [ ] **Step 5: Commit structured testing**

```bash
git add tldw_chatbook/UI/Screens/settings_endpoint_probe.py tldw_chatbook/UI/Speech/speech_settings_contracts.py tldw_chatbook/UI/Screens/settings_speech_tts.py tests/UI/test_settings_endpoint_probe.py tests/UI/test_settings_speech_tts_model.py tests/TTS/test_openai_compatible_endpoint.py
git commit -m "fix: report TTS endpoint readiness accurately"
```

### Task 5: Preserve saved/applied/active revision honesty and default atomicity

**Files:**
- Modify: `tldw_chatbook/TTS/effective_settings.py:260-335,1267-1510`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py:1985-2015,2580-2690`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py:1650-1835`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py:3170-3490`
- Modify: `tests/TTS/test_effective_settings.py`
- Modify: `tests/TTS/test_stts_settings_reconfiguration.py`
- Modify: `tests/TTS/test_global_tts_settings_events.py`

**Interfaces:**
- Extends: `TTSEffectiveSelectionRevisions` with explicit `provider_saved`, `provider_applied`, and `provider_active` accessors while retaining `provider_configuration` compatibility.
- Produces: `tts_configuration_is_active(service, provider_id, saved_revision) -> bool`.
- Produces: `commit_voice_setup_default(preferences, *, expected_saved_revision) -> bool`.

- [ ] **Step 1: Write failure-path tests**

```python
async def test_failed_adapter_handoff_keeps_previous_active_selection():
    result = await publish_settings_with_failing_adapter(new_preferences)
    assert result.persistence.file_replaced is True
    assert service.saved_configuration_revision("openai") > service.applied_configuration_revision("openai")
    assert result.provider_runtime_revisions["openai"] == prior_runtime_revision
    assert resolver.active_preferences() == old_preferences


async def test_use_as_default_waits_for_matching_active_revision():
    changed = await commit_voice_setup_default(new_preferences, expected_saved_revision=7)
    assert changed is False
    assert read_global_preferences() == old_preferences
```

- [ ] **Step 2: Run revision and reconfiguration tests**

Run: `.venv/bin/python -m pytest tests/TTS/test_effective_settings.py tests/TTS/test_stts_settings_reconfiguration.py tests/TTS/test_global_tts_settings_events.py -k "revision or handoff or active" -v`

Expected: FAIL where saved draft state is projected as active or defaults update before handoff completion.

- [ ] **Step 3: Add revision comparison and activation-gated default commit**

```python
def tts_configuration_is_active(service, provider_id, saved_revision):
    return (
        service.saved_configuration_revision(provider_id) == saved_revision
        and service.applied_configuration_revision(provider_id) == saved_revision
    )
```

Keep the existing publication generation as the saved/applied identity and the registry revision as the separate active-runtime freshness identity; never compare those unlike counters for numeric equality. On adapter failure, render `Saved, activation failed` and retain the previous active resolver snapshot. The Voice setup callback updates global provider/model/voice/format/speed only after `tts_configuration_is_active` succeeds; otherwise preserve prior defaults and retain the draft for Retry.

- [ ] **Step 4: Run all publication/effective-setting tests**

Run: `.venv/bin/python -m pytest tests/TTS/test_effective_settings.py tests/TTS/test_stts_settings_reconfiguration.py tests/TTS/test_global_tts_settings_events.py tests/TTS/test_tts_registry_service.py -v`

Expected: PASS with stale revisions rejected.

- [ ] **Step 5: Commit revision/default safeguards**

```bash
git add tldw_chatbook/TTS/effective_settings.py tldw_chatbook/TTS/TTS_Generation.py tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tests/TTS/test_effective_settings.py tests/TTS/test_stts_settings_reconfiguration.py tests/TTS/test_global_tts_settings_events.py
git commit -m "fix: activate TTS defaults only after provider handoff"
```

### Task 6: Simplify the primary Settings view and guard focused inputs

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py:1680-1765,4060-4190`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tests/UI/test_settings_speech_tts_panel.py`
- Modify: `tests/UI/test_speech_tts_settings_ownership_closeout.py`

**Interfaces:**
- Produces: collapsed `Collapsible` controls `settings-speech-details` and `settings-speech-scope-inspector`.
- Produces: `SpeechTTSSettingsPanel.command_allowed(command: str) -> bool`, returning false for letter shortcuts while an `Input` or `TextArea` owns focus.
- Preserves: clickable Save and Revert while any field is focused.

- [ ] **Step 1: Write failing hierarchy and keystroke tests**

```python
async def test_details_and_scope_start_collapsed_on_every_mount():
    async with _panel_app().run_test() as pilot:
        assert panel.query_one("#settings-speech-details", Collapsible).collapsed
        assert panel.query_one("#settings-speech-scope-inspector", Collapsible).collapsed
        assert "revision" not in _visible_primary_copy(panel).lower()


async def test_letter_shortcut_does_not_mutate_focused_endpoint():
    endpoint = panel.query_one("#settings-speech-openai-base-url", Input)
    endpoint.value = "http://127.0.0.1:8765/v1/audio/speech"
    endpoint.focus()
    await pilot.press("s")
    assert endpoint.value == "http://127.0.0.1:8765/v1/audio/speechs"
    assert save_event_count == 0
```

- [ ] **Step 2: Run panel tests and verify inspector/shortcut failures**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_speech_tts_panel.py tests/UI/test_speech_tts_settings_ownership_closeout.py -k "collapsed or shortcut or endpoint" -v`

Expected: FAIL until details are collapsed and app commands defer to text input.

- [ ] **Step 3: Move technical state behind disclosures and add command gating**

Render task language and current status in the primary provider block. Move owner IDs, raw keys, provenance, revisions, and Scope Inspector rows into collapsed `Collapsible` sections with `collapsed=True` on every construction. In key handling, let focused text-entry widgets consume printable keys before checking panel shortcuts; keep button handlers unchanged.

- [ ] **Step 4: Run Settings UI tests at desktop and narrow sizes**

Run: `.venv/bin/python -m pytest tests/UI/test_settings_speech_tts_panel.py tests/UI/test_speech_tts_settings_ownership_closeout.py -v`

Expected: PASS with Save/Revert reachable by click and no primary raw-key copy.

- [ ] **Step 5: Commit Settings hierarchy changes**

```bash
git add tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tldw_chatbook/css/components/_agentic_terminal.tcss tests/UI/test_settings_speech_tts_panel.py tests/UI/test_speech_tts_settings_ownership_closeout.py
git commit -m "fix: simplify global TTS Settings workflow"
```

### Task 7: Add the compact Voice step to Quick and Full Setup

**Files:**
- Create: `tldw_chatbook/UI/Wizards/first_run_voice_step_state.py`
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py:246-275`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:1050-1550,3100-3950`
- Modify: `tldw_chatbook/css/features/_wizards.tcss`
- Create: `tests/Wizards/test_first_run_voice_step_state.py`
- Modify: `tests/Wizards/test_first_run_setup_wizard.py`
- Modify: `tests/UI/test_first_run_wizard_live_contract.py`
- Create: `tests/integration/test_first_run_pocket_tts_flow.py`

**Interfaces:**
- Produces: `VoiceSetupDraft(endpoint, authentication_mode, model_id, voice_id, response_format, speed, sample_text, use_as_default)`.
- Produces: `validate_voice_setup_draft(draft) -> VoiceSetupValidation` with local validity independent from connection state.
- Produces: `validate_voice_sample_text(value: object) -> str`, returning trimmed text or raising `ValueError` outside 1-500 characters.
- Produces: `VoiceSetupStep.commit() -> tuple[bool, str]` using the global TTS save event and activation-gated default command.
- Produces in the new integration module: a loopback `fake_pocket_tts` fixture with `url` and captured `requests`, plus `run_quick_voice_setup(endpoint)` that drives the real wizard controls and returns the resulting effective defaults.
- Extends: `_QUICK_TRACK` and `_FULL_TRACK` with `STEP_VOICE` immediately after `STEP_MODEL`.

- [ ] **Step 1: Write draft, track, and end-to-end failing tests**

```python
def test_quick_track_contains_five_non_secret_steps():
    assert active_step_ids(TRACK_QUICK, key_entered=False) == (
        STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_VOICE, STEP_SUMMARY
    )


def test_pocket_tts_draft_is_valid_without_api_key():
    draft = VoiceSetupDraft(
        endpoint="http://127.0.0.1:8765/v1/audio/speech",
        authentication_mode="none",
        model_id="pocket-tts",
        voice_id="alba",
        response_format="wav",
        speed=1.0,
        sample_text="Hello from Chatbook.",
        use_as_default=True,
    )
    assert validate_voice_setup_draft(draft).configuration_valid


def test_voice_sample_rejects_blank_and_overlong_text():
    with pytest.raises(ValueError, match="1 to 500"):
        validate_voice_sample_text("   ")
    with pytest.raises(ValueError, match="1 to 500"):
        validate_voice_sample_text("x" * 501)


async def test_quick_voice_setup_sends_exact_sample_and_sets_default(fake_pocket_tts):
    result = await run_quick_voice_setup(fake_pocket_tts.url)
    assert result.default.provider_id == "openai"
    assert fake_pocket_tts.requests[-1].json()["model"] == "pocket-tts"
    assert "Authorization" not in fake_pocket_tts.requests[-1].headers
```

- [ ] **Step 2: Run the new tests and verify the Voice step is absent**

Run: `.venv/bin/python -m pytest tests/Wizards/test_first_run_voice_step_state.py tests/Wizards/test_first_run_setup_wizard.py tests/integration/test_first_run_pocket_tts_flow.py -k "voice or pocket" -v`

Expected: FAIL because `STEP_VOICE` and compact draft do not exist.

- [ ] **Step 3: Implement one shared compact Voice step**

Render, in order: preset (`PocketTTS`, `Official OpenAI`, `Custom compatible`), endpoint, Authentication segmented select, model, voice, sample text, `Test and Hear`, status, and `Use as default` checkbox. Bound trimmed sample text to 1-500 characters and show the count at the field; invalid sample text disables Test and Hear but does not erase other fields. Keep model/voice manually editable when catalog discovery is unsupported. A successful sample records process-scoped Verified; a failed sample keeps locally valid Save enabled and shows Needs test. Store only non-secret fields in the setup draft.

On commit, post the same `STTSSettingsSaveEvent` used by Settings. If `use_as_default` is checked, wait for the matching saved/applied publication generation and a current active-runtime snapshot, then write exact global axes. Do not compare the runtime registry revision numerically with the publication generation. On failure, retain fields and previous defaults.

- [ ] **Step 4: Run wizard, Settings, backend, and integration tests**

Run: `.venv/bin/python -m pytest tests/Wizards tests/UI/test_first_run_wizard_live_contract.py tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_global_tts_settings_events.py tests/integration/test_first_run_pocket_tts_flow.py -v`

Expected: PASS for Quick and Full tracks, manual/offline save, exact sample payload, and default assignment.

- [ ] **Step 5: Commit Voice setup**

```bash
git add tldw_chatbook/UI/Wizards/first_run_voice_step_state.py tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss tests/Wizards/test_first_run_voice_step_state.py tests/Wizards/test_first_run_setup_wizard.py tests/UI/test_first_run_wizard_live_contract.py tests/integration/test_first_run_pocket_tts_flow.py
git commit -m "feat: add PocketTTS voice setup to onboarding"
```

## Plan Verification

Run: `.venv/bin/python -m pytest tests/Chat/test_console_session_settings.py tests/UI/test_console_session_settings.py tests/TTS/test_openai_compatible_endpoint.py tests/TTS/test_effective_settings.py tests/TTS/test_stts_settings_reconfiguration.py tests/TTS/test_global_tts_settings_events.py tests/UI/test_settings_endpoint_probe.py tests/UI/test_settings_speech_tts_model.py tests/UI/test_settings_speech_tts_panel.py tests/Wizards tests/integration/test_first_run_pocket_tts_flow.py -v`

Manual checkpoint: configure `http://127.0.0.1:8765/v1/audio/speech`, Authentication None, model `pocket-tts`, a real voice, WAV, Test and Hear, and Use as default. Confirm no dummy key, no malformed models URL, exact payload, audible sample, and truthful Saved/Needs test/Verified copy.
