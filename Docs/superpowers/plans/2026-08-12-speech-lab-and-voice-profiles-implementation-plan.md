# Speech Lab and Voice Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Speech Lab reflect current global TTS configuration, verify provider-neutral OpenAI-compatible profiles, separate Voice Profiles from Kokoro Voice Blends, and remove misleading dependency/resource warnings.

**Architecture:** Reuse the existing provider-configuration-changed message, effective settings resolver, immutable profile repository, and generation artifacts. Add only a process-scoped profile evidence cache to `TTSProfileService`; no new profile persistence schema. Keep Kokoro blend JSON and provider-neutral profiles as separate navigation destinations.

**Tech Stack:** Python 3.11+, Textual, pytest, existing TTS profile repository/service, importlib.resources, setuptools package data.

## Global Constraints

- Speech Lab resolves fresh global and Studio state after successful configuration publication.
- Provider-neutral profiles store provider and synthesis axes, never endpoint URLs or credentials.
- Non-audio.cpp profiles cannot be labeled Verified without successful current-process evidence.
- Profile evidence is keyed by profile ID/revision, exact synthesis axes, and active provider configuration revision.
- Profiles remain usable after restart but return to Needs test unless capabilities re-establish availability.
- Voice Profiles and Kokoro Voice Blends remain separate tools.
- Missing local packages do not imply a configured OpenAI-compatible endpoint is unavailable.
- Installed distributions include `openai_tts_mappings.json`.

---

## File Structure

- Modify `tldw_chatbook/UI/STTS_Window.py`: refresh retained global preferences and destination labels.
- Modify `tldw_chatbook/UI/Speech/speech_settings_pane.py` and `speech_playground_pane.py`: apply fresh global defaults without reconstructing the whole window.
- Modify `tldw_chatbook/TTS/profile_types.py`: bounded process-local verification evidence value.
- Modify `tldw_chatbook/TTS/profile_service.py`: provider-correct revisions, evidence cache, and OpenAI-compatible availability.
- Modify `tldw_chatbook/UI/stts_profile_library.py`: truthful Verified/Needs test presentation and recovery.
- Modify `tldw_chatbook/UI/Speech/speech_settings_mixin.py`: Kokoro-only Voice Blends copy.
- Modify `tldw_chatbook/UI/Lab_Modules/lab_speech_status.py` and `speech_runtime_status.py`: capability-specific dependency messaging.
- Create `tldw_chatbook/Config_Files/openai_tts_mappings.json`: packaged canonical mapping resource.

### Task 1: Refresh Speech Lab global preferences after Settings publication

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py:494-505,1735-1760,2258-2275`
- Modify: `tldw_chatbook/UI/STTS_Window.py:100-180,1290-1380`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_pane.py:211-290,580-805`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py:287-485`
- Modify: `tests/UI/test_speech_playground_pane.py`
- Modify: `tests/UI/test_studio_tts_preferences.py`
- Modify: `tests/TTS/test_global_tts_settings_events.py`

**Interfaces:**
- Extends: `STTSProviderConfigurationChanged(provider_id, configuration_revision, global_preferences_revision=None)` so existing two-argument publishers remain compatible during migration.
- Produces: `SpeechSettingsPane.refresh_global_preferences(snapshot: TTSPreferencesSnapshot) -> None`.
- Produces: `SpeechPlaygroundPane.refresh_global_preferences(snapshot: TTSPreferencesSnapshot) -> None`.

- [ ] **Step 1: Write stale-cache regression tests**

```python
async def test_mounted_speech_lab_refreshes_global_defaults_after_save():
    pane = SpeechSettingsPane(global_preferences=_preferences("openai", "old", "old-voice"))
    async with _host(pane).run_test() as pilot:
        pane.refresh_global_preferences(_preferences("openai", "pocket-tts", "alba"))
        await pilot.pause()
        assert pane._global_preferences.model_id == "pocket-tts"
        assert pane._global_preferences.voice_id == "alba"


def test_configuration_changed_message_carries_global_revision():
    message = STTSProviderConfigurationChanged("openai", 8, 12)
    assert message.global_preferences_revision == 12
```

- [ ] **Step 2: Run the focused synchronization tests**

Run: `.venv/bin/python -m pytest tests/UI/test_speech_playground_pane.py tests/UI/test_studio_tts_preferences.py tests/TTS/test_global_tts_settings_events.py -k "refreshes_global or configuration_changed" -v`

Expected: FAIL because the retained pane keeps constructor-time `_global_preferences`.

- [ ] **Step 3: Refresh through the existing publication signal**

```python
def refresh_global_preferences(self, snapshot: TTSPreferencesSnapshot) -> None:
    if type(snapshot) is not TTSPreferencesSnapshot:
        raise TypeError("global preferences must be a TTS preferences snapshot")
    self._global_preferences = snapshot
    self._apply_global_preferences_to_inherited_controls()
```

After a successful settings publication, include the global preference revision in `STTSProviderConfigurationChanged`. The retained STTS handler reloads one `TTSPreferencesSnapshot` and calls refresh on mounted settings/playground panes. Preserve explicit Studio overrides; update only inherited axes and their displayed source labels.

- [ ] **Step 4: Run Speech Lab and publication tests**

Run: `.venv/bin/python -m pytest tests/UI/test_speech_playground_pane.py tests/UI/test_studio_tts_preferences.py tests/TTS/test_global_tts_settings_events.py tests/TTS/test_effective_settings.py -v`

Expected: PASS; provider, model, voice, format, speed, and source are current without reopening Lab.

- [ ] **Step 5: Commit live synchronization**

```bash
git add tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py tldw_chatbook/UI/STTS_Window.py tldw_chatbook/UI/Speech/speech_settings_pane.py tldw_chatbook/UI/Speech/speech_playground_pane.py tests/UI/test_speech_playground_pane.py tests/UI/test_studio_tts_preferences.py tests/TTS/test_global_tts_settings_events.py
git commit -m "fix: refresh Speech Lab after TTS settings changes"
```

### Task 2: Add provider-neutral process-scoped profile verification

**Files:**
- Modify: `tldw_chatbook/TTS/profile_types.py:330-430`
- Modify: `tldw_chatbook/TTS/profile_service.py:795-835,1366-1550,1700-1785,2535-2625`
- Modify: `tests/TTS/test_profile_types.py`
- Modify: `tests/TTS/test_profile_service.py`
- Modify: `tests/TTS/test_tts_profile_capabilities.py`

**Interfaces:**
- Produces: `TTSProfileVerificationEvidence(profile_id, profile_revision, provider_id, model_id, voice_id, response_format, speed, options_fingerprint, provider_configuration_revision)`.
- Produces: `TTSProfileService.record_sample_evidence(loaded: LoadedTTSProfile, artifact: STTSGeneratedAudio) -> None`.
- Extends: `TTSProfileAvailability` with `provider_configuration_revision: int | None`; newly observed rows always carry the exact provider's revision while the existing snapshot-level `configuration_revision` remains the audio.cpp compatibility value.
- Changes: `_current_configuration_revision(provider_id: str) -> int`; no hard-coded audio.cpp lookup.
- Preserves: repository schema, immutable profile rows, and `create_from_artifact` return type.

- [ ] **Step 1: Write provider revision and evidence tests**

```python
async def test_openai_profile_created_from_sample_is_available_this_process(service, artifact):
    loaded = await service.create_from_artifact("Pocket Alba", artifact)
    page = await service.list_profiles(limit=20, offset=0)
    availability = await service.observe_availability(page)
    row = next(
        item for item in availability.profiles
        if item.profile_id == loaded.profile.profile_id
    )
    assert row.state == "available"
    assert row.provider_configuration_revision == artifact.requested_selection.configuration_revision


async def test_openai_profile_evidence_invalidates_on_revision_change(service, artifact):
    loaded = await service.create_from_artifact("Pocket Alba", artifact)
    service._tts_service.set_revision("openai", artifact.requested_selection.configuration_revision + 1)
    page = await service.list_profiles(limit=20, offset=0)
    availability = await service.observe_availability(page)
    row = next(
        item for item in availability.profiles
        if item.profile_id == loaded.profile.profile_id
    )
    assert row.state == "unverified"


def test_current_revision_reads_requested_provider(service):
    service._tts_service.revisions = {"audio_cpp": 2, "openai": 9}
    assert service._current_configuration_revision("openai") == 9
```

- [ ] **Step 2: Run profile tests and observe non-audio.cpp remains Unverified**

Run: `.venv/bin/python -m pytest tests/TTS/test_profile_service.py tests/TTS/test_tts_profile_capabilities.py -k "openai_profile or current_revision" -v`

Expected: FAIL because availability short-circuits non-audio.cpp profiles and the revision helper is audio.cpp-only.

- [ ] **Step 3: Implement exact in-memory evidence admission**

```python
@dataclass(frozen=True, slots=True)
class TTSProfileVerificationEvidence:
    profile_id: UUID
    profile_revision: int
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options_fingerprint: str
    provider_configuration_revision: int


def profile_options_fingerprint(options: JsonOptions) -> str:
    payload = canonical_json_options(options).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
```

Initialize `self._sample_evidence: dict[UUID, TTSProfileVerificationEvidence] = {}`. `record_sample_evidence` accepts only a successful `STTSGeneratedAudio` whose requested selection exactly matches the loaded profile and current provider revision. `create_from_artifact` calls it after repository commit. `observe_availability` uses native capability classification for audio.cpp and exact evidence matching for other structurally supported providers. Profile edit/delete remove stale entries; a newly constructed service starts with an empty cache.

- [ ] **Step 4: Run profile lifecycle, portability, and capability tests**

Run: `.venv/bin/python -m pytest tests/TTS/test_profile_types.py tests/TTS/test_profile_service.py tests/TTS/test_tts_profile_capabilities.py tests/TTS/test_profile_portability.py -v`

Expected: PASS; imported/restarted non-native profiles are Needs test, not falsely Verified.

- [ ] **Step 5: Commit provider-neutral verification**

```bash
git add tldw_chatbook/TTS/profile_types.py tldw_chatbook/TTS/profile_service.py tests/TTS/test_profile_types.py tests/TTS/test_profile_service.py tests/TTS/test_tts_profile_capabilities.py
git commit -m "feat: verify OpenAI-compatible voice profiles from samples"
```

### Task 3: Present profile status truthfully in the library and Playground

**Files:**
- Modify: `tldw_chatbook/UI/stts_profile_library.py:96-165,1100-1650`
- Modify: `tldw_chatbook/UI/Speech/speech_profile_mixin.py:200-500`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py:2420-2785`
- Modify: `tests/UI/test_stts_profile_library.py`
- Modify: `tests/UI/test_speech_profile_port.py`
- Modify: `tests/UI/test_speech_playground_pane.py`

**Interfaces:**
- Produces UI states: `Verified`, `Needs test`, and `Unavailable` mapped from `TTSProfileAvailability`.
- Produces recovery: Open in Playground with exact provider/model/voice/format/speed for Needs test.
- Consumes: `TTSPlaygroundSelectionPreset` and process-scoped evidence from Task 2.

- [ ] **Step 1: Write profile status and recovery tests**

```python
async def test_unverified_openai_profile_reads_needs_test_and_opens_exact_preset():
    screen = _profile_library_with(_openai_profile(), availability="unverified")
    async with _host(screen).run_test() as pilot:
        assert "Needs test" in _row_copy(screen)
        await pilot.click("#stts-profile-test")
        assert screen.posted_preset.provider_id == "openai"
        assert screen.posted_preset.model_id == "pocket-tts"
        assert screen.posted_preset.voice_id == "alba"


async def test_successful_profile_sample_updates_row_to_verified():
    screen = _profile_library_with(_openai_profile(), availability="unverified")
    await screen.apply_successful_sample(_matching_artifact())
    assert "Verified" in _row_copy(screen)
```

- [ ] **Step 2: Run UI profile tests and confirm generic recovery is missing**

Run: `.venv/bin/python -m pytest tests/UI/test_stts_profile_library.py tests/UI/test_speech_profile_port.py tests/UI/test_speech_playground_pane.py -k "needs_test or successful_profile_sample" -v`

Expected: FAIL until non-audio.cpp recovery and row refresh consume the new evidence.

- [ ] **Step 3: Wire exact profile testing and bounded copy**

Map `available -> Verified`, `unverified -> Needs test`, and `unavailable -> Unavailable`. For Needs test, open the Playground with an exact immutable preset; disable Save-as-profile until one successful matching artifact exists. After sample completion, call `record_sample_evidence`, refresh only the affected library row, and preserve the selected profile.

- [ ] **Step 4: Run complete profile UI tests**

Run: `.venv/bin/python -m pytest tests/UI/test_stts_profile_library.py tests/UI/test_speech_profile_port.py tests/UI/test_speech_profile_navigation.py tests/UI/test_speech_playground_pane.py -v`

Expected: PASS with no endpoint or credential exposed in profile rows.

- [ ] **Step 5: Commit profile status UX**

```bash
git add tldw_chatbook/UI/stts_profile_library.py tldw_chatbook/UI/Speech/speech_profile_mixin.py tldw_chatbook/UI/Speech/speech_playground_pane.py tests/UI/test_stts_profile_library.py tests/UI/test_speech_profile_port.py tests/UI/test_speech_playground_pane.py
git commit -m "fix: show truthful voice profile verification status"
```

### Task 4: Separate Voice Profiles from Kokoro Voice Blends in navigation

**Files:**
- Modify: `tldw_chatbook/UI/STTS_Window.py:900-980,1640-1695`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_pane.py:70-115,450-500`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_mixin.py:360-405,500-560,830-925`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_group.py`
- Modify: `tests/UI/test_speech_profile_navigation.py`
- Modify: `tests/UI/test_voice_blend_dialog.py`

**Interfaces:**
- Produces navigation destinations with exact labels `Voice Profiles` and `Voice Blends`.
- Preserves: Kokoro blend IDs use `blend:<name>` only inside Kokoro-capable controls.
- Preserves: provider-neutral profile IDs remain UUID-backed profile selections.

- [ ] **Step 1: Write navigation identity tests**

```python
def test_voice_profiles_action_never_opens_blend_editor():
    target = resolve_speech_navigation("voice-profiles")
    assert target.view == "profiles"
    assert target.provider_id is None


def test_voice_blends_are_labeled_and_scoped_to_kokoro():
    target = resolve_speech_navigation("voice-blends")
    assert target.view == "blends"
    assert target.provider_id == "kokoro"
```

- [ ] **Step 2: Run profile/blend navigation tests**

Run: `.venv/bin/python -m pytest tests/UI/test_speech_profile_navigation.py tests/UI/test_voice_blend_dialog.py -v`

Expected: FAIL where generic labels or routes still point to Kokoro blend management.

- [ ] **Step 3: Apply distinct labels, routes, and control scopes**

Use separate destination IDs and headings. `Voice Profiles` opens `STTSProfileLibrary`; `Voice Blends` opens the existing Kokoro section and dialog. Remove copy that describes a blend as a profile. Keep blend choices out of non-Kokoro provider voice selectors.

- [ ] **Step 4: Run Speech settings/navigation tests**

Run: `.venv/bin/python -m pytest tests/UI/test_speech_profile_navigation.py tests/UI/test_voice_blend_dialog.py tests/UI/test_speech_playground_pane.py tests/UI/test_stts_profile_library.py -v`

Expected: PASS with back navigation returning to the originating view.

- [ ] **Step 5: Commit profile/blend separation**

```bash
git add tldw_chatbook/UI/STTS_Window.py tldw_chatbook/UI/Speech/speech_settings_pane.py tldw_chatbook/UI/Speech/speech_settings_mixin.py tldw_chatbook/UI/Speech/speech_settings_group.py tests/UI/test_speech_profile_navigation.py tests/UI/test_voice_blend_dialog.py
git commit -m "fix: separate voice profiles from Kokoro blends"
```

### Task 5: Correct dependency messaging and package the OpenAI mapping resource

**Files:**
- Modify: `tldw_chatbook/UI/Lab_Modules/lab_speech_status.py:45-120`
- Modify: `tldw_chatbook/UI/Speech/speech_runtime_status.py:70-140,520-710`
- Modify: `tldw_chatbook/UI/STTS_Window.py:1320-1370`
- Create: `tldw_chatbook/Config_Files/openai_tts_mappings.json`
- Modify: `tldw_chatbook/config.py:417-490`
- Modify: `tests/UI/test_stts_capability_state.py`
- Modify: `tests/UI/test_speech_runtime_status.py`
- Create: `tests/Packaging/test_openai_tts_mapping_resource.py`

**Interfaces:**
- Preserves: `SpeechLocalDependencyAvailability` independent booleans for STT, Kokoro, Chatterbox, and Higgs.
- Produces: installed resource `tldw_chatbook.Config_Files/openai_tts_mappings.json` readable through `importlib.resources`.
- Changes fallback logging from warning to informational when built-ins remain usable.

- [ ] **Step 1: Write capability-copy and built-wheel resource tests**

Add `json`, `subprocess`, `sys`, `zipfile`, and `Path` imports directly to the
new packaging test module; it has no shared wheel fixture.

```python
def test_remote_openai_compatible_remains_available_without_local_packages():
    projection = build_speech_runtime_projection(
        provider_id="openai",
        local_dependencies=_all_missing(),
        runtime_status=_ready_openai_status(),
    )
    assert projection.runtime_ready
    assert "OpenAI-compatible" in projection.summary


def test_built_wheel_contains_openai_mapping_resource(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(repository),
        ],
        check=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        payload = archive.read(
            "tldw_chatbook/Config_Files/openai_tts_mappings.json"
        )
    assert json.loads(payload)["models"]["tts-1"]
```

- [ ] **Step 2: Run capability and packaging tests**

Run: `.venv/bin/python -m pytest tests/UI/test_stts_capability_state.py tests/UI/test_speech_runtime_status.py tests/Packaging/test_openai_tts_mapping_resource.py -v`

Expected: FAIL because the JSON resource is absent and combined local dependency copy can imply remote TTS is blocked.

- [ ] **Step 3: Add exact capability copy and canonical JSON resource**

Create JSON with the current built-in `models` and `voices` mappings from `config.py`. Keep `[tool.setuptools.package-data]."tldw_chatbook.Config_Files" = ["*.json", ...]` as the packaging authority. Update copy to name Local transcription, Local Kokoro, Local Chatterbox, and Local Higgs separately, followed by the exact optional extra. Do not gate OpenAI-compatible readiness on those flags. Log fallback mapping use at INFO without raw resource exception text.

- [ ] **Step 4: Build a wheel and run the installed-resource test**

Run: `.venv/bin/python -m build --wheel --outdir /tmp/tldw-wheel-test`

Expected: one wheel builds successfully.

Run: `.venv/bin/python -m pytest tests/UI/test_stts_capability_state.py tests/UI/test_speech_runtime_status.py tests/Packaging/test_openai_tts_mapping_resource.py -v`

Expected: PASS; the test builds and inspects the wheel itself without an external fixture.

- [ ] **Step 5: Commit dependency/resource fixes**

```bash
git add tldw_chatbook/UI/Lab_Modules/lab_speech_status.py tldw_chatbook/UI/Speech/speech_runtime_status.py tldw_chatbook/UI/STTS_Window.py tldw_chatbook/Config_Files/openai_tts_mappings.json tldw_chatbook/config.py tests/UI/test_stts_capability_state.py tests/UI/test_speech_runtime_status.py tests/Packaging/test_openai_tts_mapping_resource.py
git commit -m "fix: clarify speech dependencies and package TTS mappings"
```

## Plan Verification

Run: `.venv/bin/python -m pytest tests/TTS/test_profile_types.py tests/TTS/test_profile_service.py tests/TTS/test_tts_profile_capabilities.py tests/UI/test_speech_playground_pane.py tests/UI/test_studio_tts_preferences.py tests/UI/test_stts_profile_library.py tests/UI/test_speech_profile_navigation.py tests/UI/test_voice_blend_dialog.py tests/UI/test_stts_capability_state.py tests/UI/test_speech_runtime_status.py tests/Packaging/test_openai_tts_mapping_resource.py -v`

Manual checkpoint: leave Speech Lab open, change the global PocketTTS model/voice in Settings, return without reopening Lab, generate a sample, save it as a profile, and verify that the profile is Verified only for the current configuration. Restart and confirm it reads Needs test while remaining usable.
