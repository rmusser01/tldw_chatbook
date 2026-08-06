# Voice Profiles Slice 3 — App-wide default voice profile: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A saved voice profile can be named as the app-wide default ("the assistant voice"), used whenever no character-specific voice applies — live-linked, so editing that profile changes every surface that inherits it.

**Architecture:** One new nullable setting `[app_tts] default_profile_id` (UUID string) plus one new precedence rung in the resolver, between `character_profile` and `global`. The raw Global-defaults axes stay exactly as they are and remain the fallback when no default profile is set. When a set default profile cannot be used, speech REFUSES with the existing one-tap "Speak with global defaults" override rather than silently substituting a voice.

**Tech Stack:** Python 3.11+, frozen dataclasses with `__post_init__` validation, Textual widgets, pytest + pytest-asyncio (real dataclasses + small hand-rolled fakes; Textual pilot for panel tests).

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md` §4.3 and ruling 5 (refuse + one-tap override on failure — never silently substitute).
- **`default_profile_id` must NOT be added to `TTSPreferencesSnapshot`** (`TTS/preferences.py`). That dataclass *is* the "global raw axes" precedence rung; the default profile is a distinct, higher rung. It travels via `event.settings` + a `_TTS_SETTING_BINDINGS` entry instead.
- Precedence, exactly: `explicit → character_profile → default_profile → global → provider_fallback`. **Studio is out of scope** — `resolve_studio` has its own draft/saved rungs and never took `character_profile`; do not add the new rung there.
- A malformed or dangling `default_profile_id` is a DEFINED state, never a crash: Settings still renders the saved value with an explanatory notice and never silently clears it; speech treats it as resolution failure.
- Legacy-provider profiles classify `"unverified"` (slice 1) — that is assignable, so an unverified profile is a legitimate default. Only `"unavailable"`/missing/store-down is failure.
- venv pytest only: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree root. Targeted files + blast-radius greps; no full `Tests/UI` sweeps.
- Never `git stash`; never `git checkout <file>` to undo. Strict idioms (`type(x) is Y`, frozen dataclasses, bounded error codes). Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Slice-1's headline trap applies directly:** a new state must be taught to every surface that reads it. Before declaring any task done, grep for every consumer of what you touched.

---

### Task 1: Resolver rung — `TTSDefaultProfileSelection` + `DEFAULT_PROFILE` source

**Files:**
- Modify: `tldw_chatbook/TTS/effective_settings.py` (enum `:90-98`; carrier dataclasses near `TTSCharacterProfileSelection` `:177-195`; `TTSEffectiveSelectionRevisions` `:217-250`; completeness gate `_require_complete_character_profile` `:448-498`; `resolve_non_studio` `:1183-1237`; `__all__`)
- Test: `Tests/TTS/test_effective_settings.py`

**Interfaces:**
- Produces: `TTSSelectionSource.DEFAULT_PROFILE = "default_profile"`; frozen `TTSDefaultProfileSelection(selection: TTSSelectionOverrides, repository_generation: int, profile_revision: int)` mirroring `TTSCharacterProfileSelection`'s validation; `resolve_non_studio(..., default_profile: TTSDefaultProfileSelection | None = None)`; `TTSEffectiveSelectionRevisions` gains paired `default_profile_repository` / `default_profile_revision`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/TTS/test_effective_settings.py`, following `test_normal_resolution_applies_explicit_then_character_then_global`'s exact shape — real dataclasses + the file's `_ResolutionRuntime` fake + its `_global_preferences()` helper):

```python
@pytest.mark.asyncio
async def test_default_profile_wins_over_global_and_loses_to_character() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("character-model", voices=("character-voice",))
    )
    default_profile = TTSDefaultProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp", model_mode="exact", model_id="default-model",
            voice_mode="exact", voice_id="default-voice", response_format="wav",
            speed=1.0, provider_options={},
        ),
        repository_generation=4,
        profile_revision=2,
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        default_profile=default_profile,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.voice_id == "default-voice"
    assert resolved.sources["voice_id"] is TTSSelectionSource.DEFAULT_PROFILE
    assert resolved.sources["model_id"] is TTSSelectionSource.DEFAULT_PROFILE
    assert resolved.revisions.default_profile_repository == 4
    assert resolved.revisions.default_profile_revision == 2


@pytest.mark.asyncio
async def test_character_profile_outranks_default_profile() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("character-model", voices=("character-voice",))
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp", model_mode="exact", model_id="character-model",
            voice_mode="exact", voice_id="character-voice", response_format="wav",
            speed=1.0, provider_options={},
        ),
        repository_generation=9, profile_revision=6,
    )
    default_profile = TTSDefaultProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp", model_mode="exact", model_id="default-model",
            voice_mode="exact", voice_id="default-voice", response_format="wav",
            speed=1.0, provider_options={},
        ),
        repository_generation=4, profile_revision=2,
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        character_profile=character,
        default_profile=default_profile,
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.voice_id == "character-voice"
    assert resolved.sources["voice_id"] is TTSSelectionSource.CHARACTER_PROFILE


@pytest.mark.asyncio
async def test_no_default_profile_still_falls_through_to_global() -> None:
    runtime = _ResolutionRuntime(
        capability=_audio_cpp_capability("character-model", voices=("character-voice",))
    )
    resolved = await TTSEffectiveSettingsResolver().resolve_non_studio(
        global_preferences=_global_preferences(),
        global_preferences_revision=7,
        provider_revision_reader=runtime.provider_revision,
        catalog_reader=runtime.read_catalog,
        native_capability_reader=runtime.read_native_capability,
    )

    assert resolved.sources["voice_id"] is TTSSelectionSource.GLOBAL
    assert resolved.revisions.default_profile_repository is None
    assert resolved.revisions.default_profile_revision is None


def test_default_profile_revisions_must_travel_together() -> None:
    with pytest.raises(ValueError):
        TTSEffectiveSelectionRevisions(
            global_preferences=1, studio_preferences=None,
            character_repository=None, character_profile=None,
            default_profile_repository=4, default_profile_revision=None,
            provider_configuration=1, provider_catalog=1,
        )
```

(Check `TTSEffectiveSelectionRevisions`'s real field list and `_global_preferences()`/`_audio_cpp_capability()` signatures in the file before writing — match them exactly; the constructor call above shows intent, not necessarily the exact kwargs.)

- [ ] **Step 2: Run to verify failures** — `.venv/bin/python -m pytest Tests/TTS/test_effective_settings.py -q`. Expected: ImportError/TypeError on `TTSDefaultProfileSelection` and the unknown `default_profile` kwarg. That is the right failure.

- [ ] **Step 3: Implement.**
  (a) Add `DEFAULT_PROFILE = "default_profile"` to `TTSSelectionSource`.
  (b) Add `TTSDefaultProfileSelection` immediately after `TTSCharacterProfileSelection`, copying its `__post_init__` validation verbatim (complete `TTSSelectionOverrides`, `repository_generation` int ≥ 0, `profile_revision` int ≥ 1).
  (c) `TTSEffectiveSelectionRevisions`: add `default_profile_repository: int | None` and `default_profile_revision: int | None`, and extend `__post_init__` with the same paired-nullability rule the character pair uses.
  (d) Generalize the completeness gate: `_require_complete_character_profile` currently hardcodes `source=TTSSelectionSource.CHARACTER_PROFILE` — parameterize the source (keep a thin character-named wrapper if other call sites depend on the name) so a default-profile layer fails closed identically instead of silently falling through to global.
  (e) `resolve_non_studio`: add the `default_profile` keyword, and append its layer **between** the character-profile append and the `_global_layer` append. Thread its revisions into the snapshot the same way the character pair is threaded.
  (f) Export `TTSDefaultProfileSelection` in `__all__`.

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest Tests/TTS/test_effective_settings.py -q` → all pass. Then **mutation-check**: move the default-profile layer append to *after* `_global_layer` and confirm `test_default_profile_wins_over_global_and_loses_to_character` fails; restore.

- [ ] **Step 5: Blast radius** — `grep -rn "TTSEffectiveSelectionRevisions(\|resolve_non_studio(" tldw_chatbook/ Tests/ | grep -v __pycache__`; run every test file that appears. Any positional construction of the revisions bag will break on the new fields — fix those call sites.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(tts): default-profile precedence rung between character and global (slice 3, task 1)"`

---

### Task 2: Setting plumbing — `[app_tts] default_profile_id` persists and round-trips

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py` (`GlobalSpeechTTSDefaults` `:294-305`; `load_global_speech_tts_state` `:554-771`; `build_global_speech_tts_save_proposal` `:1490-1539`)
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py` (`_TTS_SETTING_BINDINGS`)
- Test: `Tests/UI/test_settings_speech_tts_model.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent).
- Produces: `GlobalSpeechTTSDefaults.default_profile_id: str | None`; loader reads `[app_tts] default_profile_id`; save proposal emits `settings["default_profile_id"]` when it changed, or lists it in `delete_setting_keys` when cleared; `_TTS_SETTING_BINDINGS["default_profile_id"] = _app_tts_binding("default_profile_id")`.

**Critical context:** the raw defaults axes flow out as `proposal.preferences` (a `TTSPreferencesSnapshot`) and deliberately produce `proposal.settings == {}` — pinned by `test_selection_only_save_has_no_adapter_affecting_payload`. `default_profile_id` is NOT part of that snapshot, so it must be diffed and injected into `proposal.settings` explicitly, independent of `configure_provider`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_settings_speech_tts_model.py`, using its `_settings()` dict fixture + `deepcopy` draft idiom):

```python
def test_default_profile_id_round_trips_from_settings() -> None:
    settings = _settings()
    settings["app_tts"]["default_profile_id"] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    state = load_global_speech_tts_state(settings, environment={})

    assert state.defaults.default_profile_id == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def test_absent_default_profile_id_loads_as_none() -> None:
    state = load_global_speech_tts_state(_settings(), environment={})

    assert state.defaults.default_profile_id is None


def test_setting_default_profile_id_lands_in_save_settings() -> None:
    original = load_global_speech_tts_state(_settings(), environment={})
    draft = deepcopy(original)
    draft.defaults.default_profile_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert proposal.settings["default_profile_id"] == (
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    )


def test_clearing_default_profile_id_deletes_the_key() -> None:
    settings = _settings()
    settings["app_tts"]["default_profile_id"] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)
    draft.defaults.default_profile_id = None

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert "default_profile_id" in proposal.delete_setting_keys
    assert "default_profile_id" not in proposal.settings


def test_unchanged_default_profile_id_is_not_written() -> None:
    settings = _settings()
    settings["app_tts"]["default_profile_id"] = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    original = load_global_speech_tts_state(settings, environment={})
    draft = deepcopy(original)

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="audio_cpp"
    )

    assert "default_profile_id" not in proposal.settings
    assert "default_profile_id" not in proposal.delete_setting_keys
```

- [ ] **Step 2: Run to verify failures** — `.venv/bin/python -m pytest Tests/UI/test_settings_speech_tts_model.py -q`. Expected: AttributeError on `default_profile_id`.

- [ ] **Step 3: Implement.** Add the field to `GlobalSpeechTTSDefaults` (default `None`); read it in the loader from the `app_tts` mapping, normalizing to `None` when absent/empty/whitespace; **validate shape but do not reject**: a value that is not a well-formed UUID string must still load (it is a defined dangling state — Task 3 renders a notice, Task 4 refuses at speak time). Diff it in `build_global_speech_tts_save_proposal` and emit into `settings` / `delete_setting_keys`. Register the binding in `_TTS_SETTING_BINDINGS`.

- [ ] **Step 4: Run** — the model test file passes; **mutation-check** the diff logic by making the setter unconditional and confirming `test_unchanged_default_profile_id_is_not_written` fails; restore.

- [ ] **Step 5: Blast radius** — `grep -rn "GlobalSpeechTTSDefaults(" tldw_chatbook/ Tests/ | grep -v __pycache__` (positional constructions break on a new field) and `grep -rn "_TTS_SETTING_BINDINGS" tldw_chatbook/ Tests/`; run every file found.

- [ ] **Step 6: Commit** — `git commit -am "feat(tts): persist [app_tts] default_profile_id through the settings model (slice 3, task 2)"`

---

### Task 3: Settings panel — "Default voice profile" selector, honest when the store is down

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` (Global defaults `compose()` region `:1014-1211`; `_collect_visible_state()` `:1724-1771`; `has_unsaved_changes()` `:1836-1865`; `request_save()` `:1938-2031`; `_show_validation_error` field map `:1910-1936`)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (wherever the panel is constructed — it must receive the profile list)
- Test: `Tests/UI/test_settings_speech_tts_panel.py`

**Interfaces:**
- Consumes: `GlobalSpeechTTSDefaults.default_profile_id` (Task 2).
- Produces: a `Select` with id `settings-speech-default-profile`, first option `"None — use the fields below"` (value = the panel's blank sentinel), then one option per known profile labelled `f"{display_name}"`; the panel accepts an injected profile list so the pure/impure boundary is preserved.

**Purity boundary (do not violate):** `settings_speech_tts.py` is the PURE model — it must not load profiles. The impure screen loads them and passes them in as static choices. If the store is unavailable, the panel renders the saved id with an explanatory line and **never silently clears or drops the setting**.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_settings_speech_tts_panel.py`, following its `_PanelHarness` pilot idiom and the `#settings-speech-speed` save/error patterns):

```python
@pytest.mark.asyncio
async def test_selecting_a_default_voice_profile_saves_its_id() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        select = app.query_one("#settings-speech-default-profile", Select)
        select.value = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        await pilot.click("#settings-speech-save")
        await pilot.pause()

        assert app.events
        assert app.events[0].settings["default_profile_id"] == (
            "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        )


@pytest.mark.asyncio
async def test_choosing_none_clears_the_default_voice_profile() -> None:
    app = _PanelHarness(configure_provider="openai")
    async with app.run_test(size=(150, 60)) as pilot:
        # seed a saved default, then clear it
        ...
        assert "default_profile_id" in app.events[0].delete_setting_keys


@pytest.mark.asyncio
async def test_unavailable_profile_store_keeps_the_saved_id_and_says_so() -> None:
    app = _PanelHarness(configure_provider="openai")  # constructed with profiles=None
    async with app.run_test(size=(150, 60)) as pilot:
        rendered = str(app.query_one("#settings-speech-default-profile-note", Static).renderable)

        assert "3f2504e0-4f89-11d3-9a0c-0305e82c3301" in rendered
        assert "unavailable" in rendered.lower()
```

(The harness currently constructs the panel without a profile list — extend `_PanelHarness` with a `profiles` parameter mirroring how it already passes `configure_provider`, defaulting to a small fixed list so existing tests are unaffected. Fill in the seeding steps in test 2 following the file's existing "seed then act" tests.)

- [ ] **Step 2: Run to verify failures** — `.venv/bin/python -m pytest Tests/UI/test_settings_speech_tts_panel.py -q -k default_profile`. Expected: `QueryError` (widget absent).

- [ ] **Step 3: Implement.** Add the row at the TOP of the Global defaults card (it outranks the axes below it, and reading order should match precedence), using `self._row(...)` with the file's `settings-speech-draft-field` class so dirty-tracking sees it. Add its `_default_error(...)`/note node. Read it in `_collect_visible_state()` inside the file's `try/except QueryError` guard. **Add an explicit `default_profile_id` comparison to `has_unsaved_changes()` and to `request_save()`'s `defaults_changed`** — both currently compare `.snapshot()`, which cannot see this field. Add `"default_profile_id": "#settings-speech-default-profile"` to `_show_validation_error`'s field map. Wire the screen to pass the loaded profile list (and `None` when the store is unavailable).

- [ ] **Step 4: Run** — the panel test file passes. **Mutation-check** the dirty-detection addition: revert the explicit comparison and confirm a test asserting "changing only the default profile enables Save" fails; restore. (Add that test if the three above don't already cover it.)

- [ ] **Step 5: Blast radius** — `grep -rn "SpeechTTSSettingsPanel(" tldw_chatbook/ Tests/ | grep -v __pycache__`; every construction site must still work (new parameter needs a default).

- [ ] **Step 6: Commit** — `git commit -am "feat(tts): Default voice profile selector in Global defaults (slice 3, task 3)"`

---

### Task 4: Speech time — resolve the default profile, refuse honestly when it can't be used

**Files:**
- Modify: `tldw_chatbook/TTS/character_request_resolver.py` (bounded codes `:21-54`) **or** a sibling resolver — see Step 3
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` (`_resolve_message_speech_request` `:803-827`; the catch site `:513-538`; `_admit_tts_generation` wiring)
- Modify: `tldw_chatbook/TTS/request_admission.py` (thread `default_profile` into `resolve_non_studio`)
- Test: `Tests/TTS/test_character_request_resolver.py`, `Tests/TTS/test_console_speech_snapshot_admission.py`

**Interfaces:**
- Consumes: Task 1's `TTSDefaultProfileSelection` + `resolve_non_studio(default_profile=...)`; Task 2's persisted setting.
- Produces: speech with no character voice uses the default profile when one is set and loadable; when it is set but unusable, the existing `TTSCompleteEvent(error=..., global_override_token=...)` → `ConfirmationDialog` → `TTSGlobalOverrideDecisionEvent` path fires with copy that names the DEFAULT voice, not the character voice.

**Key facts:** `profile_service.get_profile(profile_id: UUID) -> LoadedTTSProfile` already exists and raises `ProfileRepositoryError` with `code == "missing"` when deleted — `Subscriptions/briefing_voices.py::resolve_roster_voices` is the precedent consumer, follow its shape. `tts_events.py`'s `_profile_service_loader` is currently only awaited when `snapshot.assistant_kind == "character"`; it must widen. The whole refuse+override UI (`app.py::_offer_tts_global_override`) needs NO change — only the raise site and its copy.

- [ ] **Step 1: Write the failing tests.** Cover: (a) no character voice + a loadable default profile → the resolved request uses the profile's provider/model/voice; (b) default profile id set but `get_profile` raises `missing` → a bounded resolution error whose `allow_global_override` is True and whose copy names the default voice (assert the copy does NOT say "character"); (c) profile store unavailable → same refuse+override; (d) no default profile set → unchanged global-axes behavior; (e) a character voice present → default profile is not consulted at all (assert `get_profile` was not called for the default id).

- [ ] **Step 2: Run to verify failures**, then implement.

- [ ] **Step 3: Implement.** Add bounded codes for the default-profile failures with their own accurate copy — do NOT reuse `"assignment_invalid"`, whose text says "The assigned voice profile… Repair or remove the assignment", which would misdescribe an app-default failure. Add e.g. `"default_profile_missing"` and `"default_profile_store_unavailable"` to the code table and to `_GLOBAL_OVERRIDE_CODES` (both are recoverable by falling back to global). Widen the profile-service load so the non-character path can resolve a default profile. Thread the loaded profile into `resolve_non_studio` via `TTSDefaultProfileSelection`.

- [ ] **Step 4: Run** the two named test files plus `Tests/Chat/test_console_speech_snapshots.py`. **Mutation-check**: make the failure path silently fall back to global instead of raising, and confirm the refuse test fails; restore.

- [ ] **Step 5: Blast radius** — `grep -rn "_RESOLUTION_COPY\|_GLOBAL_OVERRIDE_CODES\|CharacterTTSResolutionError" tldw_chatbook/ Tests/ | grep -v __pycache__`; any surface enumerating the code set must learn the new codes (slice-1's every-surface trap).

- [ ] **Step 6: Commit** — `git commit -am "feat(tts): speak with the app default voice profile, refusing honestly when it cannot be used (slice 3, task 4)"`

---

### Task 5: Deletion integrity — deleting the app default must warn

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py` and/or `tldw_chatbook/UI/stts_profile_library.py` (the Delete flow and its existing `assignment_count` machinery)
- Test: `Tests/UI/test_stts_profile_library.py`

**Interfaces:** the library's delete confirmation states that the profile is the app-wide default, in addition to any character assignments it already reports.

- [ ] **Step 1: Read first.** Find how the Delete flow obtains `assignment_count` today and where its confirmation copy is built. The app default lives in config, not the profile store, so the count machinery cannot discover it — the panel/screen must supply it. Decide the seam and say so in your report before implementing.
- [ ] **Step 2: Write the failing test** — deleting a profile that is the current `[app_tts] default_profile_id` shows a confirmation mentioning it is the app default; deleting an unrelated profile does not.
- [ ] **Step 3: Run to verify failure, implement, re-run, mutation-check** (remove the default-aware branch; confirm the test fails; restore).
- [ ] **Step 4: Blast radius + commit** — `git commit -am "feat(tts): deleting the app default voice profile warns first (slice 3, task 5)"`

---

### Task 6: Live verification, docs, backlog, ship

**Files:** `Docs/User_Guide/openai-compatible-tts.md` (+ index), `backlog/tasks/`, `backlog/decisions/` if a ruling needs recording.

- [ ] **Step 1: Live verification.** Read `backlog/docs/lessons-live-verification.md` first. Scratch `TLDW_CONFIG_PATH`; real provider key from repo-root `*-api-key.txt`. Drive the real TUI: set a Default voice profile in Settings ▸ Speech & TTS ▸ Global defaults → speak a message with NO character active → hear the default profile's voice → then delete that profile and confirm speech REFUSES with the "Use global voice?" dialog rather than silently switching. **Pick a provider that exercises the risky axis** — slice 1's live check used OpenAI, the one provider with no option keys, and missed a defect because of it.
- [ ] **Step 2: Docs.** Extend the user guide with the default-voice concept: what it is, that it is live-linked (editing the profile changes the default), and that character voices outrank it. Update the "Verified against" stamp.
- [ ] **Step 3: Backlog.** Scan ALL worktrees for the max task id (`git worktree list --porcelain` → `os.listdir` + regex over `backlog/tasks/`), leapfrog with headroom, file the slice-3 task Done with Implementation Notes. Re-verify the id at merge.
- [ ] **Step 4: Gates.** `ruff check` + `ruff format --check` on every touched file; the branch's targeted suites; repo-wide `--collect-only`.
- [ ] **Step 5: Report** — do NOT open a PR or merge; the controller owns that.

## Self-review notes

- Spec §4.3 coverage: setting (T2), resolver rung (T1), refuse+override (T4), Settings UI incl. store-unavailable honesty (T3), deletion integrity (T5), docs (T6). Reconfiguration-without-restart is inherited from the existing settings-save → service-reconfiguration path that Task 2's binding plugs into; T6's live check exercises it end to end.
- Deliberately out of scope: Studio resolution (has its own rungs), per-provider profile options (later slice), the "no catalog check" availability state (slice 2).
- Known judgment points (decision rules given, not placeholders): T4 Step 3 chooses between extending the existing resolver's code table vs a sibling resolver — extend if the character resolver's structure fits, and say which you chose and why; T5 Step 1 requires reading the delete flow before picking the seam.
