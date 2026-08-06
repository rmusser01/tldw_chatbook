# Voice Profiles Slice 1 — Gate Lifting + Per-Provider Validation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Voice profiles (and therefore per-character voice assignments) accept all seven TTS providers instead of audio.cpp only, with a per-provider validation table, a fail-closed store version bump, and the character speech path resolving legacy-provider profiles.

**Architecture:** The profile *store* is already provider-agnostic; this slice replaces four audio.cpp-only gates with one closed per-provider contract table exported from `profile_types.py`. audio.cpp keeps its strict WAV/1.0/empty-options contract; the six legacy-bridge providers get free-text model/voice, formats from the legacy catalog set, speed 0.25–4.0, and **empty options** (per spec §4.1). Capability preflight stays native-only (legacy providers have no catalog authority — they skip it).

**Tech Stack:** Python 3.11+, dataclasses with `__post_init__` validation, pytest + pytest-asyncio with pure in-process fakes (`_FakeRepository`/`_FakeTTSService` — no SQLite in service tests; real SQLite only in `test_profile_schema.py`).

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md` §4.1. Owner rulings §2 (all six providers at once; options stay empty for legacy in this slice).
- Legacy provider IDs, exactly: `openai`, `elevenlabs`, `kokoro`, `chatterbox`, `higgs`, `alltalk` (`TTS/legacy_catalogs.py:LEGACY_MODELS`). Legacy format set, exactly: `("mp3", "opus", "aac", "flac", "wav", "pcm")` (`legacy_catalogs._ALL_VISIBLE_FORMATS`).
- audio.cpp contract unchanged: `wav`, speed exactly `1.0`, empty options (`AUDIO_CPP_PROFILE_RESPONSE_FORMAT`/`AUDIO_CPP_PROFILE_SPEED`, `profile_types.py:60-61`).
- Work in a fresh worktree off `origin/dev` (concurrent-sessions rule); venv pytest only: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`; targeted test files + `--collect-only` sweep, never full `Tests/UI`.
- Error-code style: `ProfileValidationError("<field>")`, `ProfileServiceError("<code>")` — bounded codes, no dynamic text. Match `is`-based type checks and frozen-dataclass idioms of the existing files.
- Never use `git stash`. Never `git checkout <file>` to undo a scratch mutation (copy the file aside instead).

---

### Task 1: Provider contract table in `profile_types.py`

**Files:**
- Modify: `tldw_chatbook/TTS/profile_types.py` (constants block lines 48-61; `_validate_audio_cpp` lines 247-258; both `__post_init__` call sites lines 289-324 and 344-361)
- Test: `Tests/TTS/test_profile_types.py`

**Interfaces:**
- Produces: `PROFILE_PROVIDER_FORMATS: MappingProxyType[str, tuple[str, ...]]` and `PROFILE_PROVIDER_IDS: frozenset[str]` (public, imported by Tasks 2 and 3); `_validate_provider_contract(provider_id, response_format, speed, options)` replacing `_validate_audio_cpp` internally.
- Behavior change: `TTSProfileDraft`/`TTSGenerationProfile` construction now REJECTS unknown providers (`ProfileValidationError("provider_id")`), rejects formats outside the provider's list (`ProfileValidationError("response_format")`), and rejects non-empty options for legacy providers (`ProfileValidationError("options")`). audio.cpp behavior byte-identical (still `ProfileValidationError("audio_cpp")`).

- [ ] **Step 1: Write the failing tests** (append to `Tests/TTS/test_profile_types.py`, reusing its `_draft` helper which defaults to `provider_id="openai"`):

```python
def test_unknown_provider_is_rejected_at_construction() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: provider_id$"
    ):
        _draft(provider_id="future_native")


@pytest.mark.parametrize("provider_id", sorted(("openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk")))
@pytest.mark.parametrize("response_format", ["mp3", "opus", "aac", "flac", "wav", "pcm"])
def test_legacy_providers_accept_catalog_formats(provider_id: str, response_format: str) -> None:
    draft = _draft(provider_id=provider_id, response_format=response_format, speed=2.0)
    assert draft.provider_id == provider_id
    assert draft.response_format == response_format


def test_legacy_provider_rejects_format_outside_catalog_set() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: response_format$"
    ):
        _draft(provider_id="openai", response_format="ulaw_8000")


def test_legacy_provider_rejects_non_empty_options_this_slice() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: options$"
    ):
        _draft(provider_id="elevenlabs", options={"stability": 0.5})


def test_provider_table_matches_legacy_catalogs() -> None:
    from tldw_chatbook.TTS import legacy_catalogs

    assert set(profile_types_module.PROFILE_PROVIDER_FORMATS) == {"audio_cpp", *legacy_catalogs.LEGACY_MODELS}
    for provider_id in legacy_catalogs.LEGACY_MODELS:
        assert (
            profile_types_module.PROFILE_PROVIDER_FORMATS[provider_id]
            == legacy_catalogs._ALL_VISIBLE_FORMATS
        )
    assert profile_types_module.PROFILE_PROVIDER_FORMATS["audio_cpp"] == ("wav",)
    assert profile_types_module.PROFILE_PROVIDER_IDS == frozenset(
        profile_types_module.PROFILE_PROVIDER_FORMATS
    )
```

- [ ] **Step 2: Run to verify failures**

Run: `.venv/bin/python -m pytest Tests/TTS/test_profile_types.py -q`
Expected: the five new tests FAIL (unknown provider currently constructs fine; `ulaw_8000` matches the format regex so currently passes; options currently accepted; table doesn't exist → AttributeError). The three existing `audio_cpp` pin tests still PASS.

- [ ] **Step 3: Implement.** In `profile_types.py`, after the existing constants (line 61):

```python
PROFILE_PROVIDER_FORMATS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "audio_cpp": (AUDIO_CPP_PROFILE_RESPONSE_FORMAT,),
        "openai": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
        "elevenlabs": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
        "kokoro": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
        "chatterbox": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
        "higgs": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
        "alltalk": ("mp3", "opus", "aac", "flac", "wav", "pcm"),
    }
)
PROFILE_PROVIDER_IDS: frozenset[str] = frozenset(PROFILE_PROVIDER_FORMATS)
```

(`MappingProxyType` needs `from types import MappingProxyType`; `Mapping` from `collections.abc` — check existing imports first.) Replace `_validate_audio_cpp` (lines 247-258) with:

```python
def _validate_provider_contract(
    provider_id: str,
    response_format: str,
    speed: float,
    options: FrozenJsonOptions,
) -> None:
    formats = PROFILE_PROVIDER_FORMATS.get(provider_id)
    if formats is None:
        raise ProfileValidationError("provider_id")
    if response_format not in formats:
        raise ProfileValidationError("response_format")
    if provider_id == "audio_cpp":
        if speed != AUDIO_CPP_PROFILE_SPEED or bool(options):
            raise ProfileValidationError("audio_cpp")
        return
    if bool(options):
        raise ProfileValidationError("options")
```

Update BOTH call sites (`TTSProfileDraft.__post_init__` and `TTSGenerationProfile.__post_init__`) from `_validate_audio_cpp(...)` to `_validate_provider_contract(...)` — same argument list. Note: for `audio_cpp`, a non-`wav` format now raises `"response_format"` where it raised `"audio_cpp"` before — check the existing parametrized pin test (`{"response_format": "mp3"}` case, `test_profile_types.py:486-503`) and update that one parameter's expected match to `response_format`; speed/options cases keep `audio_cpp`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/TTS/test_profile_types.py -q`
Expected: all PASS (including the adjusted pin-test parameter).

- [ ] **Step 5: Run the neighbor suites that construct drafts/profiles**

Run: `.venv/bin/python -m pytest Tests/TTS/test_profile_service.py Tests/TTS/test_profile_portability.py Tests/TTS/test_character_request_resolver.py Tests/TTS/test_profile_repository.py -q`
Expected: ONE failure — `test_profile_service.py::test_generation_edit_rejects_unreviewed_native_provider` (its `provider_id="future_native"` draft now fails at construction). Fix it in Task 2 Step 1; everything else must PASS. Any other failure = stop and investigate before proceeding.

- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(tts): per-provider profile contract table replaces audio.cpp-only validation (slice 1, task 1)"`

---

### Task 2: Lift the service gates in `profile_service.py`

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py` (`_selection_is_profile_safe` :250-264; `_require_authoritative_capability` :2263-2290; `observe_portable_profile` :1277-1335; `_classify_selection` :2374-2411)
- Test: `Tests/TTS/test_profile_service.py`

**Interfaces:**
- Consumes: `PROFILE_PROVIDER_FORMATS`, `PROFILE_PROVIDER_IDS`, `AUDIO_CPP_PROFILE_SPEED` from Task 1.
- Produces: `create_from_artifact`/`update_profile`/`duplicate_profile` accept legacy-provider selections; legacy paths never call `get_native_capability_snapshot` (the fake's `get_catalog`/`get_voices` assertions enforce this for free); `_classify_selection` returns `"unverified"` for structurally-valid legacy profiles (interim honesty until Slice 2).

- [ ] **Step 1: Rewrite the now-broken pin test.** Replace `test_generation_edit_rejects_unreviewed_native_provider` (`test_profile_service.py:2477-2495`) — unknown providers are now unconstructable, so the service-level protection it proved has moved to construction time:

```python
def test_unknown_provider_draft_is_unconstructable() -> None:
    with pytest.raises(
        ProfileValidationError, match=r"^TTS profile validation failed: provider_id$"
    ):
        TTSProfileDraft(
            display_name="Future",
            provider_id="future_native",
            model_id="model",
            voice_id=None,
            response_format="wav",
            speed=1.0,
            options={},
        )
```

(`ProfileValidationError` is already imported in this test module; verify.)

- [ ] **Step 2: Write the failing service tests** (append; reuse `_service()`, `_profile()`, `_loaded = LoadedTTSProfile(repository_generation=repository.generation, profile=_profile())` patterns):

```python
@pytest.mark.asyncio
async def test_update_profile_accepts_openai_draft_without_native_calls() -> None:
    service, repository, tts_service = _service()
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation, profile=_profile()
    )
    draft = TTSProfileDraft(
        display_name="Narrator",
        provider_id="openai",
        model_id="pocket-tts",
        voice_id="marius",
        response_format="mp3",
        speed=1.25,
        options={},
    )

    await service.update_profile(loaded, draft)

    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["update"]


@pytest.mark.asyncio
async def test_duplicate_of_legacy_profile_skips_native_capability() -> None:
    service, repository, tts_service = _service()
    source = _profile(provider_id="elevenlabs", model_id="eleven_multilingual_v2",
                      voice_id="21m00Tcm4TlvDq8ikWAM", response_format="mp3", speed=1.0)
    loaded = LoadedTTSProfile(
        repository_generation=repository.generation, profile=source
    )

    await service.duplicate_profile(loaded, "Narrator copy")

    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["create"]


@pytest.mark.asyncio
async def test_create_from_artifact_accepts_legacy_selection() -> None:
    service, repository, tts_service = _service()
    selection = _selection(provider_id="openai", model_id="tts-1", voice_id="alloy",
                           response_format="mp3", speed=1.0)

    await service.create_from_artifact("OpenAI voice", _artifact(selection=selection))

    assert tts_service.revision_decisions == [("openai", 3)]
    assert tts_service.capability_calls == []
    assert [name for name, _ in repository.calls] == ["create"]
```

(Check `_selection()`/`_artifact()` factory signatures in the file first — pass provider overrides the way the factory expects; if `_selection` has no overrides, add keyword parameters to the factory following `_profile()`'s pattern.)

- [ ] **Step 3: Run to verify failures**

Run: `.venv/bin/python -m pytest Tests/TTS/test_profile_service.py -q 2>&1 | tail -8`
Expected: Step-1 test PASSES (construction rejects); Step-2 tests FAIL with `ProfileServiceError: ... unsupported_profile`.

- [ ] **Step 4: Implement.**

(a) Rewrite `_selection_is_profile_safe` (keep its loose-`object` signature — `_classify_selection` feeds it unvalidated fields):

```python
def _selection_is_profile_safe(
    provider_id: object,
    response_format: object,
    speed: object,
    options: object,
) -> bool:
    if type(provider_id) is not str or type(response_format) is not str:
        return False
    formats = PROFILE_PROVIDER_FORMATS.get(provider_id)
    if formats is None or response_format not in formats:
        return False
    if type(speed) is not float or not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        return False
    if not _mapping_is_empty(options):
        return False
    if provider_id == _PROFILE_PROVIDER_ID:
        return speed == AUDIO_CPP_PROFILE_SPEED
    return True
```

(import `PROFILE_PROVIDER_FORMATS` alongside the existing `AUDIO_CPP_*` imports from `profile_types`; `math` may need importing.)

(b) `_require_authoritative_capability` — legacy providers have no catalog authority; add at the top:

```python
        if draft.provider_id != _PROFILE_PROVIDER_ID:
            return
```

(c) `observe_portable_profile` — after the `_selection_is_profile_safe` gate (which now passes legacy), branch before the native-snapshot block:

```python
        repository_generation = self._current_repository_generation()
        if draft.provider_id != _PROFILE_PROVIDER_ID:
            revision = self._current_configuration_revision()
            self._require_repository_generation(repository_generation)
            return PortableProfileAvailabilityObservation(
                repository_generation=repository_generation,
                configuration_revision=revision,
                profile=portable,
                availability="unverified",
            )
```

(the audio.cpp tail from `exact_voice_models = ...` onward is untouched).

(d) `_classify_selection` — insert after the `_selection_is_profile_safe` gate:

```python
        if provider_id != _PROFILE_PROVIDER_ID:
            return "unverified"
```

(the incoming `snapshot` is always the native audio.cpp snapshot; classifying a legacy profile against it would say "unavailable", which is a lie — Slice 2 replaces this with the explicit no-catalog state).

- [ ] **Step 5: Run the service suite**

Run: `.venv/bin/python -m pytest Tests/TTS/test_profile_service.py -q 2>&1 | tail -5`
Expected: ALL pass (4615-line suite — any pre-existing test that asserted `unsupported_profile` for a *legacy* provider now fails; each such failure is behavior we intend — rewrite it to assert the new acceptance, and say so in the commit body. Any OTHER failure = stop and investigate.)

- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(tts): profile service accepts all seven providers; legacy skips native capability (slice 1, task 2)"`

---

### Task 3: Character speech resolver accepts the provider set

**Files:**
- Modify: `tldw_chatbook/TTS/character_request_resolver.py` (`CharacterTTSRequestResolution.__post_init__`, the `provider_id != "audio_cpp"` check at :91)
- Test: `Tests/TTS/test_character_request_resolver.py`

**Interfaces:**
- Consumes: `PROFILE_PROVIDER_IDS` from Task 1.
- Produces: an assigned legacy-provider profile resolves to `source="assigned"` with an exact `TTSRequest`, instead of surfacing `CharacterTTSResolutionError("assignment_invalid")`.

- [ ] **Step 1: Write the failing test** (reuse the file's `_character_ref()`, `_FakeProfileService`, `_loaded_assignment` helpers; `_profile()`/`_loaded_assignment` hardcode `audio_cpp` — add provider/model/voice/format keyword overrides to those factories first, defaulting to current values):

```python
@pytest.mark.asyncio
async def test_assigned_openai_profile_resolves_to_exact_request() -> None:
    character_ref = _character_ref()
    service = _FakeProfileService(
        _loaded_assignment(
            character_ref,
            revision=6,
            provider_id="openai",
            model_id="pocket-tts",
            voice_id="marius",
            response_format="mp3",
        )
    )
    resolver = CharacterTTSRequestResolver(service)

    resolved = await resolver.resolve(
        text="A character-authored reply.",
        assistant_kind="character",
        character_ref=character_ref,
    )

    assert resolved.source == "assigned"
    assert resolved.request is not None
    assert resolved.request.provider_id == "openai"
    assert resolved.request.model_id == "pocket-tts"
    assert resolved.request.voice == "marius"
    assert resolved.request.response_format == "mp3"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest Tests/TTS/test_character_request_resolver.py -q`
Expected: new test FAILS with `CharacterTTSResolutionError` ("assignment_invalid" — the `ValueError("assigned resolution")` from `__post_init__` is swallowed and re-coded at :237-240).

- [ ] **Step 3: Implement.** In `CharacterTTSRequestResolution.__post_init__`, change

```python
                or self.request.provider_id != "audio_cpp"
```

to

```python
                or self.request.provider_id not in PROFILE_PROVIDER_IDS
```

with `from tldw_chatbook.TTS.profile_types import PROFILE_PROVIDER_IDS` (module already imports from profile-adjacent modules — match import placement).

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest Tests/TTS/test_character_request_resolver.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/Chat/test_console_speech_snapshots.py -q` — Expected: ALL pass (the admission/snapshot suites exercise the downstream Console path).

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(tts): character speech resolves legacy-provider profile assignments (slice 1, task 3)"`

---

### Task 4: Store schema v2 with in-place upgrade

**Files:**
- Create: `tldw_chatbook/TTS/migrations/v1_to_v2.py`
- Modify: `tldw_chatbook/TTS/profile_schema.py` (`CURRENT_PROFILE_SCHEMA_VERSION` :44; `MIGRATIONS` :373; open flow :812-837; `_migrate_empty_store` :704-724)
- Test: `Tests/TTS/test_profile_schema.py`

**Interfaces:**
- Produces: `CURRENT_PROFILE_SCHEMA_VERSION == 2`; opening a non-empty v1 store upgrades it in place; a v2 store under pre-slice builds fails closed (`schema_unsupported`, since old code rejects `version > 1`).

**Why:** v2 has no DDL change — the bump is a downgrade fence: a store that may contain non-audio.cpp profiles must be refused by builds whose validation would misinterpret them (spec §4.1).

- [ ] **Step 1: Read the open flow and existing schema tests first** (`profile_schema.py:700-840`, `Tests/TTS/test_profile_schema.py`) — the plan below names the seams; confirm exact helper names before editing. Note the CURRENT open flow only migrates **empty v0** stores and hard-rejects any other version mismatch; there is no existing upgrade path for populated stores.

- [ ] **Step 2: Write the failing tests** (this file uses real SQLite — follow its existing fixture style for building a store; build a **populated v1 store** by momentarily monkeypatching `CURRENT_PROFILE_SCHEMA_VERSION` back to 1 is NOT acceptable — instead craft the v1 store by running the v0→v1 migration directly on a raw connection and inserting one profile row with the file's existing row-insert helper, or reuse its store-builder fixture if one exists):

```python
def test_populated_v1_store_upgrades_in_place_to_v2(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.sqlite"
    _build_populated_v1_store(db_path)  # helper written in this task, see step 4

    # open with current code — must upgrade, not reject
    connection = _open_store(db_path)  # the module's public open entrypoint
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert version == 2
        rows = connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        assert rows == 1
    finally:
        connection.close()


def test_future_version_store_still_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.sqlite"
    _build_populated_v1_store(db_path)
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA user_version = 3")
    raw.close()

    with pytest.raises(<the module's repository error type>) as caught:
        _open_store(db_path)
    assert "schema_unsupported" in str(caught.value)
```

(Resolve `_open_store` / the error type / the profiles-table name against the module and its existing tests — `test_profile_schema.py` already opens stores and asserts `schema_unsupported`; copy its exact idioms. The two assertions that matter: in-place upgrade preserves rows and sets version 2; future versions still refuse.)

- [ ] **Step 3: Run to verify failure** — the v1→v2 test fails with `schema_unsupported` (current open flow rejects version 1 != 1?  No: today CURRENT==1 so v1 opens fine — the test fails only AFTER the constant bumps. So: bump `CURRENT_PROFILE_SCHEMA_VERSION` to 2 FIRST, then run — expected: upgrade test FAILS with `schema_unsupported` (no upgrade path yet), future-version test PASSES.)

- [ ] **Step 4: Implement.** `migrations/v1_to_v2.py` mirroring `v0_to_v1.py`'s signature:

```python
"""v1 -> v2: version fence only. v2 stores may contain non-audio.cpp providers;
pre-expansion builds must refuse them (they reject user_version > 1)."""
import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA user_version = 2")
```

(match `v0_to_v1.py`'s actual exported name — if it exports `_migrate_v0_to_v1` style, follow it.) Register: `MIGRATIONS = {0: _migrate_v0_to_v1, 1: _migrate_v1_to_v2}`. In the open flow, replace the hard `elif version != CURRENT_PROFILE_SCHEMA_VERSION: raise schema_unsupported` branch with an in-place upgrade for `0 < version < CURRENT_PROFILE_SCHEMA_VERSION`: validate the schema at its current version first (`_validate_schema` guards), then run the same BEGIN IMMEDIATE loop as `_migrate_empty_store` starting from `version` (extract that loop into `_run_migrations(connection, from_version)` used by both). Keep the `version > CURRENT` refusal exactly as-is.

- [ ] **Step 5: Run** `.venv/bin/python -m pytest Tests/TTS/test_profile_schema.py Tests/TTS/test_profile_repository.py Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_store_lock.py -q` — Expected: ALL pass.

- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(tts): profile store schema v2 with in-place v1 upgrade — downgrade fence for multi-provider profiles (slice 1, task 4)"`

---

### Task 5: Playground adoption honesty + real-registry characterization

**Files:**
- Modify: `tldw_chatbook/UI/stts_playground_catalog.py` (`profile_availability_from_catalog` :304-325)
- Test: `Tests/UI/test_stts_playground_audio_cpp.py` (or the file that covers `profile_availability_from_catalog` — grep for the function name and put tests beside its existing coverage), `Tests/TTS/test_adapter_registry.py`

**Interfaces:**
- Consumes: `PROFILE_PROVIDER_IDS` from Task 1.
- Produces: adopting a legacy-provider preset shows `"unverified"` (not `"unavailable"`); a characterization test pins that the real adapter registry serves `configuration_revision("openai")` (the assumption Tasks 2's `create_from_artifact` path rests on).

- [ ] **Step 1: Write the failing test** for `profile_availability_from_catalog`: a preset with `provider_id="openai"`, `response_format="mp3"`, `speed=1.0`, empty options, `availability="available"`, catalog `None` → expect `"unverified"` (today: `"unavailable"` via the audio.cpp pin at :312-316). Construct the preset the way the existing tests in its covering file do.

- [ ] **Step 2: Run to verify failure**, then implement: replace the pinned condition

```python
    if (
        preset.provider_id != AUDIO_CPP_PROVIDER_ID
        or preset.response_format != AUDIO_CPP_PROFILE_RESPONSE_FORMAT
        or preset.speed != AUDIO_CPP_PROFILE_SPEED
        or bool(preset.options)
    ):
        return "unavailable"
```

with

```python
    if preset.provider_id != AUDIO_CPP_PROVIDER_ID:
        if preset.provider_id in PROFILE_PROVIDER_IDS and not bool(preset.options):
            return "unverified"
        return "unavailable"
    if (
        preset.response_format != AUDIO_CPP_PROFILE_RESPONSE_FORMAT
        or preset.speed != AUDIO_CPP_PROFILE_SPEED
        or bool(preset.options)
    ):
        return "unavailable"
```

- [ ] **Step 3: Registry characterization test** (append to `Tests/TTS/test_adapter_registry.py`, copying its real-registry construction):

```python
@pytest.mark.asyncio
async def test_registry_serves_configuration_revision_for_legacy_providers() -> None:
    registry = <the file's existing registry factory>
    for provider_id in ("openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk"):
        revision = registry.configuration_revision(provider_id)
        assert type(revision) is int and revision >= 0
```

If this FAILS (registry doesn't track legacy revisions): STOP — do not bolt on revision support ad hoc; report the failure, because Task 2(c)'s `create_from_artifact` acceptance depends on it and the fix belongs in a deliberate follow-up discussion.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest Tests/TTS/test_adapter_registry.py Tests/UI/test_stts_playground_audio_cpp.py -q` → ALL pass. **Step 5: Commit** `git commit -am "feat(tts): legacy presets adopt as unverified; pin registry revision coverage (slice 1, task 5)"`

---

### Task 6: Live verification, ADR amendment, backlog hygiene, ship

**Files:**
- Modify: `backlog/decisions/028-character-tts-generation-profile-ownership.md` (append amendment block)
- Create: backlog task file for slice 1 (ID from all-worktrees scan + headroom) and the sample-persona follow-up task (spec ruling 1)

- [ ] **Step 1: Live verification.** In the worktree with a scratch `TLDW_CONFIG_PATH` (dev-environment memory), drive the real TUI via tmux: create an OpenAI-provider profile via TTS Playground → Generate (real key from repo-root `*-api-key.txt`) → "Save result as profile"; assign it to a character in Roleplay ▸ Voice & Speech; speak a character message in Console with 🔊 and hear/observe the request go to OpenAI (or to a local keyless server via custom Base URL, reusing the TASK-2260 pattern). Record evidence per `backlog/docs/lessons-live-verification.md`.
- [ ] **Step 2: ADR-028 amendment** (dated block): first-release audio.cpp-only profile contract expanded to the seven-provider closed set; legacy providers = free-text exact model/voice, catalog format set, speed 0.25–4.0, empty options this slice; store schema v2 as downgrade fence; availability for legacy = "unverified" pending the Slice-2 no-catalog state.
- [ ] **Step 3: Backlog tasks:** scan ALL worktrees for max task ID (os.listdir+regex, leapfrog with headroom), file the slice-1 task (In Progress → Done with notes) and the sample-persona follow-up (To Do, dependencies: this slice's task), re-verify IDs at merge.
- [ ] **Step 4: Gates.** `ruff check` + `ruff format --check` on every touched file; targeted suites from Tasks 1-5 once more; repo-wide `--collect-only` sweep.
- [ ] **Step 5: PR to dev** titled `feat(tts): voice profiles accept all seven providers (slice 1)`, body summarizing the four lifted gates + fence + interim availability; address review; merge only after gates pass; update the spec's Status line (Slice 1 shipped) on the spec PR.

## Self-review notes

- Spec §4.1 coverage: gates P1 (Task 3), P2 (Tasks 2+5), validation table (Task 1), schema fence (Task 4), ADR amendment (Task 6). P3/P4 are Slice 2 by design; `_classify_selection`/adoption return interim `"unverified"` so Slice 1 never lies about legacy availability.
- Known judgment points for the implementer (not placeholders — decision rules given): factory-signature details in Tests (`_selection`/`_artifact`/`_loaded_assignment` overrides follow `_profile()`'s existing pattern); Task 4 helper names must be read from `profile_schema.py`/its tests before editing; Task 5 Step 3 has an explicit STOP rule.
