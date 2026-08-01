from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
from threading import Barrier
from types import MappingProxyType
import tomllib

import pytest
import toml

from tldw_chatbook import config as config_module
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.TTS.studio_preferences import (
    STUDIO_TTS_PROVIDER_OPTION_KEYS,
    STUDIO_TTS_SCHEMA_VERSION,
    StudioTTSLoadState,
    StudioTTSPreferenceStore,
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
    StudioTTSWriteStatus,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    BUILT_IN_TTS_PROVIDER_IDS as OWNERSHIP_PROVIDER_IDS,
)


def _write_config(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(values), encoding="utf-8")


def _saved_config(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolated_config_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config_module, "_CONFIG_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE", None)
    monkeypatch.setattr(config_module, "_SETTINGS_CACHE_SOURCE", None)
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 0)


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: dict[str, object]):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, raw)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    return StudioTTSPreferenceStore(), config_path


def _selection(**changes: object) -> StudioTTSSelectionOverrides:
    values = {
        "provider_id": None,
        "model_mode": None,
        "model_id": None,
        "voice_mode": None,
        "voice_id": None,
        "response_format": None,
        "speed": None,
    }
    values.update(changes)
    return StudioTTSSelectionOverrides(**values)


def test_provider_ids_are_one_shared_canonical_contract() -> None:
    assert BUILT_IN_TTS_PROVIDER_IDS == OWNERSHIP_PROVIDER_IDS
    assert BUILT_IN_TTS_PROVIDER_IDS == (
        "audio_cpp",
        "openai",
        "elevenlabs",
        "kokoro",
        "chatterbox",
        "higgs",
        "alltalk",
    )
    assert set(STUDIO_TTS_PROVIDER_OPTION_KEYS) == set(BUILT_IN_TTS_PROVIDER_IDS)
    assert STUDIO_TTS_PROVIDER_OPTION_KEYS == {
        "audio_cpp": frozenset(),
        "openai": frozenset(),
        "elevenlabs": frozenset(),
        "kokoro": frozenset(),
        "chatterbox": frozenset({"exaggeration", "cfg_weight"}),
        "higgs": frozenset(),
        "alltalk": frozenset(),
    }


def test_every_selection_axis_is_an_optional_override() -> None:
    selection = StudioTTSSelectionOverrides()

    assert selection.provider_id is None
    assert selection.model_mode is None
    assert selection.model_id is None
    assert selection.voice_mode is None
    assert selection.voice_id is None
    assert selection.response_format is None
    assert selection.speed is None


def test_snapshot_is_frozen_slotted_and_defensively_copies_options() -> None:
    mutable = {"chatterbox": {"exaggeration": 0.7}}
    snapshot = StudioTTSPreferencesSnapshot(
        revision=4,
        selection=_selection(provider_id="chatterbox"),
        provider_options=mutable,
    )
    mutable["chatterbox"]["exaggeration"] = 0.1

    assert not hasattr(snapshot, "__dict__")
    assert isinstance(snapshot.provider_options, MappingProxyType)
    assert isinstance(snapshot.provider_options["chatterbox"], MappingProxyType)
    assert snapshot.provider_options["chatterbox"]["exaggeration"] == 0.7
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = 5


@pytest.mark.parametrize("provider_id", BUILT_IN_TTS_PROVIDER_IDS)
def test_each_canonical_provider_is_accepted(provider_id: str) -> None:
    assert _selection(provider_id=provider_id).provider_id == provider_id


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"provider_id": "plugin"}, "provider"),
        ({"model_mode": "automatic"}, "model mode"),
        ({"voice_mode": "automatic"}, "voice mode"),
        ({"model_id": ""}, "model"),
        ({"voice_id": "<redacted>"}, "voice"),
        ({"voice_id": "***REDACTED***"}, "voice"),
        ({"response_format": "exe"}, "format"),
        ({"speed": 0}, "speed"),
        ({"speed": math.inf}, "speed"),
    ],
)
def test_selection_rejects_unbounded_or_unsupported_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _selection(**changes)


def test_exact_model_and_voice_identifiers_remain_opaque() -> None:
    selection = _selection(
        model_id="<Opaque:Model>",
        voice_id="C:\\voices\\sample.wav",
    )

    assert selection.model_id == "<Opaque:Model>"
    assert selection.voice_id == "C:\\voices\\sample.wav"


def test_audio_cpp_selection_enforces_current_native_constraints() -> None:
    valid = _selection(
        provider_id="audio_cpp",
        response_format="wav",
        speed=1.0,
    )
    assert valid.response_format == "wav"
    assert valid.speed == 1.0

    with pytest.raises(ValueError, match="audio.cpp.*format"):
        _selection(provider_id="audio_cpp", response_format="mp3")
    with pytest.raises(ValueError, match="audio.cpp.*speed"):
        _selection(provider_id="audio_cpp", speed=1.2)


def test_dynamic_selection_modes_cannot_persist_contradictory_exact_ids() -> None:
    with pytest.raises(ValueError, match="first_available"):
        _selection(model_mode="first_available", model_id="exact-model")
    with pytest.raises(ValueError, match="server_default"):
        _selection(voice_mode="server_default", voice_id="exact-voice")


@pytest.mark.parametrize(
    "provider_options",
    [
        {"unknown": {}},
        {"openai": {"api_key": "secret"}},
        {"alltalk": {"base_url": "https://example.test"}},
        {"higgs": {"model_path": "/tmp/model"}},
        {"audio_cpp": {"max_response_bytes": 10}},
        {"chatterbox": {"text": "submitted synthesis text"}},
        {"chatterbox": {"cfg_weight": "<masked>"}},
        {"chatterbox": {"exaggeration": -0.1}},
        {"chatterbox": {"cfg_weight": 1.1}},
    ],
)
def test_snapshot_rejects_unknown_global_runtime_secret_and_text_fields(
    provider_options: dict[str, dict[str, object]],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        StudioTTSPreferencesSnapshot(provider_options=provider_options)


def test_absent_section_loads_sparse_inheritance_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(
        tmp_path,
        monkeypatch,
        {"app_tts": {"default_provider": "openai"}},
    )
    original = config_path.read_bytes()
    write_calls = 0
    real_write = config_module.atomic_private_write_text

    def counted_write(*args: object, **kwargs: object):
        nonlocal write_calls
        write_calls += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(config_module, "atomic_private_write_text", counted_write)

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MISSING
    assert loaded.snapshot.revision == 0
    assert loaded.snapshot.selection == StudioTTSSelectionOverrides()
    assert loaded.snapshot.provider_options == {}
    assert loaded.issues == ()
    assert write_calls == 0
    assert config_path.read_bytes() == original


def test_sparse_round_trip_and_provider_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(tmp_path, monkeypatch, {"global": {"keep": True}})
    empty = store.load().snapshot
    candidate = replace(
        empty,
        selection=_selection(provider_id="chatterbox", speed=1.25),
        provider_options={
            "chatterbox": {"exaggeration": 0.7, "cfg_weight": 0.3},
        },
    )

    saved = store.save(candidate)
    reloaded = store.load()

    assert saved.status is StudioTTSWriteStatus.SAVED
    assert saved.snapshot is not None
    assert saved.snapshot.revision == 1
    assert reloaded.state is StudioTTSLoadState.LOADED
    assert reloaded.snapshot == saved.snapshot
    assert reloaded.snapshot.provider_options["chatterbox"] == {
        "exaggeration": 0.7,
        "cfg_weight": 0.3,
    }
    raw = _saved_config(config_path)
    assert raw["global"] == {"keep": True}
    assert raw["speech_studio"] == {
        "schema_version": STUDIO_TTS_SCHEMA_VERSION,
        "revision": 1,
        "selection": {"provider_id": "chatterbox", "speed": 1.25},
        "provider_options": {"chatterbox": {"exaggeration": 0.7, "cfg_weight": 0.3}},
    }


def test_provider_options_survive_switching_without_cross_provider_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(tmp_path, monkeypatch, {})
    first = store.load().snapshot
    chatterbox = replace(
        first,
        selection=_selection(provider_id="chatterbox"),
        provider_options={"chatterbox": {"cfg_weight": 0.4}},
    )
    saved = store.save(chatterbox).snapshot
    assert saved is not None

    switched = replace(saved, selection=_selection(provider_id="alltalk"))
    switched_result = store.save(switched)

    assert switched_result.status is StudioTTSWriteStatus.SAVED
    assert switched_result.snapshot is not None
    assert switched_result.snapshot.selection.provider_id == "alltalk"
    assert switched_result.snapshot.provider_options == {
        "chatterbox": {"cfg_weight": 0.4}
    }


def test_reset_to_global_deletes_all_overrides_but_keeps_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(tmp_path, monkeypatch, {"global": {"keep": "yes"}})
    initial = store.load().snapshot
    candidate = replace(
        initial,
        selection=_selection(provider_id="alltalk", voice_id="custom.wav"),
        provider_options={"chatterbox": {"exaggeration": 0.8}},
    )
    saved = store.save(candidate).snapshot
    assert saved is not None

    reset = store.reset_to_global(saved)

    assert reset.status is StudioTTSWriteStatus.SAVED
    assert reset.snapshot is not None
    assert reset.snapshot.selection == StudioTTSSelectionOverrides()
    assert reset.snapshot.provider_options == {}
    assert _saved_config(config_path)["speech_studio"] == {
        "schema_version": STUDIO_TTS_SCHEMA_VERSION,
        "revision": 2,
    }
    assert _saved_config(config_path)["global"] == {"keep": "yes"}


def test_stale_writer_gets_conflict_and_cannot_overwrite_newer_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(tmp_path, monkeypatch, {})
    first_editor = store.load().snapshot
    stale_editor = store.load().snapshot
    first_candidate = replace(
        first_editor,
        selection=_selection(provider_id="elevenlabs", model_id="eleven_turbo_v2"),
    )
    stale_candidate = replace(
        stale_editor,
        selection=_selection(provider_id="alltalk", voice_id="male_01.wav"),
    )

    first_save = store.save(first_candidate)
    stale_save = store.save(stale_candidate)

    assert first_save.status is StudioTTSWriteStatus.SAVED
    assert stale_save.status is StudioTTSWriteStatus.CONFLICT
    assert stale_save.snapshot is None
    current = store.load().snapshot
    assert current.selection.provider_id == "elevenlabs"
    assert current.selection.model_id == "eleven_turbo_v2"


def test_concurrent_writers_publish_one_snapshot_and_conflict_the_other(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(tmp_path, monkeypatch, {})
    observed = store.load().snapshot
    candidates = (
        replace(observed, selection=_selection(provider_id="openai")),
        replace(observed, selection=_selection(provider_id="alltalk")),
    )
    barrier = Barrier(2)

    def save(candidate: StudioTTSPreferencesSnapshot):
        barrier.wait(timeout=5)
        return store.save(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(save, candidates))

    assert {result.status for result in results} == {
        StudioTTSWriteStatus.SAVED,
        StudioTTSWriteStatus.CONFLICT,
    }
    assert store.load().snapshot.selection.provider_id in {"openai", "alltalk"}


def test_identical_current_snapshot_is_an_unchanged_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(tmp_path, monkeypatch, {})
    candidate = replace(
        store.load().snapshot,
        selection=_selection(provider_id="openai"),
    )
    saved = store.save(candidate).snapshot
    assert saved is not None
    before = config_path.read_bytes()

    unchanged = store.save(saved)

    assert unchanged.status is StudioTTSWriteStatus.UNCHANGED
    assert unchanged.snapshot == saved
    assert config_path.read_bytes() == before


def test_cache_reload_failure_reports_saved_file_without_claiming_full_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(tmp_path, monkeypatch, {})
    candidate = replace(
        store.load().snapshot,
        selection=_selection(provider_id="openai"),
    )

    def fail_reload(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(config_module, "load_settings", fail_reload)

    result = store.save(candidate)

    assert result.status is StudioTTSWriteStatus.SAVED_CACHE_RELOAD_FAILED
    assert result.snapshot is not None
    assert result.snapshot.revision == 1
    assert _saved_config(config_path)["speech_studio"]["revision"] == 1


def test_write_failure_publishes_no_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(tmp_path, monkeypatch, {"global": {"keep": True}})
    candidate = replace(
        store.load().snapshot,
        selection=_selection(provider_id="chatterbox"),
        provider_options={"chatterbox": {"cfg_weight": 0.2}},
    )
    original = config_path.read_bytes()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("injected")

    monkeypatch.setattr(config_module, "atomic_private_write_text", fail_write)

    result = store.save(candidate)

    assert result.status is StudioTTSWriteStatus.FAILED
    assert result.snapshot is None
    assert config_path.read_bytes() == original


def test_migration_copies_only_non_default_proven_request_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "app_tts": {
            "default_provider": "alltalk",
            "ALLTALK_TTS_VOICE_DEFAULT": "narrator.wav",
            "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": "flac",
            "CHATTERBOX_EXAGGERATION": 0.8,
            "CHATTERBOX_CFG_WEIGHT": 0.25,
            "ELEVENLABS_DEFAULT_MODEL": "eleven_turbo_v2",
            "ALLTALK_TTS_URL_DEFAULT": "https://tts.example.test",
            "CHATTERBOX_VOICE_DIR": "/private/voices",
            "audio_cpp": {
                "base_url": "http://127.0.0.1:8000",
                "max_response_bytes": 1234,
            },
        },
        "API": {"elevenlabs_api_key": "legacy-secret"},
        "character_tts": {"assignment": "must-stay-global"},
    }
    store, config_path = _store(tmp_path, monkeypatch, raw)

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MIGRATED
    assert loaded.snapshot.selection == _selection(
        voice_mode="exact",
        voice_id="narrator.wav",
        response_format="flac",
    )
    assert loaded.snapshot.provider_options == {
        "chatterbox": {"exaggeration": 0.8, "cfg_weight": 0.25}
    }
    saved = _saved_config(config_path)
    assert saved["app_tts"] == raw["app_tts"]
    assert saved["API"] == raw["API"]
    assert saved["character_tts"] == raw["character_tts"]
    studio_text = toml.dumps({"speech_studio": saved["speech_studio"]})
    for prohibited in (
        "legacy-secret",
        "tts.example.test",
        "/private/voices",
        "127.0.0.1",
        "assignment",
        "max_response_bytes",
        "eleven_turbo_v2",
    ):
        assert prohibited not in studio_text


def test_migration_uses_raw_saved_values_not_environment_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "environment-secret")
    store, config_path = _store(
        tmp_path,
        monkeypatch,
        {"app_tts": {"default_provider": "openai"}},
    )
    original = config_path.read_bytes()

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MISSING
    assert loaded.snapshot == StudioTTSPreferencesSnapshot()
    assert config_path.read_bytes() == original


def test_migration_copies_active_elevenlabs_model_without_copying_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "elevenlabs",
                "ELEVENLABS_DEFAULT_MODEL": "eleven_turbo_v2",
            }
        },
    )

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MIGRATED
    assert loaded.snapshot.selection == _selection(
        model_mode="exact",
        model_id="eleven_turbo_v2",
    )
    assert loaded.snapshot.selection.provider_id is None


def test_migration_drops_masked_eligible_value_but_keeps_valid_independent_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "alltalk",
                "ALLTALK_TTS_VOICE_DEFAULT": "<redacted>",
                "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": "flac",
            }
        },
    )

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MIGRATED
    assert loaded.snapshot.selection == _selection(response_format="flac")
    assert loaded.issues == ("app_tts.ALLTALK_TTS_VOICE_DEFAULT",)
    studio_text = toml.dumps(
        {"speech_studio": _saved_config(config_path)["speech_studio"]}
    )
    assert "redacted" not in studio_text


def test_migration_is_idempotent_and_default_only_legacy_values_do_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "chatterbox",
                "CHATTERBOX_EXAGGERATION": 0.8,
                "CHATTERBOX_CFG_WEIGHT": 0.5,
            }
        },
    )
    real_write = config_module.atomic_private_write_text
    writes = 0

    def counted_write(*args: object, **kwargs: object):
        nonlocal writes
        writes += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(config_module, "atomic_private_write_text", counted_write)

    first = store.load()
    second = store.load()

    assert first.state is StudioTTSLoadState.MIGRATED
    assert second.state is StudioTTSLoadState.LOADED
    assert first.snapshot == second.snapshot
    assert writes == 1

    no_op_store, no_op_path = _store(
        tmp_path / "defaults",
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "chatterbox",
                "CHATTERBOX_EXAGGERATION": 0.5,
                "CHATTERBOX_CFG_WEIGHT": 0.5,
            }
        },
    )
    no_op_original = no_op_path.read_bytes()
    assert no_op_store.load().state is StudioTTSLoadState.MISSING
    assert no_op_path.read_bytes() == no_op_original
    assert writes == 1


def test_migration_recovers_valid_fields_independently_from_malformed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(
        tmp_path,
        monkeypatch,
        {
            "app_tts": {
                "default_provider": "chatterbox",
                "CHATTERBOX_EXAGGERATION": 0.75,
                "CHATTERBOX_CFG_WEIGHT": "not-a-number",
            }
        },
    )

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.MIGRATED
    assert loaded.snapshot.provider_options == {"chatterbox": {"exaggeration": 0.75}}
    assert loaded.issues == ("app_tts.CHATTERBOX_CFG_WEIGHT",)


def test_loaded_record_recovers_fields_independently_and_drops_unknowns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, config_path = _store(
        tmp_path,
        monkeypatch,
        {
            "speech_studio": {
                "schema_version": 1,
                "revision": 3,
                "selection": {
                    "provider_id": "alltalk",
                    "voice_id": "male_01.wav",
                    "speed": "bad",
                    "text": "must-not-load",
                },
                "provider_options": {
                    "chatterbox": {
                        "exaggeration": 0.6,
                        "api_key": "must-not-load",
                    },
                    "plugin": {"option": True},
                },
            }
        },
    )

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.RECOVERED
    assert loaded.snapshot.revision == 3
    assert loaded.snapshot.selection == _selection(
        provider_id="alltalk",
        voice_id="male_01.wav",
    )
    assert loaded.snapshot.provider_options == {"chatterbox": {"exaggeration": 0.6}}
    assert loaded.issues == (
        "speech_studio.selection.speed",
        "speech_studio.selection.unknown_field",
        "speech_studio.provider_options.chatterbox.unknown_option",
        "speech_studio.provider_options.unknown_provider",
    )

    repaired = store.save(loaded.snapshot)

    assert repaired.status is StudioTTSWriteStatus.SAVED
    repaired_text = toml.dumps(
        {"speech_studio": _saved_config(config_path)["speech_studio"]}
    )
    assert "must-not-load" not in repaired_text
    assert "api_key" not in repaired_text
    assert "plugin" not in repaired_text


def test_unknown_loaded_names_are_bounded_in_safe_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "sensitive-" + "x" * 500
    second_secret_name = "another-sensitive-" + "y" * 500
    store, _ = _store(
        tmp_path,
        monkeypatch,
        {
            "speech_studio": {
                "schema_version": 1,
                "revision": 1,
                secret_name: "hidden",
                "selection": {
                    secret_name: "hidden",
                    second_secret_name: "hidden",
                },
                "provider_options": {
                    secret_name: {secret_name: "hidden"},
                    second_secret_name: {second_secret_name: "hidden"},
                },
            }
        },
    )

    loaded = store.load()

    rendered = " ".join(loaded.issues)
    assert secret_name not in rendered
    assert second_secret_name not in rendered
    assert loaded.issues == (
        "speech_studio.unknown_field",
        "speech_studio.selection.unknown_field",
        "speech_studio.provider_options.unknown_provider",
    )


def test_loaded_dynamic_mode_recovers_by_dropping_only_contradictory_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _store(
        tmp_path,
        monkeypatch,
        {
            "speech_studio": {
                "schema_version": 1,
                "revision": 2,
                "selection": {
                    "provider_id": "audio_cpp",
                    "model_mode": "first_available",
                    "model_id": "must-drop",
                    "voice_mode": "server_default",
                    "voice_id": "must-drop",
                },
            }
        },
    )

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.RECOVERED
    assert loaded.snapshot.selection == _selection(
        provider_id="audio_cpp",
        model_mode="first_available",
        voice_mode="server_default",
    )
    assert loaded.issues == (
        "speech_studio.selection.model_id",
        "speech_studio.selection.voice_id",
    )


@pytest.mark.parametrize(
    ("corrupt_section", "expected_reset_revision"),
    [
        ("not-a-table", 1),
        ({"schema_version": 99, "revision": 4}, 5),
        ({"schema_version": 1, "revision": "bad"}, 1),
    ],
)
def test_unrecoverable_studio_record_can_reset_without_touching_other_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_section: object,
    expected_reset_revision: int,
) -> None:
    raw = {
        "speech_studio": corrupt_section,
        "app_tts": {"default_provider": "audio_cpp", "secret": "preserved"},
        "character_tts": {"assignment": "preserved"},
    }
    store, config_path = _store(tmp_path, monkeypatch, raw)

    loaded = store.load()

    assert loaded.state is StudioTTSLoadState.CORRUPT
    assert loaded.snapshot.selection == StudioTTSSelectionOverrides()
    reset = store.reset_to_global(loaded.snapshot)
    assert reset.status is StudioTTSWriteStatus.SAVED
    saved = _saved_config(config_path)
    assert saved["speech_studio"] == {
        "schema_version": 1,
        "revision": expected_reset_revision,
    }
    assert saved["app_tts"] == raw["app_tts"]
    assert saved["character_tts"] == raw["character_tts"]
