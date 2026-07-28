from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest


def _preferences_api() -> tuple[Any, Any]:
    from tldw_chatbook.TTS.preferences import (
        TTSConfigMutation,
        TTSPreferencesSnapshot,
    )

    return TTSPreferencesSnapshot, TTSConfigMutation


def _audio_cpp_settings(**overrides: Any) -> dict[str, Any]:
    values = {
        "default_provider": "audio_cpp",
        "default_model": "",
        "default_voice": "",
        "default_format": "wav",
        "default_speed": 1.0,
    }
    values.update(overrides)
    return {"app_tts": values}


def test_legacy_blank_audio_cpp_values_resolve_to_dynamic_modes() -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()

    snapshot = TTSPreferencesSnapshot.from_settings(_audio_cpp_settings())

    assert snapshot.provider_id == "audio_cpp"
    assert snapshot.model_mode == "first_available"
    assert snapshot.model_id is None
    assert snapshot.voice_mode == "server_default"
    assert snapshot.voice_id is None
    assert snapshot.response_format == "wav"
    assert snapshot.speed == 1.0


def test_explicit_modes_override_stale_exact_alias_values() -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()

    snapshot = TTSPreferencesSnapshot.from_settings(
        _audio_cpp_settings(
            default_model_mode="first_available",
            default_model="Stale.MODEL/value",
            default_voice_mode="server_default",
            default_voice="Stale.VOICE/value",
        )
    )

    assert snapshot.model_mode == "first_available"
    assert snapshot.model_id is None
    assert snapshot.voice_mode == "server_default"
    assert snapshot.voice_id is None


@pytest.mark.parametrize(
    ("mode_key", "mode", "id_key"),
    (
        ("default_model_mode", "exact", "default_model"),
        ("default_voice_mode", "exact", "default_voice"),
    ),
)
@pytest.mark.parametrize("empty_id", ("", "   ", None))
def test_exact_mode_requires_non_empty_corresponding_id(
    mode_key: str,
    mode: str,
    id_key: str,
    empty_id: object,
) -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()
    settings = _audio_cpp_settings(
        default_model_mode="first_available",
        default_voice_mode="server_default",
    )
    settings["app_tts"][mode_key] = mode
    settings["app_tts"][id_key] = empty_id

    with pytest.raises(ValueError, match=rf"^{id_key} must be a non-empty string$"):
        TTSPreferencesSnapshot.from_settings(settings)


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    (
        ({"default_format": "mp3"}, "audio.cpp response format must be wav"),
        ({"default_speed": 0.999}, "audio.cpp speed must be exactly 1.0"),
        ({"default_speed": 1.001}, "audio.cpp speed must be exactly 1.0"),
        (
            {"default_options": {"normalize": True}},
            "audio.cpp default options must be empty",
        ),
        (
            {"options": {"temperature": 0.5}},
            "audio.cpp default options must be empty",
        ),
    ),
)
def test_audio_cpp_rejects_unsupported_global_defaults(
    overrides: dict[str, object],
    diagnostic: str,
) -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()

    with pytest.raises(ValueError, match=rf"^{diagnostic}$"):
        TTSPreferencesSnapshot.from_settings(_audio_cpp_settings(**overrides))


def test_opaque_exact_ids_are_preserved_byte_for_byte() -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()
    model_id = "Model.V1/Case_Sensitive:@sha-ABC123"
    voice_id = "Voice+Accent/EN-US:#Speaker_07"

    snapshot = TTSPreferencesSnapshot.from_settings(
        _audio_cpp_settings(
            default_model_mode="exact",
            default_model=model_id,
            default_voice_mode="exact",
            default_voice=voice_id,
        )
    )

    assert snapshot.model_id == model_id
    assert snapshot.voice_id == voice_id


@pytest.mark.parametrize("shape", ("raw", "comprehensive", "normalized"))
def test_supported_settings_shapes_parse_without_textual_state(shape: str) -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()
    app_tts = {
        "default_provider": "audio_cpp",
        "default_model_mode": "exact",
        "default_model": "Model/ONE",
        "default_voice_mode": "exact",
        "default_voice": "Voice/TWO",
        "default_format": "wav",
        "default_speed": 1.0,
    }
    if shape == "raw":
        settings = {"app_tts": app_tts}
    elif shape == "comprehensive":
        settings = {
            "COMPREHENSIVE_CONFIG_RAW": {"app_tts": app_tts},
            "APP_TTS_CONFIG": {
                **app_tts,
                "default_model": "ignored-normalized-model",
            },
        }
    else:
        settings = {"APP_TTS_CONFIG": app_tts}

    snapshot = TTSPreferencesSnapshot.from_settings(settings)

    assert snapshot.model_id == "Model/ONE"
    assert snapshot.voice_id == "Voice/TWO"


def test_reading_legacy_blanks_does_not_mutate_input_or_write_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    TTSPreferencesSnapshot, _ = _preferences_api()
    settings = _audio_cpp_settings()
    original_settings = deepcopy(settings)
    persistence_calls: list[str] = []

    def unexpected_persistence(*args: object, **kwargs: object) -> None:
        del args, kwargs
        persistence_calls.append("called")
        raise AssertionError("preference reads must not call persistence helpers")

    for helper_name in (
        "apply_settings_mutation_to_cli_config",
        "save_settings_to_cli_config",
        "save_setting_to_cli_config",
        "delete_settings_from_cli_config",
        "atomic_private_write_text",
    ):
        monkeypatch.setattr(
            config_module,
            helper_name,
            unexpected_persistence,
        )

    TTSPreferencesSnapshot.from_settings(settings)

    assert persistence_calls == []
    assert settings == original_settings


def test_preference_snapshot_is_frozen_and_slotted() -> None:
    TTSPreferencesSnapshot, _ = _preferences_api()
    snapshot = TTSPreferencesSnapshot.from_settings(_audio_cpp_settings())

    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(FrozenInstanceError):
        snapshot.speed = 2.0


def _exact_snapshot(
    *,
    model_mode: str = "exact",
    model_id: str | None = "Model/Exact",
    voice_mode: str = "exact",
    voice_id: str | None = "Voice/Exact",
) -> Any:
    TTSPreferencesSnapshot, _ = _preferences_api()
    return TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode=model_mode,
        model_id=model_id,
        voice_mode=voice_mode,
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
    )


def test_exact_modes_write_authoritative_modes_and_all_current_aliases() -> None:
    mutation = _exact_snapshot().config_mutation()

    assert mutation.sets == {
        "app_tts": {
            "default_provider": "audio_cpp",
            "default_model_mode": "exact",
            "default_model": "Model/Exact",
            "default_voice_mode": "exact",
            "default_voice": "Voice/Exact",
            "default_format": "wav",
            "default_speed": 1.0,
        },
        "tts_settings": {
            "default_tts_provider": "audio_cpp",
            "default_openai_tts_model": "Model/Exact",
            "default_tts_voice": "Voice/Exact",
            "default_openai_tts_output_format": "wav",
            "default_openai_tts_speed": 1.0,
        },
    }
    assert mutation.deletes == {}


def test_dynamic_modes_delete_exactly_the_current_exact_aliases() -> None:
    mutation = _exact_snapshot(
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
    ).config_mutation()

    assert mutation.sets == {
        "app_tts": {
            "default_provider": "audio_cpp",
            "default_model_mode": "first_available",
            "default_voice_mode": "server_default",
            "default_format": "wav",
            "default_speed": 1.0,
        },
        "tts_settings": {
            "default_tts_provider": "audio_cpp",
            "default_openai_tts_output_format": "wav",
            "default_openai_tts_speed": 1.0,
        },
    }
    assert mutation.deletes == {
        "app_tts": ("default_model", "default_voice"),
        "tts_settings": (
            "default_openai_tts_model",
            "default_tts_voice",
        ),
    }


@pytest.mark.parametrize(
    (
        "model_mode",
        "model_id",
        "voice_mode",
        "voice_id",
        "expected_sets",
        "expected_deletes",
    ),
    (
        (
            "exact",
            "Model/Exact",
            "server_default",
            None,
            {
                "app_tts": {"default_model": "Model/Exact"},
                "tts_settings": {"default_openai_tts_model": "Model/Exact"},
            },
            {
                "app_tts": ("default_voice",),
                "tts_settings": ("default_tts_voice",),
            },
        ),
        (
            "first_available",
            None,
            "exact",
            "Voice/Exact",
            {
                "app_tts": {"default_voice": "Voice/Exact"},
                "tts_settings": {"default_tts_voice": "Voice/Exact"},
            },
            {
                "app_tts": ("default_model",),
                "tts_settings": ("default_openai_tts_model",),
            },
        ),
    ),
)
def test_mixed_modes_only_write_or_delete_their_own_exact_aliases(
    model_mode: str,
    model_id: str | None,
    voice_mode: str,
    voice_id: str | None,
    expected_sets: dict[str, dict[str, str]],
    expected_deletes: dict[str, tuple[str, ...]],
) -> None:
    mutation = _exact_snapshot(
        model_mode=model_mode,
        model_id=model_id,
        voice_mode=voice_mode,
        voice_id=voice_id,
    ).config_mutation()

    for section, values in expected_sets.items():
        assert mutation.sets[section].items() >= values.items()
    assert mutation.deletes == expected_deletes
    for section, keys in expected_deletes.items():
        for key in keys:
            assert key not in mutation.sets[section]


def test_config_mutation_never_encodes_deletion_as_none_or_blank() -> None:
    mutation = _exact_snapshot(
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
    ).config_mutation()

    values = [
        value
        for section_values in mutation.sets.values()
        for value in section_values.values()
    ]
    assert None not in values
    assert "" not in values


def test_config_mutation_defensively_freezes_nested_mappings() -> None:
    _, TTSConfigMutation = _preferences_api()
    source_sets = {"app_tts": {"default_provider": "audio_cpp"}}
    source_deletes = {"app_tts": ["default_model"]}

    mutation = TTSConfigMutation(source_sets, source_deletes)
    source_sets["app_tts"]["default_provider"] = "openai"
    source_deletes["app_tts"].append("default_voice")

    assert isinstance(mutation.sets, MappingProxyType)
    assert isinstance(mutation.sets["app_tts"], MappingProxyType)
    assert isinstance(mutation.deletes, MappingProxyType)
    assert mutation.sets["app_tts"]["default_provider"] == "audio_cpp"
    assert mutation.deletes["app_tts"] == ("default_model",)
    with pytest.raises(TypeError):
        mutation.sets["app_tts"]["default_provider"] = "openai"


@pytest.mark.parametrize("delete_keys", ("default_model", b"default_model"))
def test_config_mutation_rejects_string_like_delete_collections(
    delete_keys: str | bytes,
) -> None:
    _, TTSConfigMutation = _preferences_api()

    with pytest.raises(
        ValueError,
        match="^TTS configuration delete keys must be collections$",
    ):
        TTSConfigMutation(
            sets={},
            deletes={"app_tts": delete_keys},  # type: ignore[dict-item]
        )
