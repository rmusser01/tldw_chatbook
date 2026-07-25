from __future__ import annotations

from dataclasses import replace

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    SERVER_DEFAULT_VOICE_ID,
    CatalogRequestToken,
    controls_from_catalog,
    provider_options,
    voice_id_for_request,
)


def _model(
    *,
    model_id: str = "model-a",
    display_name: str = "Model A",
    formats: tuple[str, ...] = ("wav",),
    voices: tuple[str, ...] = ("voice-a", "voice-b"),
    supports_speed: bool = True,
    server_default: bool = True,
) -> TTSModelInfo:
    return TTSModelInfo(
        model_id=model_id,
        display_name=display_name,
        family="test",
        upstream_mode="tts",
        formats=formats,
        voices=voices,
        supports_speed=supports_speed,
        omit_voice_uses_server_default=server_default,
    )


def _catalog(
    *,
    provider_id: str = "audio_cpp",
    health: ProviderHealth | None = None,
    models: tuple[TTSModelInfo, ...] | None = None,
    revision: int = 7,
) -> TTSProviderCatalog:
    return TTSProviderCatalog(
        provider_id=provider_id,
        revision=revision,
        health=health or ProviderHealth(state="available", fresh=True),
        models=models or (_model(),),
    )


def test_provider_options_preserve_descriptor_order_and_canonical_values() -> None:
    descriptors = (
        TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
        TTSProviderDescriptor("openai", "[bold]OpenAI[/]", False),
        TTSProviderDescriptor("kokoro", "<Kokoro>", False),
    )

    assert provider_options(descriptors) == (
        ("audio.cpp", "audio_cpp"),
        ("[bold]OpenAI[/]", "openai"),
        ("<Kokoro>", "kokoro"),
    )


def test_audio_cpp_initial_controls_force_wav_speed_and_server_default() -> None:
    controls = controls_from_catalog(
        _catalog(),
        selected_model_id=None,
        selected_voice_id=None,
        discovered_voices=("voice-a", "voice-b"),
        selected_format="mp3",
        speed=1.75,
    )

    assert controls.selected_model_id == "model-a"
    assert controls.voice_options == (
        ("Server default", SERVER_DEFAULT_VOICE_ID),
        ("voice-a", "voice-a"),
        ("voice-b", "voice-b"),
    )
    assert controls.selected_voice_id == SERVER_DEFAULT_VOICE_ID
    assert voice_id_for_request(controls.selected_voice_id) is None
    assert controls.format_options == ("wav",)
    assert controls.selected_format == "wav"
    assert controls.format_locked is True
    assert controls.speed == 1.0
    assert controls.speed_locked is True
    assert controls.generation_allowed is True
    assert controls.selection_changed is False


def test_audio_cpp_retains_valid_explicit_voice_and_falls_back_when_removed() -> None:
    retained = controls_from_catalog(
        _catalog(),
        selected_model_id="model-a",
        selected_voice_id="voice-b",
        discovered_voices=("voice-a", "voice-b"),
        selected_format="wav",
        speed=1.0,
    )
    removed = controls_from_catalog(
        _catalog(),
        selected_model_id="model-a",
        selected_voice_id="removed",
        discovered_voices=("voice-a",),
        selected_format="wav",
        speed=1.0,
    )

    assert retained.selected_voice_id == "voice-b"
    assert voice_id_for_request(retained.selected_voice_id) == "voice-b"
    assert retained.selection_changed is False
    assert removed.selected_voice_id == SERVER_DEFAULT_VOICE_ID
    assert removed.selection_changed is True


def test_audio_cpp_without_discovered_voices_keeps_only_server_default() -> None:
    controls = controls_from_catalog(
        _catalog(),
        selected_model_id="model-a",
        selected_voice_id=None,
        discovered_voices=(),
        selected_format="wav",
        speed=1.0,
    )

    assert controls.voice_options == (("Server default", SERVER_DEFAULT_VOICE_ID),)
    assert controls.selected_voice_id == SERVER_DEFAULT_VOICE_ID


def test_removed_model_falls_back_and_stale_health_disables_generation() -> None:
    catalog = _catalog(
        models=(
            _model(model_id="first", display_name="First"),
            _model(model_id="second", display_name="Second"),
        )
    )
    controls = controls_from_catalog(
        catalog,
        selected_model_id="removed",
        selected_voice_id=None,
        discovered_voices=(),
        selected_format="wav",
        speed=1.0,
    )
    stale = controls_from_catalog(
        replace(catalog, health=ProviderHealth(state="available", fresh=False)),
        selected_model_id="first",
        selected_voice_id=None,
        discovered_voices=(),
        selected_format="wav",
        speed=1.0,
    )

    assert controls.model_options == (("First", "first"), ("Second", "second"))
    assert controls.selected_model_id == "first"
    assert controls.selection_changed is True
    assert stale.model_options == controls.model_options
    assert stale.selected_model_id == "first"
    assert stale.generation_allowed is False


def test_legacy_controls_preserve_format_voice_and_speed_support() -> None:
    catalog = _catalog(
        provider_id="openai",
        models=(
            _model(
                formats=("mp3", "flac", "wav"),
                voices=("alloy", "nova"),
                server_default=False,
            ),
        ),
    )

    controls = controls_from_catalog(
        catalog,
        selected_model_id="model-a",
        selected_voice_id="nova",
        discovered_voices=None,
        selected_format="flac",
        speed=1.25,
    )

    assert controls.voice_options == (("alloy", "alloy"), ("nova", "nova"))
    assert controls.selected_voice_id == "nova"
    assert controls.format_options == ("mp3", "flac", "wav")
    assert controls.selected_format == "flac"
    assert controls.format_locked is False
    assert controls.speed == 1.25
    assert controls.speed_locked is False
    assert controls.generation_allowed is True


def test_remote_model_and_voice_labels_remain_plain_unparsed_strings() -> None:
    unsafe_model = "[bold red]model[/]"
    unsafe_voice = "<script>alert(1)</script>"
    controls = controls_from_catalog(
        _catalog(models=(_model(model_id="opaque", display_name=unsafe_model),)),
        selected_model_id=None,
        selected_voice_id=None,
        discovered_voices=(unsafe_voice,),
        selected_format="wav",
        speed=1.0,
    )

    assert controls.model_options == ((unsafe_model, "opaque"),)
    assert controls.voice_options[1] == (unsafe_voice, unsafe_voice)


def test_catalog_request_token_matches_every_revision_dimension() -> None:
    token = CatalogRequestToken(
        provider_id="audio_cpp",
        configuration_revision=3,
        catalog_revision=7,
        model_id="model-a",
    )
    current = {
        "provider_id": "audio_cpp",
        "configuration_revision": 3,
        "catalog_revision": 7,
        "model_id": "model-a",
    }

    assert token.matches(**current) is True
    for field, replacement_value in (
        ("provider_id", "openai"),
        ("configuration_revision", 4),
        ("catalog_revision", 8),
        ("model_id", "model-b"),
    ):
        changed = dict(current)
        changed[field] = replacement_value
        assert token.matches(**changed) is False
