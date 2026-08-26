from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
)
from tldw_chatbook.TTS.profile_service import TTSPlaygroundSelectionPreset
from tldw_chatbook.TTS.profile_types import (
    PROFILE_PROVIDER_FORMATS,
    PROFILE_PROVIDER_IDS,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    SERVER_DEFAULT_VOICE_ID,
    CatalogRequestToken,
    controls_from_catalog,
    controls_from_profile_preset,
    preset_has_no_catalog_check,
    profile_availability_from_catalog,
    provider_options,
    voice_id_for_request,
)

from Tests.UI.speech_playground_fixtures import _profile_preset


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


@pytest.mark.parametrize(
    "public_callable",
    (CatalogRequestToken.matches, provider_options),
)
def test_public_catalog_callables_have_google_style_docstrings(
    public_callable: object,
) -> None:
    docstring = getattr(public_callable, "__doc__", "") or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


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


def test_audio_cpp_retains_valid_and_missing_explicit_voices() -> None:
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
    assert removed.selected_voice_id == "removed"
    assert "removed" in {value for _label, value in removed.voice_options}
    assert removed.generation_allowed is False
    assert removed.selection_changed is False


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


def test_audio_cpp_preserves_sentinel_shaped_remote_ids() -> None:
    remote_model_id = "__opaque_model__"
    remote_voice_id = "__server_default__"
    controls = controls_from_catalog(
        _catalog(models=(_model(model_id=remote_model_id),)),
        selected_model_id=remote_model_id,
        selected_voice_id=remote_voice_id,
        discovered_voices=(remote_voice_id,),
        selected_format="wav",
        speed=1.0,
    )

    assert controls.selected_model_id == remote_model_id
    assert controls.voice_options == (
        ("Server default", SERVER_DEFAULT_VOICE_ID),
        (remote_voice_id, remote_voice_id),
    )
    assert controls.selected_voice_id == remote_voice_id
    assert voice_id_for_request(remote_voice_id) == remote_voice_id
    assert voice_id_for_request(SERVER_DEFAULT_VOICE_ID) is None


def test_audio_cpp_missing_exact_model_is_preserved_and_stale_health_disables_generation() -> (
    None
):
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

    assert "removed" in {value for _label, value in controls.model_options}
    assert controls.selected_model_id == "removed"
    assert controls.selection_changed is False
    assert controls.generation_allowed is False
    assert stale.model_options == (("First", "first"), ("Second", "second"))
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


def test_openai_custom_exact_model_and_voice_are_pinned_and_generation_allowed() -> (
    None
):
    """Saved custom OpenAI names must survive catalog resolution.

    Custom OpenAI-compatible endpoints (TASK-2260) define their own model
    and voice names; the catalog cannot verify them, so they are pinned as
    honest "(no catalog check)" options instead of being silently replaced
    with the first official entry (TASK-15421).
    """
    catalog = _catalog(
        provider_id="openai",
        models=(
            _model(
                model_id="tts-1",
                display_name="TTS-1 (Standard)",
                formats=("mp3", "wav"),
                voices=("alloy", "nova"),
                server_default=False,
            ),
        ),
    )

    controls = controls_from_catalog(
        catalog,
        selected_model_id="pocket-tts-model",
        selected_voice_id="pocket-voice",
        discovered_voices=None,
        selected_format="wav",
        speed=1.0,
    )

    assert (
        "pocket-tts-model (no catalog check)",
        "pocket-tts-model",
    ) in controls.model_options
    assert controls.selected_model_id == "pocket-tts-model"
    assert (
        "pocket-voice (no catalog check)",
        "pocket-voice",
    ) in controls.voice_options
    assert controls.selected_voice_id == "pocket-voice"
    assert controls.selection_changed is False
    assert controls.format_options == ("mp3", "wav")
    assert controls.selected_format == "wav"
    assert controls.generation_allowed is True


def test_openai_custom_voice_is_pinned_alongside_a_catalog_model() -> None:
    """A custom voice pins even when the model is an official catalog one."""
    catalog = _catalog(
        provider_id="openai",
        models=(
            _model(
                model_id="tts-1",
                display_name="TTS-1 (Standard)",
                formats=("mp3", "wav"),
                voices=("alloy", "nova"),
                server_default=False,
            ),
        ),
    )

    controls = controls_from_catalog(
        catalog,
        selected_model_id="tts-1",
        selected_voice_id="pocket-voice",
        discovered_voices=None,
        selected_format="mp3",
        speed=1.0,
    )

    assert controls.selected_model_id == "tts-1"
    assert (
        "pocket-voice (no catalog check)",
        "pocket-voice",
    ) in controls.voice_options
    assert controls.selected_voice_id == "pocket-voice"
    assert controls.selection_changed is False
    assert controls.generation_allowed is True


def test_non_openai_legacy_custom_model_still_falls_back_to_the_catalog() -> None:
    """The pin is scoped to OpenAI, the one provider with custom endpoints.

    Other legacy providers keep the existing first-catalog-entry fallback;
    their model ids are fixed by the local engine, not by a server the
    user pointed the app at.
    """
    catalog = _catalog(
        provider_id="elevenlabs",
        models=(
            _model(
                model_id="eleven_multilingual_v2",
                display_name="Eleven Multilingual v2 (Default)",
                formats=("mp3",),
                voices=("Rachel",),
                server_default=False,
            ),
        ),
    )

    controls = controls_from_catalog(
        catalog,
        selected_model_id="no-such-model",
        selected_voice_id="Rachel",
        discovered_voices=None,
        selected_format="mp3",
        speed=1.0,
    )

    assert controls.selected_model_id == "eleven_multilingual_v2"
    assert controls.selection_changed is True


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
        request_generation=9,
    )
    current = {
        "provider_id": "audio_cpp",
        "configuration_revision": 3,
        "catalog_revision": 7,
        "model_id": "model-a",
        "request_generation": 9,
    }

    assert token.matches(**current) is True
    for field, replacement_value in (
        ("provider_id", "openai"),
        ("configuration_revision", 4),
        ("catalog_revision", 8),
        ("model_id", "model-b"),
        ("request_generation", 10),
    ):
        changed = dict(current)
        changed[field] = replacement_value
        assert token.matches(**changed) is False


@pytest.mark.parametrize(
    "provider_id",
    sorted(PROFILE_PROVIDER_IDS - {"audio_cpp"}),
)
def test_legacy_provider_preset_adopts_as_unverified_without_catalog(
    provider_id: str,
) -> None:
    """A legitimate legacy-provider preset must not be forced 'unavailable'.

    Before this behavior, ``profile_availability_from_catalog`` pinned every
    non-audio.cpp preset to "unavailable" regardless of provider validity,
    which made a freshly adopted OpenAI/ElevenLabs/etc. profile look broken
    in the Playground the instant it was selected (no catalog fetched yet).
    """
    preset = TTSPlaygroundSelectionPreset(
        provider_id=provider_id,
        model_id="profile/model",
        voice_id="profile/voice",
        response_format=PROFILE_PROVIDER_FORMATS[provider_id][0],
        speed=1.0,
        options={},
        availability="available",
    )

    assert profile_availability_from_catalog(preset, None) == "unverified"


def test_legacy_provider_preset_already_unavailable_is_not_upgraded_to_unverified() -> (
    None
):
    """A legacy preset already known "unavailable" must stay "unavailable".

    Guards the ordering between the early availability short-circuit and the
    new legacy-provider branch: the new branch must never run once the
    preset's own availability is already "unavailable".
    """
    preset = TTSPlaygroundSelectionPreset(
        provider_id="openai",
        model_id="profile/model",
        voice_id="profile/voice",
        response_format="mp3",
        speed=1.0,
        options={},
        availability="unavailable",
    )

    assert profile_availability_from_catalog(preset, None) == "unavailable"


@pytest.mark.parametrize(
    "provider_id",
    sorted(PROFILE_PROVIDER_IDS - {"audio_cpp"}),
)
def test_preset_has_no_catalog_check_is_true_for_every_legacy_provider(
    provider_id: str,
) -> None:
    """Slice 2 task 3's single shared class test, exercised for every legacy
    provider PROFILE_PROVIDER_IDS knows about -- so a new legacy provider
    added to that set is automatically covered here too.
    """
    preset = TTSPlaygroundSelectionPreset(
        provider_id=provider_id,
        model_id="profile/model",
        voice_id="profile/voice",
        response_format=PROFILE_PROVIDER_FORMATS[provider_id][0],
        speed=1.0,
        options={},
        availability="unverified",
    )

    assert preset_has_no_catalog_check(preset) is True


def test_preset_has_no_catalog_check_is_false_for_audio_cpp() -> None:
    """audio.cpp is the one provider class with real catalog authority --
    its "unverified" is transient, never the permanent no-catalog story.
    """
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="<opaque:model>",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
        availability="unverified",
    )

    assert preset_has_no_catalog_check(preset) is False


# --- TASK-2951 port: controls_from_profile_preset gap coverage ---
#
# Ported from `Tests/UI/test_stts_playground_audio_cpp.py` (the retired
# legacy playground widget's test file). These two call `controls_from_
# profile_preset` directly with no App/pilot, so they belong beside this
# file's existing coverage of the same function rather than in the new
# `test_speech_playground_pane_lifecycle.py` (which is for tests that mount
# a pane). No widget/pane-specific behavior involved -- pure projection.


def test_profile_preset_projection_keeps_missing_exact_values_but_blocks_generation() -> (
    None
):
    """A profile whose exact model/voice are missing from the catalog must
    still be *shown* (so the user can see what they saved), but generation
    stays blocked.
    """
    preset = _profile_preset()

    controls = controls_from_profile_preset(
        _catalog(),
        preset=preset,
        discovered_voices=("[voice]",),
    )

    assert controls.selected_model_id == "profile/model"
    assert "profile/model" in {value for _label, value in controls.model_options}
    assert controls.selected_voice_id == "profile/voice"
    assert "profile/voice" in {value for _label, value in controls.voice_options}
    assert controls.selected_format == "wav"
    assert controls.speed == 1.0
    assert controls.generation_allowed is False
    assert controls.selection_changed is False


@pytest.mark.parametrize(
    "model",
    (
        replace(_model(), formats=("mp3",)),
        replace(_model(), omit_voice_uses_server_default=False),
    ),
    ids=("format-missing", "server-default-unsupported"),
)
def test_profile_preset_projection_blocks_incompatible_model_contract(
    model: TTSModelInfo,
) -> None:
    """A profile's exact model can be *present* in the catalog and still be
    incompatible with the persisted profile contract (wrong format list, or
    a model that requires an explicit voice) -- either must still block
    generation, matching `AUDIO_CPP_PROFILE_RESPONSE_FORMAT`/slice-1 rules.
    """
    catalog = replace(_catalog(), models=(model,))
    preset = _profile_preset(model_id=model.model_id, voice_id=None)

    controls = controls_from_profile_preset(
        catalog,
        preset=preset,
        discovered_voices=None,
    )

    assert controls.selected_model_id == model.model_id
    assert controls.selected_voice_id is SERVER_DEFAULT_VOICE_ID
    assert controls.generation_allowed is False
