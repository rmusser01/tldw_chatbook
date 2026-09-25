"""omnivoice is wired as the 8th built-in TTS provider.

Covers every TTS-package seam Task 8 touched: provider IDs, legacy bridge
routes/names/prefixes and config projection, catalogs, request routing in
both builders, profile formats, studio preferences, and backend-config
projection through TTSBackendManager.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.TTS.legacy_bridge import (
    _APP_TTS_PREFIXES,
    _BACKEND_PREFIXES,
    _DISPLAY_NAMES,
    LEGACY_PROVIDER_IDS,
    LEGACY_ROUTES,
)
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
    LEGACY_MODEL_LABELS,
    LEGACY_MODELS,
    LEGACY_REQUEST_OPTION_KEYS,
    LEGACY_VOICE_OPTIONS,
)
from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request
from tldw_chatbook.TTS.profile_types import PROFILE_PROVIDER_FORMATS
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.TTS.studio_preferences import STUDIO_TTS_PROVIDER_OPTION_KEYS
from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager


def test_provider_id_registered_as_eighth() -> None:
    assert "omnivoice" in BUILT_IN_TTS_PROVIDER_IDS
    assert len(BUILT_IN_TTS_PROVIDER_IDS) == 8
    assert BUILT_IN_TTS_PROVIDER_IDS[-1] == "omnivoice"


def test_legacy_bridge_routes_names_and_prefixes() -> None:
    assert "omnivoice" in LEGACY_PROVIDER_IDS
    assert LEGACY_ROUTES["local_omnivoice_default"] == "omnivoice"
    assert _DISPLAY_NAMES["omnivoice"] == "OmniVoice (Local)"
    assert _APP_TTS_PREFIXES["omnivoice"] == "OMNIVOICE_"
    assert _BACKEND_PREFIXES["omnivoice"] == "local_omnivoice_"


def test_catalog_entries() -> None:
    assert LEGACY_MODELS["omnivoice"] == ("omnivoice-int8hq",)
    assert LEGACY_DEFAULT_MODELS["omnivoice"] == "omnivoice-int8hq"
    assert (
        LEGACY_MODEL_LABELS["omnivoice"]["omnivoice-int8hq"]
        == "OmniVoice int8hq (ONNX)"
    )
    assert LEGACY_DEFAULT_VOICES["omnivoice"] == "default"
    voice_values = {value for _, value in LEGACY_VOICE_OPTIONS["omnivoice"]}
    assert "default" in voice_values and "custom" in voice_values
    assert "language" in LEGACY_REQUEST_OPTION_KEYS["omnivoice"]


def test_request_builder_routes_to_internal_id() -> None:
    request, internal_model_id = build_legacy_speech_request(
        provider_id="omnivoice",
        model_id="omnivoice-int8hq",
        voice="default",
        text="hello",
    )
    assert internal_model_id == "local_omnivoice_default"
    assert request.model == "omnivoice-int8hq"


def test_profile_formats_and_studio_options_present() -> None:
    assert "omnivoice" in PROFILE_PROVIDER_FORMATS
    assert PROFILE_PROVIDER_FORMATS["omnivoice"] == (
        "mp3",
        "opus",
        "aac",
        "flac",
        "wav",
        "pcm",
    )
    assert "omnivoice" in STUDIO_TTS_PROVIDER_OPTION_KEYS


def test_saved_omnivoice_section_settings_reach_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIVOICE_MODEL_ROOT", raising=False)
    manager = TTSBackendManager(
        {
            "OmniVoiceSettings": {
                "model_root": "/models/omnivoice",
                "num_steps": 16,
                "guidance_scale": 1.5,
                "max_reference_duration": 22,
                "timeout_factor": 4.0,
            }
        }
    )

    config = manager._prepare_backend_config("local_omnivoice_default")

    assert config["OMNIVOICE_MODEL_ROOT"] == "/models/omnivoice"
    assert config["OMNIVOICE_NUM_STEPS"] == 16
    assert config["OMNIVOICE_GUIDANCE_SCALE"] == 1.5
    assert config["OMNIVOICE_MAX_REFERENCE_DURATION"] == 22
    assert config["OMNIVOICE_TIMEOUT_FACTOR"] == 4.0
    # Sane defaults for everything the engine reads but the section omits
    assert config["OMNIVOICE_INTRA_OP_THREADS"] == 0
    assert config["OMNIVOICE_T_SHIFT"] == 0.1
    assert config["OMNIVOICE_LANGUAGE"] == "auto"


def test_environment_override_beats_section_for_model_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIVOICE_MODEL_ROOT", "/env/models/omnivoice")
    manager = TTSBackendManager(
        {"OmniVoiceSettings": {"model_root": "/section/models/omnivoice"}}
    )

    config = manager._prepare_backend_config("local_omnivoice_default")

    assert config["OMNIVOICE_MODEL_ROOT"] == "/env/models/omnivoice"


def test_bridge_projection_carries_omnivoice_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config

    monkeypatch.delenv("OMNIVOICE_MODEL_ROOT", raising=False)
    projected = legacy_provider_config(
        "omnivoice",
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "OmniVoiceSettings": {"num_steps": 12, "model_root": "/raw/root"},
            }
        },
    )

    settings = projected["app_config"]["OmniVoiceSettings"]
    assert settings["num_steps"] == 12
    assert settings["model_root"] == "/raw/root"
    assert projected["app_config"]["app_tts"] == {}


def test_bridge_projection_environment_overrides_model_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config

    monkeypatch.setenv("OMNIVOICE_MODEL_ROOT", "/env/omnivoice")
    projected = legacy_provider_config(
        "omnivoice",
        {"COMPREHENSIVE_CONFIG_RAW": {"OmniVoiceSettings": {}}},
    )

    assert (
        projected["app_config"]["OmniVoiceSettings"]["model_root"] == "/env/omnivoice"
    )
