"""Speech Lab surfaces omnivoice parameters, settings, and profile voices."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from tldw_chatbook.TTS.backends import omnivoice as omnivoice_backend_module
from tldw_chatbook.TTS.backends.omnivoice import OmniVoiceOnnxTTSBackend
from tldw_chatbook.TTS.omnivoice_sampler import OmniVoiceSamplerConfig
from tldw_chatbook.TTS.omnivoice_voice_manager import OmniVoiceVoiceManager
from tldw_chatbook.UI.Speech.speech_param_group import (
    PARAM_DEFAULTS,
    PARAM_LABELS,
)
from tldw_chatbook.UI.Speech.speech_playground_model import (
    PROVIDER_PARAMS,
    params_for_provider,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SPEECH_TTS_OWNERSHIP_INVENTORY,
)
from tldw_chatbook.UI.Speech.speech_settings_group import (
    PROVIDER_TITLES,
    SELECT_OPTIONS,
)
from tldw_chatbook.UI.Speech.speech_settings_model import (
    ALL_SETTINGS_CONTROLS,
    PROVIDER_SETTINGS,
    REQUIRED_SETTINGS,
    SETTINGS_ACTIONS,
    settings_for_provider,
)


def test_omnivoice_param_defaults_and_labels() -> None:
    # Blank defers to the saved [OmniVoiceSettings] value (UAT: a pre-filled
    # "32" silently overrode a Settings value of 16 on every request).
    for knob in (
        "tts-omnivoice-num-steps-input",
        "tts-omnivoice-guidance-scale-input",
        "tts-omnivoice-max-ref-duration-input",
    ):
        assert PARAM_DEFAULTS[knob]["value"] == ""
        assert "Settings default" in PARAM_DEFAULTS[knob]["placeholder"]
    assert PARAM_LABELS["tts-omnivoice-num-steps-input"] == "Diffusion steps"
    assert PARAM_LABELS["tts-omnivoice-guidance-scale-input"] == "Guidance (CFG)"


def test_omnivoice_playground_knobs_offered() -> None:
    assert set(PROVIDER_PARAMS["omnivoice"]) == {
        "tts-omnivoice-num-steps-input",
        "tts-omnivoice-guidance-scale-input",
        "tts-omnivoice-max-ref-duration-input",
    }
    offered = params_for_provider("omnivoice")
    assert "tts-omnivoice-num-steps-input" in offered


def test_omnivoice_settings_declared_everywhere() -> None:
    expected = {
        "omnivoice-guidance-scale-input",
        "omnivoice-max-ref-duration-input",
        "omnivoice-model-root-input",
        "omnivoice-voices-dir-input",
    }
    assert expected <= set(PROVIDER_SETTINGS["omnivoice"])
    assert settings_for_provider("omnivoice") == PROVIDER_SETTINGS["omnivoice"]
    assert expected <= ALL_SETTINGS_CONTROLS
    # The browse button is a command, not a persisted value (Higgs precedent).
    assert "omnivoice-voices-browse-btn" in SETTINGS_ACTIONS
    assert "omnivoice-voices-browse-btn" not in PROVIDER_SETTINGS["omnivoice"]
    assert "omnivoice-voices-browse-btn" in ALL_SETTINGS_CONTROLS
    assert REQUIRED_SETTINGS["omnivoice"] == ()
    assert PROVIDER_TITLES["omnivoice"] == "OmniVoice"
    provider_values = {
        value for _label, value in SELECT_OPTIONS["default-provider-select"]
    }
    assert "omnivoice" in provider_values


def test_omnivoice_ownership_records_declared() -> None:
    omnivoice_records = [
        record
        for record in SPEECH_TTS_OWNERSHIP_INVENTORY
        if record.owner_id == "omnivoice"
    ]
    assert {record.control_id for record in omnivoice_records} == {
        "omnivoice-guidance-scale-input",
        "omnivoice-max-ref-duration-input",
        "omnivoice-model-root-input",
        "omnivoice-voices-browse-btn",
        "omnivoice-voices-dir-input",
    }


def test_engine_loads_voice_profile_with_transcript(tmp_path: Path) -> None:
    _write_wav(tmp_path / "ref.wav")
    manager = OmniVoiceVoiceManager(tmp_path / "voices")
    ok, _msg = manager.create_profile(
        "narrator", str(tmp_path / "ref.wav"), reference_text="hello there"
    )
    assert ok

    backend = OmniVoiceOnnxTTSBackend(
        {"OMNIVOICE_VOICE_SAMPLES_DIR": str(tmp_path / "voices")}
    )
    audio, transcript = backend._load_voice_profile("narrator")
    assert audio is not None and Path(audio).is_file()
    assert transcript == "hello there"

    missing_audio, missing_text = backend._load_voice_profile("ghost")
    assert missing_audio is None and missing_text is None


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 24000)


def _make_artifact_tree(root: Path) -> None:
    for rel in (
        "omnivoice_lm_int8_hq/model.onnx",
        "omnivoice_lm_int8_hq/model.onnx_data",
        "audio_tokenizer_decoder_int8/model.onnx",
        "audio_tokenizer_decoder_int8/model.onnx_data",
        "audio_tokenizer_encoder_int8/model.onnx",
        "audio_tokenizer_encoder_int8/model.onnx_data",
        "tokenizer.json",
        "config.json",
    ):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.mark.asyncio
async def test_request_level_knobs_override_config(monkeypatch, tmp_path: Path) -> None:
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest

    _make_artifact_tree(tmp_path / "model")
    captured: dict[str, OmniVoiceSamplerConfig] = {}

    def fake_sampling(_lm, _prompt_ids, _target_len, *, config, **_kwargs):
        captured["config"] = config
        return np.zeros((8, 4), dtype=np.int64)

    monkeypatch.setattr(
        omnivoice_backend_module, "run_diffusion_sampling", fake_sampling
    )

    def make_backend() -> OmniVoiceOnnxTTSBackend:
        backend = OmniVoiceOnnxTTSBackend(
            {
                "OMNIVOICE_MODEL_ROOT": str(tmp_path / "model"),
                "OMNIVOICE_NUM_STEPS": 32,
                "OMNIVOICE_GUIDANCE_SCALE": 2.0,
            }
        )
        monkeypatch.setattr(backend, "_create_lm_runner", lambda: object())
        monkeypatch.setattr(backend, "_create_decoder_session", lambda: _FakeDecoder())
        monkeypatch.setattr(backend, "_load_tokenizer", lambda path: _FakeTokenizer())
        return backend

    request = OpenAISpeechRequest(
        model="omnivoice",
        input="hello",
        voice="",
        response_format="wav",
        extra_params={"num_steps": 8, "guidance_scale": 1.5},
    )
    [chunk async for chunk in make_backend().generate_speech_stream(request)]

    assert captured["config"].num_step == 8
    assert captured["config"].guidance_scale == 1.5

    # Without overrides the config defaults stand.
    plain = OpenAISpeechRequest(
        model="omnivoice", input="hello", voice="", response_format="wav"
    )
    [chunk async for chunk in make_backend().generate_speech_stream(plain)]

    assert captured["config"].num_step == 32
    assert captured["config"].guidance_scale == 2.0


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [i % 900 + 10 for i, _ in enumerate(text.split())]


class _FakeDecoder:
    def run(self, codes: np.ndarray) -> list[np.ndarray]:
        return [np.zeros(codes.shape[2] * 320, dtype=np.float32)]


# --- review fixes: knob parsing + configured profile dir ------------------------


def test_lab_knobs_parse_all_three_including_max_reference() -> None:
    from tldw_chatbook.UI.Speech.speech_synthesis_mixin import omnivoice_request_knobs

    params, error = omnivoice_request_knobs(
        {
            "#tts-omnivoice-num-steps-input": "16",
            "#tts-omnivoice-guidance-scale-input": "1.5",
            "#tts-omnivoice-max-ref-duration-input": "12",
        }
    )
    assert error is None
    assert params == {
        "num_steps": 16,
        "guidance_scale": 1.5,
        "max_reference_duration": 12.0,
    }
    assert isinstance(params["num_steps"], int)
    # blank fields defer to the configured defaults
    assert omnivoice_request_knobs({}) == ({}, None)


@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf"])
def test_lab_knobs_refuse_bad_steps_without_raising(bad: str) -> None:
    from tldw_chatbook.UI.Speech.speech_synthesis_mixin import omnivoice_request_knobs

    params, error = omnivoice_request_knobs({"#tts-omnivoice-num-steps-input": bad})
    assert params == {}
    assert error is not None and "num steps" in error


def test_profile_choices_read_the_configured_voices_dir(tmp_path: Path) -> None:
    from tldw_chatbook.UI.Speech.speech_catalog_mixin import SpeechCatalogMixin

    _write_wav(tmp_path / "ref.wav")
    voices = tmp_path / "custom_voices"
    ok, _msg = OmniVoiceVoiceManager(voices).create_profile(
        "narrator", str(tmp_path / "ref.wav"), reference_text="hello there"
    )
    assert ok

    class _Host(SpeechCatalogMixin):
        def _cli_setting(self, section, key, default=None):
            if (section, key) == ("OmniVoiceSettings", "voice_samples_dir"):
                return str(voices)
            return default

    assert _Host()._omnivoice_profile_choices() == [("narrator", "profile:narrator")]
