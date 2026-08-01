"""Focused coverage for the runnable Parakeet ONNX batch path."""

import json
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from tldw_chatbook.Local_Ingestion import transcription_service as service_module
from tldw_chatbook.Local_Ingestion.parakeet_v2_installer import (
    PARAKEET_V2_REPOSITORY,
    PARAKEET_V2_REVISION,
    VERIFICATION_RECEIPT,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionError,
    TranscriptionService,
)


def _write_silent_wav(path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\0\0" * 1_600)


def _write_model_bundle(path) -> None:
    path.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ):
        (path / filename).touch()


def _known_v2_receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": PARAKEET_V2_REPOSITORY,
        "revision": PARAKEET_V2_REVISION,
    }


def test_parakeet_onnx_transcribes_with_local_v2_int8_model(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)

    load_calls = []

    class FakeModel:
        def recognize(self, path):
            assert path == str(audio_path)
            return "A working local transcription."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    service = TranscriptionService()
    service.config["parakeet_onnx_model_dir"] = str(model_dir)
    result = service.transcribe(
        str(audio_path),
        provider="parakeet-onnx",
    )

    assert load_calls == [
        (
            PARAKEET_V2_MODEL,
            {
                "path": str(model_dir),
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
                "preprocessor_config": {
                    "use_numpy_preprocessors": True,
                    "max_concurrent_workers": 1,
                },
            },
        )
    ]
    assert result == {
        "text": "A working local transcription.",
        "segments": [
            {
                "start": 0.0,
                "end": 0.1,
                "text": "A working local transcription.",
                "Time_Start": 0.0,
                "Time_End": 0.1,
                "Text": "A working local transcription.",
            }
        ],
        "language": "en",
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "warnings": [],
        "provider": "parakeet-onnx",
        "model": PARAKEET_V2_MODEL,
    }


def test_parakeet_onnx_non_english_selects_v3_without_decoder_language(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            assert path == str(audio_path)
            return " Eine lokale Transkription. "

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    result = TranscriptionService().transcribe(
        str(audio_path),
        provider="parakeet-onnx",
        language="de",
        model_dir=str(model_dir),
    )

    assert load_calls == [
        (
            PARAKEET_V3_MODEL,
            {
                "path": str(model_dir),
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
                "preprocessor_config": {
                    "use_numpy_preprocessors": True,
                    "max_concurrent_workers": 1,
                },
            },
        )
    ]
    assert result["text"] == "Eine lokale Transkription."
    assert result["segments"][0]["text"] == "Eine lokale Transkription."
    assert result["language"] is None
    assert result["requested_language"] == "de"
    assert result["effective_language"] == "auto"
    assert result["detected_language"] is None
    assert result["warnings"] == ["requested_language_not_enforced"]
    assert result["model"] == PARAKEET_V3_MODEL


def test_parakeet_onnx_rejects_known_v2_bundle_when_v3_is_selected(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    (model_dir / VERIFICATION_RECEIPT).write_text(
        json.dumps(_known_v2_receipt()),
        encoding="utf-8",
    )
    load_calls = []

    def unexpected_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("load_model must not run for a known v2/v3 mismatch")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError) as exc_info:
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            language="de",
            model_dir=str(model_dir),
        )

    error_text = str(exc_info.value)
    assert "identified as Parakeet v2 by receipt metadata" in error_text
    assert "verified" not in error_text.lower()
    assert "Choose a Parakeet v3 folder" in error_text
    assert "Retry with faster-whisper" in error_text
    assert load_calls == []


@pytest.mark.parametrize(
    "receipt_text",
    [
        pytest.param("{", id="malformed"),
        pytest.param(("[" * 30_000) + ("]" * 30_000), id="excessively-nested"),
        pytest.param(
            '{"repository": ' + ("1" * 5_000) + "}",
            id="oversized-integer-token",
        ),
        pytest.param(
            json.dumps(_known_v2_receipt()) + (" " * (1024 * 1024)),
            id="oversized-valid-known-v2",
        ),
    ],
)
def test_parakeet_onnx_manual_v3_directory_ignores_untrusted_receipts(
    tmp_path, monkeypatch, receipt_text
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    (model_dir / VERIFICATION_RECEIPT).write_text(receipt_text, encoding="utf-8")
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            return "Eine lokale Transkription."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    result = TranscriptionService().transcribe(
        str(audio_path),
        provider="parakeet-onnx",
        language="de",
        model_dir=str(model_dir),
    )

    assert load_calls[0][0] == PARAKEET_V3_MODEL
    assert result["model"] == PARAKEET_V3_MODEL


@pytest.mark.parametrize(
    ("source_lang", "language", "expected_model", "expected_language"),
    [
        pytest.param(
            None,
            "de",
            PARAKEET_V3_MODEL,
            "de",
            id="explicit-language",
        ),
        pytest.param(
            "de",
            "en",
            PARAKEET_V3_MODEL,
            "de",
            id="explicit-source-language",
        ),
        pytest.param(
            None,
            None,
            PARAKEET_V2_MODEL,
            "en",
            id="missing-language-defaults-to-english",
        ),
    ],
)
def test_parakeet_onnx_request_ignores_configured_language_defaults(
    tmp_path,
    monkeypatch,
    source_lang,
    language,
    expected_model,
    expected_language,
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            assert path == str(audio_path)
            return "Eine lokale Transkription."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )
    service = TranscriptionService()
    service.config["default_source_language"] = "fr"
    service.config["default_target_language"] = "es"

    result = service.transcribe(
        str(audio_path),
        provider="parakeet-onnx",
        source_lang=source_lang,
        language=language,
        model_dir=str(model_dir),
    )

    assert load_calls[0][0] == expected_model
    assert "language" not in load_calls[0][1]
    assert result["requested_language"] == expected_language
    if expected_model == PARAKEET_V3_MODEL:
        assert result["effective_language"] == "auto"
        assert result["warnings"] == ["requested_language_not_enforced"]
    else:
        assert result["effective_language"] == "en"
        assert result["warnings"] == []


def test_parakeet_onnx_file_target_language_alias_rejects_before_model_load(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    def unexpected_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("load_model must not run for a translation request")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError, match="Retry with faster-whisper"):
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            language="en",
            target_language="fr",
            model_dir=str(model_dir),
        )

    assert load_calls == []


def test_parakeet_onnx_target_lang_argument_wins_over_alias(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            assert path == str(audio_path)
            return "English transcription."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    result = TranscriptionService().transcribe(
        str(audio_path),
        provider="parakeet-onnx",
        language="en",
        target_lang="",
        target_language="fr",
        model_dir=str(model_dir),
    )

    assert load_calls[0][0] == PARAKEET_V2_MODEL
    assert result["requested_language"] == "en"


@pytest.mark.parametrize(
    ("language", "target_lang", "model"),
    [
        pytest.param("auto", None, PARAKEET_V3_MODEL, id="auto"),
        pytest.param("ja", None, PARAKEET_V3_MODEL, id="unsupported-language"),
        pytest.param("en", "fr", PARAKEET_V2_MODEL, id="translation"),
        pytest.param("de", None, PARAKEET_V2_MODEL, id="v2-non-english-mismatch"),
        pytest.param("en", None, PARAKEET_V3_MODEL, id="v3-english-mismatch"),
    ],
)
def test_parakeet_onnx_rejects_incompatible_requests_before_model_load(
    tmp_path, monkeypatch, language, target_lang, model
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    def unexpected_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("load_model must not run for an incompatible request")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError, match="Retry with faster-whisper"):
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            model=model,
            language=language,
            target_lang=target_lang,
            model_dir=str(model_dir),
        )

    assert load_calls == []


def test_parakeet_onnx_rejects_incomplete_model_directory_before_loading(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "incomplete-model"
    model_dir.mkdir()
    _write_silent_wav(audio_path)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("load_model must not run for an incomplete local bundle")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError, match="missing required files"):
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            model_dir=str(model_dir),
        )


def test_parakeet_onnx_rejects_model_directory_that_fails_central_path_validation(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    model_dir = tmp_path / "model$(unsafe)"
    _write_model_bundle(model_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    def unexpected_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("load_model must not run for an invalid model path")

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=unexpected_load)
    )

    with pytest.raises(TranscriptionError, match="invalid local model directory"):
        TranscriptionService().transcribe(
            str(audio_path),
            provider="parakeet-onnx",
            model=PARAKEET_V2_MODEL,
            language="en",
            model_dir=str(model_dir),
        )

    assert load_calls == []


def test_parakeet_onnx_transcribes_pcm_buffer_without_staging_a_file(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)

    recognized = []

    class FakeModel:
        def recognize(self, waveform, *, sample_rate):
            recognized.append((waveform, sample_rate))
            return "  Memory only.  "

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "onnx_asr",
        SimpleNamespace(load_model=lambda *args, **kwargs: FakeModel()),
    )

    def reject_temp_file(*args, **kwargs):
        raise AssertionError("Parakeet buffer transcription must not stage a file")

    monkeypatch.setattr(service_module.tempfile, "NamedTemporaryFile", reject_temp_file)

    pcm = np.array([-32768, 0, 16384, 32767], dtype=np.int16)
    result = TranscriptionService().transcribe_buffer(
        pcm.tobytes(),
        sample_rate=8_000,
        channels=1,
        sample_width=2,
        provider="parakeet-onnx",
        model=PARAKEET_V2_MODEL,
        language="en",
        model_dir=str(model_dir),
    )

    assert len(recognized) == 1
    waveform, sample_rate = recognized[0]
    assert sample_rate == 8_000
    np.testing.assert_allclose(
        waveform,
        np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
    )
    assert result["text"] == "Memory only."
    assert result["segments"][0]["end"] == pytest.approx(4 / 8_000)


def test_parakeet_onnx_buffer_target_lang_wins_over_alias(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    load_calls = []

    class FakeModel:
        def recognize(self, waveform, *, sample_rate):
            return "Memory only."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    result = TranscriptionService().transcribe_buffer(
        np.array([0, 16384], dtype=np.int16).tobytes(),
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        provider="parakeet-onnx",
        model=PARAKEET_V2_MODEL,
        language="en",
        target_lang="",
        target_language="fr",
        model_dir=str(model_dir),
    )

    assert load_calls[0][0] == PARAKEET_V2_MODEL
    assert result["requested_language"] == "en"


def test_parakeet_onnx_v3_transcribes_pcm_buffer_with_transparent_language_result(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    _write_model_bundle(model_dir)
    load_calls = []

    class FakeModel:
        def recognize(self, waveform, *, sample_rate):
            assert waveform.dtype == np.float32
            assert sample_rate == 16_000
            return " Deutsche Aufnahme. "

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )

    result = TranscriptionService().transcribe_buffer(
        np.array([0, 16384], dtype=np.int16).tobytes(),
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        provider="parakeet-onnx",
        language="de",
        model_dir=str(model_dir),
    )

    assert load_calls == [
        (
            PARAKEET_V3_MODEL,
            {
                "path": str(model_dir),
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
                "preprocessor_config": {
                    "use_numpy_preprocessors": True,
                    "max_concurrent_workers": 1,
                },
            },
        )
    ]
    assert result["text"] == "Deutsche Aufnahme."
    assert result["language"] is None
    assert result["requested_language"] == "de"
    assert result["effective_language"] == "auto"
    assert result["detected_language"] is None
    assert result["warnings"] == ["requested_language_not_enforced"]
    assert result["model"] == PARAKEET_V3_MODEL
    assert result["segments"][0]["text"] == "Deutsche Aufnahme."
    assert result["segments"][0]["end"] == pytest.approx(2 / 16_000)


# ---------------------------------------------------------------------------
# TASK-1696: managed-first resolver for the batch path, when neither an
# explicit model_dir argument nor transcription.parakeet_onnx_model_dir is
# configured. Order: active managed artifact, then verified legacy bundle,
# then the existing "no model will be downloaded automatically" error.
# ---------------------------------------------------------------------------


def test_parakeet_onnx_batch_uses_active_managed_artifact_when_unconfigured(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    managed_dir = tmp_path / "managed"
    _write_model_bundle(managed_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            return "Managed model transcript."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )
    monkeypatch.setattr(
        service_module, "active_managed_parakeet_v2_dir", lambda: managed_dir
    )

    def unexpected_legacy_check(*_args, **_kwargs):
        raise AssertionError(
            "the verified legacy bundle must not be consulted once a managed "
            "artifact is active"
        )

    monkeypatch.setattr(
        service_module, "verify_parakeet_v2_bundle", unexpected_legacy_check
    )

    service = TranscriptionService()
    assert not service.config["parakeet_onnx_model_dir"]
    result = service.transcribe(str(audio_path), provider="parakeet-onnx")

    assert load_calls[0][1]["path"] == str(managed_dir)
    assert result["text"] == "Managed model transcript."


def test_parakeet_onnx_batch_falls_back_to_verified_legacy_bundle(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    legacy_dir = tmp_path / "legacy"
    _write_model_bundle(legacy_dir)
    _write_silent_wav(audio_path)
    load_calls = []

    class FakeModel:
        def recognize(self, path):
            return "Legacy bundle transcript."

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=fake_load_model)
    )
    monkeypatch.setattr(
        service_module, "active_managed_parakeet_v2_dir", lambda: None
    )
    monkeypatch.setattr(service_module, "parakeet_v2_install_dir", lambda: legacy_dir)
    monkeypatch.setattr(
        service_module,
        "verify_parakeet_v2_bundle",
        lambda directory: directory == legacy_dir,
    )

    service = TranscriptionService()
    assert not service.config["parakeet_onnx_model_dir"]
    result = service.transcribe(str(audio_path), provider="parakeet-onnx")

    assert load_calls[0][1]["path"] == str(legacy_dir)
    assert result["text"] == "Legacy bundle transcript."


def test_parakeet_onnx_batch_reports_missing_model_when_nothing_resolves(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "speech.wav"
    _write_silent_wav(audio_path)

    monkeypatch.setattr(service_module, "ONNX_ASR_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        service_module, "active_managed_parakeet_v2_dir", lambda: None
    )
    monkeypatch.setattr(
        service_module, "parakeet_v2_install_dir", lambda: tmp_path / "no-legacy"
    )
    monkeypatch.setattr(service_module, "verify_parakeet_v2_bundle", lambda directory: False)

    service = TranscriptionService()
    assert not service.config["parakeet_onnx_model_dir"]
    with pytest.raises(TranscriptionError, match="no model will be downloaded"):
        service.transcribe(str(audio_path), provider="parakeet-onnx")
