from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import time

import httpx
import pytest

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import BackendRegistry, TTSBackendManager
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend


EXPECTED_LEGACY_IDS = {
    "openai_official_*",
    "local_kokoro_*",
    "elevenlabs_*",
    "local_chatterbox_*",
    "alltalk_*",
    "local_higgs_*",
}


@pytest.fixture(autouse=True)
def reset_legacy_registry_state() -> Iterator[None]:
    BackendRegistry._registry.clear()
    BackendRegistry._builtins_loaded = False
    yield
    BackendRegistry._registry.clear()
    BackendRegistry._builtins_loaded = False


def test_legacy_registry_is_closed_to_new_providers() -> None:
    BackendRegistry.ensure_builtins()

    with pytest.raises(RuntimeError, match="sealed legacy registry"):
        BackendRegistry.register("new_provider_*", object)  # type: ignore[arg-type]


def test_legacy_registry_has_exact_routes_and_manager_lookup() -> None:
    manager = TTSBackendManager({})
    first = tuple(BackendRegistry.ensure_builtins())
    second = tuple(BackendRegistry.ensure_builtins())

    assert first == second
    assert len(first) == len(set(first))
    assert set(first) == EXPECTED_LEGACY_IDS
    assert set(manager.list_available_backends()) == EXPECTED_LEGACY_IDS
    assert BackendRegistry.get("openai_official_tts-1") is OpenAITTSBackend


def test_concurrent_manager_construction_loads_builtins_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = 0
    original_load = BackendRegistry._load_builtin_classes

    def counted_load(cls: type[BackendRegistry]) -> None:
        nonlocal load_calls
        load_calls += 1
        time.sleep(0.01)
        original_load()

    monkeypatch.setattr(
        BackendRegistry,
        "_load_builtin_classes",
        classmethod(counted_load),
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        managers = list(pool.map(lambda _: TTSBackendManager({}), range(16)))

    assert len(managers) == 16
    assert load_calls == 1
    assert set(BackendRegistry.list_backends()) == EXPECTED_LEGACY_IDS


def test_legacy_registry_exposes_no_test_only_reset_hook() -> None:
    assert not hasattr(BackendRegistry, "_reset_for_tests")


def test_higgs_backend_config_consumes_nested_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIGGS_MODEL_PATH", raising=False)
    manager = TTSBackendManager(
        {
            "HiggsSettings": {
                "model_path": "custom/higgs",
                "device": "cpu",
                "enable_flash_attn": False,
                "dtype": "float32",
                "voice_samples_dir": "/tmp/higgs-voices",
                "enable_voice_cloning": False,
                "max_reference_duration": 12,
                "default_language": "de",
                "enable_multi_speaker": False,
                "speaker_delimiter": "###",
                "track_performance": False,
                "max_new_tokens": 2048,
                "temperature": 0.4,
                "top_p": 0.8,
                "repetition_penalty": 1.3,
            }
        }
    )

    config = manager._prepare_backend_config("local_higgs_default")

    assert config["HIGGS_MODEL_PATH"] == "custom/higgs"
    assert config["HIGGS_DEVICE"] == "cpu"
    assert config["HIGGS_ENABLE_FLASH_ATTN"] is False
    assert config["HIGGS_DTYPE"] == "float32"
    assert config["HIGGS_VOICE_SAMPLES_DIR"] == "/tmp/higgs-voices"
    assert config["HIGGS_ENABLE_VOICE_CLONING"] is False
    assert config["HIGGS_MAX_REFERENCE_DURATION"] == 12
    assert config["HIGGS_DEFAULT_LANGUAGE"] == "de"
    assert config["HIGGS_ENABLE_MULTI_SPEAKER"] is False
    assert config["HIGGS_SPEAKER_DELIMITER"] == "###"
    assert config["HIGGS_TRACK_PERFORMANCE"] is False
    assert config["HIGGS_MAX_NEW_TOKENS"] == 2048
    assert config["HIGGS_TEMPERATURE"] == 0.4
    assert config["HIGGS_TOP_P"] == 0.8
    assert config["HIGGS_REPETITION_PENALTY"] == 1.3


def test_kokoro_backend_config_consumes_app_tts_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KOKORO_MODEL_PATH", raising=False)
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)
    manager = TTSBackendManager(
        {
            "app_tts": {
                "KOKORO_ONNX_MODEL_PATH_DEFAULT": "custom/kokoro.onnx",
                "KOKORO_ONNX_VOICES_JSON_DEFAULT": "custom/voices.json",
                "KOKORO_DEVICE_DEFAULT": "cuda",
                "KOKORO_MAX_TOKENS": 999,
                "KOKORO_ENABLE_VOICE_MIXING": True,
                "KOKORO_TRACK_PERFORMANCE": False,
            }
        }
    )

    config = manager._prepare_backend_config("local_kokoro_default_onnx")

    assert config["KOKORO_MODEL_PATH"] == "custom/kokoro.onnx"
    assert config["KOKORO_VOICES_JSON_PATH"] == "custom/voices.json"
    assert config["KOKORO_DEVICE"] == "cuda"
    assert config["KOKORO_MAX_TOKENS"] == 999
    assert config["KOKORO_ENABLE_VOICE_MIXING"] is True
    assert config["KOKORO_TRACK_PERFORMANCE"] is False


@pytest.mark.asyncio
async def test_openai_backend_uses_configured_endpoint_and_organization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"audio")

    backend = OpenAITTSBackend(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://tts.example.test/v1/audio/speech",
            "OPENAI_ORG_ID": "org-test",
        }
    )
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        chunks = [
            chunk
            async for chunk in backend.generate_speech_stream(
                OpenAISpeechRequest(
                    model="tts-1",
                    input="hello",
                    voice="alloy",
                    response_format="wav",
                )
            )
        ]
    finally:
        await backend.close()

    assert chunks == [b"audio"]
    assert str(requests[0].url) == "https://tts.example.test/v1/audio/speech"
    assert requests[0].headers["OpenAI-Organization"] == "org-test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    (
        {"OPENAI_BASE_URL": "relative/speech"},
        {"OPENAI_BASE_URL": "ftp://example.test/speech"},
        {"OPENAI_BASE_URL": "https://user:secret@example.test/speech"},
        {"OPENAI_BASE_URL": "https://example.test/speech#fragment"},
        {"OPENAI_ORG_ID": "org-test\r\nInjected: value"},
    ),
)
async def test_openai_backend_rejects_unsafe_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = None
    try:
        with pytest.raises(ValueError, match="OpenAI"):
            backend = OpenAITTSBackend({"OPENAI_API_KEY": "test-key", **config})
    finally:
        if backend is not None:
            await backend.close()
