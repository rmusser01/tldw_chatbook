from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import httpx
import pytest

from tldw_chatbook.TTS import TTS_Backends
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


def test_saved_elevenlabs_settings_reach_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    manager = TTSBackendManager(
        {
            "API": {"elevenlabs_api_key": "xi-saved"},
            "app_tts": {
                "ELEVENLABS_DEFAULT_MODEL": "eleven_turbo_v2_5",
                "ELEVENLABS_OUTPUT_FORMAT": "pcm_24000",
                "ELEVENLABS_VOICE_STABILITY": 0.4,
                "ELEVENLABS_SIMILARITY_BOOST": 0.7,
                "ELEVENLABS_STYLE": 0.2,
                "ELEVENLABS_USE_SPEAKER_BOOST": False,
            },
        }
    )

    config = manager._prepare_backend_config("elevenlabs_eleven_turbo_v2_5")

    assert config["ELEVENLABS_API_KEY"] == "xi-saved"
    assert config["ELEVENLABS_DEFAULT_MODEL"] == "eleven_turbo_v2_5"
    assert config["ELEVENLABS_OUTPUT_FORMAT"] == "pcm_24000"
    assert config["ELEVENLABS_VOICE_STABILITY"] == 0.4
    assert config["ELEVENLABS_SIMILARITY_BOOST"] == 0.7
    assert config["ELEVENLABS_STYLE"] == 0.2
    assert config["ELEVENLABS_USE_SPEAKER_BOOST"] is False


def test_saved_kokoro_app_tts_settings_reach_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KOKORO_MODEL_PATH", raising=False)
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)
    manager = TTSBackendManager(
        {
            "app_tts": {
                "KOKORO_DEVICE_DEFAULT": "mps",
                "KOKORO_USE_ONNX": True,
                "KOKORO_ONNX_MODEL_PATH_DEFAULT": "/models/kokoro.onnx",
                "KOKORO_ONNX_VOICES_JSON_DEFAULT": "/models/voices.json",
                "KOKORO_MAX_TOKENS": 777,
                "KOKORO_ENABLE_VOICE_MIXING": True,
                "KOKORO_TRACK_PERFORMANCE": False,
            }
        }
    )

    config = manager._prepare_backend_config("local_kokoro_default_onnx")

    assert config["KOKORO_DEVICE"] == "mps"
    assert config["KOKORO_USE_ONNX"] is True
    assert config["KOKORO_MODEL_PATH"] == "/models/kokoro.onnx"
    assert config["KOKORO_VOICES_JSON_PATH"] == "/models/voices.json"
    assert config["KOKORO_MAX_TOKENS"] == 777
    assert config["KOKORO_ENABLE_VOICE_MIXING"] is True
    assert config["KOKORO_TRACK_PERFORMANCE"] is False


def test_saved_higgs_section_settings_reach_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIGGS_MODEL_PATH", raising=False)
    manager = TTSBackendManager(
        {
            "HiggsSettings": {
                "model_path": "saved/higgs-model",
                "voice_samples_dir": "/voices",
                "device": "mps",
                "enable_flash_attn": False,
                "dtype": "float16",
                "max_reference_duration": 22,
                "default_language": "es",
                "enable_voice_cloning": False,
                "enable_multi_speaker": False,
                "speaker_delimiter": "::",
                "track_performance": False,
                "max_new_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.25,
            }
        }
    )
    monkeypatch.setattr(manager, "_check_cuda_available", lambda: False)

    config = manager._prepare_backend_config("local_higgs_v2")

    assert config["HIGGS_MODEL_PATH"] == "saved/higgs-model"
    assert config["HIGGS_VOICE_SAMPLES_DIR"] == "/voices"
    assert config["HIGGS_DEVICE"] == "mps"
    assert config["HIGGS_ENABLE_FLASH_ATTN"] is False
    assert config["HIGGS_DTYPE"] == "float16"
    assert config["HIGGS_MAX_REFERENCE_DURATION"] == 22
    assert config["HIGGS_DEFAULT_LANGUAGE"] == "es"
    assert config["HIGGS_ENABLE_VOICE_CLONING"] is False
    assert config["HIGGS_ENABLE_MULTI_SPEAKER"] is False
    assert config["HIGGS_SPEAKER_DELIMITER"] == "::"
    assert config["HIGGS_TRACK_PERFORMANCE"] is False
    assert config["HIGGS_MAX_NEW_TOKENS"] == 2048
    assert config["HIGGS_TEMPERATURE"] == 0.3
    assert config["HIGGS_TOP_P"] == 0.8
    assert config["HIGGS_REPETITION_PENALTY"] == 1.25


def test_missing_builtin_class_does_not_abort_remaining_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_module = "tldw_chatbook.TTS.backends.openai"
    original_import_module = importlib.import_module
    warnings: list[tuple[str, tuple[object, ...]]] = []

    def import_module(module_name: str) -> ModuleType:
        if module_name == missing_module:
            return ModuleType(module_name)
        return original_import_module(module_name)

    monkeypatch.setattr(TTS_Backends.importlib, "import_module", import_module)
    monkeypatch.setattr(
        TTS_Backends.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    loaded = set(BackendRegistry.ensure_builtins())

    assert loaded == EXPECTED_LEGACY_IDS - {"openai_official_*"}
    assert (
        "Legacy TTS backend is unavailable: {}",
        ("openai_official_*",),
    ) in warnings


def test_builtin_import_attribute_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_module = "tldw_chatbook.TTS.backends.openai"
    original_import_module = importlib.import_module

    def import_module(module_name: str) -> ModuleType:
        if module_name == missing_module:
            raise AttributeError("backend import bug")
        return original_import_module(module_name)

    monkeypatch.setattr(TTS_Backends.importlib, "import_module", import_module)

    with pytest.raises(AttributeError, match="backend import bug"):
        BackendRegistry.ensure_builtins()


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
