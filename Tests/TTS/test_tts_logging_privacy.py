from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.TTS as tts
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.TTS.legacy_bridge import LEGACY_ROUTES

GUIDE_PATH = Path(__file__).parents[2] / "Docs/Development/TTS/TTS_MODULE_GUIDE.md"


def test_tts_package_exports_only_stable_adapter_service_api() -> None:
    expected = {
        "NormalizationOptions",
        "OpenAISpeechRequest",
        "ProgressSink",
        "ProviderHealth",
        "TTSAudioResponse",
        "TTSModelInfo",
        "TTSProgress",
        "TTSProviderCatalog",
        "TTSProviderDescriptor",
        "TTSRequest",
        "TTSService",
        "bind_tts_service",
        "close_tts_resources",
        "get_tts_service",
        "reset_tts_service_binding",
    }
    forbidden = {
        "BackendRegistry",
        "LegacyBackendHost",
        "LegacyTTSAdapter",
        "OpenAITTSBackend",
        "TTSBackendBase",
        "TTSBackendManager",
    }

    assert set(tts.__all__) == expected
    assert all(hasattr(tts, name) for name in expected)
    assert all(not hasattr(tts, name) for name in forbidden)


def test_tts_guide_documents_exact_legacy_routes_and_working_example() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    architecture = guide.split("### TTS adapter service", 1)[1].split(
        "### Module Structure", 1
    )[0]
    normalized_architecture = " ".join(architecture.split())
    usage = guide.split("### Programmatic Usage", 1)[1].split(
        "### Event System Integration", 1
    )[0]
    routes = guide.split("### Exact legacy route allowlist", 1)[1].split(
        "### Audio Formats", 1
    )[0]
    documented_routes = dict(
        re.findall(r"^- `([^`]+)` → `([^`]+)`$", routes, re.MULTILINE)
    )

    assert (
        "Native adapters use canonical provider IDs and "
        "`TTSService.synthesize()`." in normalized_architecture
    )
    assert (
        "Until `audio_cpp` lands, all currently registered providers are "
        "compatibility adapters and callers use `generate_audio_stream()` "
        "with an enumerated legacy internal model ID." in normalized_architecture
    )
    assert documented_routes == LEGACY_ROUTES
    assert 'internal_model_id = "openai_official_tts-1"' in usage
    assert "generate_audio_stream(request, internal_model_id)" in usage
    assert "tts_service.synthesize(" not in usage


@pytest.mark.asyncio
async def test_openai_backend_never_logs_api_key_details(monkeypatch) -> None:
    secret = "sk-UniquePrefix-Extremely-Private-Suffix"
    messages: list[str] = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        lambda: {"api_settings": {"openai": {"api_key": secret}}},
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    backend = None
    try:
        backend = OpenAITTSBackend(config={})
    finally:
        if backend is not None:
            await backend.close()
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert secret not in rendered
    assert secret[:10] not in rendered
    assert secret[-10:] not in rendered
    assert hashlib.sha256(secret.encode()).hexdigest() not in rendered
    assert "API key length" not in rendered
