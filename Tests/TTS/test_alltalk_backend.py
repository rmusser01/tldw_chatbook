from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from loguru import logger

from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends import alltalk as alltalk_module
from tldw_chatbook.TTS.backends.alltalk import AllTalkTTSBackend
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_specs
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app_tts", "configured"),
    [
        (
            {"ALLTALK_TTS_URL_DEFAULT": "https://leased-default.example:9443"},
            "https://leased-default.example:9443",
        ),
        (
            {
                "ALLTALK_TTS_URL": "https://leased-runtime.example:9444",
                "ALLTALK_TTS_URL_DEFAULT": "https://ignored-default.example:9555",
            },
            "https://leased-runtime.example:9444",
        ),
    ],
)
async def test_alltalk_uses_immutable_leased_default_for_actual_request(
    monkeypatch: pytest.MonkeyPatch,
    app_tts: dict[str, str],
    configured: str,
) -> None:
    global_value = "https://stale-global.example.test:9555"
    monkeypatch.setattr(
        alltalk_module,
        "get_cli_setting",
        lambda *_args, **_kwargs: global_value,
        raising=False,
    )
    requested: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b"audio")

    backends: list[AllTalkTTSBackend] = []
    replaced_clients: list[httpx.AsyncClient] = []

    class Manager:
        def __init__(self, backend: AllTalkTTSBackend) -> None:
            self.backend = backend

        async def get_backend(self, _backend_id: str) -> AllTalkTTSBackend:
            return self.backend

        async def close_all_backends(self) -> None:
            await self.backend.close()

    def manager_factory(provider_id: str, config: dict):
        assert provider_id == "alltalk"
        backend = AllTalkTTSBackend(config["app_tts"])
        backends.append(backend)
        replaced_clients.append(backend.client)
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return Manager(backend)

    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": app_tts}},
            manager_factory=manager_factory,
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="alltalk",
            model_mode="exact",
            model_id="alltalk",
            voice_mode="exact",
            voice_id="female_01.wav",
            response_format="wav",
            speed=1.0,
        ),
    )
    try:
        response = await service.synthesize_default(
            text="Leased reply.",
        )
        audio = await _collect(response.byte_stream)
        await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()
        for client in replaced_clients:
            await client.aclose()

    assert audio == b"audio"
    assert requested == [f"{configured}/v1/audio/speech"]
    assert len(backends) == 1


@pytest.mark.asyncio
async def test_alltalk_production_manager_authorizes_actual_effective_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = "https://canonical-alltalk.example.test:9443"
    conflicting = "https://model-override.example.test:9555"
    requested: list[str] = []
    authorized: list[tuple[str, str]] = []
    real_async_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b"audio")

    def client_factory(*args, **kwargs) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_async_client(*args, **kwargs)

    async def initialize(_backend: AllTalkTTSBackend) -> None:
        return None

    monkeypatch.setattr(alltalk_module.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(AllTalkTTSBackend, "initialize", initialize)
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {
                "COMPREHENSIVE_CONFIG_RAW": {
                    "app_tts": {"ALLTALK_TTS_URL": canonical},
                    "alltalk_default": {"ALLTALK_TTS_URL": conflicting},
                    "alltalk_alltalk": {
                        "ALLTALK_TTS_URL_DEFAULT": conflicting,
                    },
                }
            }
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="alltalk",
            model_mode="exact",
            model_id="alltalk",
            voice_mode="exact",
            voice_id="female_01.wav",
            response_format="wav",
            speed=1.0,
        ),
    )

    def authorize(provider_id: str, endpoint: str) -> bool:
        authorized.append((provider_id, endpoint))
        return True

    try:
        response = await service.synthesize_default(
            text="Production manager route.",
            admission_authorizer=authorize,
        )
        audio = await _collect(response.byte_stream)
        await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()

    assert audio == b"audio"
    assert authorized == [("alltalk", canonical)]
    assert requested == [f"{canonical}/v1/audio/speech"]


@pytest.mark.asyncio
async def test_alltalk_logs_never_expose_raw_origin_or_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_user = "PRIVATE_USER_4e71"
    secret_password = "PRIVATE_PASSWORD_9f20"
    secret_path = "PRIVATE_PATH_b230"
    raw = (
        f"https://{secret_user}:{secret_password}@voice.example.test:9443/"
        f"{secret_path}?token=PRIVATE_QUERY_0ac1"
    )
    captured: list[str] = []
    caplog.set_level(logging.DEBUG)
    sink = logger.add(captured.append, level="DEBUG", format="{message}")
    backend = AllTalkTTSBackend({"ALLTALK_TTS_URL": raw})
    await backend.client.aclose()

    def fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            content=(f"provider repeated {raw}").encode(),
        )

    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(ValueError):
            await _collect(
                backend.generate_speech_stream(
                    OpenAISpeechRequest(input="Private reply.", response_format="wav")
                )
            )
    finally:
        await backend.close()
        logger.remove(sink)

    rendered = "".join(captured) + caplog.text
    for private_value in (
        raw,
        secret_user,
        secret_password,
        secret_path,
        "PRIVATE_QUERY_0ac1",
    ):
        assert private_value not in rendered
