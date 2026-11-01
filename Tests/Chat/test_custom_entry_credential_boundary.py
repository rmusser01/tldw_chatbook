"""Registry destinations own credentials, including an explicit keyless result."""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path

import httpx
import pytest
import requests
import toml

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderSelection,
)
from tldw_chatbook.Utils.config_encryption import ConfigEncryption

FAMILY_URL = "http://127.0.0.1:8121"
ENTRY_URL = "http://127.0.0.1:8122"
FAMILY_KEY = "synthetic-family-only-key"
ENTRY_KEY = "synthetic-entry-stored-key"
ENTRY_ENV_KEY = "synthetic-entry-env-key"
ROUTES = [
    pytest.param("llama_cpp", True, "llama_cpp", id="direct-llama"),
    pytest.param("openai_compatible", True, "custom-hosted", id="compatible-engine"),
    pytest.param(
        "openai_compatible", False, "custom-openai-api", id="compatible-legacy"
    ),
    pytest.param("ollama", True, "ollama", id="ollama"),
]


pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def selected_config():
    """Load a real TOML document through the admitted isolated bootstrap source."""
    from tldw_chatbook import config as module

    def select(document):
        path = module.get_cli_config_path()
        assert path.resolve().is_relative_to(
            Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).resolve()
        )
        path.write_text(toml.dumps(document), encoding="utf-8")
        path.chmod(0o600)
        module.load_settings(force_reload=True)
        return module.load_cli_config_and_ensure_existence(force_reload=True)

    return select


@pytest.fixture
def transport(monkeypatch):
    """Capture HTTPX and real Requests adapter output at the transport boundary."""
    seen = []

    def handle(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "selected",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        return httpx.Response(200, json={"data": [{"id": "selected"}]})

    mock = httpx.MockTransport(handle)
    with httpx.Client(transport=mock) as client:

        def send(_session, request, **_kwargs):
            # Keep real provider serialization/header selection; replace only I/O.
            reply = client.request(
                request.method,
                request.url,
                headers=dict(request.headers),
                content=request.body,
            )
            response = requests.Response()
            response.status_code = reply.status_code
            response._content = reply.content
            response.raw = io.BytesIO(reply.content)
            response.headers.update(reply.headers)
            response.url = str(reply.url)
            response.request = request
            response.encoding = "utf-8"
            return response

        monkeypatch.setattr(requests.Session, "send", send)
        yield mock, seen


def document(family, engine, *, family_stranded=False, entry_credential="keyless"):
    family_key = "custom" if family == "openai_compatible" else family
    key = (
        ConfigEncryption().encrypt_value(FAMILY_KEY, "synthetic-master")
        if family_stranded
        else FAMILY_KEY
    )
    entry = {
        "display_name": "Entry",
        "family": family,
        "base_url": ENTRY_URL,
        "models": ["selected"],
    }
    if entry_credential == "stored":
        entry["api_key"] = ENTRY_KEY
    elif entry_credential == "environment":
        entry.update(api_key=ENTRY_KEY, api_key_env="SYNTHETIC_ENTRY_KEY")
    elif entry_credential == "stranded":
        entry["api_key"] = ConfigEncryption().encrypt_value(
            ENTRY_KEY, "synthetic-master"
        )
    return {
        "console": {"custom_endpoints_use_engine": engine},
        "api_settings": {
            family_key: {
                "api_url": FAMILY_URL,
                "api_key": key,
                "credential_source": "stored",
                "model": "selected",
            }
        },
        "custom_endpoints": {"entry": entry},
    }


async def send_chat(gateway, resolution):
    output = [
        item
        async for item in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "synthetic hello"}],
            ephemeral=True,
        )
    ]
    assert output == ["ok"]
    pending = tuple(gateway._reasoning_metadata_refreshes.values())
    if pending:
        await asyncio.gather(*pending)


@pytest.mark.asyncio
@pytest.mark.parametrize("family,engine,execution", ROUTES)
@pytest.mark.parametrize(
    "family_stranded", [False, True], ids=["family-plain", "family-stranded"]
)
@pytest.mark.parametrize("credential", ["keyless", "stored", "environment"])
async def test_registry_entry_owns_resolution_health_and_chat_credentials(
    selected_config,
    transport,
    family,
    engine,
    execution,
    family_stranded,
    credential,
):
    config = selected_config(
        document(
            family, engine, family_stranded=family_stranded, entry_credential=credential
        )
    )
    expected_key = {"keyless": None, "stored": ENTRY_KEY, "environment": ENTRY_ENV_KEY}[
        credential
    ]
    expected_source = {
        "keyless": None,
        "stored": "config:custom_endpoints.entry.api_key",
        "environment": "env:SYNTHETIC_ENTRY_KEY",
    }[credential]
    mock, seen = transport
    async with httpx.AsyncClient(transport=mock) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            config_provider=lambda: config,
            environ={"SYNTHETIC_ENTRY_KEY": ENTRY_ENV_KEY},
        )
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider="custom-ep:entry",
                explicit_model="selected",
                max_tokens=64,
                streaming=False,
            )
        )
        assert resolution.ready, resolution.visible_copy
        assert resolution.execution_key == execution
        if family == "llama_cpp":
            health = [request for request in seen if request.url.path == "/health"]
            assert len(health) == 1
            assert str(health[0].url) == ENTRY_URL + "/health"
            assert health[0].headers.get("Authorization") == (
                f"Bearer {expected_key}" if expected_key else None
            )
        assert resolution.api_key == expected_key
        assert resolution.api_key_source == expected_source
        await send_chat(gateway, resolution)

    posts = [request for request in seen if request.method == "POST"]
    assert len(posts) == 1
    assert str(posts[0].url) == ENTRY_URL + "/v1/chat/completions"
    assert posts[0].headers.get("Authorization") == (
        f"Bearer {expected_key}" if expected_key else None
    )
    assert seen
    assert all(
        request.url.host == "127.0.0.1" and request.url.port == 8122 for request in seen
    )
    assert all(
        request.headers.get("Authorization")
        == (f"Bearer {expected_key}" if expected_key else None)
        for request in seen
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("family,engine,execution", ROUTES)
async def test_stranded_entry_still_blocks_before_any_transport(
    selected_config,
    transport,
    family,
    engine,
    execution,
):
    config = selected_config(
        document(family, engine, family_stranded=True, entry_credential="stranded")
    )
    mock, seen = transport
    async with httpx.AsyncClient(transport=mock) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: config, environ={}
        )
        result = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider="custom-ep:entry",
                explicit_model="selected",
                max_tokens=64,
            )
        )
    assert not result.ready
    assert result.api_key is None
    assert "still encrypted" in result.visible_copy
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["llama_cpp", "openai_compatible", "ollama"])
async def test_ordinary_family_keeps_its_own_destination_and_stored_credential(
    selected_config,
    transport,
    family,
):
    config = selected_config(document(family, True))
    provider = "custom" if family == "openai_compatible" else family
    mock, seen = transport
    async with httpx.AsyncClient(transport=mock) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: config, environ={}
        )
        result = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=provider,
                base_url=FAMILY_URL,
                explicit_model="selected",
                max_tokens=64,
                streaming=False,
            )
        )
        assert result.ready, result.visible_copy
        assert result.api_key == FAMILY_KEY
        assert result.api_key_source == f"config:api_settings.{provider}.api_key"
        await send_chat(gateway, result)
    posts = [request for request in seen if request.method == "POST"]
    assert len(posts) == 1
    assert str(posts[0].url) == FAMILY_URL + "/v1/chat/completions"
    assert posts[0].headers.get("Authorization") == f"Bearer {FAMILY_KEY}"
    assert seen
    assert all(request.url.port == 8121 for request in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["llama_cpp", "openai_compatible", "ollama"])
async def test_ordinary_stranded_family_still_blocks_before_any_transport(
    selected_config,
    transport,
    family,
):
    config = selected_config(document(family, True, family_stranded=True))
    provider = "custom" if family == "openai_compatible" else family
    mock, seen = transport
    async with httpx.AsyncClient(transport=mock) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: config, environ={}
        )
        result = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=provider,
                base_url=FAMILY_URL,
                explicit_model="selected",
                max_tokens=64,
            )
        )
    assert not result.ready
    assert result.api_key is None
    assert "still encrypted" in result.visible_copy
    assert seen == []
