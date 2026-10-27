"""TASK-34100.2 AC#4 (gap-01): setup lists models for engine presets.

The first-run review found NVIDIA NIM and Novita (and, by harness, all 32
engine presets) reporting "Couldn't reach the server (request failed)" for a
VALID key: setup staged the discovery endpoint as the chat URL
(``.../v1/chat/completions``), and the hosted engine refuses a pasted
``/chat/completions`` base, so discovery failed as ``invalid_provider_settings``
before any request. Gemini's base was mangled the same way
(``.../v1beta/v1/chat/completions``). Setup must hand discovery the base URL --
what persistence writes for these providers -- and the catalog service must
then reach the provider. The discovery client here records the request
instead of sending it.
"""

from __future__ import annotations

import copy

import pytest

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ModelDiscoveryResult,
)
from tldw_chatbook.provider_registry import ENGINE_RECORDS
from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state

_PRESETS = [record for record in ENGINE_RECORDS if record.key != "custom-hosted"]


def _draft(provider: str) -> setup_state.FirstRunProviderDraft:
    return setup_state.FirstRunProviderDraft(
        provider=provider,
        endpoint="",
        credential=setup_state.ProviderCredentialDraft(
            "draft", "test-key-engine-preset-discovery", 1
        ),
    )


def _staged(resolved: setup_state.FirstRunProviderDraft, key) -> dict:
    """What the wizard stages for discovery: the wizard's own function.

    Review round 1 (TE-2): this used to be a hand-written copy of the
    staging, so flipping the real function back to the chat URL (the gap-01
    defect) left every test here green.
    """
    from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
        _first_run_discovery_staged_settings,
    )

    return _first_run_discovery_staged_settings(resolved, key)


@pytest.mark.parametrize("record", _PRESETS, ids=lambda record: record.key)
def test_engine_preset_discovery_is_staged_as_the_base_url(record) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    resolved = setup_state.resolve_first_run_provider_draft(_draft(record.key), config)

    if record.default_base_url is None:
        pytest.skip(f"{record.key} has no shipped base URL (per-account host)")
    assert not resolved.discovery_endpoint.endswith("/chat/completions")
    assert resolved.discovery_endpoint.rstrip("/") == record.default_base_url.rstrip(
        "/"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("record", _PRESETS, ids=lambda record: record.key)
async def test_engine_preset_discovery_reaches_the_provider(record) -> None:
    if record.default_base_url is None:
        pytest.skip(f"{record.key} has no shipped base URL (per-account host)")
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    resolved = setup_state.resolve_first_run_provider_draft(_draft(record.key), config)
    key = setup_state.build_first_run_model_discovery_key(resolved)
    sent: list[dict] = []

    async def client(**kwargs) -> ModelDiscoveryResult:
        sent.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=kwargs["endpoint"],
            status="success",
        )

    service = LocalLLMProviderCatalogService(
        settings_loader=lambda: config,
        provider_catalog_loader=lambda: copy.deepcopy(config["providers"]),
        discovery_client=client,
        environ={},
    )
    result = await service.discover_models(
        provider=record.key,
        staged_settings=_staged(resolved, key),
        use_shared_cache=False,
    )

    if record.discovery_route is None:
        # Seeded-only presets document no models route: "listing
        # unavailable" (type a model id), never "couldn't reach the server".
        assert result.status == "unsupported"
        assert result.error.kind == "unsupported_endpoint"
        assert sent == []
        return
    assert result.error is None, result.error
    assert result.status == "success"
    assert sent and sent[0]["api_key"] == "test-key-engine-preset-discovery"
    assert not sent[0]["endpoint"].endswith("/chat/completions")


def test_gemini_discovery_is_staged_as_googles_own_api_base() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    resolved = setup_state.resolve_first_run_provider_draft(_draft("google"), config)

    assert resolved.discovery_endpoint == (
        "https://generativelanguage.googleapis.com/v1beta"
    )
    # The discovery identity is unchanged: the same connection the send uses.
    key = setup_state.build_first_run_model_discovery_key(resolved)
    assert key.provider_key == "google"


def test_openai_compatible_cloud_discovery_staging_is_unchanged() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    resolved = setup_state.resolve_first_run_provider_draft(_draft("openai"), config)

    assert resolved.discovery_endpoint == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "models_url"),
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1/models"),
        ("novita", "https://api.novita.ai/openai/v1/models"),
        ("together", "https://api.together.xyz/v1/models"),
    ],
)
async def test_the_real_client_lists_each_preset_at_its_models_url(
    provider: str, models_url: str
) -> None:
    """Review round 1 (C11): the test above stubs the discovery client, so it
    cannot show the real client builds each preset's models URL from the staged
    base (Novita's lives under /openai/v1). Here the real client runs; only the
    transport is an ``httpx.MockTransport``."""
    import httpx

    from tldw_chatbook.LLM_Provider_Catalog import (
        openai_compatible_model_discovery as discovery_module,
    )

    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    resolved = setup_state.resolve_first_run_provider_draft(_draft(provider), config)
    key = setup_state.build_first_run_model_discovery_key(resolved)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "vendor/model-a"}]})

    async def client(**kwargs) -> ModelDiscoveryResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await discovery_module.discover_openai_compatible_models(
                **kwargs, client=http
            )

    service = LocalLLMProviderCatalogService(
        settings_loader=lambda: config,
        provider_catalog_loader=lambda: copy.deepcopy(config["providers"]),
        discovery_client=client,
        environ={},
    )
    result = await service.discover_models(
        provider=provider, staged_settings=_staged(resolved, key), use_shared_cache=False
    )

    assert result.status == "success", result.error
    assert [str(request.url) for request in seen] == [models_url]
    assert seen[0].headers["Authorization"] == "Bearer test-key-engine-preset-discovery"
