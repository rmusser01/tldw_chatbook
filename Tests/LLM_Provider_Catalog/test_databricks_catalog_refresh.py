"""Databricks model catalog auto-refresh + native tool membership (ADR-179 Task 12).

Ports the strict-hosted catalog-service pattern the moonshot/zai suites use
(``test_local_llm_provider_catalog_service.py``): Databricks resolves its
workspace endpoint and credential through the SAME engine seam the chat path
dispatches through (``resolve_hosted_engine_request``), so discovery's base
URL and key match a real send exactly -- including the ``/openai/v1`` suffix
append a bare workspace host goes through.
"""

import pytest

from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    ModelCatalogSettings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ModelDiscoveryResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    build_models_url,
)


def test_auto_refresh_list_includes_databricks():
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        AUTO_REFRESH_PROVIDER_LIST_KEYS,
    )

    # Entry style is the [providers] spelling, matching the registry parity
    # test's config-key comparison ("Databricks", like "Moonshot"/"ZAI").
    assert "Databricks" in AUTO_REFRESH_PROVIDER_LIST_KEYS


def test_native_tools_includes_databricks():
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS

    assert "databricks" in NATIVE_TOOLS_PROVIDERS


def _databricks_service(*, api_base_url: str, seen: list[dict]) -> LocalLLMProviderCatalogService:
    """Build a catalog service with one Databricks workspace configured."""

    async def fake_client(**kwargs):
        seen.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint="fp",
            status="success",
            models=(),
        )

    return LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"Databricks": ["databricks-gpt-4o"]},
        settings_loader=lambda: {
            "providers": {"Databricks": ["databricks-gpt-4o"]},
            "api_settings": {
                "databricks": {"api_base_url": api_base_url},
            },
        },
        discovery_client=fake_client,
        environ={"DATABRICKS_TOKEN": "catalog-secret-canary"},
    )


@pytest.mark.asyncio
async def test_catalog_service_builds_models_url_from_workspace_host():
    seen: list[dict] = []
    service = _databricks_service(
        api_base_url="https://dbc-1.cloud.databricks.com", seen=seen
    )

    result = await service.discover_models(provider="Databricks")

    assert result.status == "success"
    assert len(seen) == 1
    # The engine resolver appends the /openai/v1 suffix to the bare workspace
    # host; build_models_url then appends the discovery route ("models").
    assert (
        build_models_url(seen[0]["endpoint"], seen[0]["provider"])
        == "https://dbc-1.cloud.databricks.com/openai/v1/models"
    )
    assert seen[0]["api_key"] == "catalog-secret-canary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_base_url",
    [
        # Bare workspace host: the engine's base_url_suffix append must fire.
        "https://dbc-1.cloud.databricks.com",
        # Already-suffixed base: passed through unchanged (no double suffix).
        "https://dbc-1.cloud.databricks.com/openai/v1",
    ],
)
async def test_bare_workspace_host_resolution_goes_through_engine_suffix_append(
    configured_base_url,
):
    seen: list[dict] = []
    service = _databricks_service(api_base_url=configured_base_url, seen=seen)

    result = await service.discover_models(provider="Databricks")

    assert result.status == "success"
    assert len(seen) == 1
    assert seen[0]["endpoint"] == "https://dbc-1.cloud.databricks.com/openai/v1"


@pytest.mark.asyncio
async def test_auto_refresh_loop_refreshes_databricks(tmp_path):
    """The refresh loop refreshes databricks through the engine resolution."""
    seen: list[dict] = []
    service = _databricks_service(
        api_base_url="https://dbc-1.cloud.databricks.com", seen=seen
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")

    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(refresh_consent_recorded=True),
        disk_store=store,
        provider_list_keys=("Databricks",),
    )

    assert [(o.provider_list_key, o.status) for o in report.outcomes] == [
        ("Databricks", "refreshed")
    ]
    assert len(seen) == 1
    assert seen[0]["endpoint"] == "https://dbc-1.cloud.databricks.com/openai/v1"
