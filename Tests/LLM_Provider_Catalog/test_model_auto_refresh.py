from datetime import UTC, datetime

import pytest
from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
    ProviderRefreshOutcome,
    RefreshReport,
    format_refresh_notification,
)
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    ModelCatalogSettings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    fingerprint_endpoint,
)

_SIX = ("OpenAI", "Anthropic", "MistralAI", "Moonshot", "OpenRouter", "ZAI")


def _discovered(list_key: str, *ids: str) -> tuple[DiscoveredModel, ...]:
    return tuple(
        DiscoveredModel(
            provider=list_key, provider_list_key=list_key, model_id=m,
            display_name=m, source="runtime_discovered",
            endpoint_fingerprint="fp", discovered_at="2026-07-17T00:00:00Z",
        )
        for m in ids
    )


def _service(models_by_provider, saved_calls, **overrides):
    async def fake_client(**kwargs):
        models = models_by_provider.get(kwargs["provider_list_key"], ())
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
            models=models,
        )

    def fake_save(section_values):
        saved_calls.append(section_values)
        return True

    default_settings = {
        "providers": {k: ["saved-1"] for k in _SIX}
    }

    return LocalLLMProviderCatalogService(
        provider_catalog_loader=overrides.get(
            "catalog_loader",
            lambda: {k: ["saved-1"] for k in _SIX},
        ),
        settings_loader=overrides.get("settings_loader", lambda: default_settings),
        discovery_client=overrides.get("discovery_client", fake_client),
        save_discovered_models_callback=overrides.get("save_callback", fake_save),
        environ=overrides.get("environ", {"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk",
                                          "MISTRAL_API_KEY": "sk", "MOONSHOT_API_KEY": "sk",
                                          "ZAI_API_KEY": "sk"}),  # no OPENROUTER key: public catalog
    )


def _tracking_client(calls, models_by_provider):
    async def client(**kwargs):
        calls.append(kwargs)
        models = models_by_provider.get(kwargs["provider_list_key"], ())
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
            models=models,
        )

    return client


@pytest.mark.asyncio
async def test_openrouter_refreshes_without_api_key(tmp_path):
    saved_calls = []
    service = _service({"OpenRouter": _discovered("OpenRouter", "a/b")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenRouter",),
    )
    assert report.outcomes[0].status == "refreshed"


@pytest.mark.asyncio
async def test_baseline_guard_suppresses_oversized_first_write(tmp_path):
    big = _discovered("OpenRouter", *[f"vendor/m{i}" for i in range(60)])
    saved_calls = []
    service = _service({"OpenRouter": big}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openrouter"})),
        disk_store=store,
        provider_list_keys=("OpenRouter",),
    )
    assert report.outcomes[0].status == "baseline"
    assert saved_calls == []  # nothing written on oversized first fetch


@pytest.mark.asyncio
async def test_small_catalog_backfills_on_first_write(tmp_path):
    saved_calls = []
    service = _service({"OpenAI": _discovered("OpenAI", "saved-1", "new-1", "new-2")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openai"})),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "refreshed"
    assert saved_calls == [{"providers": {"OpenAI": ["saved-1", "new-1", "new-2"]}}]


@pytest.mark.asyncio
async def test_second_fetch_appends_only_new_since_baseline(tmp_path):
    saved_calls = []
    service = _service({"OpenRouter": _discovered("OpenRouter", *[f"v/m{i}" for i in range(60)])}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    settings = ModelCatalogSettings(write_to_config=frozenset({"openrouter"}))
    await service.refresh_stale_configured_providers(
        catalog_settings=settings, disk_store=store, provider_list_keys=("OpenRouter",))
    # second fetch adds one model
    service2 = _service(
        {"OpenRouter": _discovered("OpenRouter", *[f"v/m{i}" for i in range(60)], "v/new")},
        saved_calls,
    )
    service2.discovery_cache = service.discovery_cache  # share prior cache state
    report = await service2.refresh_stale_configured_providers(
        catalog_settings=settings, disk_store=store, provider_list_keys=("OpenRouter",),
        force=True,
    )
    assert report.outcomes[0].saved_model_ids == ("v/new",)


@pytest.mark.asyncio
async def test_auth_failure_is_quiet_not_ready(tmp_path):
    async def failing_client(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="error",
            error=ModelDiscoveryError(
                kind="missing_credentials",
                message="The models endpoint rejected the configured credentials.",
                recovery_hint="Check the API key configured for this provider.",
            ),
        )

    saved_calls = []
    service = _service({}, saved_calls, discovery_client=failing_client)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "skipped_not_ready"
    assert format_refresh_notification(report) is None


def test_notification_none_when_nothing_happened():
    report = RefreshReport((
        ProviderRefreshOutcome(provider_list_key="OpenAI", status="skipped_fresh"),
    ))
    assert format_refresh_notification(report) is None


def test_notification_reports_cached_and_failed():
    report = RefreshReport((
        ProviderRefreshOutcome(provider_list_key="OpenAI", status="refreshed", new_model_ids=("gpt-x",)),
        ProviderRefreshOutcome(provider_list_key="ZAI", status="failed", error_kind="request_failed"),
    ))
    message = format_refresh_notification(report)
    assert "OpenAI" in message and "cached" in message
    assert "ZAI" in message and "using cached list" in message


@pytest.mark.asyncio
async def test_opted_out_provider_is_skipped_without_calling_client(tmp_path):
    client_calls = []
    saved_calls = []
    service = _service(
        {"OpenAI": _discovered("OpenAI", "saved-1", "new-1")},
        saved_calls,
        discovery_client=_tracking_client(client_calls, {"OpenAI": _discovered("OpenAI", "saved-1", "new-1")}),
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(auto_refresh_disabled=frozenset({"openai"})),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "skipped_disabled"
    assert client_calls == []
    assert saved_calls == []


@pytest.mark.asyncio
async def test_missing_credentials_skip_as_not_ready(tmp_path):
    saved_calls = []
    service = _service(
        {"OpenAI": _discovered("OpenAI", "saved-1", "new-1")},
        saved_calls,
        environ={},  # no keys anywhere
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "skipped_not_ready"


@pytest.mark.asyncio
async def test_fresh_entry_is_skipped_without_calling_client(tmp_path):
    client_calls = []
    saved_calls = []
    service = _service(
        {},
        saved_calls,
        discovery_client=_tracking_client(client_calls, {}),
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    store.record(
        "OpenAI",
        fingerprint_endpoint("https://api.openai.com/v1"),
        ["saved-1"],
        fetched_at=datetime.now(UTC),
    )
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "skipped_fresh"
    assert client_calls == []


@pytest.mark.asyncio
async def test_request_failure_is_reported_as_failed(tmp_path):
    async def failing_client(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="error",
            error=ModelDiscoveryError(
                kind="request_failed",
                message="The models endpoint could not be reached.",
                recovery_hint="Check the endpoint and try again.",
            ),
        )

    saved_calls = []
    service = _service({}, saved_calls, discovery_client=failing_client)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_kind == "request_failed"
    message = format_refresh_notification(report)
    assert "OpenAI" in message and "using cached list" in message


@pytest.mark.asyncio
async def test_write_through_failure_is_flagged_and_notified(tmp_path):
    saved_calls = []

    def failing_save(section_values):
        saved_calls.append(section_values)
        return False

    service = _service(
        {"OpenAI": _discovered("OpenAI", "saved-1", "new-1")},
        saved_calls,
        save_callback=failing_save,
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openai"})),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    outcome = report.outcomes[0]
    assert outcome.write_failed is True
    assert outcome.saved_model_ids == ()
    message = format_refresh_notification(report)
    assert "config save failed" in message
    assert "new cached" not in message  # save-failed clause already covers the diff


@pytest.mark.asyncio
async def test_zero_stale_after_hours_refetches_fresh_entry(tmp_path):
    saved_calls = []
    service = _service({"OpenAI": _discovered("OpenAI", "saved-1", "new-1")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    store.record(
        "OpenAI",
        fingerprint_endpoint("https://api.openai.com/v1"),
        ["saved-1"],
        fetched_at=datetime.now(UTC),
    )
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(stale_after_hours=0),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "refreshed"


@pytest.mark.asyncio
async def test_disk_save_failure_still_returns_report(tmp_path, monkeypatch):
    saved_calls = []
    service = _service({"OpenAI": _discovered("OpenAI", "saved-1", "new-1")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")

    def failing_save():
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", failing_save)
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "refreshed"
    assert report.outcomes[0].new_model_ids == ("new-1",)


@pytest.mark.asyncio
async def test_unresolved_provider_is_skipped_not_ready(tmp_path):
    saved_calls = []
    service = _service(
        {},
        saved_calls,
        catalog_loader=lambda: {"OpenAI": ["saved-1"]},  # requested key absent
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenRouter",),
    )
    assert report.outcomes[0].status == "skipped_not_ready"
    assert report.outcomes[0].provider_list_key == "OpenRouter"


@pytest.mark.asyncio
async def test_unsupported_endpoint_is_reported_as_failed(tmp_path):
    async def unsupported_client(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="unsupported",
            error=ModelDiscoveryError(
                kind="unsupported_endpoint",
                message="This endpoint is not a valid OpenAI-compatible models endpoint.",
                recovery_hint="Configure an explicit http:// or https:// /v1 models endpoint.",
            ),
        )

    saved_calls = []
    service = _service({}, saved_calls, discovery_client=unsupported_client)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_kind == "unsupported_endpoint"


@pytest.mark.asyncio
async def test_failed_refresh_leaves_disk_entry_unset(tmp_path):
    async def failing_client(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="error",
            error=ModelDiscoveryError(
                kind="request_failed",
                message="The models endpoint could not be reached.",
                recovery_hint="Check the endpoint and try again.",
            ),
        )

    saved_calls = []
    service = _service({}, saved_calls, discovery_client=failing_client)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "failed"
    fingerprint = fingerprint_endpoint("https://api.openai.com/v1")
    assert store.fetched_at("OpenAI", fingerprint) is None  # retried next launch
    assert store.is_stale("OpenAI", fingerprint, stale_after_hours=24)


@pytest.mark.asyncio
async def test_prune_drops_providers_no_longer_configured(tmp_path):
    saved_calls = []
    # Catalog holds only OpenAI; "Ghost" lingered in the disk cache from a
    # past session where it was still present in [providers].
    service = _service(
        {"OpenAI": _discovered("OpenAI", "saved-1")},
        saved_calls,
        catalog_loader=lambda: {"OpenAI": ["saved-1"]},
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    fingerprint = fingerprint_endpoint("https://api.openai.com/v1")
    store.record("Ghost", fingerprint, ["ghost-1"])
    store.record("OpenAI", fingerprint, ["saved-1"], fetched_at=datetime.now(UTC))
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI", "Ghost"),
    )
    assert report.outcomes[0].status == "skipped_fresh"  # fresh entry, no refetch
    assert report.outcomes[1].status == "skipped_not_ready"  # Ghost not configured
    # Prune keep-set is the configured catalog: Ghost is dropped even though it
    # was requested, while the configured provider's entry is retained.
    assert store.fetched_at("Ghost", fingerprint) is None
    assert store.fetched_at("OpenAI", fingerprint) is not None


@pytest.mark.asyncio
async def test_unexpected_client_error_becomes_failed_outcome_and_loop_continues(tmp_path):
    async def raising_client(**kwargs):
        if kwargs["provider_list_key"] == "OpenAI":
            raise RuntimeError("boom")
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
            models=_discovered("OpenRouter", "a/b"),
        )

    saved_calls = []
    service = _service({}, saved_calls, discovery_client=raising_client)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI", "OpenRouter"),
    )
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_kind == "unexpected"
    assert report.outcomes[1].status == "refreshed"


@pytest.mark.asyncio
async def test_unexpected_error_logs_no_traceback_or_secrets(tmp_path):
    secret = "sk-leak-probe-value"

    async def raising_client(**kwargs):
        raise ValueError("boom")

    saved_calls = []
    service = _service(
        {},
        saved_calls,
        discovery_client=raising_client,
        environ={"OPENAI_API_KEY": secret},
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    logged: list[str] = []
    # str(message) renders the fully formatted record, including any traceback
    # (the test sink, like the app's file sink, defaults to diagnose=True).
    sink_id = logger.add(lambda message: logged.append(str(message)), level="WARNING")
    try:
        report = await service.refresh_stale_configured_providers(
            catalog_settings=ModelCatalogSettings(),
            disk_store=store,
            provider_list_keys=("OpenAI",),
        )
    finally:
        logger.remove(sink_id)
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_kind == "unexpected"
    text = "".join(logged)
    assert "Model catalog refresh failed for OpenAI: ValueError" in text
    assert "Traceback" not in text
    assert secret not in text
    assert "api_key" not in text


@pytest.mark.asyncio
async def test_setup_error_becomes_failed_outcome_and_loop_continues(tmp_path):
    default_settings = {"providers": {k: ["saved-1"] for k in _SIX}}
    settings_calls = {"count": 0}

    def flaky_settings_loader():
        settings_calls["count"] += 1
        if settings_calls["count"] == 2:
            # First in-loop settings read (OpenAI's endpoint fingerprint) blows up.
            raise RuntimeError("settings read blew up")
        return default_settings

    saved_calls = []
    service = _service(
        {"OpenRouter": _discovered("OpenRouter", "a/b")},
        saved_calls,
        settings_loader=flaky_settings_loader,
    )
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenAI", "OpenRouter"),
    )
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_kind == "unexpected"
    assert report.outcomes[0].provider_list_key == "OpenAI"
    assert report.outcomes[1].status == "refreshed"
    assert report.outcomes[1].provider_list_key == "OpenRouter"


def test_notification_reports_baseline_diff_as_cached():
    report = RefreshReport((
        ProviderRefreshOutcome(
            provider_list_key="OpenRouter", status="baseline", new_model_ids=("a/b", "c/d")),
        ProviderRefreshOutcome(provider_list_key="ZAI", status="baseline"),
    ))
    message = format_refresh_notification(report)
    assert "OpenRouter: 2 new cached" in message
    assert "ZAI: catalog cached" in message
