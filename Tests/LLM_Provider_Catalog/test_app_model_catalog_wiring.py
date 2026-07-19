"""App wiring tests for the ADR-019 startup model catalog auto-refresh.

Covers two things:

1. Message routing: Textual messages bubble UP only, so an App-posted
   ModelCatalogRefreshed never reaches a Screen's @on handler. The
   forward_model_catalog_refreshed helper bridges that gap via duck typing.
2. The TldwCli._refresh_model_catalogs startup coroutine, exercised unbound
   against a stub self (the full app is too heavy to construct in a unit
   test).
"""

import pytest
from textual.app import App
from textual.screen import Screen

from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
    ModelCatalogRefreshed,
    ProviderRefreshOutcome,
    RefreshReport,
    format_refresh_notification,
    forward_model_catalog_refreshed,
)
from tldw_chatbook.app import TldwCli


# ---------------------------------------------------------------------------
# Routing: App.post_message must be forwarded DOWN to a mounted screen handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forward_reaches_mounted_screen_handler():
    calls = []

    class StubScreen(Screen):
        async def handle_model_catalog_refreshed(self, event):
            calls.append(set(event.providers))

    class StubApp(App):
        def on_mount(self):
            self.push_screen(StubScreen())

    app = StubApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        event = ModelCatalogRefreshed(providers={"OpenAI"})
        assert await forward_model_catalog_refreshed(app, event) is True
    assert calls == [{"OpenAI"}]


@pytest.mark.asyncio
async def test_forward_returns_false_without_handler():
    app = App()
    async with app.run_test():
        event = ModelCatalogRefreshed(providers={"OpenAI"})
        assert await forward_model_catalog_refreshed(app, event) is False


# ---------------------------------------------------------------------------
# Refresh loop: TldwCli._refresh_model_catalogs against a stub self
# ---------------------------------------------------------------------------


class _StubCatalogService:
    """Stands in for LocalLLMProviderCatalogService.refresh_stale_configured_providers."""

    def __init__(self, report=None, error=None):
        from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
            ModelDiscoveryCache,
        )

        self.calls = []
        self.discovery_cache = ModelDiscoveryCache()
        self._report = report
        self._error = error

    async def refresh_stale_configured_providers(self, **kwargs):
        self.calls.append(kwargs)
        on_config_saved = kwargs.get("on_config_saved")
        if on_config_saved is not None:
            on_config_saved()
        if self._error is not None:
            raise self._error
        return self._report


class _StubApp:
    """Minimal self for the unbound TldwCli._refresh_model_catalogs coroutine."""

    def __init__(self, *, disk_store, service):
        self.model_catalog_disk_store = disk_store
        self.local_llm_provider_catalog_service = service
        self.init_providers_models_calls = 0
        self.posted_messages = []
        self.notifications = []

    def _init_providers_models(self):
        self.init_providers_models_calls += 1

    def post_message(self, message):
        self.posted_messages.append(message)

    def notify(self, message, *, title=None, severity=None):
        self.notifications.append((message, title, severity))


_DEFAULT_DISK_STORE = object()


def _stub(
    monkeypatch, *, settings=None, report=None, error=None, disk_store=_DEFAULT_DISK_STORE
):
    """Build a stub app + service and pin tldw_chatbook.app.load_settings."""
    monkeypatch.setattr("tldw_chatbook.app.load_settings", lambda: settings or {})
    service = _StubCatalogService(report=report, error=error)
    app = _StubApp(
        disk_store=object() if disk_store is _DEFAULT_DISK_STORE else disk_store,
        service=service,
    )
    return app, service


@pytest.mark.asyncio
async def test_refresh_skips_when_auto_refresh_disabled(monkeypatch):
    app, service = _stub(
        monkeypatch,
        settings={"model_catalog": {"auto_refresh_enabled": False}},
    )
    await TldwCli._refresh_model_catalogs(app)
    assert service.calls == []
    assert app.posted_messages == []
    assert app.notifications == []


@pytest.mark.asyncio
async def test_refresh_skips_when_disk_store_missing(monkeypatch):
    app, service = _stub(monkeypatch, disk_store=None)
    await TldwCli._refresh_model_catalogs(app)
    assert service.calls == []
    assert app.posted_messages == []
    assert app.notifications == []


@pytest.mark.asyncio
async def test_refresh_notifies_once_with_formatted_message(monkeypatch):
    report = RefreshReport(
        outcomes=(
            ProviderRefreshOutcome(
                provider_list_key="OpenAI",
                status="refreshed",
                new_model_ids=("gpt-9",),
                saved_model_ids=("gpt-9",),
            ),
        )
    )
    app, _service = _stub(monkeypatch, report=report)
    await TldwCli._refresh_model_catalogs(app)
    expected = format_refresh_notification(report)
    assert expected is not None
    assert app.notifications == [(expected, "Model catalog", "information")]


@pytest.mark.asyncio
async def test_refresh_notifies_with_warning_severity_on_failure(monkeypatch):
    report = RefreshReport(
        outcomes=(
            ProviderRefreshOutcome(
                provider_list_key="OpenAI",
                status="failed",
                error_kind="network",
            ),
        )
    )
    app, _service = _stub(monkeypatch, report=report)
    await TldwCli._refresh_model_catalogs(app)
    expected = format_refresh_notification(report)
    assert expected is not None
    assert app.notifications == [(expected, "Model catalog", "warning")]


@pytest.mark.asyncio
async def test_refresh_notifies_with_warning_severity_on_write_failure(monkeypatch):
    report = RefreshReport(
        outcomes=(
            ProviderRefreshOutcome(
                provider_list_key="OpenAI",
                status="refreshed",
                new_model_ids=("gpt-9",),
                write_failed=True,
            ),
        )
    )
    app, _service = _stub(monkeypatch, report=report)
    await TldwCli._refresh_model_catalogs(app)
    expected = format_refresh_notification(report)
    assert expected is not None
    assert app.notifications == [(expected, "Model catalog", "warning")]


@pytest.mark.asyncio
async def test_refresh_posts_model_catalog_refreshed_for_refreshed_and_baseline(
    monkeypatch,
):
    report = RefreshReport(
        outcomes=(
            ProviderRefreshOutcome(
                provider_list_key="OpenAI",
                status="refreshed",
                new_model_ids=("gpt-9",),
            ),
            ProviderRefreshOutcome(provider_list_key="Anthropic", status="baseline"),
            ProviderRefreshOutcome(
                provider_list_key="MistralAI", status="skipped_fresh"
            ),
        )
    )
    app, _service = _stub(monkeypatch, report=report)
    await TldwCli._refresh_model_catalogs(app)
    events = [
        m for m in app.posted_messages if isinstance(m, ModelCatalogRefreshed)
    ]
    assert len(events) == 1
    assert events[0].providers == frozenset({"OpenAI", "Anthropic"})


@pytest.mark.asyncio
async def test_refresh_passes_config_saved_callback_and_disk_store(monkeypatch):
    report = RefreshReport()
    app, service = _stub(monkeypatch, report=report)
    await TldwCli._refresh_model_catalogs(app)
    # The stub service invokes the callback it was handed; it must be the
    # app's own _init_providers_models so selectors pick up newly saved models.
    assert app.init_providers_models_calls == 1
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["disk_store"] is app.model_catalog_disk_store
    assert call["on_config_saved"] == app._init_providers_models
    assert call["catalog_settings"].auto_refresh_enabled is True


@pytest.mark.asyncio
async def test_refresh_swallows_and_logs_errors(monkeypatch):
    from loguru import logger

    app, _service = _stub(monkeypatch, error=RuntimeError("boom-secret"))
    logged = []
    # str(message) renders the fully formatted record, including any traceback
    # (the test sink, like the app's file sink, defaults to diagnose=True).
    sink_id = logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        await TldwCli._refresh_model_catalogs(app)  # must not raise
    finally:
        logger.remove(sink_id)
    text = "".join(logged)
    # Context: the provider list keys being refreshed + the exception type name.
    assert (
        "Model catalog auto-refresh failed "
        "(OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, ZAI): RuntimeError"
    ) in text
    # No traceback (diagnose=True would dump frame locals) and no exception
    # message, which may carry endpoint URLs or credentials.
    assert "boom-secret" not in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------
# Disk store init: TldwCli._init_model_catalog_disk_store against a stub self
# ---------------------------------------------------------------------------


def test_disk_store_builds_for_cache_path_inside_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tldw_chatbook.app.get_user_data_dir", lambda: tmp_path)
    app, _service = _stub(monkeypatch)
    store = TldwCli._init_model_catalog_disk_store(app)
    assert store is not None
    assert store.path == tmp_path / "model_catalog_cache.json"


def test_disk_store_rejected_when_cache_path_escapes_data_dir(tmp_path, monkeypatch):
    from loguru import logger

    monkeypatch.setattr("tldw_chatbook.app.get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.path_validation.get_safe_relative_path",
        lambda path, base: None,
    )
    app, _service = _stub(monkeypatch)
    logged = []
    sink_id = logger.add(lambda message: logged.append(str(message)), level="WARNING")
    try:
        store = TldwCli._init_model_catalog_disk_store(app)
    finally:
        logger.remove(sink_id)
    assert store is None
    text = "".join(logged)
    assert "user data dir" in text
    assert str(tmp_path / "model_catalog_cache.json") in text


def test_disk_store_load_failure_logs_path_without_traceback(tmp_path, monkeypatch):
    from loguru import logger

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
        ModelCatalogDiskStore,
    )

    monkeypatch.setattr("tldw_chatbook.app.get_user_data_dir", lambda: tmp_path)

    def failing_load_into(self, cache):
        raise RuntimeError("boom-secret")

    monkeypatch.setattr(ModelCatalogDiskStore, "load_into", failing_load_into)
    app, _service = _stub(monkeypatch)
    logged = []
    sink_id = logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        store = TldwCli._init_model_catalog_disk_store(app)
    finally:
        logger.remove(sink_id)
    assert store is None
    text = "".join(logged)
    # Context: the cache file path + the exception type name.
    assert str(tmp_path / "model_catalog_cache.json") in text
    assert "RuntimeError" in text
    # No traceback (diagnose=True would dump frame locals) and no exception
    # message, which may carry sensitive details.
    assert "boom-secret" not in text
    assert "Traceback" not in text
