"""Real checked display reads cover prices without leaking into live caches."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _actual_calls, _screen, _warm
from tldw_chatbook import config
from tldw_chatbook.Chat import console_cost_tracker
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls import pricing_catalog as pricing
from tldw_chatbook.LLM_Provider_Catalog import models_dev_catalog as metadata
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Console_Modules.pricing_display import DisplayPricingCatalog
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(autouse=True)
def restore_catalog_owners():
    old = pricing._global_catalog, pricing._global_catalog_owner, metadata._MEMORY_CACHE
    yield
    pricing._global_catalog, pricing._global_catalog_owner, metadata._MEMORY_CACHE = old


def _seed_metadata(*, enabled=True, input_rate=2.0):
    assert config.save_setting_to_cli_config("model_catalog", "use_models_dev", enabled)
    blob = {
        "display_fixture": {
            "models": {
                "historical": {"cost": {"input": input_rate, "output": 3.0}},
                "selected": {"cost": {"input": 4.0, "output": 5.0}},
                "incomplete": {"cost": {"input": 9.0}},
            }
        }
    }
    metadata.fetch_models_dev(
        disk_path=metadata.default_cache_path(),
        http_get=lambda url, headers: (200, {}, json.dumps(blob).encode()),
    )
    metadata.reset_memory_cache()
    pricing.reload_pricing_catalog()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_checked_prices_and_historical_usage_need_no_main_native_read(
    tmp_path, enabled, monkeypatch
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    _seed_metadata(enabled=enabled)
    rows = [
        SimpleNamespace(
            content="",
            role=ConsoleMessageRole.ASSISTANT,
            usage=ProviderUsage(
                uncached_input=1_000_000,
                output=1_000_000,
                provider="display_fixture",
                model="historical",
            ),
        )
    ]
    values = []
    try:
        projection = await _warm(screen, tasks)
        assert type(projection.pricing) is DisplayPricingCatalog
        live = pricing.get_pricing_catalog()

        def render():
            catalog = spend.pricing_catalog_for_display(
                screen, pricing.get_pricing_catalog
            )
            assert catalog is projection.pricing and catalog is not live
            values.append(
                (
                    catalog.get_pricing("display_fixture", "selected"),
                    console_cost_tracker.build_cost_snapshot(
                        rows,
                        provider="display_fixture",
                        model="selected",
                        **spend.pricing_snapshot_options(catalog),
                    ),
                )
            )
            assert catalog.get_pricing("display_fixture", "incomplete") is None
            assert catalog.get_pricing("llama_cpp", "unlisted").input_per_mtok == 0
            assert (
                catalog.get_pricing("anthropic", "claude-haiku-4-5").as_of
                != "models.dev"
            )

        with _actual_calls() as calls:
            assert ChatScreen._run_console_config_sync(screen, render)
        assert calls["main_scopes"] == calls["main_opens"] == 0, calls
        price, total = values[0]
        if enabled:
            assert price.input_per_mtok == 4.0
            assert total.total_usd == 5.0  # Historical model's own rates.
        else:
            assert price is None and total.total_usd is None
        assert pricing.get_pricing_catalog() is live
        live_calls = []

        def current_entry(provider, model):
            live_calls.append((provider, model))
            return metadata.ModelsDevEntry(None, False, 17.0, 19.0)

        monkeypatch.setattr(metadata, "models_dev_entry", current_entry)
        assert live.get_pricing("display_fixture", "selected").input_per_mtok == 17.0
        assert live_calls == [("display_fixture", "selected")]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_catalog_only_change_invalidates_display_but_equal_prices_reuse_key(
    tmp_path,
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    _seed_metadata()
    try:
        projection = await _warm(screen, tasks)
        first = projection.pricing
        await projection._refresh(projection._key())
        assert projection.pricing == first
        old_mapping = dict(projection.value)
        # Keep the exact global catalog/config owner; only upstream disk data changes.
        metadata.fetch_models_dev(
            disk_path=metadata.default_cache_path(),
            http_get=lambda url, headers: (
                200,
                {},
                json.dumps(
                    {
                        "display_fixture": {
                            "models": {
                                "historical": {"cost": {"input": 7.0, "output": 3.0}}
                            }
                        }
                    }
                ).encode(),
            ),
        )
        metadata.reset_memory_cache()
        before_tasks = len(tasks)
        await projection._refresh(projection._key())
        assert projection.value == old_mapping
        assert projection.pricing != first
        assert (
            len(tasks) > before_tasks
        ), "metadata-only change did not schedule publication"
        assert ("same history", first) != ("same history", projection.pricing)
        assert (
            projection.pricing.get_pricing(
                "display_fixture", "historical"
            ).input_per_mtok
            == 7.0
        )
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_injected_catalog_retains_original_reader_and_native_scope(
    tmp_path, monkeypatch
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    _seed_metadata()
    calls = []

    class CustomCatalog(pricing.PricingCatalog):
        def get_pricing(self, provider, model):
            calls.append((provider, model))
            return "custom-price"

    custom = CustomCatalog(config={})
    monkeypatch.setattr(pricing, "_global_catalog", custom)
    try:
        projection = await _warm(screen, tasks)
        assert projection.pricing is None
        values = []
        with _actual_calls() as native:
            assert ChatScreen._run_console_config_sync(
                screen,
                lambda: values.append(
                    spend.pricing_catalog_for_display(
                        screen, pricing.get_pricing_catalog
                    ).get_pricing("custom", "model")
                ),
            )
        assert native["main_scopes"] > 0
        assert values == ["custom-price"] and calls == [("custom", "model")]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.parametrize("kind", ["direct", "pattern"])
def test_configured_unknown_price_does_not_fall_through_to_upstream(monkeypatch, kind):
    configured = (
        {"models": {"display_fixture:selected": {"input_per_mtok": 1}}}
        if kind == "direct"
        else {
            "patterns": {
                "display_fixture": [{"pattern": "^selected$", "input_per_mtok": 1}]
            }
        }
    )
    catalog = pricing.PricingCatalog(config=configured)
    monkeypatch.setattr(catalog, "_to_model_pricing", lambda entry: None)
    upstream_calls = []
    monkeypatch.setattr(
        pricing, "_models_dev_pricing", lambda *args: upstream_calls.append(args)
    )
    assert catalog.get_pricing("display_fixture", "selected") is None
    assert upstream_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        "entry",
        "lookup",
        "adapter_body",
        "helper_body",
        "adapter_class",
        "display_module",
        "pricing_module",
        "metadata_module",
        "historical_getter",
    ],
)
async def test_replaced_pricing_sources_refuse_checked_display_before_dispatch(
    tmp_path, monkeypatch, changed
):
    import sys
    from types import ModuleType
    from tldw_chatbook.UI.Console_Modules import pricing_display

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    _seed_metadata()
    replacement_calls = []

    def replacement(*args, **kwargs):
        replacement_calls.append((args, kwargs))
        return None

    def replacement_body(*args, **kwargs):
        raise AssertionError("changed optional display body was invoked")

    try:
        projection = await _warm(screen, tasks)
        assert type(projection.pricing) is DisplayPricingCatalog
        statuses = []
        assert projection.run(
            lambda: statuses.append(spend._checked_display_status(projection))
        )
        assert statuses == [True]
        if changed == "entry":
            monkeypatch.setattr(metadata, "models_dev_entry", replacement)
        elif changed == "lookup":
            monkeypatch.setattr(metadata.ModelsDevCache, "lookup", replacement)
        elif changed == "adapter_body":
            monkeypatch.setattr(
                DisplayPricingCatalog.get_pricing, "__code__", replacement_body.__code__
            )
        elif changed == "helper_body":
            monkeypatch.setattr(
                pricing_display.pricing_display_current,
                "__code__",
                replacement_body.__code__,
            )
        elif changed == "adapter_class":
            monkeypatch.setattr(pricing_display, "DisplayPricingCatalog", replacement)
        elif changed == "historical_getter":
            monkeypatch.setattr(
                console_cost_tracker, "get_pricing_catalog", replacement
            )
        else:
            module = {
                "display_module": pricing_display,
                "pricing_module": pricing,
                "metadata_module": metadata,
            }[changed]
            monkeypatch.setitem(
                sys.modules, module.__name__, ModuleType(module.__name__)
            )
        assert projection.run(
            lambda: statuses.append(spend._checked_display_status(projection))
        )
        assert statuses == [True, None]
        assert replacement_calls == []
    finally:
        monkeypatch.undo()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_changed_adapter_equality_is_not_invoked_after_background_read(
    tmp_path, monkeypatch
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    _seed_metadata()
    try:
        projection = await _warm(screen, tasks)
        old_pricing, old_time = projection.pricing, projection.at
        original_read = projection.read_current
        equality_calls = []

        def changed_equality(*args, **kwargs):
            equality_calls.append(True)
            return True

        def read_then_change():
            result = original_read()
            monkeypatch.setattr(DisplayPricingCatalog, "__eq__", changed_equality)
            return result

        monkeypatch.setattr(projection, "read_current", read_then_change)
        await projection._refresh(projection._key())
        assert projection.pricing is old_pricing and projection.at == old_time
        assert equality_calls == []
        assert not projection.pending and projection._settled.is_set()
    finally:
        monkeypatch.undo()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.parametrize("changed", ["class", "helper", "converter"])
def test_pricing_native_return_rechecks_before_helper_and_constructor(
    monkeypatch, changed
):
    import sys
    from tldw_chatbook.UI.Console_Modules import pricing_display

    _seed_metadata()
    original = metadata._memory_cache.__code__
    tool = next(i for i in (3, 4, 2) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool, "pricing-native-return-control")
    returned = []
    replacement_calls = []

    def replacement(*args, **kwargs):
        replacement_calls.append(True)
        return None

    def on_return(code, offset, result):
        assert type(result) is metadata.ModelsDevCache
        returned.append(True)
        if changed == "converter":
            monkeypatch.setattr(pricing, "_pricing_from_models_dev_entry", replacement)
        else:
            name = (
                "DisplayPricingCatalog"
                if changed == "class"
                else "_stock_catalog_current"
            )
            monkeypatch.setattr(pricing_display, name, replacement)

    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, on_return)
    sys.monitoring.set_local_events(tool, original, sys.monitoring.events.PY_RETURN)
    try:
        assert pricing_display.prepare_pricing_display() is None
        assert returned == [True] and replacement_calls == []
    finally:
        sys.monitoring.set_local_events(tool, original, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, None)
        sys.monitoring.free_tool_id(tool)
