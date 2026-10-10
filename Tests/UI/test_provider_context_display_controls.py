"""Evidence-only controls for the checked context-capacity consumer."""

from __future__ import annotations

from tldw_chatbook.UI.Console_Modules.context_spend import ConsoleContextSpendController
from Tests.UI.console_controller_stubs import context_spend_for_test

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from types import MethodType

import pytest

from Tests.UI.test_provider_context_display import (
    _actual_capacity_calls,
    _bind_estimate_fixture,
)
from Tests.UI.test_console_checked_display_scope import _screen, _warm
from tldw_chatbook.Chat import console_runtime
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@asynccontextmanager
async def _actual_display_owner(tmp_path, *, metadata_cache=False, new_model=False):
    database, store, controller, screen, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(screen.app_instance)
    gateway = None
    try:
        settings = ConsoleSessionSettings(provider="deepseek", model="deepseek-chat")
        store.replace_session_settings("session-1", settings)
        runtime.set_chat_store(store)
        gateway = runtime.ensure_provider_gateway()
        gateway._display_test_owned_gateways = []
        _bind_estimate_fixture(screen, store, controller, gateway)
        if metadata_cache:
            gateway._get_context_windows()
        # Preserve original first-use token resources separately from the render.
        gateway.cached_context_window(settings)
        if new_model:
            settings = ConsoleSessionSettings(
                provider="deepseek", model="deepseek-reasoner"
            )
            store.replace_session_settings("session-1", settings)
        projection = await _warm(screen, tasks)
        assert projection._display_proof is not None
        yield gateway, settings, projection, screen
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        if gateway is not None:
            for receiver in gateway._display_test_owned_gateways:
                await receiver.aclose()
        await runtime.dispose()
        database.close()


def _render(screen, settings, estimates):
    return ChatScreen._run_console_config_sync(
        screen,
        lambda: estimates.append(
            ConsoleContextSpendController._console_settings_context_estimate_for_session(
                context_spend_for_test(screen), "session-1", settings=settings
            )
        ),
    )


@pytest.mark.asyncio
async def test_display_metadata_does_not_publish_into_live_target_memo(tmp_path):
    async with _actual_display_owner(tmp_path, metadata_cache=True, new_model=True) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        memo = gateway._context_window_target_memo
        before = dict(memo)
        serving_cache = gateway._context_windows._cache
        before_serving = dict(serving_cache)
        estimates = []
        with _actual_capacity_calls(screen.app_instance) as calls:
            assert _render(screen, settings, estimates) is True
        assert len(estimates) == 1 and estimates[0].token_limit is not None
        assert calls["getter"] == calls["loader"] == 0
        assert gateway._context_window_target_memo is memo
        assert memo.keys() == before.keys()
        assert all(value is before[key] for key, value in memo.items())
        assert gateway._context_windows._cache is serving_cache
        assert serving_cache.keys() == before_serving.keys()
        assert all(value is before_serving[key] for key, value in serving_cache.items())


def _drifted_capacity_body(self, settings):
    # This code is placed on the original unguarded metadata function only;
    # its original defining-module globals include threading too. It preserves
    # a legitimate custom callback's one-argument/caller-thread contract.
    self._display_contract_calls.append(threading.get_ident())
    self._config_provider()
    return self._display_contract_resolution


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ["instance", "class", "subclass", "borrowed", "descriptor", "body", "provider"],
)
async def test_custom_metadata_callbacks_keep_original_fresh_contract(
    tmp_path, monkeypatch, route
):
    async with _actual_display_owner(tmp_path) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        calls_seen = []
        original = ConsoleProviderGateway.cached_context_window
        original_bound = MethodType(original, gateway)

        def custom(receiver, value):
            calls_seen.append(threading.get_ident())
            return original(receiver, value)

        if route == "instance":
            gateway.cached_context_window = MethodType(custom, gateway)
        elif route == "class":
            monkeypatch.setattr(ConsoleProviderGateway, "cached_context_window", custom)
        elif route == "subclass":

            class CustomGateway(ConsoleProviderGateway):
                cached_context_window = custom

            monkeypatch.setattr(gateway, "__class__", CustomGateway)
        elif route == "borrowed":
            receiver = ConsoleProviderGateway(config_provider=gateway._config_provider)
            gateway._display_test_owned_gateways.append(receiver)
            gateway.cached_context_window = MethodType(original, receiver)
        elif route == "descriptor":

            def descriptor(receiver):
                calls_seen.append(threading.get_ident())
                return MethodType(original, receiver)

            monkeypatch.setattr(
                ConsoleProviderGateway, "cached_context_window", property(descriptor)
            )
        elif route == "body":
            gateway._display_contract_resolution = original_bound(settings)
            gateway._display_contract_calls = calls_seen
            monkeypatch.setattr(original, "__code__", _drifted_capacity_body.__code__)
        else:
            source = gateway._config_provider

            def custom_source():
                calls_seen.append(threading.get_ident())
                return source()

            gateway._config_provider = custom_source
        estimates = []
        with _actual_capacity_calls(screen.app_instance) as calls:
            assert _render(screen, settings, estimates) is True
        assert len(estimates) == 1 and estimates[0].token_limit is not None
        assert calls["getter"] == calls["normal_getter_returns"] == 1
        assert calls["loader"] == calls["normal_loader_returns"] == 1
        assert calls["same_app"] and calls["same_source"]
        assert calls["unmatched_selected_frames"] == 0
        assert calls["local_masks_and_callbacks_retired"]
        if route != "borrowed":
            assert calls_seen == [threading.get_ident()]


def _drifted_cache_body(self, target):
    self._display_contract_calls.append(threading.get_ident())
    return self._display_contract_resolution


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["projector", "cache", "cache_body"])
async def test_custom_serving_metadata_callbacks_keep_their_original_signatures(
    tmp_path, monkeypatch, route
):
    async with _actual_display_owner(tmp_path, metadata_cache=True) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        calls_seen = []
        if route == "projector":
            original = gateway._project_context_window_target

            def custom(value):
                calls_seen.append(threading.get_ident())
                return original(value)

            gateway._project_context_window_target = custom
            # Exercise the original custom projection rather than an old target.
            gateway._context_window_target_memo.clear()
            expected_reads = 2
        else:
            cache = gateway._context_windows
            original = cache.cached
            if route == "cache_body":
                cache._display_contract_resolution = gateway.cached_context_window(
                    settings
                )
                cache._display_contract_calls = calls_seen
                monkeypatch.setattr(
                    original.__func__, "__code__", _drifted_cache_body.__code__
                )
            else:

                def custom(value):
                    calls_seen.append(threading.get_ident())
                    return original(value)

                cache.cached = custom
            expected_reads = 1
        estimates = []
        with _actual_capacity_calls(screen.app_instance) as calls:
            assert _render(screen, settings, estimates) is True
        assert len(estimates) == 1 and estimates[0].token_limit is not None
        assert calls_seen == [threading.get_ident()]
        assert calls["getter"] == calls["normal_getter_returns"] == expected_reads
        assert calls["loader"] == calls["normal_loader_returns"] == expected_reads
        assert calls["same_app"] and calls["same_source"]
        assert calls["unmatched_selected_frames"] == 0


@pytest.mark.asyncio
async def test_mutated_runtime_partial_keeps_original_argument_failure(tmp_path):
    async with _actual_display_owner(tmp_path) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        source = gateway._config_provider
        assert type(source) is partial and not source.keywords
        source.keywords["app"] = screen.app_instance
        with pytest.raises(TypeError, match="multiple values.*app"):
            _render(screen, settings, [])


@contextmanager
def _on_original_return(function, callback):
    import sys

    monitoring = sys.monitoring
    tool = next(
        (value for value in (5, 4, 3) if monitoring.get_tool(value) is None), None
    )
    assert tool is not None
    code = function.__code__
    monitoring.use_tool_id(tool, "context-display-source-drift")
    assert monitoring.get_events(tool) == 0
    observed = []

    def returned(actual_code, _offset, _value):
        if actual_code is code:
            observed.append(threading.get_ident())
            callback()

    monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
    monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
    try:
        yield observed
    finally:
        assert monitoring.get_events(tool) == 0
        assert monitoring.get_local_events(tool, code) == monitoring.events.PY_RETURN
        monitoring.set_local_events(tool, code, 0)
        assert monitoring.get_local_events(tool, code) == 0
        assert (
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            is returned
        )
        monitoring.free_tool_id(tool)
        assert monitoring.get_tool(tool) is None


@pytest.mark.asyncio
async def test_equal_runtime_partial_replacement_during_projection_refuses_publication(
    tmp_path,
):
    async with _actual_display_owner(tmp_path, metadata_cache=True) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        # Observe the original projector too: a prior memo hit would hide the
        # selected return boundary and turn this into a setup-only assertion.
        gateway._context_window_target_memo.clear()
        old_source = gateway._config_provider
        replacement = partial(
            console_runtime._provider_config_for_app, screen.app_instance
        )
        assert replacement is not old_source
        assert replacement.func is old_source.func
        assert replacement.args[0] is old_source.args[0]
        retries = []
        screen._request_console_control_bar_sync = lambda **_: retries.append(True)
        estimates = []
        with _on_original_return(
            gateway._project_context_window_target.__func__,
            lambda: setattr(gateway, "_config_provider", replacement),
        ) as returns:
            accepted = _render(screen, settings, estimates)
        assert returns == [threading.get_ident()]
        assert accepted is False
        assert estimates == []
        assert retries
        assert gateway._config_provider is replacement


@pytest.mark.asyncio
async def test_custom_metadata_failure_keeps_the_original_exception(tmp_path):
    async with _actual_display_owner(tmp_path) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        failure = ArithmeticError("custom metadata failure")

        def custom(value):
            assert value is settings
            raise failure

        gateway.cached_context_window = custom
        with pytest.raises(ArithmeticError) as caught:
            _render(screen, settings, [])
        assert caught.value is failure


@pytest.mark.asyncio
async def test_borrowed_cache_reader_after_original_return_refuses_publication(
    tmp_path,
):
    async with _actual_display_owner(tmp_path, metadata_cache=True) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        cache = gateway._context_windows
        original = cache.cached
        receiver = type(cache)()
        borrowed = MethodType(original.__func__, receiver)
        retries = []
        screen._request_console_control_bar_sync = lambda **_: retries.append(True)
        estimates = []
        with _on_original_return(
            original.__func__, lambda: setattr(cache, "cached", borrowed)
        ) as returns:
            accepted = _render(screen, settings, estimates)
        assert returns == [threading.get_ident()]
        assert accepted is False
        assert estimates == []
        assert retries
        assert cache.cached is borrowed


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["field_property", "attribute_hook"])
async def test_custom_field_access_stays_on_the_original_fresh_route(
    tmp_path, monkeypatch, route
):
    async with _actual_display_owner(tmp_path) as (
        gateway,
        settings,
        _projection,
        screen,
    ):
        source = gateway._config_provider
        visits = []
        if route == "field_property":

            def source_property(receiver):
                assert receiver is gateway
                visits.append(threading.get_ident())
                return source

            monkeypatch.setattr(
                ConsoleProviderGateway,
                "_config_provider",
                property(source_property),
                raising=False,
            )
        else:

            def attribute_hook(receiver, name):
                if name == "_config_provider":
                    visits.append(threading.get_ident())
                return object.__getattribute__(receiver, name)

            monkeypatch.setattr(
                ConsoleProviderGateway,
                "__getattribute__",
                attribute_hook,
            )
        estimates = []
        with _actual_capacity_calls(screen.app_instance) as calls:
            assert _render(screen, settings, estimates) is True
        assert len(estimates) == 1 and estimates[0].token_limit is not None
        # Qualification itself cannot call this field accessor, then promote
        # what it returned into the private checked-copy route. Only the
        # original public metadata method is allowed to consume the accessor.
        assert visits == [threading.get_ident()]
        assert calls["getter"] == calls["normal_getter_returns"] == 1
        assert calls["loader"] == calls["normal_loader_returns"] == 1
        assert calls["same_app"] and calls["same_source"]
        assert calls["unmatched_selected_frames"] == 0
        assert calls["local_masks_and_callbacks_retired"]
