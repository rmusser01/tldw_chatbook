"""Evidence draft: actual context-capacity rendering stays inside issued data."""

from __future__ import annotations

from tldw_chatbook.UI.Console_Modules.context_spend import ConsoleContextSpendController
from Tests.UI.console_controller_stubs import context_spend_for_test

import asyncio
import inspect
import json
import os
import sys
import threading
from contextlib import contextmanager
from types import FunctionType, MethodType, SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _screen, _warm
from tldw_chatbook import config
from tldw_chatbook.Chat import console_runtime
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


pytestmark = pytest.mark.bootstrap_profile


@contextmanager
def _actual_capacity_calls(app):
    """Observe actual installed bodies, without requiring a native cache miss."""
    monitoring = sys.monitoring
    getter = console_runtime._provider_config_for_app
    loader = config.load_settings
    assert type(getter) is FunctionType and type(loader) is FunctionType
    assert getter.__globals__ is vars(console_runtime)
    assert loader.__globals__ is vars(config)
    actors = (threading.current_thread(), threading.get_ident())
    codes = {getter.__code__: "getter", loader.__code__: "loader"}
    native = None
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        native = windows_files._native()
        installed = native.open_handle
        assert type(installed) is MethodType and installed.__self__ is native
        assert inspect.getattr_static(type(native), "open_handle") is installed.__func__
        codes[installed.__func__.__code__] = "native"
    calls = {
        "getter": 0,
        "loader": 0,
        "native": 0,
        "normal_getter_returns": 0,
        "normal_loader_returns": 0,
        "same_app": True,
        "same_source": True,
        "native_ancestry": {},
        "native_ancestry_overflow_calls": 0,
        "unmatched_selected_frames": 0,
        "global_events_zero": True,
        "local_masks_and_callbacks_retired": False,
    }
    frames = {}
    active = True

    def started(code, _offset):
        frame = sys._getframe(1)
        assert frame.f_code is code
        if not active or threading.current_thread() is not actors[0]:
            return
        label = codes[code]
        calls[label] += 1
        if label == "getter":
            calls["same_app"] &= frame.f_locals.get("app") is app
            calls["same_source"] &= frame.f_globals is vars(console_runtime)
            frames[id(frame)] = label
        elif label == "loader":
            calls["same_source"] &= frame.f_globals is vars(config)
            frames[id(frame)] = label
        else:
            labels = []
            current = frame.f_back
            for _ in range(18):
                if current is None:
                    break
                body = current.f_code
                labels.append(
                    f"{body.co_filename.rsplit(chr(92), 1)[-1].rsplit('/', 1)[-1]}:{body.co_name}:{current.f_lineno}"
                )
                current = current.f_back
            key = " > ".join(labels)
            if key in calls["native_ancestry"] or len(calls["native_ancestry"]) < 16:
                calls["native_ancestry"][key] = calls["native_ancestry"].get(key, 0) + 1
            else:
                calls["native_ancestry_overflow_calls"] += 1

    def returned(code, _offset, _value):
        frame = sys._getframe(1)
        assert frame.f_code is code
        if not active or threading.current_thread() is not actors[0]:
            return
        label = frames.pop(id(frame), None)
        if label is not None:
            calls[f"normal_{label}_returns"] += 1

    tool = next(
        (number for number in (5, 4, 3) if monitoring.get_tool(number) is None), None
    )
    assert tool is not None
    name = f"provider-capacity-draft-{id(calls)}"
    monitoring.use_tool_id(tool, name)
    assert monitoring.get_events(tool) == 0
    assert (
        monitoring.register_callback(tool, monitoring.events.PY_START, started) is None
    )
    assert (
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
        is None
    )
    masks = {}
    try:
        for code, label in codes.items():
            mask = monitoring.events.PY_START
            if label != "native":
                mask |= monitoring.events.PY_RETURN
            assert monitoring.get_local_events(tool, code) == 0
            monitoring.set_local_events(tool, code, mask)
            masks[code] = mask
        yield calls
    finally:
        assert monitoring.get_tool(tool) == name and monitoring.get_events(tool) == 0
        assert all(
            monitoring.get_local_events(tool, code) == mask
            for code, mask in masks.items()
        )
        for code in masks:
            monitoring.set_local_events(tool, code, 0)
        assert all(monitoring.get_local_events(tool, code) == 0 for code in masks)
        assert (
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            is started
        )
        assert (
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            is returned
        )
        monitoring.free_tool_id(tool)
        assert monitoring.get_tool(tool) is None
        active = False
        calls["unmatched_selected_frames"] = len(frames)
        calls["local_masks_and_callbacks_retired"] = True
        assert getter is console_runtime._provider_config_for_app
        assert loader is config.load_settings
        assert getter.__code__ in codes and loader.__code__ in codes


def _bind_estimate_fixture(screen, store, controller, gateway):
    # Only empty UI inputs are declared here. The tested screen estimate,
    # runtime gateway creation, config source, display proof and native readers
    # are the actual original product routes.
    screen._ensure_console_chat_store = lambda: store
    screen._ensure_console_chat_controller = lambda: controller
    screen._ensure_console_provider_gateway = lambda: gateway
    screen._console_composer_or_none = lambda: None
    screen._pending_console_launch_context = None
    screen._workspace = SimpleNamespace(_current_console_workspace_context=lambda: None)
    context_spend_for_test(screen)
    screen._context_spend._console_display_history = MethodType(
        ConsoleContextSpendController._console_display_history, screen._context_spend
    )
    screen._build_console_staged_context_state = MethodType(
        ChatScreen._build_console_staged_context_state, screen
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_cache", [False, True])
async def test_actual_context_estimate_opens_no_second_native_config_route(
    tmp_path, record_property, metadata_cache
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(screen.app_instance)
    gateway = None
    try:
        store.replace_session_settings(
            "session-1",
            ConsoleSessionSettings(provider="deepseek", model="deepseek-chat"),
        )
        runtime.set_chat_store(store)
        gateway = runtime.ensure_provider_gateway()
        _bind_estimate_fixture(screen, store, controller, gateway)
        if metadata_cache:
            gateway._get_context_windows()
        settings = store.session_settings("session-1")
        assert settings is not None
        # Attribute real first-use work separately, then warm its original
        # resources before measuring the repeated render. No disk miss is forced.
        with _actual_capacity_calls(screen.app_instance) as first_use:
            gateway.cached_context_window(settings)
        record_property("original_first_use", json.dumps(first_use))
        assert first_use["same_app"] and first_use["same_source"]
        projection = await _warm(screen, tasks)
        assert projection._display_proof is not None
        estimates = []

        def render():
            estimates.append(
                ConsoleContextSpendController._console_settings_context_estimate_for_session(
                    context_spend_for_test(screen), "session-1", settings=settings
                )
            )

        with _actual_capacity_calls(screen.app_instance) as calls:
            assert ChatScreen._run_console_config_sync(screen, render) is True
        record_property("actual_context_render", json.dumps(calls))
        assert len(estimates) == 1 and estimates[0].token_limit is not None
        assert calls["same_app"] and calls["same_source"]
        assert calls["unmatched_selected_frames"] == 0
        assert calls["local_masks_and_callbacks_retired"]
        assert calls["getter"] == 0, calls
        assert calls["loader"] == 0, calls
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
async def test_actual_gateway_direct_capacity_read_keeps_original_fresh_source(
    tmp_path,
):
    database, store, _controller, screen, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(screen.app_instance)
    try:
        store.replace_session_settings(
            "session-1",
            ConsoleSessionSettings(provider="deepseek", model="deepseek-chat"),
        )
        runtime.set_chat_store(store)
        gateway = runtime.ensure_provider_gateway()
        settings = store.session_settings("session-1")
        assert settings is not None
        gateway.cached_context_window(settings)
        await _warm(screen, tasks)
        with _actual_capacity_calls(screen.app_instance) as calls:
            actual = gateway.cached_context_window(settings)
        assert actual is not None
        assert calls["getter"] == calls["normal_getter_returns"] == 1
        assert calls["loader"] == calls["normal_loader_returns"] == 1
        assert calls["same_app"] and calls["same_source"]
        assert calls["unmatched_selected_frames"] == 0
        assert calls["local_masks_and_callbacks_retired"]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        database.close()
