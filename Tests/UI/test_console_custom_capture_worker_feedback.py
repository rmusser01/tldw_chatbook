"""Custom MCP fallback retains its finite worker capture and responsive UI."""

from __future__ import annotations

import sys
import threading

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    REPLY,
    ConsoleComposerBar,
    _painted_lines,
    _select_llamacpp_console,
    _wait_for_selector,
    build,
    eager_tasks,
    mark_send_start,
    press,
    until,
)
from Tests.UI.test_console_send_admission_off_pump import ENTRY_SECONDS, HOLD_SECONDS

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_custom_mcp_capture_keeps_worker_feedback_and_reads_once(monkeypatch):
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_mcp_definition_maximum,
    )
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    host, gateway, _timeline = build()
    entered, release = threading.Event(), threading.Event()
    capture_threads = []
    completed_reads = []
    timed_out = []
    selector_threads = []
    painted_while_held = []
    painted_ready = []
    loop_thread = threading.get_ident()
    target_code = capture_mcp_definition_maximum.__code__
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            runtime = None
            original_display = host._display
            try:
                console = host.screen_stack[-1]
                runtime = console._console_runtime()
                await _wait_for_selector(console, pilot, "#console-native-composer")
                await until(
                    lambda: console._console_attach_reconciled
                    and not console._console_attach_reconcile_running
                )
                _select_llamacpp_console(console)
                await pilot.pause(0.3)
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                store = console._ensure_console_chat_store()
                session_id = store.active_session_id

                def display(screen, renderable):
                    original_display(screen, renderable)
                    if screen is console and renderable is not None:
                        # Only inspect naturally delivered frames, never force paint.
                        region = composer.region.intersection(console.region)
                        lines = _painted_lines(host)
                        visible = "\n".join(
                            line[max(0, region.x) : region.right]
                            for line in lines[max(0, region.y) : region.bottom]
                        )
                        if DRAFT in visible:
                            painted_ready.append(True)
                        if (
                            entered.is_set()
                            and not release.is_set()
                            and not timed_out
                            and DRAFT + "x" in visible
                        ):
                            painted_while_held.append(True)

                monkeypatch.setattr(host, "_display", display)
                composer.load_draft(DRAFT)
                composer.focus()
                await until(
                    lambda: console._console_visible_draft_session_id == session_id
                    and composer.draft_text() == DRAFT
                    and store.session_draft(session_id) == DRAFT
                )
                await pilot.pause()
                # Deferred mount/setup work may choose focus after the first focus.
                composer.focus()
                await until(lambda: host.focused is composer and bool(painted_ready))
                assert host.screen is console
                assert console._console_composer_or_none() is composer
                assert store.active_session_id == session_id
                assert console._console_visible_draft_session_id == session_id
                for selector in (
                    "#console-command-visible-text",
                    "#console-send-message",
                ):
                    region = console.query_one(selector).region
                    assert region and region.overlaps(console.region)
                assert not store.messages_for_session(session_id)
                assert gateway.stream_calls == 0

                service = host.app_instance.unified_mcp_service
                original_kill_switch = service.get_kill_switch

                def custom_kill_switch():
                    # Count every original maximum capture for this real service,
                    # not later policy checks made by provider composition.
                    capture = sys._getframe(1).f_code is target_code
                    if capture:
                        capture_threads.append(threading.get_ident())
                    result = original_kill_switch()
                    if capture:
                        completed_reads.append(True)
                        if len(capture_threads) == 1:
                            entered.set()
                            timed_out.append(not release.wait(timeout=HOLD_SECONDS))
                    return result

                monkeypatch.setattr(service, "get_kill_switch", custom_kill_switch)
                assert not standard_console_sources(service)
                owner = console._session
                builder_code = (
                    owner._build_console_turn_execution_context.__func__.__code__
                )
                original_selection = owner._build_provider_selection_fn

                def selection(session_id):
                    if sys._getframe(1).f_code is builder_code:
                        selector_threads.append(threading.get_ident())
                    return original_selection(session_id)

                monkeypatch.setattr(owner, "_build_provider_selection_fn", selection)
                send_started = mark_send_start(console, _timeline)
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(lambda: bool(send_started), timeout=ENTRY_SECONDS)
                    assert len(send_started) == 1, "driver Enter did not own one Send"
                    await until(entered.is_set, timeout=ENTRY_SECONDS)
                    # If the fallback blocks the loop, this can resume only after
                    # its wait times out. Do not mistake that for a held worker.
                    assert capture_threads == [capture_threads[0]]
                    assert (
                        capture_threads[0] != loop_thread
                    ), "MCP capture blocked the UI loop"
                    assert not timed_out, "the original read was no longer held"
                    assert completed_reads == [True]
                    assert gateway.stream_calls == 0
                    assert not store.messages_for_session(session_id)
                    assert composer.draft_text() == DRAFT
                    press(host, "x", "x")
                    await until(
                        lambda: composer.draft_text() == DRAFT + "x", timeout=10
                    )
                    await until(lambda: bool(painted_while_held), timeout=10)
                    assert not timed_out
                    assert not store.messages_for_session(session_id)
                finally:
                    release.set()
                    gateway.validation_release.set()
                    monkeypatch.setattr(host, "_display", original_display)
                await until(lambda: gateway.stream_calls == 1)
                await until(lambda: REPLY in "\n".join(_painted_lines(host)))
                await until(lambda: not runtime.has_custodied_turns(session_id))
                assert timed_out == [False]
                assert capture_threads == [capture_threads[0]]
                assert completed_reads == [
                    True
                ], "worker preparation was reread on publication"
                assert selector_threads and set(selector_threads) == {loop_thread}
                assert composer.draft_text() == "x"
                assert store.session_draft(session_id) == "x"
                assert [
                    message.content
                    for message in store.messages_for_session(session_id)
                    if message.role.value == "user"
                ] == [DRAFT]
            finally:
                release.set()
                gateway.validation_release.set()
                monkeypatch.setattr(host, "_display", original_display)
                if runtime is not None:
                    await runtime.dispose()


@pytest.mark.parametrize("fault", ["source", "cancel"])
async def test_custom_mcp_capture_retires_before_refusal(monkeypatch, fault):
    import asyncio

    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_mcp_definition_maximum,
    )
    from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    host, gateway, timeline = build()
    entered, release, returned = (threading.Event() for _ in range(3))
    calls, timeouts, builder_calls, painted = [], [], [], []
    loop_thread = threading.get_ident()
    target_code = capture_mcp_definition_maximum.__code__
    runtime = service = read = None
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            original_display = host._display
            try:
                console = host.screen_stack[-1]
                runtime = console._console_runtime()
                await _wait_for_selector(console, pilot, "#console-native-composer")
                await until(
                    lambda: console._console_attach_reconciled
                    and not console._console_attach_reconcile_running
                )
                _select_llamacpp_console(console)
                await pilot.pause(0.3)
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                controller = console._ensure_console_chat_controller()
                store = controller.store
                session_id = store.active_session_id

                def display(screen, renderable):
                    original_display(screen, renderable)
                    if screen is console and renderable is not None:
                        region = composer.region.intersection(console.region)
                        visible = "\n".join(
                            line[max(0, region.x) : region.right]
                            for line in _painted_lines(host)[
                                max(0, region.y) : region.bottom
                            ]
                        )
                        if DRAFT in visible:
                            painted.append(True)

                monkeypatch.setattr(host, "_display", display)
                composer.load_draft(DRAFT)
                composer.focus()
                await until(
                    lambda: console._console_visible_draft_session_id == session_id
                    and store.session_draft(session_id) == DRAFT
                )
                await pilot.pause()
                composer.focus()
                await until(lambda: host.focused is composer and bool(painted))
                assert host.screen is console
                assert composer.draft_text() == DRAFT
                assert not store.messages_for_session(session_id)

                service = host.app_instance.unified_mcp_service
                original_kill_switch = service.get_kill_switch

                def custom_kill_switch():
                    capture = sys._getframe(1).f_code is target_code
                    result = original_kill_switch()
                    if capture:
                        calls.append(threading.get_ident())
                        if len(calls) == 1:
                            entered.set()
                            try:
                                timeouts.append(not release.wait(timeout=HOLD_SECONDS))
                            finally:
                                returned.set()
                    return result

                monkeypatch.setattr(service, "get_kill_switch", custom_kill_switch)
                assert not standard_console_sources(service)
                owner = console._session
                builder_code = (
                    owner._build_console_turn_execution_context.__func__.__code__
                )
                original_selection = owner._build_provider_selection_fn

                def selection(session_id):
                    if sys._getframe(1).f_code is builder_code:
                        builder_calls.append(threading.get_ident())
                    return original_selection(session_id)

                monkeypatch.setattr(owner, "_build_provider_selection_fn", selection)
                send_started = mark_send_start(console, timeline)
                gateway.validation_release.set()
                press(host, "enter", "\r")
                await until(lambda: bool(send_started), timeout=ENTRY_SECONDS)
                await until(entered.is_set, timeout=ENTRY_SECONDS)
                assert calls == [calls[0]] and calls[0] != loop_thread
                assert not timeouts and not returned.is_set()
                (read,) = preparation_reads_for(
                    controller._preparation_reads, session_id
                )
                assert read in runtime._preparation_reads
                assert read.creator is controller and read.session_id == session_id
                assert not read.task.done() and not read.retired.done()
                assert read._producer is not None and not read._producer.done()
                assert not builder_calls and gateway.stream_calls == 0
                assert not store.messages_for_session(session_id)
                press(host, "x", "x")
                await until(lambda: composer.draft_text() == DRAFT + "x", timeout=10)
                await until(lambda: store.session_draft(session_id) == DRAFT + "x")
                if fault == "source":
                    # Removal has no successor reader; the original owner must refuse.
                    host.app_instance.unified_mcp_service = None
                    await asyncio.sleep(0)
                else:
                    for _ in range(2):
                        read.task.cancel()
                        await asyncio.sleep(0.05)
                        assert not read.task.done()
                assert not read.retired.done() and not read._producer.done()
                assert read in runtime._preparation_reads
                assert not returned.is_set() and not timeouts
                assert not builder_calls and gateway.stream_calls == 0
                release.set()
                await asyncio.wait_for(asyncio.shield(read.retired), ENTRY_SECONDS)
                await asyncio.wait_for(
                    asyncio.gather(read.task, return_exceptions=True), ENTRY_SECONDS
                )
                assert returned.is_set() and timeouts == [False]
                assert not read.retired.cancelled() and read._producer.done()
                assert read not in runtime._preparation_reads
                assert read not in controller._preparation_reads
                assert calls == [calls[0]], "refusal retried the native capture inline"
                assert not builder_calls and gateway.stream_calls == 0
                assert not store.messages_for_session(session_id)
                assert (
                    composer.draft_text()
                    == store.session_draft(session_id)
                    == DRAFT + "x"
                )
            finally:
                release.set()
                gateway.validation_release.set()
                monkeypatch.setattr(host, "_display", original_display)
                try:
                    if read is not None:
                        await asyncio.wait_for(
                            asyncio.gather(read.task, return_exceptions=True),
                            ENTRY_SECONDS,
                        )
                finally:
                    if service is not None:
                        host.app_instance.unified_mcp_service = service
                    if runtime is not None:
                        await runtime.dispose()
