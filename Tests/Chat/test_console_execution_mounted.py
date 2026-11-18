"""Actual selected parent route; this does not qualify journey timing."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from Tests.private_profile import private_profile_test


def _pilot_barrier_at_failure(error, pilot, current_pumps):
    """Read the cancelled Pilot frame before teardown releases its original pumps."""
    from functools import partial
    from traceback import walk_tb

    from textual.events import Callback

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for frame, _line in walk_tb(error.__traceback__):
            if frame.f_code is not pilot._wait_for_screen.__func__.__code__:
                continue
            state = frame.f_locals
            children = state.get("children", ())
            decrement = state.get("decrement_counter")
            current = {id(pump) for pump in current_pumps}
            detached = []
            queued = []
            for pump in children:
                if id(pump) not in current:
                    detached.append(type(pump).__name__)
                messages = (*pump._message_queue.values, pump._pending_message)
                if any(
                    isinstance(message, Callback)
                    and isinstance(message.callback, partial)
                    and message.callback.func is decrement
                    for message in messages
                ):
                    queued.append((type(pump).__name__, id(pump) in current, pump._closed))
            return {
                "remaining": state.get("count"),
                "original_pumps": len(children),
                "detached_count": len(detached),
                "detached_types": sorted(set(detached))[:16],
                "queued_callback_count": len(queued),
                "queued_examples": queued[:8],
            }
        error = error.__cause__ or error.__context__
    return None


@private_profile_test
@pytest.mark.asyncio
async def test_selected_process_home_to_chat_composes_original_surface(request):
    from textual.worker import WorkerFailed

    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import _ObservedStderr
    from Tests.Chat.console_execution_mount_wait import wait_for_selected_composer
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    assert config.save_settings_to_cli_config({
        "first_run": {"setup_completed": True},
        "splash_screen": {"enabled": False},
        "general": {"default_tab": "home"},
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings.llama_cpp": {
            "api_url": "http://127.0.0.1:9099", "model": "local-model"
        },
    })
    parent_runtime_entries = []

    def runtime_entry(frame):
        parent_runtime_entries.append(frame.f_locals["self"])

    stderr_tail = bytearray()
    composition_failures = []

    def error_panel_yield(frame):
        import traceback

        error = frame.f_locals.get("exc")
        if error is not None:
            stack = traceback.extract_tb(error.__traceback__)
            composition_failures.append((
                type(error).__name__, getattr(error, "name", None),
                [(os.path.basename(f.filename), f.name, f.lineno) for f in stack],
            ))

    try:
        # Observe the original parent constructor without replacing its definition.
        # A real child has a separate interpreter and owns its actual runtime.
        with _observe(ConsoleRuntime.__init__.__code__, sys.monitoring.events.PY_START,
                      runtime_entry):
            app = TldwCli(console_execution="process")
            client = app.console_execution_client
            assert type(client) is ConsoleExecutionClient
            def created(frame):
                caller = frame.f_back
                if (caller.f_code is ConsoleExecutionClient._launch.__code__
                    and caller.f_locals.get("self") is client):
                    process = frame.f_locals["self"]
                    assert type(process) is subprocess.Popen and process.stderr is not None
                    process.stderr = _ObservedStderr(process.stderr, stderr_tail)

            assert app.console_runtime is None
            with (
                _observe(subprocess.Popen.__init__.__code__, sys.monitoring.events.PY_RETURN, created),
                _observe(BaseAppScreen._compose_content_or_failure.__code__,
                         sys.monitoring.events.PY_YIELD, error_panel_yield),
            ):
                async with app.run_test(size=(120, 40)) as pilot:
                    await app._initial_screen_setup_task
                    await app._console_execution_start_task
                    assert app.current_tab == TAB_HOME
                    assert client._process is not None
                    assert client._process.pid != os.getpid()
                    assert client.projection_view is None
                    try:
                        async with asyncio.timeout(10):
                            await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
                            await wait_for_selected_composer(app)
                    except BaseException as error:
                        import inspect

                        selected_screen = app.screen

                        def task_frames(task):
                            current = task.get_coro() if type(task) is asyncio.Task else None
                            frames = []
                            for _ in range(24):
                                if inspect.iscoroutine(current):
                                    frame, following = current.cr_frame, current.cr_await
                                elif inspect.isgenerator(current):
                                    frame, following = current.gi_frame, current.gi_yieldfrom
                                else:
                                    break
                                if frame is not None:
                                    frames.append((frame.f_globals.get("__name__"), frame.f_code.co_name, frame.f_lineno))
                                current = following
                            return frames

                        # Pilot's barrier includes the App and every child pump,
                        # even when the selected Screen itself is already idle.
                        pumps = [app, *selected_screen.walk_children(with_self=True)]
                        nonidle_pumps = []
                        nonidle_count = 0
                        for pump in pumps:
                            pump_task = getattr(pump, "_task", None)
                            pump_frames = task_frames(pump_task)
                            locations = [frame[:2] for frame in pump_frames]
                            idle = any(
                                left == ("textual.message_pump", "_process_messages_loop")
                                and right == ("textual.message_pump", "_get_message")
                                for left, right in zip(locations, locations[1:], strict=False)
                            )
                            queued = pump.message_queue_size
                            if idle and not queued:
                                continue
                            nonidle_count += 1
                            if len(nonidle_pumps) < 8:
                                nonidle_pumps.append({
                                    "owner_type": type(pump).__name__,
                                    "is_app": pump is app,
                                    "is_screen": pump is selected_screen,
                                    "task_done": pump_task.done() if type(pump_task) is asyncio.Task else None,
                                    "closing": bool(pump._closing),
                                    "closed": bool(pump._closed),
                                    "queued": queued,
                                    "frames": pump_frames[-16:],
                                    "frames_omitted": max(0, len(pump_frames) - 16),
                                })
                        task = getattr(selected_screen, "_task", None)
                        composers = tuple(selected_screen.query("#console-native-composer"))
                        reconcile_worker = getattr(selected_screen, "_console_attach_reconcile_worker", None)
                        diagnostic = {
                            "app_exception": type(getattr(app, "_exception", None)).__name__,
                            "screen": type(selected_screen).__name__,
                            "composition_failure_count": len(composition_failures),
                            "composition_failures": [
                                (kind, locations[-8:])
                                for kind, _name, locations in composition_failures[-4:]
                            ],
                            "compose_exception": type(getattr(selected_screen, "_console_process_compose_error", None)).__name__,
                            "channel_exception": type(client._channel_error).__name__,
                            "screen_task_done": task.done() if type(task) is asyncio.Task else None,
                            "screen_mounted": selected_screen.is_mounted,
                            "screen_attached": selected_screen.is_attached,
                            "selected_view_matches_screen": client._view is selected_screen,
                            "composer_count": len(composers),
                            "composer_mounted": any(composer.is_mounted for composer in composers),
                            "surface_reconciled": bool(getattr(selected_screen, "_console_process_surface_reconciled", False)),
                            "attach_reconciled": bool(getattr(selected_screen, "_console_attach_reconciled", False)),
                            "attach_reconcile_running": bool(getattr(selected_screen, "_console_attach_reconcile_running", False)),
                            "reconcile_worker_running": reconcile_worker.is_running if reconcile_worker is not None else None,
                            "reconcile_worker_finished": reconcile_worker.is_finished if reconcile_worker is not None else None,
                            "prepare_failed": bool(getattr(selected_screen, "_console_process_prepare_failed", False)),
                            "content_failed": bool(getattr(selected_screen, "_console_process_content_failed", False)),
                            "projection_available": client.projection_view is not None,
                            "projection_reads_pending": sum(not read.done() for read in client._projection_reads),
                            "frames": task_frames(task),
                            "pump_count": len(pumps),
                            "original_pilot_barrier": _pilot_barrier_at_failure(error, pilot, pumps),
                            "nonidle_pump_count": nonidle_count,
                            "nonidle_pumps": nonidle_pumps,
                            "request_count": len(client._requests),
                            "requests": [
                                (item.operation, item.received.done(), item.applied.done(), item.retired.done())
                                for item in tuple(client._requests.values())[:16]
                            ],
                        }
                        # Child failure XML preserves notes even when later
                        # backend teardown output displaces this stderr line.
                        error.add_note("Selected mount failure state: " + repr(diagnostic))
                        request.node.user_properties.append(("selected_mount_failure_state", repr(diagnostic)))
                        print("Selected mount failure state:", diagnostic, file=sys.stderr)
                        raise
                    screen = app.screen
                    assert type(screen) is ChatScreen
                    composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
                    assert composer.is_mounted
                    client.require_view(screen, client._visit)
                    projection = client.projection_view
                    assert projection is not None
                    row = projection.session(projection.active_session_id)
                    assert screen._console_chat_store is client.store_reader
                    assert client.store_reader._current() is projection
                    assert screen._session._ensure_active_console_session_settings() == row.settings
                    assert screen._active_console_settings_readiness() is not None
                    assert screen.query_one("#console-shell").is_mounted
                    assert not parent_runtime_entries
                    assert app.console_runtime is None
                    request.node.user_properties.append(
                        ("selected_original_compose_mounted", True)
                    )
                    # The original navigation ends only the visit; its backend survives.
                    async with asyncio.timeout(10):
                        await app.handle_screen_navigation(NavigateToScreen(TAB_HOME))
                    assert client._view is None
                    assert client._process.poll() is None
                    assert client.projection_view.active_session_id == row.id
                    assert app.console_runtime is None
                    await pilot.pause()
                assert client._disposed
                assert client._physically_closed
                assert client._process.poll() == 0
                assert not parent_runtime_entries
                request.node.user_properties.append(
                    ("selected_app_shutdown_physically_closed", True)
                )
    except WorkerFailed as error:
        # Textual keeps the original worker exception here and raises a new
        # WorkerFailed at run_test exit, outside the mounted wait above.
        from traceback import walk_tb

        locations = [
            (module, frame.f_code.co_qualname, line)
            for frame, line in walk_tb(error.error.__traceback__)
            if (module := frame.f_globals.get("__name__", "")).startswith("tldw_chatbook.")
        ]
        error.add_note("Original mounted Console worker locations: " + repr(locations[-16:]))
        raise
    finally:
        if stderr_tail:
            diagnostic = stderr_tail.decode("utf-8", errors="replace")
            request.node.user_properties.append(("actual_backend_stderr_tail", diagnostic))
            print("Actual mounted backend diagnostic tail (bounded):", diagnostic, file=sys.stderr)
