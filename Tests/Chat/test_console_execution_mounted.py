"""Actual selected parent route; this does not qualify journey timing."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_selected_process_home_to_chat_composes_original_surface(request):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import _ObservedStderr
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
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
            with _observe(subprocess.Popen.__init__.__code__,
                          sys.monitoring.events.PY_RETURN, created):
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
                            await _wait_for_selector(app, pilot, "#console-composer", timeout=10)
                            while not getattr(app.screen, "_console_process_surface_reconciled", False):
                                await pilot.pause()
                    except BaseException:
                        import inspect

                        selected_screen = app.screen
                        task = getattr(selected_screen, "_task", None)
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
                                frames.append((frame.f_globals.get("__name__"), frame.f_code.co_name))
                            current = following
                        diagnostic = {
                            "app_exception": type(getattr(app, "_exception", None)).__name__,
                            "screen": type(selected_screen).__name__,
                            "compose_exception": type(getattr(selected_screen, "_console_process_compose_error", None)).__name__,
                            "channel_exception": type(client._channel_error).__name__,
                            "screen_task_done": task.done() if type(task) is asyncio.Task else None,
                            "frames": frames,
                            "requests": [
                                (item.operation, item.received.done(), item.applied.done(), item.retired.done())
                                for item in client._requests.values()
                            ],
                        }
                        request.node.user_properties.append(("selected_mount_failure_state", repr(diagnostic)))
                        print("Selected mount failure state:", diagnostic, file=sys.stderr)
                        raise
                    screen = app.screen
                    assert type(screen) is ChatScreen
                    composer = screen.query_one("#console-composer", ConsoleComposerBar)
                    assert composer.is_mounted
                    client.require_view(screen, client._visit)
                    projection = client.projection_view
                    assert projection is not None
                    row = projection.session(projection.active_session_id)
                    assert screen._console_chat_store is projection
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
    finally:
        if stderr_tail:
            diagnostic = stderr_tail.decode("utf-8", errors="replace")
            request.node.user_properties.append(("actual_backend_stderr_tail", diagnostic))
            print("Actual mounted backend diagnostic tail (bounded):", diagnostic, file=sys.stderr)
