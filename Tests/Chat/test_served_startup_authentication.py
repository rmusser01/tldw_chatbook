"""Served authentication owns startup ordering; UI waits never own its lifetime."""

import asyncio
import sys
import threading
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest


def _startup_view(client, splash, calls, cold):
    async def remove():
        cold("remove")
        calls.append("remove")

    async def push():
        cold("push")
        calls.append("push")

    async def post():
        calls.append("post")

    def chrome():
        cold("chrome")
        calls.append("chrome")
        return []

    app = SimpleNamespace(
        served_canvas_control=client,
        _served_canvas_mode=client is not None,
        _served_canvas_control_start_task=None,
        _served_control_lifetime_task=None,
        served_relay_publisher=None,
        console_runtime=None,
        _shutting_down=False,
        _exit=False,
        splash_screen_active=splash,
        _splash_screen_widget=SimpleNamespace(remove=remove),
        screen=SimpleNamespace(_nodes=[]),
        _push_initial_screen=push,
        _post_mount_setup=post,
        _create_main_ui_widgets=chrome,
        _setup_buffered_logging=lambda: calls.append("logging"),
        call_after_refresh=lambda callback: calls.append("after_refresh"),
        _close_boot_worker_gate=lambda reason: calls.append("fenced"),
    )
    app.exit = lambda: setattr(app, "_exit", True)
    from loguru import logger

    app.loguru_logger = logger
    return app


def _startup(app, splash):
    from tldw_chatbook.app import TldwCli

    return (
        TldwCli.on_splash_screen_closed(app, None)
        if splash
        else TldwCli._run_no_splash_post_mount_setup(app)
    )


async def _retire_task(task):
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("splash", [False, True])
async def test_native_authentication_finishes_before_cold_startup(
    splash, record_property
):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Canvas.control_protocol import (
        CanvasControlBroker,
        CanvasControlClient,
        ControlProtocolError,
    )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="startup-control-broker")
    client_began = threading.Event()
    accepted = threading.Event()
    auth_timeout = threading.Event()
    starts = []
    held = []
    calls = []
    monitor = sys.monitoring
    tool = next(i for i in (5, 4, 3, 2, 1, 0) if monitor.get_tool(i) is None)
    client_code = CanvasControlClient.start.__code__
    broker_code = CanvasControlBroker._accept_connection.__code__
    monitor.use_tool_id(tool, "served-startup-order")

    def started(code, offset):
        if code is client_code:
            starts.append(code)
            client_began.set()
        elif code is broker_code:
            accepted.set()

    def broker_line(code, number):
        if code is broker_code:
            frame = sys._getframe(1)
            assert frame.f_code is broker_code
            if type(frame.f_locals.get("error")) is TimeoutError:
                auth_timeout.set()

    def retire_observer():
        for code in (client_code, broker_code):
            monitor.set_local_events(tool, code, 0)
        for event in (monitor.events.PY_START, monitor.events.LINE):
            monitor.register_callback(tool, event, None)
        monitor.free_tool_id(tool)

    def retire_loop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()

    async with AsyncExitStack() as cleanup:
        cleanup.callback(retire_observer)
        monitor.register_callback(tool, monitor.events.PY_START, started)
        monitor.register_callback(tool, monitor.events.LINE, broker_line)
        monitor.set_local_events(tool, client_code, monitor.events.PY_START)
        monitor.set_local_events(
            tool, broker_code, monitor.events.PY_START | monitor.events.LINE
        )
        thread.start()
        cleanup.callback(retire_loop)

        async def make_broker():
            broker = CanvasControlBroker(runtime_snapshot_id="0" * 64)
            try:
                await broker.start()
                return broker, broker.issue_child("startup-control")
            except BaseException:
                await broker.aclose()
                raise

        broker, launch = await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(make_broker(), loop)
        )

        async def close_broker():
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(broker.aclose(), loop)
            )

        cleanup.push_async_callback(close_broker)
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="0" * 64)

        def cold(name):
            if not auth_task.done() and not held:
                assert client_began.is_set()
                assert client._reader_task is None
                assert client._writer is None
                held.append(name)
                # Release comes from the real broker's original two-second timer.
                assert auth_timeout.wait(5), (
                    "native authentication timeout not observed"
                )
                assert accepted.is_set()

        app = _startup_view(client, splash, calls, cold)
        TldwCli._start_served_canvas_control(app)
        auth_task = app._served_canvas_control_start_task
        assert auth_task.get_coro().cr_code is client_code
        cleanup.push_async_callback(TldwCli._stop_served_canvas_control, app)
        startup = asyncio.create_task(_startup(app, splash))
        cleanup.push_async_callback(_retire_task, startup)
        try:
            await startup
            await asyncio.shield(auth_task)
        except ControlProtocolError as error:
            assert error.code == "authentication_failed"
            assert held and auth_timeout.is_set() and accepted.is_set()
            raise
        finally:
            record_property("held_cold_before_auth", bool(held))
            record_property("native_broker_auth_timeout", auth_timeout.is_set())
            record_property("native_client_start_count", len(starts))
        assert not held and not auth_timeout.is_set()
        assert len(starts) == 1 and client._reader_task is not None
        assert calls == (
            ["remove", "chrome", "push", "logging", "after_refresh"]
            if splash
            else ["push", "post"]
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("splash", [False, True])
async def test_local_startup_does_not_need_a_served_task(splash):
    calls = []
    app = _startup_view(None, splash, calls, lambda name: None)
    await _startup(app, splash)
    assert calls == (
        ["remove", "chrome", "push", "logging", "after_refresh"]
        if splash
        else ["push", "post"]
    )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["missing", "failed"])
async def test_served_startup_withholds_cold_work_without_authentication(state):
    calls = []
    app = _startup_view(object(), False, calls, lambda name: None)
    if state == "failed":
        from tldw_chatbook.Canvas.control_protocol import ControlProtocolError

        async def refused():
            raise ControlProtocolError("authentication_failed")

        task = asyncio.create_task(refused())
        await asyncio.gather(task, return_exceptions=True)
        app._served_canvas_control_start_task = task
    await _startup(app, False)
    assert calls == []


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_cancelled_startup_waiter_does_not_cancel_owned_authentication():
    release = asyncio.Event()
    calls = []
    app = _startup_view(object(), False, calls, lambda name: None)
    auth = asyncio.create_task(release.wait())
    app._served_canvas_control_start_task = auth
    startup = asyncio.create_task(_startup(app, False))
    try:
        marker = asyncio.Event()
        asyncio.get_running_loop().call_soon(marker.set)
        await marker.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert not auth.cancelled() and not auth.done()
        assert calls == []
    finally:
        release.set()
        await _retire_task(startup)
        await auth


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["shutdown", "exit", "task", "client"])
async def test_completed_authentication_cannot_resume_a_replaced_or_closed_startup(
    change,
):
    release = asyncio.Event()
    calls = []
    app = _startup_view(object(), True, calls, lambda name: None)
    auth = asyncio.create_task(release.wait())
    app._served_canvas_control_start_task = auth
    startup = asyncio.create_task(_startup(app, True))
    try:
        marker = asyncio.Event()
        asyncio.get_running_loop().call_soon(marker.set)
        await marker.wait()
        if change == "shutdown":
            app._shutting_down = True
        elif change == "exit":
            app._exit = True
        elif change == "task":
            app._served_canvas_control_start_task = None
        else:
            app.served_canvas_control = object()
        release.set()
        await auth
        await startup
        assert calls == []
    finally:
        release.set()
        await _retire_task(startup)
        await auth
