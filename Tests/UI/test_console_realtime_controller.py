"""No-mount coverage for realtime callback fencing and resource release."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.realtime import ConsoleRealtimeController


def _controller(app=None):
    screen = SimpleNamespace(run_worker=lambda: "original")
    owner = ConsoleRealtimeController(
        screen,
        app_instance=app or SimpleNamespace(),
        ensure_active_console_session_settings=lambda: None,
        ensure_console_chat_store=lambda: None,
        dictation_state=lambda: "idle",
        request_console_dictation_stop=lambda **kwargs: None,
        sync_native_console_chat_ui=lambda: None,
        repaint_console_realtime_chip=lambda: None,
        restore_console_voice_chip=lambda: None,
        console_pipeline_hands_free_blocker=lambda: None,
        enter_console_hands_free_pipeline_loop=lambda **kwargs: None,
    )
    return owner, screen


def test_queued_callback_checks_attempt_and_session_at_delivery():
    callbacks = []
    app = SimpleNamespace(
        _thread_id=-1,
        _loop=SimpleNamespace(call_soon_threadsafe=callbacks.append),
    )
    owner, _ = _controller(app)
    session = SimpleNamespace(connect_attempt=1)
    owner._console_realtime = session
    delivered = []
    owner._console_realtime_marshal(lambda *args: delivered.append(args), session, 1)
    session.connect_attempt = 2
    callbacks.pop()()
    assert delivered == []
    owner._console_realtime_marshal(lambda *args: delivered.append(args), session, 2)
    owner._console_realtime = SimpleNamespace(connect_attempt=2)
    callbacks.pop()()
    assert delivered == []
    owner._console_realtime = session
    owner._console_realtime_marshal(lambda *args: delivered.append(args), session, 2)
    callbacks.pop()()
    assert delivered == [(session,)]


def test_framework_service_replacement_is_read_at_call_time():
    owner, screen = _controller()
    screen.run_worker = lambda: "replacement"
    assert owner.run_worker() == "replacement"


@pytest.mark.asyncio
async def test_resource_release_keeps_tap_provider_sink_queue_order():
    owner, _ = _controller()
    order = []

    async def close():
        order.append("provider")

    queue = asyncio.Queue()
    await owner._close_console_realtime_resources(
        SimpleNamespace(stop=lambda: order.append("tap")),
        SimpleNamespace(close=close),
        SimpleNamespace(stop=lambda: order.append("sink")),
        queue,
    )
    assert order == ["tap", "provider", "sink"]
    assert queue.get_nowait() is None
