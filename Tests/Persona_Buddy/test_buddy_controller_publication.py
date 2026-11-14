"""Injected controllers must share lazy construction's publication lock."""

import threading

import pytest


@pytest.mark.unit
@pytest.mark.bootstrap_profile
def test_controller_setter_waits_for_construction_lock():
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    attempted = threading.Event()
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            attempted.set()
            lock.acquire()

        def __exit__(self, *_args):
            lock.release()

    host = ServiceWiringMixin()
    host._persona_buddy_controller_lock = ObservedLock()
    host._persona_buddy_controller = None
    injected = object()
    writer = threading.Thread(
        target=lambda: setattr(host, "persona_buddy_controller", injected)
    )
    lock.acquire()
    writer.start()
    try:
        assert attempted.wait(2), "setter bypassed the construction lock"
        assert host._persona_buddy_controller is None
    finally:
        lock.release()
        writer.join(5)
    assert not writer.is_alive()
    assert host.persona_buddy_controller is injected
    host.persona_buddy_controller = None
    assert host._persona_buddy_controller is None
