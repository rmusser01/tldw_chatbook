"""Exact native retirement evidence for a PCM output attempt."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Audio.test_streaming_sink import (
    _reset_live_sink_registry as _reset_live_sink_registry,  # noqa: PLC0414 - pytest fixture
)
from tldw_chatbook.Audio import streaming_sink as audio


class Device:
    def __init__(self):
        self.starts = 0
        self.aborts = 0
        self.closes = 0
        self.closed = False
        self.fail_close = False
        self.fail_abort = False

    def start(self):
        self.starts += 1

    def abort(self):
        self.aborts += 1
        if self.fail_abort:
            raise OSError("abort unconfirmed")

    def close(self):
        self.closes += 1
        if self.fail_close:
            raise OSError("close unconfirmed")
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("held_at", ["factory", "start"])
async def test_retirement_joins_the_committed_native_operation(held_at):
    entered, resume, revoked = (threading.Event() for _ in range(3))
    stream = Device()

    def hold():
        entered.set()
        assert resume.wait(3)

    def factory(**kwargs):
        if held_at == "factory":
            hold()
        return stream

    def start():
        stream.starts += 1
        if held_at == "start":
            hold()

    stream.start = start
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None,
        stream_factory=factory,
        start_authority=lambda: not revoked.is_set(),
    )
    opening = asyncio.create_task(asyncio.to_thread(sink.open, 24000))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        revoked.set()
        assert not sink.stop()
        retirement = asyncio.create_task(sink.retire())
        await asyncio.sleep(0)
        assert not retirement.done() and not sink.retirement_confirmed
        retirement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retirement
    finally:
        resume.set()
        await opening
    assert await sink.retire()
    assert stream.starts == (held_at == "start")
    assert stream.closed and stream.closes == 1
    assert sink.retirement_confirmed and not sink.quarantined


def test_rejected_authority_never_enters_factory():
    attempts = []
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None,
        stream_factory=lambda **_: attempts.append(True),
        start_authority=lambda: False,
    )
    sink.open(24000)
    assert not attempts and sink.retirement_confirmed


def test_managed_sink_is_independent_of_the_standalone_displacement_slot():
    standalone, managed = Device(), Device()
    first = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: standalone
    )
    second = audio.StreamingPcmSink(
        on_event=lambda _: None,
        stream_factory=lambda **_: managed,
        start_authority=lambda: True,
    )
    try:
        first.open(24000)
        second.open(24000)
        assert not standalone.closed
        audio.stop_live_sink()
        assert standalone.closed and not managed.closed
    finally:
        first.stop()
        second.stop()


def test_unknown_factory_outcome_remains_quarantined_with_no_stream():
    def incomplete(**kwargs):
        raise OSError("factory lost the handle")

    sink = audio.StreamingPcmSink(on_event=lambda _: None, stream_factory=incomplete)
    sink.open(24000)
    assert not sink.stop() and sink.quarantined and not sink.retirement_confirmed


def test_factory_returning_no_handle_cannot_confirm_retirement():
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: None
    )
    sink.open(24000)
    assert sink.state == "failed" and sink.quarantined
    assert not sink.stop() and not sink.retirement_confirmed


def test_cancelled_opaque_factory_retains_uncertain_native_custody():
    def interrupted(**kwargs):
        raise asyncio.CancelledError

    sink = audio.StreamingPcmSink(on_event=lambda _: None, stream_factory=interrupted)
    with pytest.raises(asyncio.CancelledError):
        sink.open(24000)
    assert sink.quarantined and not sink.stop()


def test_unknown_handle_after_close_failure_prevents_an_unsafe_retry():
    class UnknownDevice(Device):
        @property
        def closed(self):
            raise OSError("native status unreadable")

        @closed.setter
        def closed(self, value):
            pass

    stream = UnknownDevice()
    stream.fail_close = True
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    sink.open(24000)
    assert not sink.stop() and not sink.stop()
    assert sink.quarantined and stream.closes == 1


@pytest.mark.asyncio
async def test_cancelled_observer_keeps_one_committed_close_in_custody():
    stream = Device()
    entered, release = threading.Event(), threading.Event()

    def close():
        stream.closes += 1
        entered.set()
        assert release.wait(2)
        stream.closed = True

    stream.close = close
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    sink.open(24000)
    observer = asyncio.create_task(sink.retire())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        joined = asyncio.create_task(sink.retire())
        await asyncio.sleep(0)
        assert not joined.done() and not sink.retirement_confirmed
    finally:
        release.set()
    assert await joined
    assert stream.closes == 1 and stream.closed


def test_failed_close_retries_only_the_still_open_stream():
    stream = Device()
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    sink.open(24000)
    stream.fail_close = True
    assert not sink.stop()
    assert not sink.retirement_confirmed and sink.quarantined
    stream.fail_close = False
    assert sink.stop() and sink.retirement_confirmed
    assert stream.aborts == 1 and stream.closes == 2


def test_failed_abort_and_successful_close_preserves_uncertain_interruption():
    stream = Device()
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    sink.open(24000)
    stream.fail_abort = True
    assert not sink.stop()
    assert stream.closed and stream.closes == 1
    assert not sink.stop() and sink.quarantined
    assert stream.closes == 1


@pytest.mark.parametrize("failure", ["constructor", "close"])
def test_default_backend_retains_partial_resource_and_checks_native_status(
    monkeypatch, failure
):
    calls = []
    handle = object()

    class Output:
        def __init__(self, **kwargs):
            self._ptr = handle
            if failure == "constructor":
                raise OSError("error after Pa_OpenStream")

        def start(self):
            calls.append("start")

        def abort(self, *, ignore_errors=True):
            calls.append(("abort", ignore_errors))

        def close(self, *, ignore_errors=True):
            assert self._ptr is handle
            self._ptr = None
            calls.append(("close", ignore_errors))
            if failure == "close":
                raise OSError("Pa_CloseStream status uncertain")

    ffi = SimpleNamespace(NULL=None, typeof=lambda obj: type(obj))
    backend = SimpleNamespace(OutputStream=Output, _ffi=ffi)
    monkeypatch.setattr(audio, "_import_sounddevice", lambda: backend)
    sink = audio.StreamingPcmSink(on_event=lambda _: None)
    sink.open(24000)
    if failure == "constructor":
        assert sink.state == "failed" and sink.retirement_confirmed
        assert calls == [("close", False)]
    else:
        assert not sink.stop() and sink.quarantined
        assert not sink.stop() and not sink.retirement_confirmed
        assert calls == ["start", ("abort", False), ("close", False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["source_error", "cancellation"])
async def test_pump_retirement_does_not_block_the_event_loop(exit_kind):
    entered, release = threading.Event(), threading.Event()
    feeding = asyncio.Event()
    loop_thread = threading.get_ident()
    calls = []
    stream = Device()

    def close():
        entered.set()
        calls.append(threading.get_ident())
        assert release.wait(2)
        stream.closed = True

    stream.close = close
    sink = audio.StreamingPcmSink(
        on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    sink.open(24000)

    async def source():
        yield bytes(12)
        feeding.set()
        if exit_kind == "cancellation":
            await asyncio.Future()
        raise ValueError("source failed")

    pump = asyncio.create_task(audio.pump(sink, source()))
    try:
        await feeding.wait()
        if exit_kind == "cancellation":
            pump.cancel()
        assert await asyncio.to_thread(entered.wait, 1)
        assert calls != [loop_thread]
        assert not sink.retirement_confirmed and not pump.done()
    finally:
        release.set()
        if exit_kind == "cancellation":
            with pytest.raises(asyncio.CancelledError):
                await pump
        else:
            assert (await pump).outcome == "source_error"
    assert stream.closed and sink.retirement_confirmed
