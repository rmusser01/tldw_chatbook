"""Tests for the `pump` helper and one-voice live-sink registry displacement.

`pump` bridges an async source of PCM chunks (as produced incrementally by a
streaming TTS adapter) into a `StreamingPcmSink`'s synchronous, non-blocking
`feed()`, handling backpressure (buffer-full retry), an optional
skip-bytes prefix (WAV headers from providers that can't be told to omit
one), early exit when the sink is stopped out from under the pump, and
normal end-of-stream draining. The registry tests cover Task 2's other
half: opening a new sink stops whichever sink was previously "live" (the
one-voice contract), and MUST do so without ever touching a stream directly
from within the registry lock (see the module docstring / task brief).
"""
import asyncio

import pytest

from Tests.Audio.test_streaming_sink import RATE, _mk, _pcm
from tldw_chatbook.Audio.streaming_sink import SinkStopped, pump


async def _aiter(chunks, delay_between=0):
    for c in chunks:
        if delay_between:
            await asyncio.sleep(0)
        yield c


@pytest.mark.asyncio
async def test_pump_feeds_everything_closes_and_reports_drained():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    result_task = asyncio.ensure_future(pump(sink, _aiter([_pcm(8), _pcm(8)])))
    await asyncio.sleep(0)                     # let pump feed
    h["s"].tick(20)                            # drain everything
    result = await result_task
    assert result.outcome == "drained"
    assert result.bytes_fed == len(_pcm(8)) * 2


@pytest.mark.asyncio
async def test_pump_skip_bytes_drops_wav_header():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    header = b"RIFF" + b"\x00" * 40            # 44 bytes
    body = _pcm(16)
    task = asyncio.ensure_future(pump(sink, _aiter([header + body[:100], body[100:]]),
                                      skip_bytes=44))
    await asyncio.sleep(0)
    h["s"].tick(1)
    played = b"".join(h["s"].out)
    assert b"RIFF" not in played
    sink.stop()
    await task


@pytest.mark.asyncio
async def test_pump_exits_promptly_when_sink_stopped_midstream():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)

    async def endless():
        while True:
            await asyncio.sleep(0)
            yield _pcm(1)

    task = asyncio.ensure_future(pump(sink, endless()))
    await asyncio.sleep(0)
    sink.stop()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result.outcome == "stopped"


@pytest.mark.asyncio
async def test_pump_source_error_stops_sink_and_reports():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)

    async def broken():
        yield _pcm(1)
        raise ValueError("backend died")

    result = await pump(sink, broken())
    assert result.outcome == "source_error"
    assert any(isinstance(e, SinkStopped) for e in events)


def test_opening_a_second_sink_displaces_the_first():
    e1, e2 = [], []
    s1, h1 = _mk(e1)
    s1.open(sample_rate=RATE)
    s2, h2 = _mk(e2)
    s2.open(sample_rate=RATE)
    assert s1.state == "stopped", "one-voice: prior sink must be stopped on new open"
    assert h1["s"].aborted is True
    assert s2.state == "open"
