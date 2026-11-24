"""Console synthesis and file playback share the app's exact audio authority."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as events
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
from tldw_chatbook.TTS.audio_output_runs import FileOutputRun, PcmOutputRun
from tldw_chatbook.TTS.pcm_stream import SinkPlan


@pytest.mark.asyncio
async def test_stop_during_file_preparation_prevents_the_late_native_start(
    tmp_path, monkeypatch
):
    handler = events.TTSEventHandler()
    host = AudioOutputOwnership()
    handler.app = SimpleNamespace(audio_output=host)
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")
    handler._audio_files["m"] = path
    preparing, ready = asyncio.Event(), asyncio.Event()
    started = []

    class Prepared:
        external = False
        handed_off = False
        lease = None
        retirement_confirmed = False

        async def start(self, lease):
            started.append(lease)
            return False

        async def retire(self):
            self.retirement_confirmed = True
            return True

    async def prepare(cls, selected):
        assert selected == path
        preparing.set()
        await ready.wait()
        return Prepared()

    monkeypatch.setattr(FileOutputRun, "prepare", classmethod(prepare))
    # The baseline must never dispatch a real player.
    monkeypatch.setattr(events, "play_audio_file", lambda *_: None, raising=False)
    playing = asyncio.create_task(
        handler.handle_tts_playback(events.TTSPlaybackEvent("play", "m"))
    )
    try:
        await asyncio.wait_for(preparing.wait(), 1)
        await handler.handle_tts_playback(events.TTSPlaybackEvent("stop", "m"))
        assert path.exists()
        ready.set()
        await playing
        await asyncio.gather(*tuple(handler._active_tasks))
        assert not started and host.owner is None
    finally:
        ready.set()
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_preempted_pcm_open_never_requests_file_fallback(monkeypatch):
    handler = events.TTSEventHandler()
    host = AudioOutputOwnership()
    handler.app = SimpleNamespace(audio_output=host)
    entered, release = asyncio.Event(), asyncio.Event()

    async def failed_after_transfer(self, lease):
        self.lease = lease
        entered.set()
        await release.wait()
        return False

    async def confirmed(self):
        self.retirement_confirmed = True
        return True

    async def source():
        yield b""

    monkeypatch.setattr(PcmOutputRun, "start", failed_after_transfer)
    monkeypatch.setattr(PcmOutputRun, "retire", confirmed)
    from tldw_chatbook.Audio.streaming_sink import StreamingPcmSink

    monkeypatch.setattr(StreamingPcmSink, "open", lambda *a: None)
    task = asyncio.create_task(
        handler._stream_response_via_sink(
            SinkPlan(sample_rate=24000, channels=1, skip_bytes=0, data_bytes=None),
            source(),
            message_id="m",
            fallback_on_failure=True,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        successor = await host.acquire("other", lambda: asyncio.sleep(0, result=True))
        release.set()
        assert await task in {"interrupted", "superseded"}
        assert successor.is_current()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_delayed_bare_stop_and_old_cleanup_cannot_touch_new_same_path_owner(
    tmp_path, monkeypatch
):
    from tldw_chatbook.TTS.audio_output_runs import PreparedOutputRun
    from tldw_chatbook.TTS.audio_player import PlaybackState

    handler = events.TTSEventHandler()
    app = handler.app = SimpleNamespace(
        audio_output=AudioOutputOwnership(), _tts_handler=handler
    )
    path = tmp_path / "shared.wav"
    path.write_bytes(b"clip")
    starts = [asyncio.Event(), asyncio.Event()]
    runs = []

    class Player(PreparedOutputRun):
        external = False
        handed_off = False

        def __init__(self):
            super().__init__()
            self.ended = asyncio.Event()

        async def start(self, lease):
            self._bind(lease)
            starts[runs.index(self)].set()
            return True

        async def get_state(self):
            await self.ended.wait()
            return PlaybackState.FINISHED

        async def _close_native(self):
            self.ended.set()
            return True

    async def prepare(cls, selected):
        assert selected == path
        player = Player()
        runs.append(player)
        return player

    monkeypatch.setattr(FileOutputRun, "prepare", classmethod(prepare))
    handler._audio_files["m"] = path
    try:
        await handler.handle_tts_playback(events.TTSPlaybackEvent("play", "m"))
        await asyncio.wait_for(starts[0].wait(), 1)
        old_stop = events.TTSPlaybackEvent.captured_stop(app, app._tts_latest_intent)
        assert runs[0].lease.revoked
        await handler.handle_tts_playback(events.TTSPlaybackEvent("play", "m"))
        await asyncio.wait_for(starts[1].wait(), 1)
        await handler.handle_tts_playback(old_stop)
        assert runs[1].lease.is_current() and not runs[1].ended.is_set()
        assert path.read_bytes() == b"clip"
    finally:
        for run in runs:
            run.ended.set()
        await asyncio.gather(*tuple(handler._active_tasks))
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_uncertain_retirement_retains_file_until_late_positive_receipt(tmp_path):
    from tldw_chatbook.TTS.audio_output_runs import PreparedOutputRun

    class HeldFile(PreparedOutputRun):
        can_close = False

        async def _close_native(self):
            return self.can_close

    handler = events.TTSEventHandler()
    handler.app = SimpleNamespace(audio_output=AudioOutputOwnership())
    path = tmp_path / "held.wav"
    path.write_bytes(b"held")
    handler._audio_files["message"] = path
    attempt = handler._prepare_output("message", None, path=path)
    run = attempt.run = HeldFile()
    run._bind(
        await handler.app.audio_output.acquire(
            "speech", run.retire, intent=attempt.ticket
        )
    )

    assert not await handler._retire_output(attempt)
    await handler.cleanup_tts_resources()
    assert handler._audio_files["message"] == path and path.exists()
    assert not handler.maintenance_ready and handler.app.audio_output.blocked
    run.can_close = True
    assert await handler.app.audio_output.stop(run.lease)
    await asyncio.gather(*tuple(handler._pending_legacy_cleanup_timers))
    assert not path.exists() and handler.maintenance_ready


@pytest.mark.asyncio
@pytest.mark.parametrize("replaced", [False, True])
async def test_handed_off_file_timer_rechecks_the_absolute_deadline_after_early_wake(
    tmp_path, monkeypatch, replaced
):
    handler = events.TTSEventHandler()
    path = tmp_path / "delegated.wav"
    path.write_bytes(b"clip")
    handler._audio_files["old"] = path
    now = [10.0]
    sleeps = []

    class Clock:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        async def sleep(self, delay):
            sleeps.append(delay)
            now[0] += delay / 2 if len(sleeps) == 1 else delay

    monkeypatch.setattr(events, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(events, "asyncio", Clock())
    handler._hand_off_retained_until[path] = 12.0
    if replaced:
        await handler._discard_tts_artifact("old", path)
    else:
        await handler._cleanup_audio_file("old")
    assert path.exists()
    await asyncio.gather(*tuple(handler._pending_legacy_cleanup_timers))
    assert sleeps == [2.0, 1.0]
    assert not path.exists()


@pytest.mark.asyncio
async def test_logical_pcm_drain_with_unconfirmed_close_cannot_complete_or_fall_back(
    monkeypatch,
):
    from Tests.TTS_Events.test_spoken_feedback_streaming import _RecordingSink
    from tldw_chatbook.TTS import audio_output_runs

    class FailedClose(_RecordingSink):
        async def retire(self):
            return False

    async def audio():
        yield bytes(480)

    handler = events.TTSEventHandler()
    host = AudioOutputOwnership()
    handler.app = SimpleNamespace(audio_output=host)
    finished = []
    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", FailedClose)
    monkeypatch.setattr(
        handler, "_post_tts_message", lambda *_: asyncio.sleep(0, result=True)
    )
    result = await handler._stream_response_via_sink(
        SinkPlan(24000, 1, 0, None),
        audio(),
        message_id="held",
        on_finished=finished.append,
        fallback_on_failure=True,
    )
    assert result == "streaming_failed" and finished == [False] and host.blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("during_lookup", [True, False])
async def test_cancelled_file_play_settles_pending_action_before_native_dispatch(
    tmp_path, monkeypatch, during_lookup
):
    handler = events.TTSEventHandler()
    handler.app = SimpleNamespace(audio_output=AudioOutputOwnership())
    path = tmp_path / "pending.wav"
    path.write_bytes(b"clip")
    handler._audio_files["msg"] = path
    outcomes = []
    event = events.TTSPlaybackEvent("play", "msg", outcome_callback=outcomes.append)

    async def unexpected(*_args):
        pytest.fail("Cancelled play reached native preparation")

    monkeypatch.setattr(FileOutputRun, "prepare", unexpected)
    if during_lookup:
        async with handler._audio_files_lock:
            task = asyncio.create_task(handler.handle_tts_playback(event))
            await asyncio.sleep(0)
            assert handler._speech_outputs
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    else:
        await handler.handle_tts_playback(event)
        (task,) = handler._active_tasks
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert outcomes == [False] and not handler._speech_outputs and path.exists()
    await handler.cleanup_tts_resources()
