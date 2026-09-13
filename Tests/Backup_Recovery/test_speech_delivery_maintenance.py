"""Accepted app speech delivery survives reversible maintenance."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time, threading
from pathlib import Path
from types import SimpleNamespace
from textual.app import App
from loguru import logger
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSEventHandler, TTSCompleteEvent, TTSPlaybackEvent, TTSPlaybackLifecycle,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSProviderConfigurationChanged, STTSEventHandler, STTSSettingsSaveEvent
case = sys.argv[1]
async def main():
    app = object.__new__(TldwCli)
    App.__init__(app)
    app.loguru_logger = logger
    app._tts_handler = TTSEventHandler()
    app._tts_handler.app = app
    app._stts_handler = None
    app._speech_delivery_close_admission()
    app._speech_delivery_resume()
    app.notify = lambda *a, **k: None
    app.query = lambda *a, **k: ()
    path = Path.home() / 'accepted.wav'
    path.write_bytes(b'accepted artifact')
    states = []
    lifecycle = TTSPlaybackLifecycle(message_id='m', request_id=1,
        validator=lambda: True, callback=states.append)
    complete = TTSCompleteEvent('m', path, playback_lifecycle=lifecycle)
    played = []
    async def playback(event):
        played.append(event.action)
        event.report_outcome(True)
    app._tts_handler.handle_tts_playback = playback
    if case == 'monitor':
        from tldw_chatbook.Backup_Recovery import runtime_maintenance
        entered, release = threading.Event(), threading.Event()
        cleaned = []
        class ReachedCleanup(BaseException): pass
        async def monitor(owner):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                entered.set()
                await asyncio.to_thread(release.wait, 4)
                path.write_bytes(b'resumed')
                raise
        async def cleanup():
            cleaned.append(True)
            raise ReachedCleanup()
        runtime_maintenance.monitor_app = monitor
        app._shutdown_app_owned_lifecycles = cleanup
        app._start_backup_maintenance_monitor()
        retained = app._backup_maintenance_monitor_task
        app._start_backup_maintenance_monitor()
        assert app._backup_maintenance_monitor_task is retained
        await asyncio.sleep(0)
        stopping = asyncio.create_task(app.on_unmount())
        assert await asyncio.to_thread(entered.wait, 2)
        stopping.cancel()
        await asyncio.sleep(.02)
        assert not stopping.done() and not cleaned
        release.set()
        result = await asyncio.gather(stopping, return_exceptions=True)
        assert isinstance(result[0], ReachedCleanup)
        assert cleaned and path.read_bytes() == b'resumed'
        assert app._backup_maintenance_monitor_task is None
    elif case == 'cancel':
        entered, release = asyncio.Event(), asyncio.Event()
        async def delivered(event):
            entered.set(); await release.wait(); path.write_bytes(b'published')
        app._deliver_tts_complete_event = delivered
        assert await app._tts_handler._post_tts_message(complete)
        event = await app._message_queue.get()
        task = asyncio.create_task(app._settle_speech_delivery(event, delivered))
        await entered.wait()
        app._speech_delivery_close_admission()
        task.cancel()
        await asyncio.sleep(0)
        assert not await app._speech_delivery_drain(time.monotonic())
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert path.read_bytes() == b'published'
        assert await app._speech_delivery_drain(time.monotonic()+1)
    elif case == 'settings_lease':
        released, replies = [], []
        handler = app._stts_handler = STTSEventHandler()
        handler.app = app
        event = STTSSettingsSaveEvent({}, request_id=1,
            reply_to=SimpleNamespace(receive_stts_settings_save_result=replies.append),
            publication_lease=SimpleNamespace(abandon=lambda:released.append(True)))
        handler.maintenance_close_admission()
        app._speech_initialization_close_admission()
        await app.handle_stts_settings_save_event(event)
        assert released == [True] and event.publication_lease is None
        assert len(replies) == 1 and not replies[0].persisted
    elif case == 'settings':
        handler = app._stts_handler = STTSEventHandler()
        handler.app = app
        publication = SimpleNamespace(generation=1, published=True,
            provider_statuses={'openai':'applied'}, provider_revisions={'openai':1})
        handler._post_applied_settings_changes(None, publication)
        app._speech_delivery_close_admission()
        assert not await app._speech_delivery_drain(time.monotonic())
        app.handle_stts_provider_configuration_changed(await app._message_queue.get())
        assert await app._speech_delivery_drain(time.monotonic()+1)
    elif case == 'queued_playback':
        event = TTSPlaybackEvent('play', 'm', playback_lifecycle=lifecycle)
        assert app._post_speech_delivery(event)
        app._speech_delivery_close_admission()
        await app.handle_tts_playback_event(await app._message_queue.get())
        assert not played and not states
        assert await app._speech_delivery_drain(time.monotonic()+1)
        app._speech_delivery_resume()
        await app.handle_tts_playback_event(await app._message_queue.get())
        assert played == ['play']
    elif case == 'new_playback':
        outcomes = []
        app._speech_delivery_close_admission()
        event = TTSPlaybackEvent('play', 'm', outcome_callback=outcomes.append)
        await app.handle_tts_playback_event(event)
        assert outcomes == [False] and not played
        assert not getattr(app, '_speech_delivery_deferred', [])
        assert await app._speech_delivery_drain(time.monotonic()+1)
    elif case == 'rejected':
        app.post_message = lambda event: False
        assert not await app._tts_handler._post_tts_message(complete)
        app._speech_delivery_close_admission()
        assert await app._speech_delivery_drain(time.monotonic()+1)
    else:
        assert await app._tts_handler._post_tts_message(complete)
        app._speech_initialization_close_admission()
        app._speech_delivery_close_admission()
        assert not await app._speech_delivery_drain(time.monotonic())
        await app.handle_tts_complete_event(await app._message_queue.get())
        assert path.exists() and not states and not played
        assert await app._speech_delivery_drain(time.monotonic()+1)
        if case == 'stop':
            await app.control_tts_playback(TTSPlaybackEvent('stop', 'm'))
            assert played == ['stop']
        app._speech_initialization_resume()
        app._speech_delivery_resume()
        if case == 'autoplay':
            await app.handle_tts_playback_event(await app._message_queue.get())
            assert played == ['play']
        else:
            assert app._message_queue.empty()
            assert states == ['stopped']
    print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "case",
    [
        "autoplay",
        "stop",
        "cancel",
        "settings",
        "queued_playback",
        "rejected",
        "settings_lease",
        "monitor",
        "new_playback",
    ],
)
def test_accepted_speech_delivery(tmp_path, case):
    _run(tmp_path, case, "delivery", script=_SCRIPT)
