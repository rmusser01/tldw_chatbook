"""Deferred and on-demand speech initialization remains owned during capture."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, threading, time, sys
from types import SimpleNamespace, MethodType
from loguru import logger
import tldw_chatbook.app as module
from tldw_chatbook.app import TldwCli

kind, scenario = sys.argv[1:]
async def main():
    entered, release = threading.Event(), threading.Event()
    made, scheduled = [], []
    class Handler:
        def __init__(self, **kwargs): made.append(self)
        async def initialize_tts(self):
            entered.set()
            await asyncio.to_thread(release.wait, 5)
        initialize_stts = initialize_tts
    module.TTSEventHandler = module.STTSEventHandler = Handler
    app = SimpleNamespace(_tts_handler=None, _stts_handler=None,
        _tts_initialization_task=None, _stts_initialization_task=None,
        loguru_logger=logger, _ensure_tts_profile_service=lambda: None)
    for name in (
        '_schedule_tts_initialization', '_schedule_stts_initialization',
        '_initialize_tts_service', '_initialize_stts_service',
        '_initialize_tts_service_owned', '_initialize_stts_service_owned',
        '_ensure_tts_handler', '_ensure_stts_handler',
        '_run_speech_initialization', '_speech_initialization_allowed', '_speech_initialization_close_admission',
        '_speech_initialization_drain', '_speech_initialization_resume',
        '_settle_speech_initialization',
    ):
        if hasattr(TldwCli, name):
            setattr(app, name, MethodType(getattr(TldwCli, name), app))
    def schedule(coro, **kwargs):
        task=asyncio.create_task(coro); scheduled.append(task); return task
    app._create_deferred_startup_task = schedule
    ensure = getattr(app, '_ensure_'+kind+'_handler')
    schedule_init = getattr(app, '_schedule_'+kind+'_initialization')
    if scenario.startswith('shutdown_'):
        class ReachedCleanup(BaseException): pass
        observed = []
        async def cleanup_boundary():
            observed.append(getattr(app, '_'+kind+'_handler'))
            raise ReachedCleanup()
        app._shutdown_app_owned_lifecycles = cleanup_boundary
        module.persist_event = lambda *args: None
        if scenario == 'shutdown_deferred':
            app._speech_initialization_close_admission()
            schedule_init()
            release.set()
            try:
                await TldwCli.on_unmount(app)
            except ReachedCleanup:
                pass
            app._speech_initialization_resume()
            assert await ensure() is None
            assert not made and not scheduled
        else:
            if scenario == 'shutdown_scheduled':
                schedule_init()
                first = scheduled[-1]
            else:
                first = asyncio.create_task(ensure())
            for _ in range(100):
                if entered.is_set(): break
                await asyncio.sleep(.01)
            assert entered.is_set()
            if scenario == 'shutdown_scheduled':
                first.cancel()
                await asyncio.sleep(0)
                assert not first.done(), 'cancelled startup wrapper detached native initialization'
            shutdown = asyncio.create_task(TldwCli.on_unmount(app))
            try:
                await asyncio.sleep(.02)
                assert not shutdown.done(), 'cleanup passed unretired on-demand initializer'
                assert await ensure() is None
                schedule_init()
                assert len(made) == 1
                if scenario == 'shutdown_cancel':
                    shutdown.cancel()
                    await asyncio.sleep(0)
                    assert not shutdown.done()
                release.set()
                await asyncio.gather(first, return_exceptions=True)
                result = await asyncio.gather(shutdown, return_exceptions=True)
                assert isinstance(result[0], ReachedCleanup)
                assert observed == [made[0]]
                app._speech_initialization_resume()
                assert await ensure() is None
                before = len(scheduled)
                schedule_init()
                assert len(scheduled) == before
            finally:
                release.set()
                await asyncio.gather(first, shutdown, return_exceptions=True)
    elif scenario == 'deferred':
        app._speech_initialization_paused = True
        schedule_init()
        assert not scheduled, 'paused schedule admitted initialization'
        assert await ensure() is None
        assert not made
        app._speech_initialization_resume()
        release.set()
        await asyncio.gather(*scheduled)
        assert len(made) == 1
        assert getattr(app, '_'+kind+'_handler') is made[0]
    else:
        first = asyncio.create_task(ensure())
        for _ in range(100):
            if entered.is_set(): break
            await asyncio.sleep(.01)
        assert entered.is_set()
        second = asyncio.create_task(ensure()) if scenario == 'concurrent' else None
        await asyncio.sleep(0)
        app._speech_initialization_close_admission()
        assert not await app._speech_initialization_drain(time.monotonic())
        if scenario == 'cancel':
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done()
            assert not await app._speech_initialization_drain(time.monotonic())
        assert await ensure() is None
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        assert await app._speech_initialization_drain(time.monotonic()+2)
        assert len(made) == 1
        assert getattr(app, '_'+kind+'_handler') is made[0]
        app._speech_initialization_resume()
        assert await ensure() is made[0]
    await asyncio.gather(*scheduled, return_exceptions=True)
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("kind", ["tts", "stts"])
@pytest.mark.parametrize("scenario", ["deferred", "active", "cancel", "concurrent"])
def test_speech_initialization_is_retained_or_deferred(tmp_path, kind, scenario):
    _run(tmp_path, kind, scenario, script=_SCRIPT)


@pytest.mark.parametrize("kind", ["tts", "stts"])
@pytest.mark.parametrize(
    "scenario",
    ["shutdown_ondemand", "shutdown_scheduled", "shutdown_deferred", "shutdown_cancel"],
)
def test_shutdown_settles_actual_initialization_before_cleanup(
    tmp_path, kind, scenario
):
    _run(tmp_path, kind, scenario, script=_SCRIPT)
