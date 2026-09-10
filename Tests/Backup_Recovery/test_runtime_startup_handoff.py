"""Actual app owners yield startup only across a native exclusive capture."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import asyncio, sys, threading, time
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

async def main():
    app = TldwCli()
    audio_tasks = []
    audio_release = asyncio.Event()
    if sys.argv[2] == 'audio':
        from tldw_chatbook.Audio_Services_Interop import local_audio_services_service as audio
        audio_entered = asyncio.Event()
        async def generate(**kwargs):
            audio_entered.set()
            await audio_release.wait()
            return b'accepted speech'
        owner = app.local_audio_services_service
        owner.tts_audio_generator = generate
        audio_tasks.append(asyncio.create_task(owner.create_audio_speech({'input':'before capture'})))
        await audio_entered.wait()
        async def finish_accepted_audio():
            while not audio._history_closed:
                await asyncio.sleep(.001)
            assert not app.tts_service._maintenance_paused, 'TTS closed before accepted audio finished'
            audio_release.set()
        audio_tasks.append(asyncio.create_task(finish_accepted_audio()))
    runtime = RuntimeMaintenance(app)
    app._backup_runtime_maintenance = runtime
    entered, release = threading.Event(), threading.Event()
    errors = []
    worker = None
    try:
        await runtime.settle_producers(time.monotonic()+10)
        if audio_tasks:
            results = await asyncio.gather(*audio_tasks)
            assert results[0]['content'] == b'accepted speech'
            assert (await owner.list_tts_history())['total'] == 1
            with __import__('pytest').raises(RuntimeError, match='audio_history_paused_for_maintenance'):
                await owner.create_audio_speech({'input':'during capture'})
        runtime.retire_local_caches()
        assert runtime.pause.drain(time.monotonic()+1)
        startup = next(iter(storage._startups.values()))
        holder = storage._holds[startup._key]
        with __import__('pytest').raises(RecoveryRequired, match='participant_runtime_coverage_incomplete'):
            runtime.pause.require_runtime_coverage()
        def capture_owner():
            try:
                with holder.authority.maintenance(holder.names, 10):
                    entered.set()
                    assert release.wait(10)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=capture_owner)
        worker.start()
        runtime.pause.retire_startup(runtime)
        for _ in range(200):
            if entered.is_set() or errors: break
            await asyncio.sleep(.01)
        assert entered.is_set(), errors
        assert not storage._startups
        with __import__('pytest').raises(RecoveryRequired, match='storage_locally_paused'):
            storage.acquire_storage()
        with __import__('pytest').raises(RecoveryRequired, match='startup_reacquisition_required'):
            runtime.pause.resume()
        loop = asyncio.get_running_loop()
        loop.call_later(.1, release.set)
        if sys.argv[2] == 'cancel':
            loop.call_later(.02, asyncio.current_task().cancel)
            with __import__('pytest').raises(asyncio.CancelledError):
                await runtime.resume()
        else:
            await runtime.resume()
        assert storage._pause is None
        assert len(storage._startups) == 1
        app.chachanotes_db.get_connection().execute('SELECT 1').fetchone()
        assert not errors
        assert not blocked_attempts()
        if audio_tasks:
            assert (await owner.create_audio_speech({'input':'after capture'}))['content'] == b'accepted speech'
    finally:
        audio_release.set()
        if audio_tasks:
            await asyncio.gather(*audio_tasks, return_exceptions=True)
        release.set()
        if worker is not None:
            worker.join(2)
            assert not worker.is_alive()
        if runtime.pause is not None or runtime.closed:
            await runtime.resume()
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        for participant in tuple(__import__('tldw_chatbook.Backup_Recovery.participants', fromlist=['_installed_repositories'])._installed_repositories):
            repository = participant.repository()
            if repository is not None:
                close = getattr(repository, 'close_connection', None) or getattr(repository, 'close', None)
                if close is not None:
                    close()
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("outcome", ["resume", "cancel", "audio"])
def test_actual_app_startup_returns_before_ordinary_admission(tmp_path, outcome):
    _run(tmp_path, "startup", outcome, script=_SCRIPT)


_MONITOR_SCRIPT = r"""
import asyncio, sys, threading
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery import participants
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app

async def main():
    app = TldwCli()
    startup = next(iter(storage._startups.values()))
    holder = storage._holds[startup._key]
    entered, release = threading.Event(), threading.Event()
    errors = []
    def capture_owner():
        try:
            with holder.authority.maintenance(holder.names, 10):
                entered.set()
                assert release.wait(10)
        except BaseException as error:
            errors.append(error)
    monitoring = asyncio.create_task(monitor_app(app))
    worker = threading.Thread(target=capture_owner)
    worker.start()
    try:
        for _ in range(500):
            if entered.is_set() or errors: break
            await asyncio.sleep(.01)
        assert entered.is_set(), (errors, getattr(app, '_backup_maintenance_error', None))
        assert not storage._startups
        if sys.argv[2] == 'cancel':
            monitoring.cancel()
            await asyncio.sleep(.05)
            assert not monitoring.done()
        release.set()
        for _ in range(500):
            if storage._pause is None: break
            await asyncio.sleep(.01)
        assert storage._pause is None
        assert len(storage._startups) == 1
        app.chachanotes_db.get_connection().execute('SELECT 1').fetchone()
        assert not errors
        assert not blocked_attempts()
    finally:
        release.set()
        monitoring.cancel()
        try:
            await monitoring
        except asyncio.CancelledError:
            pass
        worker.join(2)
        assert not worker.is_alive()
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        for participant in tuple(participants._installed_repositories):
            repository = participant.repository()
            if repository is not None:
                close = getattr(repository, 'close_connection', None) or getattr(repository, 'close', None)
                if close is not None: close()
asyncio.run(main())
print('retired and reopened')
"""


@pytest.mark.parametrize("outcome", ["resume", "cancel"])
def test_native_intent_monitor_settles_actual_app_and_reopens(tmp_path, outcome):
    _run(tmp_path, "monitor", outcome, script=_MONITOR_SCRIPT)
