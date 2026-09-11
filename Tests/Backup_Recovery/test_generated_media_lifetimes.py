"""Actual generated-file owners participate in live maintenance."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_PAUSED = r"""
import asyncio, sys, time
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for module in ('sounddevice', 'pyaudio'):
    sys.modules[module] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from tldw_chatbook.Media_Creation.image_generation_service import (
    ImageGenerationService, GenerationResult,
)
from tldw_chatbook.Video_Generation.video_store import VideoStore

async def main():
    app = TldwCli()
    images = ImageGenerationService()
    videos = VideoStore()
    source = images.output_dir / 'temp' / 'fixture.png'
    source.write_bytes(b'original image bytes')
    result = GenerationResult(True, [str(source)], 'fixture', '', {})
    target = (images.output_dir / 'saved' / 'fixture.png' if sys.argv[1] == 'image'
              else videos.root / 'message' / 'fixture.mp4')
    runtime = RuntimeMaintenance(app)
    async def save():
        if sys.argv[1] == 'image':
            return await images.save_generation(result, name='fixture')
        return videos.save('message', 'fixture', b'video bytes', extension='mp4')
    try:
        await runtime.settle_producers(time.monotonic() + 15)
        runtime.retire_local_caches()
        assert runtime.pause.drain(time.monotonic())
        try:
            await save()
        except RecoveryRequired:
            pass
        else:
            raise AssertionError('generated owner wrote during maintenance')
        assert not target.exists()
        assert source.read_bytes() == b'original image bytes'
        await runtime.resume()
        await save()
        assert target.read_bytes() == (
            b'original image bytes' if sys.argv[1] == 'image' else b'video bytes'
        )
    finally:
        await runtime.resume()
        try:
            await app._shutdown_app_owned_lifecycles()
        except asyncio.CancelledError:
            pass
        try:
            await app.tts_service.close()
        except asyncio.CancelledError:
            pass
    assert not blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize("owner", ("image", "video"))
def test_actual_generated_owner_refuses_pause_then_resumes(tmp_path, owner):
    _run(tmp_path, owner, "success", script=_PAUSED)


_ACCEPTED = r"""
import asyncio, sys, threading, time
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for module in ('sounddevice', 'pyaudio'):
    sys.modules[module] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.generated_media_lifetime import participant
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from tldw_chatbook.Media_Creation.image_generation_service import ImageGenerationService
from tldw_chatbook.Video_Generation.video_store import VideoStore

async def until(condition):
    deadline = time.monotonic() + 15
    while not condition():
        assert time.monotonic() < deadline, 'native operation did not reach barrier'
        await asyncio.sleep(.01)

async def main():
    app = TldwCli()
    started, release, native_finished = (threading.Event() for _ in range(3))
    resumed = asyncio.Event()
    state = {}
    if sys.argv[1] == 'image':
        import aiofiles.threadpool
        images = ImageGenerationService()
        class Client:
            async def generate_image(self, **kwargs):
                return {'success': True, 'images': ['fixture']}
            async def get_image(self, path):
                return b'actual image payload'
        images.client = Client()
        original_open = aiofiles.threadpool.sync_open
        def held_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if Path(path).parent == images.output_dir / 'temp':
                started.set()
                if not release.wait(20):
                    handle.close()
                    raise TimeoutError('test native open barrier')
                native_finished.set()
            return handle
        aiofiles.threadpool.sync_open = held_open
        work = asyncio.create_task(images.generate_custom('fixture'))
    else:
        videos = VideoStore()
        original_publish = VideoStore._atomic_publish
        def held_publish(self, *args, **kwargs):
            started.set()
            assert release.wait(20), 'test native publication barrier'
            try:
                return original_publish(self, *args, **kwargs)
            finally:
                native_finished.set()
        VideoStore._atomic_publish = held_publish
        work = asyncio.create_task(asyncio.to_thread(
            videos.save, 'message', 'fixture', b'actual video payload', extension='mp4'
        ))
    maintenance = None
    async def maintain():
        runtime = RuntimeMaintenance(app)
        try:
            await runtime.settle_producers(time.monotonic() + 20)
            assert native_finished.is_set()
            runtime.retire_local_caches()
            assert runtime.pause.drain(time.monotonic())
            state['paused'] = True
            await resumed.wait()
        finally:
            await runtime.resume()
    try:
        await until(started.is_set)
        work.cancel()
        await asyncio.sleep(0)
        maintenance = asyncio.create_task(maintain())
        await until(lambda: participant.calls.closed or maintenance.done())
        assert participant.calls.closed, 'runtime skipped generated native producer'
        assert not maintenance.done(), 'runtime passed the retained native worker'
        assert not state.get('paused')
        release.set()
        try:
            await work
        except asyncio.CancelledError:
            pass
        await until(lambda: state.get('paused') or maintenance.done())
        assert state.get('paused')
        if sys.argv[1] == 'image':
            outputs = list((images.output_dir / 'temp').glob('*.png'))
            assert len(outputs) == 1
            assert outputs[0].read_bytes() == b'actual image payload'
        else:
            assert (videos.root / 'message' / 'fixture.mp4').read_bytes() == b'actual video payload'
        resumed.set()
        await maintenance
        assert not participant.calls.closed
    finally:
        release.set()
        resumed.set()
        try:
            await work
        except asyncio.CancelledError:
            pass
        if maintenance is not None:
            await maintenance
        try:
            await app._shutdown_app_owned_lifecycles()
        except asyncio.CancelledError:
            pass
        try:
            await app.tts_service.close()
        except asyncio.CancelledError:
            pass
    assert not blocked_attempts()
    print('retired and reopened')
asyncio.run(main())
"""


@pytest.mark.parametrize("owner", ("image", "video"))
def test_runtime_drains_cancelled_generated_native_work(tmp_path, owner):
    _run(tmp_path, owner, "cancel", script=_ACCEPTED)
