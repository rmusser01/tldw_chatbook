"""Installed Console video operations retain native work through transcript commit."""

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
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules import video as video_module
from tldw_chatbook.Video_Generation import worker
from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Chat.console_generate_video import _stage_pending_video
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import (
    RuntimeMaintenance, _console_hooks, _settle_stage, _resume_hooks,
)

async def until(condition):
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, 'native operation did not reach barrier'
        await asyncio.sleep(.01)

async def main():
    app = TldwCli()
    store = app.console_runtime.ensure_chat_store()
    from Tests.Chat.test_console_agent_bridge import _ChunkGateway
    app.console_runtime.ensure_chat_controller(
        store=store, provider_gateway=_ChunkGateway([['unused']])
    )
    session = store.ensure_session(title='Accepted video settlement')
    screen = ChatScreen(app)
    video = screen._video
    # Rendering is not needed for this exact installed controller/native-store proof.
    async def render_only():
        pass
    video._sync_native_console_chat_ui_fn = render_only
    video._ensure_console_chat_store_fn = lambda: store
    videos = video._ensure_console_video_store()
    message = 'accepted-' + sys.argv[1]
    body = b'private deterministic native video bytes'
    entered, release = threading.Event(), threading.Event()
    outcomes = []
    artifact = None
    if sys.argv[1] == 'generation':
        original = video_module.run_video_generation
        worker.run_generation = lambda *a, **kw: VideoGenResult(
            body, 'video/mp4', 'mp4', len(body)
        )
        def held(*args, **kwargs):
            result = original(*args, **kwargs)
            outcomes.append(result)
            entered.set()
            assert release.wait(10), 'native worker was not released'
            return result
        video_module.run_video_generation = held
        def start():
            return video._run_console_video_generation_operation(
                session_id=session.id, message_id=message,
                backend='comfyui', prompt='settlement proof', video_store=videos,
            )
    else:
        metadata = VideoGenerationMetadata(
            name='adopted-clip', prompt='settlement proof', backend='comfyui'
        )
        artifact = _stage_pending_video(
            metadata=metadata, message_id=message, slug=metadata.name,
            extension='mp4', content=body, max_bytes=1, reason='over_capacity',
        )
        video._pending_video_artifacts[message] = artifact
        def held(*args, **kwargs):
            result = videos.adopt_oversized(*args, **kwargs)
            outcomes.append((metadata, result))
            entered.set()
            assert release.wait(10), 'native adoption was not released'
            return result
        async def finalize(path):
            resolved = await asyncio.to_thread(
                videos.resolve, message, metadata.name, extension='mp4'
            )
            assert resolved == path
            store.append_video_message(
                session.id, video_metadata=metadata, persist=True, message_id=message
            )
            return path
        def start():
            return video._run_pending_console_video_operation(
                artifact, held, message, metadata.name, artifact.stream,
                artifact.size_bytes, extension='mp4', result_callback=finalize,
            )
    accepted = asyncio.create_task(start())
    closed = []
    settlement = None
    runtime = RuntimeMaintenance(app)
    try:
        await until(entered.is_set)
        accepted.cancel()
        await asyncio.sleep(.02)
        assert not accepted.done(), 'cancelled waiter abandoned native completion'
        assert outcomes[0][1].read_bytes() == body
        producers, _ = _console_hooks(app, (screen,))
        hooks = [hook for hook in producers if hook and hook.owner is video]
        assert len(hooks) == 1, 'runtime skipped actual screen video producer'
        settlement = asyncio.create_task(_settle_stage(hooks, closed, time.monotonic()+5))
        await asyncio.sleep(.05)
        assert not settlement.done(), 'maintenance skipped accepted native work'
        assert not await video._maintenance_drain(time.monotonic())
        assert app.chachanotes_db.get_connection().execute(
            'SELECT COUNT(*) FROM messages WHERE id=?', (message,)
        ).fetchone()[0] == 0
        try:
            await start()
        except RecoveryRequired:
            pass
        else:
            raise AssertionError('Console accepted video while fenced')
        release.set()
        result = await asyncio.gather(accepted, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        await settlement
        rows = app.chachanotes_db.get_connection().execute(
            'SELECT metadata_json FROM messages WHERE id=?', (message,)
        ).fetchall()
        assert len(rows) == 1 and 'video_generation' in rows[0][0]
        assert not video._pending_video_active_operations
        # Run the real remaining app settlement only after this selected producer.
        await runtime.settle_producers(time.monotonic()+10)
        runtime.retire_local_caches()
        assert runtime.pause.drain(time.monotonic()+1)
        await runtime.resume()
        await _resume_hooks(closed)
        message += '-resumed'
        if artifact is not None:
            video._pending_video_artifacts.pop(artifact.message_id)
            artifact.message_id = message
            video._pending_video_artifacts[message] = artifact
        await start()
        assert outcomes[-1][1].read_bytes() == body
    finally:
        release.set()
        await asyncio.gather(accepted, return_exceptions=True)
        if settlement is not None:
            await asyncio.gather(settlement, return_exceptions=True)
        await runtime.resume()
        await _resume_hooks(closed)
        if artifact is not None:
            artifact.close()
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


@pytest.mark.parametrize("route", ("generation", "adoption"))
def test_console_video_native_completion_precedes_core_pause(tmp_path, route):
    _run(tmp_path, route, "cancel", script=_SCRIPT)
