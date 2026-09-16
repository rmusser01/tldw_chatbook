"""Saved generated images join a real complete capture without retaining temp data."""

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SAVED = r"""
import asyncio, json, os, sys, threading
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home = Path.home()
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Media_Creation.image_generation_service import (
    ImageGenerationService, GenerationResult,
)

async def main():
    app = TldwCli()
    images = ImageGenerationService()
    temporary = images.output_dir / 'temp' / 'generated.png'
    temporary.write_bytes(b'saved image payload')
    result = GenerationResult(True, [str(temporary)], 'fixture', '', {})
    saved = await images.save_generation(result, name='kept')
    temporary.write_bytes(b'temporary image remains disposable')
    options = {'staging_parent': home, 'temporary_media': False}
    preview = preview_capture((selector,), options=options)
    bad = [(i.owner, str(i.path), i.status) for i in preview.items
           if i.status not in ('included', 'included_directory', 'unused',
                               'intentionally_excluded', 'intentionally_deleted')]
    assert preview.complete, (preview.issues, bad)
    destination = home / 'saved-images.tldw-backup.zip'
    monitoring = asyncio.create_task(monitor_app(app))
    cancel = threading.Event()
    watchdog = asyncio.get_running_loop().call_later(20, cancel.set)
    try:
        captured = await asyncio.to_thread(
            capture, (selector,), preview.scope_digest, destination,
            options=options, cancel=cancel,
        )
        for _ in range(500):
            if storage._pause is None and app._backup_runtime_maintenance is None:
                break
            await asyncio.sleep(.01)
        assert storage._pause is None and app._backup_runtime_maintenance is None
        assert temporary.read_bytes() == b'temporary image remains disposable'
        later = await images.save_generation(result, name='after-capture')
        assert Path(later[0]).read_bytes() == b'temporary image remains disposable'
        manifest = json.loads(captured.manifest_bytes)
        assert captured.inventory.complete
        assert manifest['consistency'] == 'coherent'
        members = [f for f in manifest['files'] if f['owner_id'] == 'generation.assets']
        assert len(members) == 1, members
        assert members[0]['relative_path'] == 'saved/kept.png'
        assert (captured.root / members[0]['payload']).read_bytes() == b'saved image payload'
        assert Path(saved[0]).read_bytes() == b'saved image payload'
        assert not destination.exists()
        sealed = await asyncio.to_thread(
            write_archive, captured, destination, password=None,
            cancel=threading.Event(),
        )
        assert sealed.path == destination and destination.is_file()
        assert not blocked_attempts()
    finally:
        watchdog.cancel()
        cancel.set()
        monitoring.cancel()
        try:
            await monitoring
        except asyncio.CancelledError:
            pass
        try:
            await app._shutdown_app_owned_lifecycles()
        except asyncio.CancelledError:
            pass
        try:
            await app.tts_service.close()
        except asyncio.CancelledError:
            pass
asyncio.run(main())
print('retired and reopened')
"""


def test_saved_images_complete_capture_excludes_temp_and_resumes_before_packaging(
    tmp_path,
):
    _run(tmp_path, "saved", "complete", script=_SAVED)
