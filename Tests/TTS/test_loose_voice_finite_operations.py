"""Installed loose voice behavior under exact-root admission and live draining."""

import os
import subprocess
import sys

import pytest

_SETUP = r"""
from Tests.network_guard import install
install()
import asyncio, json, os, sys, threading, time, wave
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, bind_profile
from tldw_chatbook.Backup_Recovery.profile_paths import default_base_data_dir
home=Path.home(); config=Path(os.environ['TLDW_CONFIG_PATH']); root=home/'exact-voices'
root.mkdir(mode=0o700); data=default_base_data_dir(); data.mkdir(parents=True,mode=0o700)
exports=home/'exports'; exports.mkdir(mode=0o700)
control=bootstrap.default_bootstrap_root(); authority=admission_authority(control)
authority.register('voice-profile',(root,data,config.parent,exports))
bind_profile(control,config,('voice-profile',),control/'admission')
from tldw_chatbook.TTS import loose_voice_lifetime as lifetime
source=home/'input.wav'
with wave.open(str(source),'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(b'\0\0'*800)
source.chmod(0o600)
def settled():
    assert not storage._raw_operations, storage._raw_operations
    assert set(storage._live_leases)==set(storage._startups.values())
"""


def _child(tmp_path, code):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_dir = home / "selected-config"
    config_dir.mkdir(mode=0o700)
    config = config_dir / "config.toml"
    config.write_text('[general]\nusers_name="default_user"\n')
    config.chmod(0o600)
    env = dict(
        os.environ,
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / "xdg-config"),
        XDG_DATA_HOME=str(home / "xdg-data"),
        TLDW_CONFIG_PATH=str(config),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _SETUP + code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["chatterbox", "higgs", "kokoro"])
def test_installed_backend_exact_root_roundtrip(tmp_path, backend):
    _child(
        tmp_path,
        {
            "chatterbox": r"""
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend
backend=ChatterboxTTSBackend({'CHATTERBOX_VOICE_DIR':str(root)})
assert asyncio.run(backend.list_voices())==['default']
assert asyncio.run(backend.save_reference_voice_with_metadata('one',str(source),{'label':'saved'}))
assert (root/'one.wav').read_bytes()==source.read_bytes()
assert json.loads((root/'one_metadata.json').read_text())['label']=='saved'
assert 'one' in asyncio.run(backend.list_voices())
assert asyncio.run(backend.list_voices_with_metadata())[1]['metadata']['label']=='saved'
settled()
""",
            "higgs": r"""
from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
backend=HiggsAudioTTSBackend({'HIGGS_VOICE_SAMPLES_DIR':str(root)})
assert len(backend.list_voice_profiles())==6
assert asyncio.run(backend.create_voice_profile('one',str(source)))
saved=json.loads((root/'voice_profiles.json').read_text())
assert Path(saved['one']['reference_audio']).read_bytes()==source.read_bytes()
assert backend.delete_voice_profile('one')
assert 'one' not in json.loads((root/'voice_profiles.json').read_text())
assert not (root/'one').exists()
settled()
""",
            "kokoro": r"""
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
backend=KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
assert len(backend.list_voice_blends())==8
assert backend.save_voice_blend('one',[('af_bella',1)])
assert backend.get_voice_blend('one') is backend.saved_blends['one']
assert 'one' in json.loads((root/'voice_blends.json').read_text())
assert backend.delete_voice_blend('one')
assert 'one' not in json.loads((root/'voice_blends.json').read_text())
assert not backend.save_voice_blend('invalid',[('af_bella',0)])
settled()
""",
        }[backend],
    )


@pytest.mark.parametrize("manager", ["chatterbox", "higgs"])
def test_installed_manager_create_export_import_delete_backup(tmp_path, manager):
    _child(
        tmp_path,
        f"kind={manager!r}\n"
        + r"""
if kind=='chatterbox':
    from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager as Manager
else:
    from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager as Manager
manager=Manager(root)
assert manager.create_profile('one',str(source))[0]
assert manager.export_profile('one',str(exports))[0]
package=exports/(kind+'_voice_one')
assert manager.import_profile(str(package),'two')[0]
profile=manager.get_profile('two')
assert Path(profile['reference_audio']).read_bytes()==source.read_bytes()
assert manager.delete_profile('one')[0]
assert manager.get_profile('one') is None
assert manager.get_profile('two') is not None
if kind=='higgs':
    assert list((root/'backups').glob('*.json'))
    assert manager.restore_from_backup()[0]
settled()
""",
    )


def test_active_worker_drains_after_cancelled_waiter_and_resumes(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
from tldw_chatbook.Utils import private_paths
backend=KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
entered=threading.Event(); release=threading.Event()
original=os.fsync
def fsync(*args,**kwargs):
    if lifetime.native() is not None:
        entered.set(); assert release.wait(5)
    return original(*args,**kwargs)
os.fsync=fsync
async def exercise():
    task=asyncio.create_task(asyncio.to_thread(backend.save_voice_blend,'worker',[('af_bella',1)]))
    while not entered.is_set():
        if task.done():
            task.result()
            raise AssertionError("worker returned before publication")
        await asyncio.sleep(.005)
    pause=storage._begin_local_pause()
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    assert not pause.drain(time.monotonic()+.03)
    try: backend.save_voice_blend('denied',[('af_bella',1)])
    except bootstrap.RecoveryRequired: pass
    else: raise AssertionError('new writer entered')
    release.set()
    assert pause.drain(time.monotonic()+3)
    pause.resume()
asyncio.run(exercise())
os.fsync=original
assert 'worker' in json.loads((root/'voice_blends.json').read_text())
assert 'denied' not in backend.saved_blends
assert backend.save_voice_blend('resumed',[('af_bella',1)])
settled()
""",
    )


def test_ambiguous_native_close_blocks_capture_drain(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
backend=KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
original=os.close; triggered=False
# Fail only a regular output descriptor; directory traversal is unaffected.
import stat
def close(fd):
    global triggered
    if lifetime.native() is not None and not triggered and stat.S_ISREG(os.fstat(fd).st_mode):
        triggered=True; raise OSError('uncertain native close')
    return original(fd)
os.close=close
assert not backend.save_voice_blend('uncertain',[('af_bella',1)])
os.close=original
assert triggered
pause=storage._begin_local_pause()
assert not pause.drain(time.monotonic()+.03)
pause.resume()
""",
    )


def test_observed_root_drift_and_path_escape_refused(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend
backend=ChatterboxTTSBackend({'CHATTERBOX_VOICE_DIR':str(root)})
try: asyncio.run(backend.save_reference_voice('../escape',str(source)))
except ValueError: pass
else: raise AssertionError('unsafe name accepted')
settled()
root.rename(root.with_name('prior')); root.mkdir(mode=0o700)
try: asyncio.run(backend.list_voices())
except bootstrap.RecoveryRequired: pass
else: raise AssertionError('observed replacement accepted')
pause=storage._begin_local_pause()
assert not pause.drain(time.monotonic()+.03)
pause.resume()
""",
    )


def test_failed_related_publication_keeps_bytes_and_blocks_drain(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend
backend=ChatterboxTTSBackend({'CHATTERBOX_VOICE_DIR':str(root)})
assert not asyncio.run(backend.save_reference_voice_with_metadata('partial',str(source),{'bad':object()}))
assert (root/'partial.wav').read_bytes()==source.read_bytes()
assert not (root/'partial_metadata.json').exists()
pause=storage._begin_local_pause()
assert not pause.drain(time.monotonic()+.03)
pause.resume()
""",
    )


def test_rejected_write_preserves_cache_without_fictitious_writer(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends import kokoro
backend=kokoro.KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
path=root/'voice_blends.json'; before=path.read_bytes(); cache=dict(backend.saved_blends)
def refuse(*args,**kwargs): raise ValueError('rejected before native publication')
kokoro.write_private_json=refuse
assert not backend.save_voice_blend('rejected',[('af_bella',1)])
assert backend.saved_blends==cache and path.read_bytes()==before
settled()
""",
    )


def test_import_reference_cannot_read_or_write_package_sibling(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
manager=HiggsVoiceProfileManager(root)
package=exports/'package'; package.mkdir(mode=0o700)
(package/'profile.json').write_text(json.dumps({'reference_audio':'../input.wav'}))
(package/'profile.json').chmod(0o600)
(exports/'input.wav').write_bytes(source.read_bytes())
assert not manager.import_profile(str(package),'escape')[0]
assert not (root/'input.wav').exists()
assert manager.get_profile('escape') is None
settled()
""",
    )


@pytest.mark.parametrize(
    "kind", ["chatterbox", "higgs", "kokoro", "chatterbox_manager", "higgs_manager"]
)
def test_fresh_constructor_denied_before_filesystem_effects(tmp_path, kind):
    _child(
        tmp_path,
        f"kind={kind!r}\n"
        + r"""
from tldw_chatbook.TTS.backends.chatterbox import ChatterboxTTSBackend
from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
constructors={
'chatterbox':lambda:ChatterboxTTSBackend({'CHATTERBOX_VOICE_DIR':str(root/'new')}),
'higgs':lambda:HiggsAudioTTSBackend({'HIGGS_VOICE_SAMPLES_DIR':str(root/'new')}),
'kokoro':lambda:KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root/'new')}),
'chatterbox_manager':lambda:ChatterboxVoiceManager(root/'new'),
'higgs_manager':lambda:HiggsVoiceProfileManager(root/'new')}
pause=storage._begin_local_pause()
try: constructors[kind]()
except bootstrap.RecoveryRequired: pass
else: raise AssertionError('constructor entered after pause')
assert not (root/'new').exists()
assert pause.drain(time.monotonic()+.03)
pause.resume()
""",
    )


def test_accepted_default_chain_finishes_after_intake_closes(tmp_path):
    _child(
        tmp_path,
        r"""
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
original=os.fsync; pause=None
def fsync(fd):
    global pause
    if lifetime.native() is not None and pause is None:
        pause=storage._begin_local_pause()
        assert not pause.drain(time.monotonic()+.01)
    return original(fd)
os.fsync=fsync
backend=KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
os.fsync=original
assert pause is not None and len(backend.saved_blends)==8
assert len(json.loads((root/'voice_blends.json').read_text()))==8
assert pause.drain(time.monotonic()+.03)
pause.resume()
settled()
""",
    )


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_native_write_retires_only_after_temporary_cleanup(
    tmp_path, cleanup_fails
):
    _child(
        tmp_path,
        f"cleanup_fails={cleanup_fails!r}\n"
        + r"""
from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
backend=KokoroTTSBackend({'KOKORO_VOICE_BLENDS_DIR':str(root)})
original_write=os.write; original_unlink=os.unlink; triggered=False

def failing_write(fd,payload):
    global triggered
    if lifetime.native() is not None:
        triggered=True
        original_write(fd,payload[:8])
        if cleanup_fails: os.unlink=failing_unlink
        raise OSError('injected partial disk write')
    return original_write(fd,payload)

def failing_unlink(path,*args,**kwargs):
    if lifetime.native() is not None:
        raise OSError('injected temp cleanup failure')
    return original_unlink(path,*args,**kwargs)

before=set(root.iterdir()); original_bytes=(root/'voice_blends.json').read_bytes()
original_cache=dict(backend.saved_blends)
os.write=failing_write
try: result=backend.save_voice_blend('partial',[('af_bella',1)])
finally: os.write=original_write; os.unlink=original_unlink
assert triggered and not result
assert (root/'voice_blends.json').read_bytes()==original_bytes
assert backend.saved_blends==original_cache
remaining=set(root.iterdir())-before
if cleanup_fails:
    assert len(remaining)==1
    temporary=remaining.pop()
    assert temporary.name.startswith('.voice_blends.json.') and temporary.suffix=='.tmp'
    assert len(temporary.read_bytes())==8
else:
    assert not remaining
    settled()
pause=storage._begin_local_pause()
assert pause.drain(time.monotonic()+.03) is not cleanup_fails
pause.resume()
""",
    )


@pytest.mark.parametrize("action", ["delete", "retention"])
def test_deletion_traversal_close_uncertainty_blocks_drain(tmp_path, action):
    _child(
        tmp_path,
        f"action={action!r}\n"
        + r"""
from tldw_chatbook.Utils import private_paths
if action=='delete':
    from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
    manager=ChatterboxVoiceManager(root)
    assert manager.create_profile('one',str(source))[0]
    target=root/'one'
    invoke=lambda:manager.delete_profile('one')
else:
    from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
    manager=HiggsVoiceProfileManager(root)
    assert manager.create_profile('one',str(source))[0]
    for number in range(11):
        (root/'backups'/f'voice_profiles_backup_000{number:02}.json').write_text('{}')
    target=sorted((root/'backups').glob('*.json'))[0]
    invoke=manager._create_backup
original_close=os.close; original_parent=private_paths._open_verified_parent
failed_fd=None; traversing=False

def selected_parent(path,**kwargs):
    global traversing
    if Path(path)!=target: return original_parent(path,**kwargs)
    traversing=True
    try: return original_parent(path,**kwargs)
    finally: traversing=False

def fail_close(fd):
    global failed_fd
    if failed_fd is None and traversing:
        failed_fd=fd
        raise OSError('injected actual deletion traversal close uncertainty')
    return original_close(fd)

private_paths._open_verified_parent=selected_parent; os.close=fail_close
try: invoke()
finally: os.close=original_close; private_paths._open_verified_parent=original_parent
assert failed_fd is not None
os.fstat(failed_fd)
assert target.exists()
pause=storage._begin_local_pause()
assert not pause.drain(time.monotonic()+.03)
pause.resume()
""",
    )
