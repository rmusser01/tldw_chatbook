"""Classify only the actual selected TTS repository's persistent empty lock."""

import json
import os
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _CHILD_ENVIRONMENT_KEYS,
    _run_profile_child,
)

_SCRIPT = r"""
import asyncio,json,os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
mode=os.environ['TTS_LOCK_CASE']
from tldw_chatbook.config import get_tts_profiles_db_path
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS import profile_repository
from Tests.Backup_Recovery.thread_diagnostics import _error_metadata
from tldw_chatbook.TTS.profile_types import TTSProfileDraft
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
selector=Path(os.environ['TLDW_CONFIG_PATH'])
database=get_tts_profiles_db_path()
lock=database.with_name(database.name+'.lock')
diagnostic_errors=[]
inventory_errors=[]
def observe_mapper(original):
 def observed(*errors):
  for error in errors:
   if isinstance(error,BaseException):
    diagnostic_errors.append(_error_metadata(error))
    del diagnostic_errors[:-16]
  return original(*errors)
 return observed
for mapper in ('_raise_operation_error','_raise_with_cleanup_precedence'):
 setattr(profile_repository,mapper,observe_mapper(getattr(profile_repository,mapper)))
async def main():
 repository=TTSProfileRepository(database)
 await repository.open()
 try:
  profile=(await repository.create_profile(TTSProfileDraft(display_name='Native lock',provider_id='openai',model_id='tts-1',voice_id='alloy',response_format='wav',speed=1.0,options={}))).value
  assert profile.revision==1
 finally:await repository.close()
 assert lock.is_file() and lock.read_bytes()==b'' and lock.stat().st_nlink==1
 other=database.parent/'unrelated.lock'
 if mode=='absent':lock.unlink()
 elif mode=='nonempty':lock.write_bytes(b'not a runtime lock')
 elif mode=='directory':lock.unlink();lock.mkdir()
 elif mode=='symlink':
  lock.unlink();other.write_bytes(b'');lock.symlink_to(other)
 elif mode=='hardlink':os.link(lock,other)
 elif mode=='neighbor':other.write_bytes(b'')
 elif mode=='native_custom':
  assert database==selector.parent/'custom'/'selected.sqlite'
  other=database.parent/'not-selected.sqlite.lock';other.write_bytes(b'')
 before=lock.lstat() if mode!='absent' else None
 def trace_inventory(frame,event,arg):
  if event=='exception' and frame.f_code.co_name=='walk' and Path(frame.f_code.co_filename).name=='file_inventory.py':
   error=arg[1]
   if not isinstance(error,FileNotFoundError):
    inventory_errors.append(_error_metadata(error));del inventory_errors[:-16]
  return trace_inventory
 previous_trace=sys.gettrace();sys.settrace(trace_inventory)
 try:inventory=preview_capture((selector,),options={'staging_parent':selector.parent.parent})
 finally:sys.settrace(previous_trace)
 rows=[item for item in inventory.items if item.path==lock]
 assert len(rows)==1,[(item.owner,item.logical_id,item.status,str(item.path)) for item in rows]
 row=rows[0]
 assert row.owner=='tts.profile_store' and row.logical_id.endswith(':lock'),row
 assert not [item for item in inventory.items if item.owner=='tts.references' and item.path==lock]
 expected='intentionally_excluded' if mode in {'native_default','native_custom','absent','neighbor'} else 'unsupported'
 assert row.status==expected,row
 if mode in {'neighbor','native_custom'}:
  if mode=='neighbor':assert any(item.owner=='unknown' and item.path==other for item in inventory.items)
  assert not any(item.owner.startswith('tts.') and item.path==other for item in inventory.items)
 aliases=[item for item in inventory.items if item.path==database]
 assert {item.owner for item in aliases}=={'tts.profile_store','tts.references'}
 assert all(item.status=='included' for item in aliases)
 assert len({item.shared_group for item in aliases})==1 and aliases[0].shared_group
 if before is None:assert not lock.exists()
 else:
  after=lock.lstat();assert (after.st_dev,after.st_ino,after.st_mode,after.st_size)==(before.st_dev,before.st_ino,before.st_mode,before.st_size)
 assert not blocked_attempts(),blocked_attempts()
 print('NATIVE_TTS_LOCK',mode,row.status)
try:asyncio.run(main())
except BaseException:
 print('TTS_REPOSITORY_FAILURE_METADATA',json.dumps(diagnostic_errors),flush=True)
 print('TTS_INVENTORY_FAILURE_METADATA',json.dumps(inventory_errors),flush=True)
 raise
"""


@pytest.mark.parametrize(
    "mode",
    [
        "native_default",
        "native_custom",
        "absent",
        "nonempty",
        "directory",
        "symlink",
        "hardlink",
        "neighbor",
    ],
)
def test_actual_tts_profile_lock_inventory(tmp_path, mode):
    root = tmp_path.resolve()
    for name in ("home", "config", "data", "cache", "tmp"):
        (root / name).mkdir(mode=0o700)
    selector = root / "config" / "config.toml"
    text = (
        '[general]\nusers_name="test"\n'
        + f"[paths]\ndata_dir={json.dumps(str(root / 'data'))}\n"
    )
    if mode == "native_custom":
        (selector.parent / "custom").mkdir(mode=0o700)
        text += (
            "[database]\ntts_profiles_db_path="
            + json.dumps(str(selector.parent / "custom" / "selected.sqlite"))
            + "\n"
        )
    selector.write_text(text)
    selector.chmod(0o600)
    environment = {
        key: os.environ[key]
        for key in _CHILD_ENVIRONMENT_KEYS
        if key in os.environ
    }
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        TTS_LOCK_CASE=mode,
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    _run_profile_child(root, mode, _SCRIPT, environment)
