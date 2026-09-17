"""Real isolated pair crashes recover under exact native operation ownership."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_recovery_service import _SERVICE_ISOLATED

_HOOK = """from tldw_chatbook.Backup_Recovery import control_records
original_pair=control_records._publish_activation_record
def crash_pair(*args,**kwargs):
 original_pair(*args,**kwargs)
 os._exit(91)
control_records._publish_activation_record=crash_pair
"""
assert _SERVICE_ISOLATED.count("try:\n inspection=") == 1
_START = _SERVICE_ISOLATED.replace("try:\n inspection=", _HOOK + "try:\n inspection=")
_FINISH = r"""
import json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor
root=bootstrap.default_bootstrap_root()
service=RecoveryService(Path.home()/'custom-control')
damage=sys.argv[1]
try:
 operation=service.pending_operations()[0]['operation_id']
 assert service.status(operation)['actions']==('finish',)
 intent=next(root.glob('activation-update-*.json'))
 if damage!='none':
  record=json.loads(intent.read_text())
  if damage=='malformed':record['version']=True
  elif damage=='foreign':record['operation_id']='another-operation'
  elif damage=='generation':record['after'][0]['activation']['generation']='f'*32
  intent.write_text(json.dumps(record))
 before={p:p.read_bytes() for p in root.iterdir() if p.is_file() and p.name.startswith(('activation-','profile-','pending-'))}
 try:bootstrap._records(root)
 except ValueError as error:assert error.args==(('record_version',) if damage=='malformed' else ('activation_update_pending',)),error
 else:raise AssertionError('ordinary startup accepted split pair')
 running=service.start_recovery(operation,action='finish')
 state=service.wait(running,timeout=35)
 print('FINISH_RESULT',dict(state),flush=True)
 if damage!='none':
  assert state['state']=='recovery_required',dict(state)
  assert before=={p:p.read_bytes() for p in before}
  assert intent.exists()
  assert service.pending_operations()[0]['operation_id']==operation
 else:
  assert state['state']=='succeeded' and state['result']['restoration_validated'],dict(state)
  assert not service.pending_operations() and not intent.exists()
  pending,_=bootstrap._records(root)
  assert not pending
  # A completed idempotent engine retry also has no activation pair to repair.
  from threading import Event
  from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
  from tldw_chatbook.Backup_Recovery.plan_records import load_plan
  from tldw_chatbook.Backup_Recovery.isolated_restore import _finish_isolated
  from tldw_chatbook.Backup_Recovery.journal import _Prepared
  journal=_journal(service.control_root,operation)
  with journal._locked(exclusive=False) as parent:rows=journal._records(parent)
  prepared=_Prepared.model_validate(next(row.evidence for row in rows if row.event=='prepared'))
  _finish_isolated(Path(rows[0].evidence['stage']['path']),load_plan(journal),journal,tuple(prepared.publication.namespaces),prepared.generation,Event())
  profile=service.profiles()[0]['profile_id']
  entry=_launch_descriptor(profile,service.control_root)
  assert bootstrap.startup_permission(Path(entry.config),root)[0]
  (Path.home()/'opened-profile.json').write_text(json.dumps({'profile':profile,'installation':entry.installation_id}))
 assert not blocked_attempts()
finally:service.close()
"""
_OPEN = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
receipt=json.loads((Path.home()/'opened-profile.json').read_text())
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(receipt['profile'],Path.home()/'custom-control')
from tldw_chatbook.config import CLI_APP_CLIENT_ID
assert CLI_APP_CLIENT_ID==receipt['installation']
from tldw_chatbook.app import TldwCli
async def main():
 app=TldwCli()
 try:
  note=app.chachanotes_db.add_note('After recovered isolated pair','Ordinary native write and read.')
  row=app.chachanotes_db.get_note_by_id(note)
  assert row['content']=='Ordinary native write and read.',row
  assert not blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
  await app.tts_service.wait_closed()
asyncio.run(main())
print('fresh selected app read and wrote native data')
"""


def _child(root, script, args, environment, expected=0):
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    (root / (args[0] + ".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == expected, result.stdout[-3000:] + result.stderr[-5000:]


@pytest.mark.parametrize("damage", ["none", "malformed", "foreign", "generation"])
def test_actual_isolated_pair_crash_finish_or_refusal(tmp_path, damage):
    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    _child(tmp_path, _START, ["isolated", "service"], environment, expected=91)
    _child(tmp_path, _FINISH, [damage], environment)
    if damage == "none":
        _child(tmp_path, _OPEN, ["open"], environment)
