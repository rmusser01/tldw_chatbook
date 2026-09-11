"""Only an actual mounted restored app can report a successful profile open."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _SERVICE_ISOLATED


@pytest.mark.parametrize(
    "args",
    [
        ["--recovery-launch-attempt", "a" * 32],
        ["--recovery-profile", "one", "--recovery-launch-attempt", "a" * 32],
        [
            "--recovery-profile",
            "one",
            "--recovery-control-root",
            "/tmp/control",
            "--recovery-launch-attempt",
            "a" * 32,
            "--recovery-launch-attempt=" + "b" * 32,
        ],
    ],
)
def test_launch_attempt_requires_unique_paired_selectors(monkeypatch, args):
    import sys

    from tldw_chatbook.cli import main_cli_runner

    monkeypatch.setattr(sys, "argv", ["tldw-cli", *args])
    with pytest.raises(SystemExit) as error:
        main_cli_runner()
    assert error.value.code == 2


@pytest.mark.parametrize("attempt", ["", "a" * 31, "../receipt", "A" * 32])
def test_invalid_attempt_refuses_before_profile_access(tmp_path, attempt):
    from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt

    with pytest.raises(ValueError, match="invalid_profile_launch_attempt"):
        opened_receipt("not-created", tmp_path, attempt)


_CHILD = r"""
import asyncio,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
args=sys.argv[1:];mode=args.pop()
sys.argv=['tldw-chatbook',*args,'--help']
from tldw_chatbook.cli import main_cli_runner
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0),error
if mode!='exit_only':
 from tldw_chatbook.app import TldwCli
 from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt
 profile=args[args.index('--recovery-profile')+1]
 control=Path(args[args.index('--recovery-control-root')+1])
 attempt=args[args.index('--recovery-launch-attempt')+1]
 assert opened_receipt(profile,control,attempt) is None
 async def mount():
  app=TldwCli()
  app.app_config['_first_run']=False
  app.app_config.setdefault('first_run',{})['setup_completed']=True
  app.app_config.setdefault('splash_screen',{})['enabled']=False
  if mode=='post_mount_failure':
   def post_failure():raise OSError('fixture post-mount incomplete')
   app._schedule_deferred_startup_work=post_failure
  reached=[]
  if mode=='read_failure':
   def fail():
    reached.append('read_failure')
    raise OSError('fixture native read unavailable')
   app.chachanotes_db.count_notes=fail
  if mode=='database_failure':
   execute=app.chachanotes_db.execute_query
   def missing(query,*args,**kwargs):
    reached.append('database_failure')
    return execute(query.replace('FROM notes','FROM missing_fixture_notes'),*args,**kwargs)
   app.chachanotes_db.execute_query=missing
  async with app.run_test(size=(100,36)) as pilot:
   async with asyncio.timeout(20):
    while not getattr(app,'_recovery_open_checked',False):
     if mode=='post_mount_failure' and app._ui_ready:
      await asyncio.sleep(.3);break
     await asyncio.sleep(.03)
   receipt=opened_receipt(profile,control,attempt)
   if mode in ('read_failure','database_failure'):assert mode in reached,reached
   assert (receipt is not None)==(mode not in ('read_failure','database_failure','post_mount_failure')),receipt
   if receipt:
    assert app._initial_screen_pushed and app._ui_ready
    assert receipt.installation_id==app.client_id
 asyncio.run(mount())
assert not blocked_attempts(),blocked_attempts()
"""

_LAUNCH = r"""
 import subprocess
 from tldw_chatbook.Backup_Recovery import isolated_restore
 original_call=subprocess.call
 def headless(argv,**kwargs):
  # Only replace the terminal driver; execute actual paired CLI selection and app.
  with (base/'open-child.log').open('w') as output:
   result=subprocess.run([sys.executable,'-c',CHILD,*argv[4:],sys.argv[2]],
       **kwargs,stdout=output,stderr=output,text=True,timeout=45)
  assert result.returncode==0,(base/'open-child.log').read_text()[-7000:]
  if sys.argv[2]=='tampered':
   import json
   from tldw_chatbook.Backup_Recovery.profile_open import _expected,_name
   from tldw_chatbook.Backup_Recovery.activation import ActivationStore
   attempt=argv[argv.index('--recovery-launch-attempt')+1]
   receipt=_expected(profile,control,attempt)
   path=ActivationStore(control/'activation')._generation(receipt.generation)/_name(attempt)
   raw=json.loads(path.read_text());raw['installation_id']='foreign'
   path.write_text(json.dumps(raw))
  return 4 if sys.argv[2]=='nonzero' else result.returncode
 subprocess.call=headless
 try:
  operation=service.start_open_profile(profile)
  state=service.wait(operation,timeout=50)
  if sys.argv[2] in ('tampered','nonzero'):
   assert state['state']=='failed',dict(state)
   assert not state['result'].get('opened_successfully',False),dict(state)
  else:
   assert state['state']=='succeeded',dict(state)
   assert state['result'].get('opened_successfully')==(sys.argv[2]=='mounted'),dict(state)
   assert state['result']['exit_code']==0
   assert state['result']['needs_setup']
 finally:subprocess.call=original_call
"""


@pytest.mark.parametrize(
    "mode",
    [
        "exit_only",
        "mounted",
        "read_failure",
        "database_failure",
        "post_mount_failure",
        "tampered",
        "nonzero",
    ],
)
def test_profile_open_requires_actual_mounted_local_reads(tmp_path, mode):
    script = (
        "CHILD="
        + repr(_CHILD)
        + "\n"
        + _SERVICE_ISOLATED.replace(
            'users_name="original"\\n',
            'users_name="original"\\ndefault_tab="settings"\\n[first_run]\\nsetup_completed=true\\n[splash_screen]\\nenabled=false\\n',
        ).replace(
            " assert service.profiles()[0]['status']=='restoration_validated'",
            " assert service.profiles()[0]['status']=='restoration_validated'"
            + _LAUNCH,
        )
    )
    _run(tmp_path, "isolated", mode, script=script, timeout=70)


def test_mounted_open_reads_actual_restored_notes_chat_and_media(tmp_path):
    from Tests.Backup_Recovery.test_isolated_restore import _DATA_RESTORE, _SEED

    child = _CHILD.replace(
        "  if mode=='read_failure':",
        """  import json
  identities=json.loads((Path.home()/'seed'/'identities.json').read_text())
  assert app.chachanotes_db.get_note_by_id(identities['note'])['content']=='Exact saved note content'
  assert app.chachanotes_db.get_message_by_id(identities['message'])['content']=='Exact saved chat content'
  assert app.media_db.get_media_by_id(identities['media'])['content']=='Exact saved media content'
  if mode=='read_failure':""",
    )
    script = (
        "SEED="
        + repr(_SEED)
        + "\nCHILD="
        + repr(child)
        + "\n"
        + _DATA_RESTORE.replace(
            'users_name="original"\\n',
            'users_name="original"\\ndefault_tab="settings"\\n[first_run]\\nsetup_completed=true\\n[splash_screen]\\nenabled=false\\n',
        )
    )
    script += (
        "\nfrom tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService\nservice=RecoveryService(control)\ntry:\n"
        + _LAUNCH
        + "\nfinally:service.close()\n"
    )
    _run(tmp_path, "data", "mounted", script=script, timeout=70)
