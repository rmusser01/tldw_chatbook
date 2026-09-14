"""Only an actual mounted restored app can report a successful profile open."""

from textwrap import indent

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
import asyncio,os,sys
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
  before_mount=Path(os.environ["TLDW_CONFIG_PATH"]).read_bytes()
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
   if mode in ("console_quit","console_edit"):
    assert Path(os.environ["TLDW_CONFIG_PATH"]).read_bytes()==before_mount,"Console mount rewrote the recovered config"
   receipt=opened_receipt(profile,control,attempt)
   if mode in ('read_failure','database_failure'):assert mode in reached,reached
   assert (receipt is not None)==(mode not in ('read_failure','database_failure','post_mount_failure')),receipt
   if receipt:
    assert app._initial_screen_pushed and app._ui_ready
    assert receipt.installation_id==app.client_id
   if mode in ('ordinary_quit','changed_config_quit','console_quit','console_edit'):
    from tldw_chatbook import config
    selected=config.get_cli_config_path()
    if mode=='console_edit':
     import tomllib
     from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
     console=next(screen for screen in app.screen_stack if isinstance(screen,ChatScreen))
     console._set_console_rail_preference(left_open=False)
     async with asyncio.timeout(10):
      while tomllib.loads(selected.read_text()).get('console',{}).get('rail_state',{}).get('console_rail_state:global:shared-layout-v1',{}).get('left_open') is not False:
       await asyncio.sleep(.03)
     assert selected.read_bytes()!=before_mount,'Explicit rail edit was not persisted'
    if mode=='changed_config_quit':
     assert config.save_setting_to_cli_config('general','default_theme','textual-light')
    before=selected.read_bytes()
    await app._confirm_and_quit()
    assert selected.read_bytes()==before,'Shutdown rewrote the recovered config'
 asyncio.run(mount())
assert not blocked_attempts(),blocked_attempts()
"""

_LAUNCH = r"""
 import subprocess
 from Tests.Backup_Recovery.thread_diagnostics import _error_metadata,_write
 from tldw_chatbook.Backup_Recovery import isolated_restore
 original_call=subprocess.call
 def headless(argv,**kwargs):
  if sys.argv[2]=='spawn_failure':raise OSError('fixture spawn unavailable')
  # Only replace the terminal driver; execute actual paired CLI selection and app.
  with (base/'open-child.log').open('w') as output:
   try:
    result=subprocess.run([sys.executable,'-c',CHILD,*argv[4:],sys.argv[2]],
        **kwargs,stdout=output,stderr=output,text=True,timeout=45)
   except BaseException as error:
    try:_write(base/'open-launch-error.json.log',[_error_metadata(error)])
    except OSError:pass  # Diagnostic I/O must not replace the launch exception.
    raise
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
  if sys.argv[2] in ('tampered','nonzero','spawn_failure'):
   assert state['state']=='failed',dict(state)
   assert not state['result'].get('opened_successfully',False),dict(state)
  else:
   assert state['state']=='succeeded',dict(state)
   assert state['result'].get('opened_successfully')==(sys.argv[2] in ('mounted','ordinary_quit','console_quit','console_edit','changed_config_quit')),dict(state)
   assert state['result']['exit_code']==0
   assert state['result']['needs_setup']
 finally:subprocess.call=original_call
"""


def _observed_child(script):
    """Add content-free timing/error observations to the unchanged child driver."""
    prefix = r'''
import threading,time
from pathlib import Path
from Tests.Backup_Recovery.thread_diagnostics import observe_threads,_error_metadata,_write
_diagnostic_root=Path.home()
_diagnostic_started=time.monotonic()
_diagnostic_records=[]
_diagnostic_lock=threading.Lock()
_restore_local_reads=None
_child_failed=False
def _emit(name,records):
 try:_write(_diagnostic_root/name,records)
 except OSError:pass  # Preserve the observed outcome if diagnostics cannot write.
def _phase(name):
 with _diagnostic_lock:
  _diagnostic_records.append({'phase':name,'elapsed_seconds':round(time.monotonic()-_diagnostic_started,6)})
  del _diagnostic_records[:-32]
  _emit('open-child-phases.json.log',_diagnostic_records)
_stop_stacks=observe_threads(_diagnostic_root/'open-child-stacks.log',interval=10)
'''
    observed_reads = r'''
 from tldw_chatbook.Backup_Recovery import profile_open as _profile_open
 _original_local_reads=_profile_open._check_local_content
 _restore_local_reads=(_profile_open,_original_local_reads)
 def _observed_local_reads(*args,**kwargs):
  _phase('local_reads_and_receipt_begin')
  try:
   result=_original_local_reads(*args,**kwargs)
  except BaseException as error:
   _emit('open-local-read-error.json.log',[_error_metadata(error)])
   _phase('local_reads_and_receipt_failed')
   raise
  _phase('local_reads_and_receipt_complete')
  return result
 _profile_open._check_local_content=_observed_local_reads
'''
    observations = (
        ("from tldw_chatbook.cli import main_cli_runner", "_phase('cli_import_begin')\nfrom tldw_chatbook.cli import main_cli_runner\n_phase('cli_import_complete')"),
        ("if mode!='exit_only':", "_phase('cli_selection_complete')\nif mode!='exit_only':"),
        (" from tldw_chatbook.app import TldwCli", " _phase('app_import_begin')\n from tldw_chatbook.app import TldwCli\n _phase('app_import_complete')"),
        (" from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt", " from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt" + observed_reads),
        ("  app=TldwCli()", "  _phase('construct_begin')\n  app=TldwCli()\n  _phase('construct_complete')"),
        ("  async with app.run_test(size=(100,36)) as pilot:", "  _phase('mount_begin')\n  async with app.run_test(size=(100,36)) as pilot:\n   _phase('mount_yield')\n   _phase('recovery_check_wait_begin')"),
        ('   if mode in ("console_quit","console_edit"):', '   _phase("recovery_check_wait_complete")\n   if mode in ("console_quit","console_edit"):'),
        ("   receipt=opened_receipt(profile,control,attempt)", "   receipt=opened_receipt(profile,control,attempt)\n   _phase('receipt_read_complete')"),
        ("    await app._confirm_and_quit()", "    _phase('quit_begin')\n    await app._confirm_and_quit()\n    _phase('quit_complete')"),
        (" asyncio.run(mount())", " asyncio.run(mount())\n _phase('mount_context_complete')"),
    )
    for before, after in observations:
        if script.count(before) != 1:
            raise ValueError("profile_open_diagnostic_boundary_changed")
        script = script.replace(before, after)
    return prefix + "\ntry:\n _phase('child_entry')\n" + indent(script, " ") + r'''
except BaseException as error:
 _child_failed=True
 _emit('open-child-error.json.log',[_error_metadata(error)])
 raise
finally:
 if _restore_local_reads is not None:
  _restore_local_reads[0]._check_local_content=_restore_local_reads[1]
 try:_stop_stacks()
 except (OSError,RuntimeError):
  if not _child_failed:raise
'''


@pytest.mark.parametrize("write_failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_child_diagnostics_preserve_exception_without_values(
    tmp_path, write_failure, cleanup_failure
):
    import json
    import os
    import subprocess  # nosec B404 - fixed local test interpreter.
    import sys
    from pathlib import Path

    private_value = "synthetic-value-must-not-enter-metadata"
    child = _observed_child(
        _CHILD.replace(
            "from Tests.network_guard import install,blocked_attempts",
            f"raise RuntimeError({private_value!r})\nfrom Tests.network_guard import install,blocked_attempts",
        )
    )
    if write_failure or cleanup_failure:
        child = child.replace(
            "_diagnostic_lock=threading.Lock()",
            """_diagnostic_lock=threading.Lock()
_real_write=_write
def _failed_write(*args,**kwargs):raise OSError('synthetic diagnostic I/O failure')
""" + ("_write=_failed_write\n" if write_failure else ""),
        ).replace(
            "_stop_stacks=observe_threads(_diagnostic_root/'open-child-stacks.log',interval=10)",
            """_stop_stacks=observe_threads(_diagnostic_root/'open-child-stacks.log',interval=10)
_real_stop=_stop_stacks
def _checked_stop():
 _real_stop()
 _real_write(_diagnostic_root/'diagnostic-stopped.json',[True])
""" + (" raise RuntimeError('synthetic cleanup failure')\n" if cleanup_failure else "")
            + "_stop_stacks=_checked_stop\n",
        )
    result = subprocess.run(  # nosec B603 - generated local test driver only.
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1 and private_value in result.stderr  # nosec B101
    assert result.stderr.strip().splitlines()[-1] == f"RuntimeError: {private_value}"  # nosec B101
    if write_failure:
        assert json.loads((tmp_path / "diagnostic-stopped.json").read_text()) == [True]  # nosec B101
        return
    metadata = (tmp_path / "open-child-error.json.log").read_text()
    assert private_value not in metadata  # nosec B101
    assert json.loads(metadata)[0]["error_class"] == "RuntimeError"  # nosec B101
    phases = json.loads((tmp_path / "open-child-phases.json.log").read_text())
    assert phases[0]["phase"] == "child_entry"  # nosec B101
    assert phases[0]["elapsed_seconds"] >= 0  # nosec B101


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
        "ordinary_quit",
        "console_quit",
        "console_edit",
        "changed_config_quit",
        "spawn_failure",
    ],
)
def test_profile_open_requires_actual_mounted_local_reads(tmp_path, mode):
    script = (
        "CHILD="
        + repr(_observed_child(_CHILD))
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
    if mode in ("console_quit", "console_edit"):
        script = script.replace('default_tab="settings"', 'default_tab="chat"')
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
        + repr(_observed_child(child))
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


@pytest.mark.parametrize("source_scope", [None, "workspace", "legacy"])
def test_normal_console_seed_still_persists_defaults_or_stored_source(source_scope):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import profile_open
    from tldw_chatbook.Chat.console_rail_state import (
        build_console_rail_preference_key,
        serialize_console_rail_stored_preferences,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert profile_open._selected is None  # nosec B101
    selected = build_console_rail_preference_key(layout_scope="global")
    workspace = build_console_rail_preference_key(
        workspace_id="Research", conversation_id="saved-chat", layout_scope="workspace"
    )
    source = None if source_scope is None else {"left_open": False, "model_open": True}
    records = {}
    if source_scope is not None:
        key = workspace.value if source_scope == "workspace" else workspace.fallback_value
        records[key] = source
    writes = []
    screen = SimpleNamespace(
        _console_rail_state_config=lambda: records,
        _save_console_rail_preferences=lambda key, value, **kwargs: writes.append((key, value)),
    )
    expected = serialize_console_rail_stored_preferences(source)
    seeded = ChatScreen._ensure_console_rail_scope_seed(screen, selected, workspace)
    assert seeded == expected  # nosec B101
    assert writes == [(selected.value, expected)]  # nosec B101
    ChatScreen._ensure_console_rail_scope_seed(screen, selected, workspace)
    assert writes == [(selected.value, expected)]  # nosec B101
    if source_scope is not None:
        assert records[key] is source  # nosec B101
