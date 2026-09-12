"""Actual terminal F9 backup, inspected isolated restore and fresh profile open."""

import errno
import json
import os
import select

# Fixed Python argv, private fixtures, no shell.
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - shared native package fixture
)
from Tests.ProductionApp.test_backup_restore_composition import _ENTRY


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The private child guards network before importing the production app."""


_OPEN = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(sys.argv[-1]);args=sys.argv[1:-1]
expected=json.loads((fixture/'ui-installed.json').read_text())
profile=args[args.index('--recovery-profile')+1]
control=Path(args[args.index('--recovery-control-root')+1])
attempt=args[args.index('--recovery-launch-attempt')+1]
sys.argv=['tldw-chatbook',*args,'--help']
from tldw_chatbook.cli import main_cli_runner
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0),error
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt
from tldw_chatbook.config import get_cli_config_path
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==Path(expected['package'])/'tldw_chatbook'/'__init__.py'
assert opened_receipt(profile,control,attempt) is None
async def main():
 app=TldwCli()
 async with app.run_test(size=(100,36)):
  async with asyncio.timeout(20):
   while not getattr(app,'_recovery_open_checked',False):await asyncio.sleep(.03)
  assert opened_receipt(profile,control,attempt) is not None
  assert app._initial_screen_pushed and app._ui_ready
  assert str(get_cli_config_path())==expected['config']
  assert str(app.chachanotes_db.db_path)==expected['core']
  assert app.chachanotes_db.get_note_by_id(expected['note'])['content']=='Captured through F9'
  assert app.chachanotes_db.get_note_by_id(expected['after']) is None
  assert 'synthetic-f9-secret' not in get_cli_config_path().read_text()
  assert not blocked_attempts(),blocked_attempts()
 (fixture/'ui-opened.json').write_text(json.dumps({'profile':profile,'attempt':attempt,'opened':True}))
asyncio.run(main())
"""


_FLOW = (
    _ENTRY.split("async def main():", 1)[0].replace(
        "selector.chmod(0o600)",
        "with selector.open('a') as file:"
        "file.write('[api_settings.openai]\\napi_key=\"synthetic-f9-secret\"\\n')\n"
        "selector.chmod(0o600)",
    )
    + r"""
import faulthandler,hashlib,json,subprocess,threading,tomllib,zipfile
from textual.widgets import Input,Button,Static,Checkbox
from tldw_chatbook.Backup_Recovery import archive_reader,storage_admission
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
fixture=Path.home().parent
package=Path(os.environ['TLDW_F9_PACKAGE'])
test_root=Path(os.environ['TLDW_F9_TEST_ROOT'])
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve()==package/'tldw_chatbook'/'__init__.py'
encrypted=os.environ['TLDW_F9_TEST_ENCRYPTED']=='1'
credentials=os.environ['TLDW_F9_TEST_CREDENTIALS']=='1'
assert not credentials or encrypted
password=b'private F9 archive passphrase' if encrypted else None
credential_observations=[]
credential_shapes=[]
credential_read_errors=[]
if credentials:
 from collections import Counter
 from tldw_chatbook.Backup_Recovery import credentials as credential_owner
 process_credentials=credential_owner.process_credentials
 def observed_credentials(stage,inventory,**options):
  issues=process_credentials(stage,inventory,**options)
  if options['mode']=='include':
   try:material=json.loads(credential_owner._read(stage/'credential-recovery.json'))
   except Exception as error:
    credential_read_errors.append({'type':type(error).__name__,'issues':list(issues)})
    raise
   records=material['records']
   credential_shapes.append({'issues':list(issues),'records':[{key:value for key,value in row.items() if key!='value'} for row in records]})
   assert material['version']==1 and material['mode']=='include'
   assert len(records)==17 and len({row['id'] for row in records})==17
   assert all(row['status']=='unreadable' and row['remappable'] is False and 'value' not in row for row in records)
   assert Counter(row['kind'] for row in records)=={'citation':5,'generation':8,'server':4}
   owners={item.path.relative_to(stage).as_posix():item.owner for item in inventory.items if item.path is not None}
   citations=[row for row in records if row['kind']=='citation']
   assert {owners[row['file']] for row in citations}=={'db.chachanotes.primary','chat.attachments','study.local','quiz.local','notes.sync_bindings'}
   assert all(row['service']=='tldw_chatbook.citation-provenance.v1' for row in citations)
   generation=[row for row in records if row['kind']=='generation']
   assert {(row['service'],row['username']) for row in generation}=={('tldw_chatbook_imagegen',name) for name in ('swarmui','openrouter','novita','together','modelstudio','fal','gemini')}|{('tldw_chatbook_videogen','minimax')}
   servers=[row for row in records if row['kind']=='server']
   assert {row['purpose'] for row in servers}=={'api_key','bearer_token','access_token','refresh_token'}
   assert {row['server_id'] for row in servers}=={'http://127.0.0.1:8000'} and all('binding' not in row for row in servers)
   assert all(owners[row['file']]=='config' for row in generation)
   assert all(owners[row['file']]=='mcp.targets' for row in servers)
   assert set(issues)=={'credential_unreadable:'+row['id'] for row in records} and len(issues)==17
   credential_observations.append({'issues':list(issues),'records':records})
  return issues
 credential_owner.process_credentials=observed_credentials
diagnostics=(fixture/'ui-stacks.log').open('w')
faulthandler.dump_traceback_later(100,file=diagnostics)
async def main():
 app=TldwCli()
 app.app_config['_first_run']=False
 app.app_config.setdefault('first_run',{})['setup_completed']=True
 note=app.chachanotes_db.add_note('UI captured note','Captured through F9')
 destination=Path.home()/('ui.tldw-backup.zip.age' if encrypted else 'ui.tldw-backup.zip')
 restored=fixture/'restored';restored.mkdir(mode=0o700)
 async with app.run_test(headless=False,size=(120,42)) as pilot:
  async def press(query):
   button=app.screen.query_one(query,Button)
   async with asyncio.timeout(2):
    while button.has_class('-active'):await asyncio.sleep(.02)
   assert not button.disabled,query
   button.scroll_visible(immediate=True);button.focus()
   await pilot.press('enter')
  async def ready(predicate,timeout=30):
   try:
    async with asyncio.timeout(timeout):
     while not predicate():await asyncio.sleep(.03)
   except TimeoutError:
    (fixture/'ui-timeout.json').write_text(json.dumps({'screen':type(app.screen).__name__,'mode':getattr(app.screen,'_mode',None),'revision':getattr(app.screen,'_revision',None)}))
    faulthandler.dump_traceback(file=diagnostics)
    raise
  async def finished(kind,previous=None,timeout=65):
   service=app.recovery_service
   await ready(lambda:service.current() is not None and service.current()['kind']==kind and service.current()['operation_id']!=previous)
   operation=service.current()['operation_id']
   state=await asyncio.to_thread(service.wait,operation,timeout=timeout)
   assert state['state']=='succeeded',dict(state)
   return operation,state
  # The synthetic configured key triggers the ordinary startup consent dialog.
  # Explicitly decline it before recording the unchanged backup source baseline.
  await ready(lambda:bool(app.screen.query('#model-catalog-consent-deny')))
  await press('#model-catalog-consent-deny')
  await ready(lambda:not app.screen.query('#model-catalog-consent-deny'))
  await ready(lambda:tomllib.loads(selector.read_text()).get('model_catalog',{}).get('refresh_consent_recorded') is True)
  assert tomllib.loads(selector.read_text())['model_catalog']['auto_refresh_enabled'] is False
  original_config=selector.read_bytes()
  await pilot.press('f9')
  await ready(lambda:isinstance(app.screen,SettingsScreen))
  await press('#settings-backup-restore')
  await ready(lambda:isinstance(app.screen,BackupRestoreScreen))
  screen=app.screen
  await press('#backup-open-create')
  screen.query_one('#backup-destination',Input).value=str(destination)
  if encrypted:
   screen.query_one('#backup-encrypted',Checkbox).focus();await pilot.press('space')
   assert screen.query_one('#backup-encrypted',Checkbox).value
   screen.query_one('#backup-password',Input).value=password.decode()
   screen.query_one('#backup-password-confirm',Input).value=password.decode()
  if credentials:
   screen.query_one('#backup-credentials',Checkbox).focus();await pilot.press('space')
   assert screen.query_one('#backup-credentials',Checkbox).value
  await press('#backup-review')
  await ready(lambda:not screen.query_one('#backup-create',Button).disabled)
  assert 'Complete coverage' in str(screen.query_one('#backup-coverage',Static).render())
  await press('#backup-create')
  refused=None
  if credentials:
   await ready(lambda:app.recovery_service.current() is not None and app.recovery_service.current()['kind']=='backup')
   refused=app.recovery_service.current()['operation_id']
   rejection=await asyncio.to_thread(app.recovery_service.wait,refused,timeout=65)
   (fixture/'ui-credential-observed-shape.json').write_text(json.dumps({'captures':credential_shapes,'read_errors':credential_read_errors},indent=2))
   assert rejection['state']=='failed' and rejection['issues']==('review_required',),dict(rejection)
   assert len(credential_observations)==1
   reviewed=set(credential_observations[0]['issues'])
   assert set(rejection['review_issues'])==reviewed
   assert not destination.exists() and selector.read_bytes()==original_config
   await ready(lambda:len(screen.query('.backup-acknowledge-credential'))==17)
   boxes=list(screen.query('.backup-acknowledge-credential'))
   assert {box.name for box in boxes}==reviewed and all(not box.value for box in boxes)
   for box in boxes:
    box.scroll_visible(immediate=True);box.focus();await pilot.press('space')
    assert box.value
   screen.query_one('#backup-password',Input).value=password.decode()
   screen.query_one('#backup-password-confirm',Input).value=password.decode()
   await press('#backup-review')
   await ready(lambda:not screen.query_one('#backup-create',Button).disabled)
   assert set(screen._reviewed[2]['acknowledged_credential_issues'])==reviewed
   await press('#backup-create')
  created,state=await finished('backup',refused)
  assert state['result']['archive_verified'] and state['result']['complete'] is (not credentials),dict(state)
  if credentials:
   assert len(credential_observations)==2 and set(credential_observations[1]['issues'])==reviewed
   (fixture/'ui-credential-observations.json').write_text(json.dumps(credential_observations,indent=2))
  assert not state['result'].get('restoration_validated',False)
  assert not screen.query_one('#backup-password',Input).value
  assert not screen.query_one('#backup-password-confirm',Input).value
  with destination.open('rb') as file:assert file.read(24).startswith(b'age-encryption.org/') is encrypted
  assert storage_admission._pause is None
  after=app.chachanotes_db.add_note('After UI capture','Resumed native writer')
  acquired=await asyncio.to_thread(archive_reader.acquire,destination,fixture/'readback',ArchiveLimits(),password,threading.Event())
  assert (acquired.encrypted_source is not None) is encrypted
  doc=archive_reader.verify_sealed(acquired)
  assert doc.consistency==('partial' if credentials else 'coherent') and len(doc.profile_ids)==1
  if credentials:assert reviewed.issubset(doc.report.lines)
  assert doc.credential_policy==('include' if credentials else 'exclude')
  assert selector.read_bytes()==original_config
  with zipfile.ZipFile(acquired.path) as archive:
   for row in doc.files:
    if row.owner_id=='config':assert (b'synthetic-f9-secret' in archive.read(row.payload)) is credentials
    if row.owner_id=='config.history' and not credentials:assert b'synthetic-f9-secret' not in archive.read(row.payload)
  await press('#backup-open-inspect')
  screen.query_one('#backup-source',Input).value=str(destination)
  if encrypted:screen.query_one('#backup-inspect-password',Input).value=password.decode()
  await press('#backup-inspect')
  await ready(lambda:screen.query_one('#backup-restore-form').display)
  assert screen._inspection_summary['archive_verified']
  assert not screen.query_one('#backup-inspect-password',Input).value
  roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
  producers={row.logical_id:row for row in doc.producer_inventory}
  ordinary={'db.chachanotes.primary','chat.attachments','notes.sync_bindings','quiz.local','study.local','db.media.primary','research.local','db.prompts.primary','chatbooks.registry','db.evals','db.library_collections','db.library_ingest_jobs','db.scheduled_tasks','db.subscriptions','db.workspaces','kanban.local','mcp.targets','notifications.client','runtime.event_state','runtime.sync_state','writing.local'}
  trees={'chat.dictionaries':'chat_dicts','chatbooks.archives':'chatbooks','rag.definitions':'rag_profiles'}
  assert not any(tuple(slot.get('owners',()))==('recovery.credentials',) for slot in screen._inspection_summary['destination_slots'])
  for index,slot in enumerate(screen._inspection_summary['destination_slots']):
   key=slot['logical_id']
   if slot['kind']=='data_root':target=restored/'data'
   else:
    root=roots[key];owner=producers[key].owner_id
    if root.synthetic:
     if owner in {'config','config.history','runtime.source_state','ui.state','ui.emoji_recents'}:target=restored/'config'
     elif owner=='eval.definitions':target=restored/'inactive-eval'
     else:
      assert owner in ordinary,(key,owner)
      target=restored/'data'/'recovered-ui'
    elif owner=='persona.visual_identity_builtin':target=restored/'inactive-builtin'
    else:
     assert owner in trees,(key,owner)
     target=restored/'data'/'recovered-ui'/trees[owner]
   screen.query_one(f'#backup-root-{index}',Input).value=str(target)
  screen.query_one('#backup-profile-name-0',Input).value='recovered-ui'
  await press('#backup-review-restore')
  await ready(lambda:not screen.query_one('#backup-start-restore',Button).disabled)
  plan=screen._restore_plan
  await press('#backup-start-restore')
  restored_operation,state=await finished('restore',created)
  assert state['result']['restoration_validated'] and not state['result'].get('opened',False),dict(state)
  if credentials:
   retained=list(app.recovery_service.control_root.glob('isolated-*/credentials.age'))
   assert len(retained)==1
   assert hashlib.sha256(retained[0].read_bytes()).digest()==hashlib.sha256(destination.read_bytes()).digest()
  config_path,data_path=ProfileCatalog(app.recovery_service.control_root).resolve(state['result']['profile_id'])
  core=next(row for row in doc.files if row.owner_id=='db.chachanotes.primary')
  installed={'config':str(config_path),'core':str(dict(plan.restore)[core.logical_id]),'note':note,'after':after,'package':str(package),'credentials':credentials}
  (fixture/'ui-installed.json').write_text(json.dumps(installed))
  assert config_path==restored/'config'/'config.toml' and data_path==restored/'data'
  original_call=subprocess.call
  def child(argv,**kwargs):
   assert argv[:4]==[sys.executable,'-P','-m','tldw_chatbook']
   assert kwargs['env']['PYTHONPATH']==str(package)
   reader='import sys;sys.path.append('+repr(str(test_root))+')\n'+OPEN_CHILD
   with (fixture/'ui-child.log').open('w') as output:
    result=subprocess.run([sys.executable,'-c',reader,*argv[4:],str(fixture)],**kwargs,stdout=output,stderr=subprocess.STDOUT,text=True,timeout=45)
   assert result.returncode==0,(fixture/'ui-child.log').read_text()[-10000:]
   return result.returncode
  subprocess.call=child
  try:
   (fixture/'ui-phase.txt').write_text('opening profiles list')
   await press('#backup-open-profiles')
   await ready(lambda:len(screen.query('.backup-open-profile'))==1)
   (fixture/'ui-phase.txt').write_text('opening fresh profile')
   assert app._driver.can_suspend and not app.is_headless
   await press('.backup-open-profile')
   opened_operation,state=await finished('open_profile',restored_operation,timeout=55)
   assert state['result']['opened_successfully'] and state['result']['needs_setup'],dict(state)
   assert json.loads((fixture/'ui-opened.json').read_text())['profile']==state['result']['profile_id']
  finally:subprocess.call=original_call
  assert selector.read_bytes()==original_config
  assert app.chachanotes_db.get_note_by_id(after)['content']=='Resumed native writer'
  assert app.chachanotes_db.get_note_by_id(note)['content']=='Captured through F9'
  assert not blocked_attempts(),blocked_attempts()
  (fixture/'ui-result.json').write_text(json.dumps({'backup':created,'restore':restored_operation,'open':opened_operation,'archive_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'source_preserved':True,'opened':True,'encrypted':encrypted,'credentials':credentials}))
 assert app.recovery_service._closed
asyncio.run(main())
faulthandler.cancel_dump_traceback_later()
diagnostics.close()
"""
)


_WINDOWS_CONSOLE = r"""
import ctypes
import json
import runpy
import sys
import traceback
from pathlib import Path

receipt = Path(sys.argv[2])
state = {"status": "starting", "handles": {}}
def record():
    payload = json.dumps(state)
    if len(payload) > 32000:
        state["traceback"] = state["traceback"][-4000:]
        payload = json.dumps(state)
    receipt.write_text(payload, encoding="utf-8")
record()
try:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [ctypes.c_uint32]
    kernel.GetStdHandle.restype = ctypes.c_void_p
    kernel.GetFileType.argtypes = [ctypes.c_void_p]
    kernel.GetFileType.restype = ctypes.c_uint32
    kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetConsoleMode.restype = ctypes.c_int
    for name, identifier, stream in (("stdin", -10, sys.stdin), ("stdout", -11, sys.stdout)):
        handle = kernel.GetStdHandle(identifier)
        mode = ctypes.c_uint32()
        kind = kernel.GetFileType(handle)
        console = bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)))
        state["handles"][name] = {
            "file_type": kind, "console_mode_available": console,
            "console_mode": mode.value, "isatty": stream.isatty(),
        }
        record()
        if kind != 2 or not console or not stream.isatty():
            raise RuntimeError("native Windows console preflight failed: " + name)
    state["status"] = "console_verified"
    record()
    runpy.run_path(sys.argv[1], run_name="__main__")
    state["status"] = "completed"
    record()
except BaseException:
    state["status"] = "failed"
    state["traceback"] = traceback.format_exc()[-24000:]
    record()
    raise
"""


def _run_windows_console(
    tmp_path: Path, script: str, environment: dict[str, str]
) -> None:
    """Run the unchanged product flow with real, unredirected Windows console I/O."""
    child_script = tmp_path / "ui-console-flow.py"
    wrapper = tmp_path / "ui-console-launcher.py"
    receipt = tmp_path / "ui-console.log"
    for path, contents in ((child_script, script), (wrapper, _WINDOWS_CONSOLE)):
        with path.open("x", encoding="utf-8") as output:
            output.write(contents)
    # CREATE_NEW_CONSOLE supplies new standard handles; redirecting any of them
    # would prevent the production WindowsDriver from using its real console.
    process = subprocess.Popen(  # nosec B603
        [sys.executable, str(wrapper), str(child_script), str(receipt)],
        cwd=tmp_path,
        env=environment,
        close_fds=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    try:
        try:
            returncode = process.wait(timeout=180)
        except subprocess.TimeoutExpired as error:
            diagnostic = (
                receipt.read_text(encoding="utf-8")
                if receipt.exists()
                else "No console receipt"
            )
            raise AssertionError(
                "Native UI child timed out: " + diagnostic[-32000:]
            ) from error
        diagnostic = (
            receipt.read_text(encoding="utf-8")
            if receipt.exists()
            else "No console receipt"
        )
        assert returncode == 0, diagnostic[-32000:]
        assert json.loads(diagnostic)["status"] == "completed", diagnostic[-32000:]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux", "win32"}, reason="Native terminal workflow"
)
@pytest.mark.parametrize(
    "encrypted,credentials",
    [(False, False), (True, False), (True, True)],
    ids=["plain", "encrypted", "encrypted_credentials"],
)
def test_f9_created_archive_restores_and_opens_through_actual_controls(
    tmp_path, encrypted, credentials, native_package
):
    """Keep the real terminal suspend path while driving finite native fixtures."""
    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        TERM="xterm-256color",
        TLDW_F9_TEST_ENCRYPTED="1" if encrypted else "0",
        TLDW_F9_TEST_CREDENTIALS="1" if credentials else "0",
        TLDW_F9_PACKAGE=str(native_package),
        TLDW_F9_TEST_ROOT=str(Path(__file__).resolve().parents[2]),
        PYTHONPATH=os.pathsep.join(
            (str(native_package), str(Path(__file__).resolve().parents[2]))
        ),
    )
    script = "OPEN_CHILD=" + repr(_OPEN) + "\n" + _FLOW
    if sys.platform == "win32":
        _run_windows_console(tmp_path, script, environment)
    else:
        import pty

        master, slave = pty.openpty()
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        log = tmp_path / "ui-terminal.log"
        deadline = time.monotonic() + 180
        try:
            with log.open("wb") as output:
                while True:
                    assert time.monotonic() < deadline, "Native UI child timed out"
                    readable, _, _ = select.select([master], [], [], 0.1)
                    if readable:
                        try:
                            data = os.read(master, 65536)
                        except OSError as error:
                            if error.errno != errno.EIO:
                                raise
                            break
                        if not data:
                            break
                        output.write(data)
                    elif process.poll() is not None:
                        break
            assert process.wait(timeout=5) == 0, log.read_text(errors="replace")[
                -12000:
            ]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
    result = json.loads((tmp_path / "ui-result.json").read_text())
    assert result["source_preserved"] and result["opened"]
    assert result["encrypted"] is encrypted
    assert result["credentials"] is credentials
