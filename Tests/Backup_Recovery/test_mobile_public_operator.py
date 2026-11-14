"""Ordinary native operator paths preceding a fresh mobile review."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_START_CANONICAL = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.real_profile_guard import install as install_profile_guard,take_violations
install_profile_guard()
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Utils.launch_options import adopt_config_flag
adopt_config_flag(sys.argv[1:])
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path,legacy_mobile_dir,canonical_console_paths
from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
selector=effective_config_path()
values=read_profile_config(selector)
source=legacy_mobile_dir(values)
targets=canonical_console_paths(values)
assert not source.exists()
assert not any(path.exists() for path in targets)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
opened=[]
for owner in (CharactersRAGDB,WorkspaceDB,AgentRunsDB):
 original=owner.__init__
 def observed(self,*args,_original=original,_name=owner.__name__,**kwargs):
  import inspect
  bound=inspect.signature(_original).bind(self,*args,**kwargs)
  actual=Path(bound.arguments['db_path'])
  assert source not in actual.parents,(str(actual),str(source))
  opened.append((_name,str(actual)))
  return _original(self,*args,**kwargs)
 owner.__init__=observed
from tldw_chatbook.app import TldwCli
async def main():
 app=TldwCli()
 try:
  assert app.notes_service is not None
  assert app.chachanotes_db.db_path==targets[0]
  assert targets[0].is_file() and targets[1].is_file()
  assert not source.exists()
  assert not blocked_attempts(),blocked_attempts()
  assert not take_violations()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
  await app.tts_service.wait_closed()
 Path(sys.argv[-1]).write_text(json.dumps({'selector':str(selector),'source':str(source),'targets':[str(path) for path in targets],'opened':opened}))
asyncio.run(main())
"""

_FRESH_REVIEW = r"""
import json,sys
from pathlib import Path
from Tests.real_profile_guard import install as install_profile_guard,take_violations
install_profile_guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Utils.launch_options import adopt_config_flag
adopt_config_flag(sys.argv[1:])
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.Backup_Recovery.mobile_consolidation import preview_selected_mobile
selector=effective_config_path()
inventory=preview_capture((selector,),options={'staging_parent':Path.home()})
preview=preview_selected_mobile(selector,selector)
result={'complete':inventory.complete,'issues':inventory.issues,'owners':[(row.owner,row.status) for row in inventory.items if row.status in {'unsupported','unavailable','missing_required'}],'has_data':preview.has_data,'preview_issues':preview.issues,'source_exists':preview.source_dir.exists()}
Path(sys.argv[-1]).write_text(json.dumps(result))
assert inventory.complete,result
assert not preview.has_data and not preview.source_dir.exists(),result
assert not blocked_attempts(),blocked_attempts()
assert not take_violations()
"""


_NO_DATA = (
    _FRESH_REVIEW
    + r"""
import builtins,os,keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
class EmptyIsolatedKeyring:
 def get_password(self,service,username):return None
store=KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
credentials._credential_store=lambda:store
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Backup_Recovery import recovery_service
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation import require_mobile_cutover
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
opened=[]
for owner in (CharactersRAGDB,WorkspaceDB,AgentRunsDB):
 original=owner.__init__
 def observed(self,*args,_original=original,_name=owner.__name__,**kwargs):
  import inspect
  actual=Path(inspect.signature(_original).bind(self,*args,**kwargs).arguments['db_path'])
  assert preview.source_dir not in actual.parents
  opened.append((_name,str(actual)))
  return _original(self,*args,**kwargs)
 owner.__init__=observed
errors=[]
native_issue=recovery_service.issue_code
def observe_issue(error,**kwargs):
 import traceback
 errors.append((type(error).__name__,str(error),traceback.format_tb(error.__traceback__)))
 return native_issue(error,**kwargs)
recovery_service.issue_code=observe_issue
service=RecoveryService(default_control_root())
cli_mode=os.environ.get('TLDW_TEST_MOBILE_CLI')=='1'
if cli_mode:
 from tldw_chatbook.Backup_Recovery import launcher
 launcher._secret=lambda prompt:'test-private-mobile-rollback'
 issued=[]
 original_start=RecoveryService.start_mobile_consolidation
 def actual_start(self,*args,**kwargs):
  operation=original_start(self,*args,**kwargs)
  issued.append((self,operation))
  return operation
 RecoveryService.start_mobile_consolidation=actual_start
try:
 fresh=service.preview_mobile_consolidation(selector,selector)
 assert fresh==preview
 assert not fresh.destination.run_path.exists()
 if cli_mode:
  service.close()
  builtins.input=lambda prompt:'consolidate'
  first_cli_exit=launcher.recovery_main(['consolidate-mobile','--source-config',str(selector),'--target-config',str(selector),'--apply'])
  assert len(issued)==1
  service,operation=issued[0]
 else:operation=service.start_mobile_consolidation(fresh,destination_confirmed=True,rollback_password=b'test-private-mobile-rollback')
 state=service.wait(operation)
 Path(sys.argv[-1]).write_text(json.dumps({'state':state['state'],'issues':state['issues'],'review_issues':state.get('review_issues',()),'worker_review_issues':service.current().get('review_issues',()),'opened':opened,'operation':operation,'errors':errors}))
 first_call={key:state[key] for key in ('state','phase','issues')}
 if state['state']!='succeeded':
  assert state['state']=='recovery_required' and errors[-1][0]=='_TraceReadLimit',(dict(state),errors)
  service.close()
  try:require_mobile_cutover(preview.source_dir,preview.destination)
  except (ValueError,RuntimeError):pass
  else:raise AssertionError('pending cutover accepted')
  service=RecoveryService(default_control_root())
  if cli_mode:
   service.close()
   builtins.input=lambda prompt:'finish'
   assert launcher.recovery_main(['recover',operation,'--finish','--ask-password'])==0
   service=RecoveryService(default_control_root())
   state=service.status(operation)
  else:
   recovery=service.start_recovery(operation,action='finish',rollback_password=b'test-private-mobile-rollback')
   state=service.wait(recovery)
  assert state['state']=='succeeded',(dict(state),errors)
 assert service.pending_operations()==()
 journal=_journal(service.control_root,operation)
 with journal._locked(exclusive=False) as parent:records=journal._records(parent)
 terminal=records[-1]
 assert terminal.event=='mobile_committed' and terminal.evidence['kind']=='no_data'
 assert terminal.evidence['owners']==[]
 assert [row['purpose'] for row in terminal.evidence['captures']]==['canonical']
 assert [row.evidence['target'] for row in records if row.event=='mobile_path_staged']==[str(fresh.destination.run_path)]
finally:service.close()
assert not preview.source_dir.exists()
for _ in range(2):assert require_mobile_cutover(preview.source_dir,preview.destination) is None
chat=CharactersRAGDB(preview.destination.chat_path,'ordinary-no-data-reader')
try:
 assert chat.get_all_conversation_ids()==[]
 with chat.transaction() as cursor:assert cursor.execute('SELECT count(*) FROM mobile_consolidation_batches').fetchone()[0]==0
finally:chat.close_connection()
assert require_mobile_cutover(preview.source_dir,preview.destination) is None
assert not blocked_attempts(),blocked_attempts()
assert not take_violations()
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import bootstrap
import hashlib
physical=[(path.stat().st_dev,path.stat().st_ino) for path in preview.destination.paths]
root=bootstrap.default_bootstrap_root()
before=bootstrap._binding(selector,bootstrap._records(root)[1],bootstrap._registry(root))
assert config.save_setting_to_cli_config('general','default_theme','textual-light')
after=bootstrap._binding(selector,bootstrap._records(root)[1],bootstrap._registry(root))
assert after==dict(before,fingerprint=hashlib.sha256(selector.read_bytes()).hexdigest())
assert before['fingerprint']!=after['fingerprint']
assert physical==[(path.stat().st_dev,path.stat().st_ino) for path in preview.destination.paths]
assert require_mobile_cutover(preview.source_dir,preview.destination) is None
Path(sys.argv[-1]).write_text(json.dumps({'entrypoint':'cli' if cli_mode else 'service','first_cli_exit':first_cli_exit if cli_mode else None,'first_call':first_call,'first_call_errors':errors,'finished_state':state['state'],'operation':operation,'native_config_saved':True,'opened':opened,'source_absent':not preview.source_dir.exists()}))
assert not blocked_attempts(),blocked_attempts()
assert not take_violations()
"""
)

_AFTER_CONFIG = r"""
import json,sys
from pathlib import Path
from Tests.real_profile_guard import install as install_profile_guard,take_violations
install_profile_guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Utils.launch_options import adopt_config_flag
adopt_config_flag(sys.argv[1:])
from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path,legacy_mobile_dir,canonical_console_paths
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores,require_mobile_cutover
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
selector=effective_config_path();values=read_profile_config(selector)
source=legacy_mobile_dir(values);stores=CanonicalConsoleStores(*canonical_console_paths(values))
assert require_mobile_cutover(source,stores) is None
chat=CharactersRAGDB(stores.chat_path,'ordinary-config-restart')
try:assert chat.get_all_conversation_ids()==[]
finally:chat.close_connection()
assert require_mobile_cutover(source,stores) is None
# A real installed owner write may advance the binding but cannot redirect the
# already completed operation to a different store.
from tldw_chatbook import config
assert config.save_setting_to_cli_config('database','workspaces_db_path',str(stores.workspace_path.with_name('redirected.db')))
try:require_mobile_cutover(source,stores)
except ValueError as error:assert str(error) in {'mobile_cutover_target_changed','mobile_selector_review_changed'}
else:raise AssertionError('native storage redirect accepted')
assert config.save_setting_to_cli_config('database','workspaces_db_path',str(stores.workspace_path))
assert require_mobile_cutover(source,stores) is None
# A genuine publication during the observer must not escape its final proof.
from tldw_chatbook.Backup_Recovery import mobile_consolidation_recovery as observer
original_verify=observer._verify_terminal_owners
published=[]
def concurrent_publication(*args,**kwargs):
 result=original_verify(*args,**kwargs)
 assert config.save_setting_to_cli_config('general','default_theme','textual-dark')
 published.append(True)
 return result
observer._verify_terminal_owners=concurrent_publication
try:
 try:require_mobile_cutover(source,stores)
 except ValueError as error:assert str(error)=='mobile_cutover_changed'
 else:raise AssertionError('concurrent publication accepted')
finally:observer._verify_terminal_owners=original_verify
assert published==[True]
assert require_mobile_cutover(source,stores) is None
# Substituting the private selector parent cannot inherit its old witness.
import os
original_parent=selector.parent.with_name('retained-original-selector')
substituted_parent=selector.parent.with_name('retained-substituted-selector')
raw=selector.read_bytes()
os.rename(selector.parent,original_parent)
selector.parent.mkdir(mode=0o700)
selector.write_bytes(raw);selector.chmod(0o600)
try:
 try:require_mobile_cutover(source,stores)
 except (ValueError,RuntimeError):pass
 else:raise AssertionError('ancestor replacement accepted')
finally:
 os.rename(selector.parent,substituted_parent)
 os.rename(original_parent,selector.parent)
assert require_mobile_cutover(source,stores) is None
# Direct publication without the installed binding writer cannot inherit it.
with selector.open('a') as stream:stream.write('\n# external edit\n')
try:require_mobile_cutover(source,stores)
except ValueError as error:assert str(error)=='mobile_cutover_profile_changed'
else:raise AssertionError('external edit accepted')
assert not source.exists()
assert not blocked_attempts(),blocked_attempts()
assert not take_violations()
Path(sys.argv[-1]).write_text(json.dumps({'restart_gate':True,'ordinary_read':True,'native_redirect_refused':True,'concurrent_publication_refused':True,'ancestor_substitution_refused':True,'external_edit_refused':True}))
"""


@pytest.mark.parametrize("complete_cutover", [False, True, "cli"])
def test_ordinary_canonical_startup_precedes_fresh_no_data_review(
    tmp_path, complete_cutover
):
    home = tmp_path / "home"
    selector = home / "selected" / "config.toml"
    selector.parent.mkdir(parents=True, mode=0o700)
    (home / "data").mkdir(mode=0o700)
    (home / "canonical").mkdir(mode=0o700)
    selector.write_text(
        '[general]\nusers_name="operator"\n[first_run]\nsetup_completed=true\n'
        "[splash_screen]\nenabled=false\n"
        '[tldw_api]\nbase_url=""\napi_url=""\nurl=""\n'
        f'[paths]\ndata_dir="{home / "data"}"\n'
        f'[database]\nchachanotes_db_path="{home / "canonical" / "chat.db"}"\n'
        f'workspaces_db_path="{home / "canonical" / "workspaces.db"}"\n'
    )
    selector.chmod(0o600)
    repository = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_TEST_CONFIG_ROOT=str(home / ".test-config"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(repository),
    )
    environment.pop("TLDW_CONFIG_PATH", None)
    if complete_cutover == "cli":
        environment["TLDW_TEST_MOBILE_CLI"] = "1"
    steps = [("startup", _START_CANONICAL), ("review", _FRESH_REVIEW)]
    if complete_cutover:
        steps.append(("cutover", _NO_DATA))
        steps.append(("config-restart", _AFTER_CONFIG))
    for name, script in steps:
        log = tmp_path / f"native-{name}.log"
        result_path = tmp_path / f"native-{name}.json"
        with log.open("w") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-c",
                    script,
                    "--config",
                    str(selector),
                    str(result_path),
                ],
                cwd=repository,
                env=environment,
                stdout=output,
                stderr=output,
                timeout=120,
                check=False,
            )
        assert result.returncode == 0, log.read_text()[-12000:]
    started = json.loads((tmp_path / "native-startup.json").read_text())
    reviewed = json.loads((tmp_path / "native-review.json").read_text())
    assert started["selector"] == str(selector)
    assert {row[0] for row in started["opened"]} >= {"CharactersRAGDB", "WorkspaceDB"}
    assert reviewed["complete"] and not reviewed["has_data"]
