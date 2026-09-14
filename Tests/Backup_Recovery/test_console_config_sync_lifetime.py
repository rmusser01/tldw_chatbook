"""Console control refresh retains one checked configuration lifetime."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SYNC = _SCRIPT.split("assert config.get_cli_setting")[0] + r'''
import sys,time
from types import SimpleNamespace,MethodType
from textual.css.query import NoMatches
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Backup_Recovery import raw_participants as raw,storage_admission as storage
case=sys.argv[1]
operations=[];rendered=[];pauses=[]
fail=ValueError('synthetic refresh failure')
cleanup=RuntimeError('synthetic observer cleanup failure')
def fail_native_config_write():
 from tldw_chatbook.Utils import private_paths
 original_write=private_paths.os.write
 before=selected.read_bytes()
 def failed_write(fd,payload):
  if raw._runtime_operation() is not None:raise OSError('synthetic native write failure')
  return original_write(fd,payload)
 private_paths.os.write=failed_write
 try:assert not config.save_setting_to_cli_config('general','users_name','unsaved')
 finally:private_paths.os.write=original_write
 assert selected.read_bytes()==before
 assert config._CONFIG_PERSISTENCE_ERROR is not None
if case=='preexisting_error':fail_native_config_write()
if case=='cleanup_error':
 from contextlib import contextmanager
 from tldw_chatbook.Backup_Recovery import config_participants
 original_operation=config_participants.operation
 depth=0
 @contextmanager
 def cleanup_failure(*args,**kwargs):
  global depth
  depth+=1
  try:
   with original_operation(*args,**kwargs) as operation:yield operation
   if depth==1 and case=='cleanup_error':raise cleanup
  finally:depth-=1
 config_participants.operation=cleanup_failure
class Surface(SimpleNamespace):
 def __getattr__(self,name):return MethodType(getattr(ChatScreen,name),self)
screen=Surface()
screen._pending_console_launch_context=None
screen._library_activity=SimpleNamespace(sync_projection=lambda:None)
screen._console_auto_speak=SimpleNamespace(sync_controls=lambda:None)
for name in ('_sync_console_pending_delete_confirmation','_sync_console_transcript_guidance','_sync_console_composer_action_state','_sync_console_rail_visibility_if_changed','_sync_console_cost_chip'):
 setattr(screen,name,lambda *a,**k:None)
def missing(*a,**k):raise NoMatches('fixture has no rendered inspector')
screen.query_one=missing
screen._build_console_workbench_state=lambda value:value
screen._push_console_control_state_if_changed=lambda value,other:rendered.append(value)
screen._build_console_inspector_state=lambda context:SimpleNamespace(can_save_chatbook=False)
screen._current_console_rail_state=lambda **kwargs:None

def read(context):
 operation=getattr(raw._local,'operation',None)
 operations.append(operation)
 first=config.get_cli_setting('general','users_name')
 if case=='config_error':fail_native_config_write()
 if case in {'error','config_error','cleanup_error'}:raise fail
 if case=='pause_inside':
  pauses.append(storage._begin_local_pause())
  assert not pauses[-1].drain(time.monotonic())
 if case=='selector_inside':os.environ['TLDW_CONFIG_PATH']=str(home/'other.toml')
 if case=='database':
  assert db.add_note('Independent owner','Native content')
  assert getattr(raw._local,'operation',None) is operation
 second=config.get_cli_setting('general','users_name')
 operations.append(getattr(raw._local,'operation',None))
 return first,second
screen._build_console_control_state=read
if case=='database':
 from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
 db=CharactersRAGDB(data/'fixture'/'notes.db',client_id='fixture')
 connection=db.get_connection()
if case=='pause_before':pauses.append(storage._begin_local_pause())
try:
 if case in {'pause_before','selector_inside'}:
  try:screen._sync_console_control_bar()
  except bootstrap.RecoveryRequired:pass
  else:raise AssertionError('changed admission or selector must refuse')
  assert rendered==[]
 elif case=='pause_inside':
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  assert pauses[-1].drain(time.monotonic()+1)
  try:screen._sync_console_control_bar()
  except bootstrap.RecoveryRequired:pass
  else:raise AssertionError('new refresh must refuse while paused')
 elif case in {'error','config_error','cleanup_error'}:
  failed_case=case
  try:screen._sync_console_control_bar()
  except (ValueError,RuntimeError) as error:
   if case=='cleanup_error':assert error is cleanup and error.__cause__ is fail
   else:assert error is fail
  else:raise AssertionError('original refresh failure must propagate')
  assert operations[0] is not None
  assert operations[0] not in raw._states and not storage._raw_operations
  case='current'
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  participant=raw._raw_participant(config)
  participant.close_admission()
  try:assert participant.drain(time.monotonic()+1) is (failed_case!='config_error'),'UI or native error misclassified for config retirement'
  finally:participant.resume()
 elif case=='preexisting_error':
  existing=config._CONFIG_PERSISTENCE_ERROR
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  assert config._CONFIG_PERSISTENCE_ERROR==existing
  participant=raw._raw_participant(config)
  participant.close_admission()
  try:assert not participant.drain(time.monotonic()+1)
  finally:participant.resume()
 else:
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  assert operations[0] is not None,'control refresh reacquires config for each getter'
  assert all(value is operations[0] for value in operations)
  assert operations[0] not in raw._states
  previous=operations[0]
  assert config.save_setting_to_cli_config('general','users_name','updated')
  operations.clear()
  screen._sync_console_control_bar()
  assert rendered[-1]==('updated','updated')
  assert operations[0] is not None and operations[0] is not previous
  if case=='database':
   assert db.get_connection() is connection
   assert db.get_note_by_title('Independent owner')['content']=='Native content'
finally:
 for pause in reversed(pauses):pause.resume()
 os.environ['TLDW_CONFIG_PATH']=str(selected)
 if case=='database':db.close_connection()
assert not storage._raw_operations
assert getattr(raw._local,'operation',None) is None
assert all(operation not in raw._states for operation in operations)
assert not list(parent.glob('*.tmp'))
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "case", ["current", "unbound", "error", "config_error", "preexisting_error", "cleanup_error", "pause_before", "pause_inside", "selector_inside", "database"]
)
def test_console_control_refresh_keeps_native_config_lifetime(tmp_path, case):
    script = _SYNC
    if case == "unbound":
        script = script.replace(
            "bind_profile(root,selected,('profile',),root/'admission')", ""
        ).replace("assert str(parent) not in before[0]['roots']", "assert before==[]")
    _run(tmp_path, case, "config-sync", script=script, timeout=40)
