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
scheduled=[]
screen.set_timer=lambda delay,callback:scheduled.append((delay,callback))
screen.call_after_refresh=lambda callback:scheduled.append((0,callback))
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
 if case=='pause_before':
  screen._sync_console_control_bar()
  assert rendered==[] and len(scheduled)==1 and scheduled[0][0]>0
 elif case=='selector_inside':
  try:screen._sync_console_control_bar()
  except bootstrap.RecoveryRequired:pass
  else:raise AssertionError('changed admission or selector must refuse')
  assert rendered==[]
 elif case=='pause_inside':
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  assert pauses[-1].drain(time.monotonic()+1)
  screen._sync_console_control_bar()
  assert rendered==[('fixture','fixture')]
  assert len(scheduled)==1 and scheduled[0][0]>0
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


_NATIVE_RETRY = _SYNC.split("if case=='pause_before':")[0] + r'''
import threading
from tldw_chatbook.Backup_Recovery.admission import Admission,AdmissionCancelled
storage.admit_startup()
scheduled=[]
screen.set_timer=lambda delay,callback:scheduled.append((delay,callback))
screen.call_after_refresh=lambda callback:scheduled.append((0,callback))
original_probe=Admission.pause_requested
cancel=threading.Event();finished=threading.Event();native_requested=[];errors=[]
publisher=Admission(authority.control_root)
def publish():
 try:
  with publisher.maintenance(('profile',),10,cancel=cancel):
   raise AssertionError('startup lease must exclude maintenance publication')
 except AdmissionCancelled:pass
 except BaseException as error:errors.append(error)
 finally:finished.set()
thread=threading.Thread(target=publish)
def request_during_entry(self,names):
 if native_requested:return original_probe(self,names)
 thread.start()
 deadline=time.monotonic()+5
 while not original_probe(self,names):
  assert time.monotonic()<deadline,'native maintenance did not close its gate'
  time.sleep(.005)
 native_requested.append(True)
 assert storage._pause is None
 assert not getattr(screen,'_console_sync_maintenance_paused',False)
 return True  # The immediately preceding native gate probe returned True.
Admission.pause_requested=request_during_entry
try:
 screen._sync_console_control_bar('stale rail snapshot')
 assert native_requested and not rendered and not operations
 assert len(scheduled)==1 and scheduled[0][0]>0
 if case=='native-duplicates':
  for _ in range(5):screen._sync_console_control_bar('another stale snapshot')
  assert len(scheduled)==1 and not rendered
 if case=='native-local-pause':
  screen._console_sync_in_progress=False;screen._console_sync_requested=False
  screen._console_sync_maintenance_close_admission()
  pause=storage._begin_local_pause()
  try:
   delay,callback=scheduled.pop();callback()
   assert len(scheduled)==1 and not rendered and not operations
  finally:pause.resume()
  screen._console_sync_maintenance_resume()
 cancel.set();thread.join(5)
 assert not thread.is_alive() and finished.is_set() and not errors
 assert not original_probe(authority,('profile',))
 Admission.pause_requested=original_probe
 assert config.save_setting_to_cli_config('general','users_name','after canceled intent')
 original_render=screen._sync_console_control_bar_under_config
 fresh=[]
 def render_fresh(rail_state=None):
  fresh.append(rail_state)
  return original_render(rail_state)
 screen._sync_console_control_bar_under_config=render_fresh
 delay,callback=scheduled.pop()
 callback()
 assert rendered==[('after canceled intent','after canceled intent')]
 assert fresh==[None]
 assert not scheduled
 assert not storage._raw_operations and getattr(raw._local,'operation',None) is None
finally:
 Admission.pause_requested=original_probe
 cancel.set()
 if thread.ident is not None:thread.join(5)
 storage._shutdown()
assert not errors
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["native-canceled", "native-duplicates", "native-local-pause"])
def test_console_refresh_retries_after_native_pause_without_stale_state(tmp_path, case):
    _run(tmp_path, case, "config-sync", script=_NATIVE_RETRY, timeout=40)


def test_console_body_pause_error_is_not_mistaken_for_entry_deferral(tmp_path):
    script = _SYNC.replace(
        "fail=ValueError('synthetic refresh failure')",
        "fail=bootstrap.RecoveryRequired('storage_locally_paused')",
    ).replace("assert not storage._raw_operations\n", "assert not scheduled\nassert not storage._raw_operations\n")
    _run(tmp_path, "error", "config-sync", script=script, timeout=40)


_SCREEN_RETRY = _NATIVE_RETRY.split("try:\n screen._sync_console_control_bar('stale")[0] + r'''
import asyncio
from textual.app import App
from textual.screen import Screen
class RetryHost(Screen):
 def on_unmount(self):screen._closing=True
async def main():
 app=App()
 async with app.run_test() as pilot:
  host=RetryHost()
  await app.push_screen(host)
  callbacks=[];timers=[]
  def set_timer(delay,callback):
   def run():
    callbacks.append(True)
    callback()
   timer=host.set_timer(delay,run)
   timers.append(timer)
   return timer
  screen.set_timer=set_timer
  screen.call_after_refresh=host.call_after_refresh
  screen._sync_console_control_bar('old state')
  assert native_requested and not rendered
  for _ in range(5):screen._request_console_control_bar_sync()
  assert len(timers)==1 and not callbacks
  if case=='timer-teardown':
   await app.pop_screen()
  cancel.set();thread.join(5)
  assert not thread.is_alive() and not errors
  Admission.pause_requested=original_probe
  assert config.save_setting_to_cli_config('general','users_name','fresh timer state')
  if case=='timer-teardown':
   await asyncio.sleep(.35)
   assert not callbacks and not rendered
   screen._request_console_control_bar_sync(delayed=True)
   assert len(timers)==1
  else:
   async with asyncio.timeout(3):
    while not rendered:await asyncio.sleep(.01)
   assert rendered==[('fresh timer state','fresh timer state')]
   assert len(callbacks)==len(timers)==1
   assert not screen._console_control_bar_sync_scheduled
try:asyncio.run(main())
finally:
 Admission.pause_requested=original_probe
 cancel.set()
 if thread.ident is not None:thread.join(5)
 storage._shutdown()
assert not errors and not storage._raw_operations
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["timer-retry", "timer-teardown"])
def test_console_native_retry_uses_one_screen_owned_timer(tmp_path, case):
    _run(tmp_path, case, "config-sync", script=_SCREEN_RETRY, timeout=40)


_WORKER_RETRY = _NATIVE_RETRY.split("try:\n screen._sync_console_control_bar('stale")[0] + r'''
import asyncio
from contextlib import nullcontext
async def no_async():pass
def no_sync(*args,**kwargs):pass
screen._console_sync_in_progress=False;screen._console_sync_requested=False
screen._console_chat_store=None
screen._message=SimpleNamespace(reconcile_console_speech_context=no_sync)
screen._session=SimpleNamespace(_sync_console_session_draft=no_sync)
screen._retrieval=SimpleNamespace(
 _warm_console_effective_scope_cache_if_stale=no_async,
 _refresh_active_dictionaries_summary_if_scope_changed=no_async,
 _refresh_active_world_books_summary_if_scope_changed=no_async)
screen._character=SimpleNamespace(_refresh_active_character_avatar_if_scope_changed=no_async)
screen._character_context=SimpleNamespace(refresh_if_scope_changed=no_async)
screen._workspace=SimpleNamespace(tick_workspace_build_scope=nullcontext)
for name in ('_record_ui_worker_started','_record_ui_worker_finished','_sync_console_chat_core_state','_sync_console_settings_summary','_sync_console_settings_recovery_surfaces','_sync_console_live_work_readiness_rows','_dispatch_active_console_roleplay_refresh','_sync_console_workspace_context','_dispatch_console_rail_preference_prune'):
 setattr(screen,name,no_sync)
screen._sync_console_native_session_tabs=no_async
screen._sync_native_console_transcript=no_async
screen._console_mode_summary=lambda state:'fixture'
screen._native_run_status_copy=lambda:None
def mode_bar(selector,*args):
 if selector=='#console-mode-bar':return SimpleNamespace(update=no_sync)
 return missing(selector,*args)
screen.query_one=mode_bar
import tldw_chatbook.UI.Screens.chat_screen as module
module.project_instruction_ui.sync_project_instruction_status_for_screen=no_sync
workers=[]
screen.run_worker=lambda coro,**kwargs:workers.append(coro)
async def main():
 screen._console_sync_requested=True
 await screen._sync_native_console_chat_ui()
 assert native_requested and not rendered and not operations
 assert len(scheduled)==1 and not workers
 assert not screen._console_sync_in_progress
 if case=='native-worker-real-pause':
  screen._console_sync_maintenance_close_admission()
  pause=storage._begin_local_pause()
  try:
   delay,callback=scheduled.pop();callback()
   assert len(scheduled)==1 and not workers and not operations
  finally:pause.resume()
  screen._console_sync_maintenance_resume()
  assert not workers
 if case=='native-worker-inprogress':
  screen._console_sync_in_progress=True
  delay,callback=scheduled.pop();callback()
  assert len(scheduled)==1 and not workers and not operations
  screen._console_sync_in_progress=False
 cancel.set();thread.join(5)
 assert not thread.is_alive() and not errors
 Admission.pause_requested=original_probe
 if case=='native-worker-external':
  assert config.save_setting_to_cli_config('general','users_name','new external request')
  await screen._sync_native_console_chat_ui()
  assert not rendered and not workers and len(scheduled)==1
 if case=='native-worker-teardown':
  screen._closing=True
  delay,callback=scheduled.pop();callback()
  assert not scheduled and not workers and not rendered
  assert not screen._console_control_bar_replay_whole_sync
  return
 delay,callback=scheduled.pop();callback()
 assert len(workers)==1
 await workers.pop()
 expected='new external request' if case=='native-worker-external' else 'fixture'
 assert rendered==[(expected,expected)]
 assert not scheduled and not workers and not screen._console_sync_requested
try:asyncio.run(main())
finally:
 Admission.pause_requested=original_probe
 cancel.set()
 if thread.ident is not None:thread.join(5)
 for worker in workers:worker.close()
 storage._shutdown()
assert not errors and not storage._raw_operations
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "case", ["native-worker", "native-worker-real-pause", "native-worker-inprogress", "native-worker-teardown", "native-worker-external"]
)
def test_console_native_worker_defers_later_native_reads_with_its_refresh(tmp_path, case):
    _run(tmp_path, case, "config-sync", script=_WORKER_RETRY, timeout=40)
