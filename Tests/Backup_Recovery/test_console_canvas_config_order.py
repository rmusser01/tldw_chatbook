"""Native config reads must not wait while holding the Canvas publication lock."""

import pytest

from Tests.Backup_Recovery.test_console_config_sync_lifetime import _SYNC
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_BASE = (
    _SYNC.split("if case=='pause_before':")[0]
    + r"""
import asyncio,threading
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
runtime=ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
runtime.ensure_chat_store()
resolver=lambda requested:SimpleNamespace()
runtime.bind_canvas_native_view(scope_resolver=resolver)
assert runtime._canvas_enabled_reader is config.get_canvas_execution_enabled
"""
)

_CONCURRENT = (
    _BASE
    + r"""
original_lock_getter=config._config_file_lock
entered=threading.Event();finished=threading.Event()
results=[];errors=[]
def observed_lock_getter():
 if threading.current_thread().name.startswith('native-authority'):entered.set()
 return original_lock_getter()
config._config_file_lock=observed_lock_getter
def ensure():
 try:
  if case=='materialize':result=runtime._materialize_canvas_native_authority()
  else:result=runtime.ensure_canvas_native_authority(scope_resolver=resolver)
  results.append(result)
 except BaseException as error:errors.append(error)
workers=[threading.Thread(target=ensure,name=f'native-authority-{i}',daemon=True) for i in range(2)]
def contend(context):
 assert raw._runtime_operation() is not None
 for worker in workers:worker.start()
 assert entered.wait(3),'native reader did not enter'
 runtime.bind_canvas_native_view(scope_resolver=resolver)
 return ('fixture','fixture')
screen._build_console_control_state=contend
def watchdog():
 if not finished.wait(8):
  print('native_config_canvas_lock_cycle',flush=True)
  os._exit(9)
threading.Thread(target=watchdog,daemon=True).start()
try:
 screen._sync_console_control_bar()
 for worker in workers:worker.join(3)
 assert all(not worker.is_alive() for worker in workers)
 assert not errors,errors
 assert len(results)==2 and results[0] is results[1] is runtime._canvas_native_authority
 assert results[0] is not None and rendered==[('fixture','fixture')]
 assert runtime.canvas_controller._settlement_listeners==[runtime._canvas_settlement_listener]
 assert results[0]._events=={} and results[0]._selection=={}
 assert not storage._raw_operations and raw._runtime_operation() is None
finally:
 finished.set()
 config._config_file_lock=original_lock_getter
 asyncio.run(runtime.dispose())
print('retired and reopened')
"""
)


@pytest.mark.parametrize("case", ["ensure", "materialize"])
def test_native_control_refresh_and_canvas_authority_do_not_deadlock(tmp_path, case):
    _run(tmp_path, case, "config-canvas-order", script=_CONCURRENT, timeout=20)


_LIFETIME = (
    _BASE
    + r"""
method,outcome=case.split(':')
read_finished=threading.Event();continue_call=threading.Event();results=[];errors=[]
original_enabled=runtime._canvas_enabled
def observed_enabled():
 result=original_enabled()
 if threading.current_thread().name=='canvas-caller':
  read_finished.set()
  assert continue_call.wait(5)
 return result
runtime._canvas_enabled=observed_enabled
original_binding=runtime._canvas_native_view_binding
def invoke():
 try:
  if method=='bind':result=runtime.bind_canvas_native_view(scope_resolver=resolver)
  else:result=runtime._materialize_canvas_native_authority()
  results.append(result)
 except BaseException as error:errors.append(error)
worker=threading.Thread(target=invoke,name='canvas-caller',daemon=True)
try:
 worker.start()
 assert read_finished.wait(3),'policy read did not complete'
 if outcome=='latch':runtime.latch_canvas_disabled()
 else:asyncio.run(runtime.dispose())
 continue_call.set()
 worker.join(3)
 assert not worker.is_alive() and not errors
 assert results==[None] and runtime._canvas_native_authority is None
 assert runtime._canvas_native_view_binding is (original_binding if outcome=='latch' else None)
finally:
 continue_call.set()
 worker.join(3)
 runtime._canvas_enabled=original_enabled
 asyncio.run(runtime.dispose())
assert not storage._raw_operations
print('retired and reopened')
"""
)


@pytest.mark.parametrize("method", ["bind", "materialize"])
@pytest.mark.parametrize("outcome", ["latch", "dispose"])
def test_completed_canvas_lifetime_change_wins_after_policy_read(
    tmp_path, method, outcome
):
    _run(
        tmp_path, f"{method}:{outcome}", "canvas-lifetime", script=_LIFETIME, timeout=20
    )


_STALE = (
    _BASE
    + r"""
from tldw_chatbook.Canvas.models import CanvasScope
scope=CanvasScope(session_id='temporary',conversation_id='temporary',active_message_ids=('user-1','assistant-1'),selected_canvas_id=None,selected_revision_id=None,run_id='run-1')
controller=runtime.canvas_controller
controller.activate_session(scope.session_id)
opened=[]
runtime.bind_canvas_native_view(scope_resolver=lambda requested:scope,auto_open=lambda *args:opened.append(args))
run=controller.register_run(scope,assistant_message_id='assistant-1',temporary=True)
run.create_canvas(scope,tool_call_id='create-1',title='Preserved',html='<!doctype html><p>preserved</p>')
settlement=run.finish_assistant_run('assistant-1',actual_run_id=scope.run_id,terminal_status='done')
assert settlement is not None
original_enabled=runtime._canvas_enabled
read_finished=threading.Event();results=[];errors=[]
def observed_enabled():
 result=original_enabled()
 if threading.current_thread().name=='canvas-caller':read_finished.set()
 return result
runtime._canvas_enabled=observed_enabled
def materialize():
 try:results.append(runtime._materialize_canvas_native_authority())
 except BaseException as error:errors.append(error)
worker=threading.Thread(target=materialize,name='canvas-caller',daemon=True)
try:
 with runtime._canvas_native_lock:
  worker.start()
  assert read_finished.wait(3)
  assert config.save_setting_to_cli_config('canvas','enabled',False)
  assert not runtime._canvas_disabled_latched
 worker.join(3)
 assert not worker.is_alive() and not errors
 authority=results[0]
 assert authority is not None and authority._events=={} and opened==[]
 controller.add_settlement_listener(authority.on_settlement_publication)
 assert controller.confirm_exact_settlement(settlement) is True
 assert authority._events=={} and opened==[]
 assert runtime._canvas_disabled_latched
 assert controller.promotion_contribution(scope.session_id).revision_count==1
finally:
 worker.join(3)
 runtime._canvas_enabled=original_enabled
 asyncio.run(runtime.dispose())
assert not storage._raw_operations
print('retired and reopened')
"""
)


def test_stale_true_policy_allows_only_inert_authority_and_refuses_live_effects(
    tmp_path,
):
    _run(tmp_path, "current", "canvas-stale-policy", script=_STALE, timeout=20)


_REFUSAL = (
    _BASE
    + r"""
method,outcome=case.split(':')
calls=[]
original_binding=runtime._canvas_native_view_binding
pause=None
def reader():
 calls.append(True)
 if outcome=='error':raise OSError('synthetic policy read failure')
 if outcome=='base_error':raise KeyboardInterrupt('synthetic cancellation')
 return False
if outcome=='native_pause':pause=storage._begin_local_pause()
else:runtime._canvas_enabled_reader=reader
try:
 try:
  if method=='bind':result=runtime.bind_canvas_native_view(scope_resolver=resolver)
  else:result=runtime._materialize_canvas_native_authority()
 except KeyboardInterrupt:
  assert outcome=='base_error' and not runtime._canvas_disabled_latched
 else:
  assert result is None
  assert runtime._canvas_disabled_latched is (outcome!='native_pause')
 assert runtime._canvas_native_authority is None and runtime._canvas_native_view_binding is original_binding
 assert calls==([] if outcome=='native_pause' else [True])
finally:
 if pause is not None:pause.resume()
 runtime._canvas_enabled_reader=config.get_canvas_execution_enabled
 asyncio.run(runtime.dispose())
assert not storage._raw_operations
print('retired and reopened')
"""
)


@pytest.mark.parametrize("method", ["bind", "materialize"])
@pytest.mark.parametrize("outcome", ["false", "error", "base_error", "native_pause"])
def test_canvas_policy_refusal_keeps_existing_latch_and_error_semantics(
    tmp_path, method, outcome
):
    _run(
        tmp_path,
        f"{method}:{outcome}",
        "canvas-policy-refusal",
        script=_REFUSAL,
        timeout=20,
    )
