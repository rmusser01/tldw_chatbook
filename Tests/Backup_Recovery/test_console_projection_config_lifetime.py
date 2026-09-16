"""A Console tick shares checked config reads and retires before yielding."""

import pytest

from Tests.Backup_Recovery.test_console_config_sync_lifetime import _WORKER_RETRY
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_PROJECTION = _WORKER_RETRY.split("async def main():")[0] + r'''
Admission.pause_requested=original_probe
events=[];owners=[];visible=[]
def rail(**kwargs):
 owner=getattr(raw._local,'operation',None)
 value=config.get_cli_setting('general','users_name')
 events.append(('rail',value));owners.append(owner)
 return value
def summary():
 owners.append(getattr(raw._local,'operation',None))
 events.append(('summary',config.get_cli_setting('general','users_name')))
 if case=='config_error':fail_native_config_write()
 if case in {'ui_error','config_error'}:raise fail
 if case=='selector_change':
  os.environ['TLDW_CONFIG_PATH']=str(home/'other.toml')
  config.get_cli_setting('general','users_name')
screen._current_console_rail_state=rail
screen._sync_console_settings_summary=summary
screen._sync_console_rail_visibility_if_changed=visible.append
original_read=screen._build_console_control_state
def control(context):
 events.append(('control',config.get_cli_setting('general','users_name')))
 return original_read(context)
screen._build_console_control_state=control
async def tabs():
 assert getattr(raw._local,'operation',None) is None,'config retained across await'
 assert not storage._raw_operations
 assert all(owner not in raw._states for owner in owners)
 assert config.save_setting_to_cli_config('general','users_name','after session change')
 await asyncio.sleep(0)
screen._sync_console_native_session_tabs=tabs
async def main():
 try:await screen._sync_native_console_chat_ui()
 except ValueError as error:
  assert case in {'ui_error','config_error'} and error is fail
 except bootstrap.RecoveryRequired:
  assert case=='selector_change'
 else:assert case=='current'
 assert owners[0] is not None,'rail reads precede the checked projection lifetime'
 assert owners[1] is owners[0],'summary reacquires config outside the rail lifetime'
 if case=='current':
  assert events==[('rail','fixture'),('summary','fixture'),('control','fixture'),('control','fixture'),('rail','after session change')],events
  assert all(owner is owners[0] for owner in operations[:2])
  assert operations[2:]==[None,None],'mode-bar reads must remain outside the projection lifetime'
  assert owners[-1] is None,'post-await rail must use fresh independent config'
  assert visible==['fixture','after session change']
  assert rendered==[('fixture','fixture')]
 else:
  assert not rendered and not operations and not visible
  assert events==[('rail','fixture'),('summary','fixture')]
 assert not scheduled and not workers and not screen._console_sync_in_progress
 assert getattr(raw._local,'operation',None) is None and not storage._raw_operations
 if case!='selector_change':
  participant=raw._raw_participant(config)
  participant.close_admission()
  try:assert participant.drain(time.monotonic()+1) is (case!='config_error')
  finally:participant.resume()
try:asyncio.run(main())
finally:
 os.environ['TLDW_CONFIG_PATH']=str(selected)
 storage._shutdown()
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["current", "ui_error", "config_error", "selector_change"])
def test_console_projection_retires_config_before_async_session_change(tmp_path, case):
    _run(tmp_path, case, "config-sync", script=_PROJECTION, timeout=40)


_PAUSED_PROJECTION = _WORKER_RETRY.replace(
    "async def main():",
    r'''
projection_reads=[]
def rail(**kwargs):
 projection_reads.append('rail')
 return config.get_cli_setting('general','users_name')
def summary():
 projection_reads.append('summary')
 config.get_cli_setting('general','users_name')
screen._current_console_rail_state=rail
screen._sync_console_settings_summary=summary
async def main():''',
).replace(
    "assert native_requested and not rendered and not operations",
    "assert native_requested and not rendered and not operations and not projection_reads",
).replace(
    "assert rendered==[(expected,expected)]",
    "assert rendered==[(expected,expected)]\n assert projection_reads==['rail','summary','rail']",
)


def test_native_pause_defers_rail_summary_and_controls_until_fresh_replay(tmp_path):
    _run(tmp_path, "native-worker", "config-sync", script=_PAUSED_PROJECTION, timeout=40)
