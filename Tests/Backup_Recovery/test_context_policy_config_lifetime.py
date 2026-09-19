"""Contiguous context settings reads retain one checked config operation."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_POLICY = _SCRIPT.split("assert config.get_cli_setting")[0] + r'''
import sys,time
from tldw_chatbook.Backup_Recovery import config_participants,raw_participants as raw,storage_admission as storage
from tldw_chatbook.Chat import console_chat_controller as controller
case=sys.argv[1]
assert config.save_setting_to_cli_config('console','conversation_budget_tokens',1234)
read=controller.ConsoleChatController._global_context_policy_overrides
getter=controller.get_cli_setting
parser=controller.context_policy_overrides_from_console_config
owners=[];keys=[];pauses=[]
failure=ValueError('synthetic parser failure')
def observed(section,key,default=None):
 value=getter(section,key,default)
 owners.append(raw._runtime_operation());keys.append(key)
 if len(owners)==1:
  if case=='pause_inside':
   pauses.append(storage._begin_local_pause())
   assert not pauses[-1].drain(time.monotonic())
  elif case=='selector':os.environ['TLDW_CONFIG_PATH']=str(home/'other.toml')
 return value
def parsed(values):
 assert raw._runtime_operation() is None,'parser retained config admission'
 if case=='parser_error':raise failure
 return parser(values)
controller.get_cli_setting=observed
controller.context_policy_overrides_from_console_config=parsed
if case=='pause_before':pauses.append(storage._begin_local_pause())
try:
 if case in ('pause_before','selector'):
  try:read(None)
  except bootstrap.RecoveryRequired as error:
   if case=='pause_before':assert error.args==('storage_locally_paused',)
  else:raise AssertionError('changed selector or closed admission must refuse')
 elif case=='parser_error':
  before=config._CONFIG_PERSISTENCE_ERROR
  try:read(None)
  except ValueError as error:assert error is failure
  else:raise AssertionError('parser error lost')
  assert config._CONFIG_PERSISTENCE_ERROR is before
 else:
  result=read(None)
  assert result.custom_budget_tokens==1234
  assert len(owners)==9 and owners[0] is not None
  assert all(owner is owners[0] for owner in owners)
  assert keys==['conversation_budget_mode','conversation_budget_tokens','compaction_mode','compaction_representation','compaction_trigger_ratio','compaction_target_ratio','compaction_summary_max_tokens','compaction_failure_behavior','compaction_carry_forward_mode']
  if case=='pause_inside':assert pauses[-1].drain(time.monotonic()+1)
finally:
 for pause in reversed(pauses):pause.resume()
 os.environ['TLDW_CONFIG_PATH']=str(selected)
 controller.context_policy_overrides_from_console_config=parser
assert not storage._raw_operations and raw._runtime_operation() is None
assert all(owner not in raw._states for owner in owners if owner is not None)
if case!='selector':
 previous=owners[0] if owners else None
 case='current';owners.clear();keys.clear()
 assert config.save_setting_to_cli_config('console','conversation_budget_tokens',4321)
 assert read(None).custom_budget_tokens==4321
 assert owners[0] is not previous
 assert not storage._raw_operations and raw._runtime_operation() is None
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "case", ["current", "pause_before", "pause_inside", "selector", "parser_error"]
)
def test_context_policy_reads_retire_one_current_native_config_operation(tmp_path, case):
    _run(tmp_path, case, "context-policy", script=_POLICY, timeout=30)


_ENCLOSING = _SCRIPT.split('assert config.get_cli_setting')[0] + r'''
import sys,time
from contextlib import nullcontext
from tldw_chatbook.Backup_Recovery import config_participants,raw_participants as raw,storage_admission as storage
from tldw_chatbook.Chat import console_chat_controller as controller
from tldw_chatbook.Utils import private_paths
case=sys.argv[1]
assert config.save_setting_to_cli_config('console','conversation_budget_tokens',1234)
before=selected.read_bytes()
read=controller.ConsoleChatController._global_context_policy_overrides
getter=controller.get_cli_setting;parser=controller.context_policy_overrides_from_console_config
owners=[];parsed=[];outside_raw=None;outside_core=None
error=ValueError('synthetic parser failure')
def fail_write():
 original=private_paths.os.write
 def reject(fd,payload):
  if raw._runtime_operation() is not None:raise OSError('synthetic native write failure')
  return original(fd,payload)
 private_paths.os.write=reject
 try:assert not config.save_setting_to_cli_config('console','conversation_budget_tokens',9999)
 finally:private_paths.os.write=original
 assert selected.read_bytes()==before
 assert config._CONFIG_PERSISTENCE_ERROR is not None
if case=='preexisting_failure':fail_write()
initial_error=config._CONFIG_PERSISTENCE_ERROR
def observed(section,key,default=None):
 value=getter(section,key,default)
 owner=raw._runtime_operation();assert owner is not None
 owners.append(owner)
 assert storage._operation_local.operation is None
 if case=='nested_failure' and len(owners)==1:fail_write()
 return value
def parsed_policy(values):
 assert raw._runtime_operation() is outside_raw
 assert getattr(storage._operation_local,'operation',None) is outside_core
 parsed.append(True)
 if case.endswith('_parser'):raise error
 return parser(values)
controller.get_cli_setting=observed;controller.context_policy_overrides_from_console_config=parsed_policy
db=None
if case.startswith('core_'):
 from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
 db=CharactersRAGDB(data/'fixture'/'core.db',client_id='fixture')
 db.add_note('Preserved core','Native readback')
 scope=storage._repository_operation(db._maintenance_participant)
elif case.startswith('raw_'):scope=config_participants.operation(config)
else:scope=nullcontext()
try:
 with scope:
  outside_raw=raw._runtime_operation()
  outside_core=getattr(storage._operation_local,'operation',None)
  try:result=read(None)
  except ValueError as caught:assert case.endswith('_parser') and caught is error
  else:assert not case.endswith('_parser') and result.custom_budget_tokens==1234
  assert len(owners)==9 and all(owner is owners[0] for owner in owners)
  assert parsed==[True]
  assert raw._runtime_operation() is outside_raw
  assert getattr(storage._operation_local,'operation',None) is outside_core
  if outside_core is not None:
   storage._check_operation(outside_core,outside_core.path)
   assert db.get_note_by_title('Preserved core')['content']=='Native readback'
  if outside_raw is not None:assert owners[0] is outside_raw
 assert raw._runtime_operation() is None
 assert getattr(storage._operation_local,'operation',None) is None
 assert not storage._raw_operations and not storage._pending_acquisitions
 assert all(owner not in raw._states for owner in owners)
 assert selected.read_bytes()==before
 if case in ('nested_failure','preexisting_failure'):
  assert config._CONFIG_PERSISTENCE_ERROR is not None
 else:assert config._CONFIG_PERSISTENCE_ERROR is initial_error
 participant=raw._raw_participant(config)
 participant.close_admission()
 try:assert participant.drain(time.monotonic()+1) is (case not in ('nested_failure','preexisting_failure'))
 finally:participant.resume()
finally:
 if db is not None:db.close_connection()
 storage._shutdown()
print('retired and reopened')
'''

@pytest.mark.parametrize('case', ['nested_failure','preexisting_failure','raw_success','raw_parser','core_success','core_parser'])
def test_context_policy_preserves_enclosing_owners_and_native_failures(tmp_path,case):
    _run(tmp_path,case,'policy-review',script=_ENCLOSING,timeout=30)
