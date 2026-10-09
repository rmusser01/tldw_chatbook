"""Accepted restored MCP writes retain separate exact canonical custody."""

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP


def test_actual_restored_permission_update_finishes_pause_with_exact_canonical_lease(
    tmp_path,
):
    script = (
        _SETUP
        + r"""
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery.mcp_source_participants import canonical_path
service=plane()
service.approve_recovery_review(service.capture_recovery_review())
permission=service.permission_store
canonical=canonical_path(permission)
assert canonical != permission.path, 'fixture did not exercise restored selection'
permission.set_kill_switch(True)
permission.set_kill_switch(False)
assert permission.path.exists()
original=json.loads
pauses=[]
leases=[]
def loaded(value,*args,**kwargs):
 result=original(value,*args,**kwargs)
 if isinstance(result,dict) and 'kill_switch' in result and not pauses:
  operation=raw._local.operation
  state=raw._states[operation]
  assert state.mcp_canonical == canonical
  observer=state.mcp_observation_lease
  assert sum(lease is observer for lease in state.leases) == 1
  members=[lease for lease in state.leases if lease is not observer]
  assert len(members) == 1
  members[0].execution_context(state.selected)
  observer.execution_context(canonical)
  for lease,path in ((members[0],canonical),(observer,state.selected)):
   try:lease.execution_context(path)
   except bootstrap.RecoveryRequired:pass
   else:raise AssertionError('canonical and recovered selections were conflated')
  leases.append(observer)
  pauses.append(storage._begin_local_pause())
 return result
json.loads=loaded
try:
 permission.set_kill_switch(True)
 assert pauses and leases
 assert original(permission.path.read_text())['kill_switch'] is True
 assert all((user/name).read_bytes()==value for name,value in history.items())
finally:
 json.loads=original
 for pause in pauses:pause.resume()
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "retained-restored", script=script, timeout=90)
