"""Unverified core reads never invent required optional-owner dependencies."""

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
import threading,tomllib,json
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY,DiscoveryContext,storage_logical_id
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.DB.recovery_core import core_adapters,_CoreAdapter
from tldw_chatbook.Notes.recovery import recovery_adapters as notes_adapters
from tldw_chatbook.Persona_Visual.recovery import recovery_adapters as asset_adapters
core=core_adapters()[0]
db=config.get_chachanotes_db_lazy()
db.add_note('Before probe','Fixture only')
assert not core.validate(db.db_path)
document=tomllib.loads(selected.read_text())
document[DISCOVERY_CONTEXT_KEY]=DiscoveryContext(selected,'native-probe')
with storage._preview_reads():before=core.discover(document)[0]
validations=[]
original_validate=_CoreAdapter.validate
def observed_validate(self,path):
 result=original_validate(self,path)
 validations.append(result)
 return result
_CoreAdapter.validate=observed_validate
write=threading.Event();done=threading.Event();errors=[]
def writer():
 try:
  assert write.wait(5)
  db.add_note('Concurrent ordinary note','Fixture only')
 except BaseException as error:errors.append(type(error).__name__)
 finally:
  db.close_connection()
  done.set()
thread=threading.Thread(target=writer)
original=storage._PreviewScope._source_state
calls=[]
def interleaved(source):
 if source==db.db_path:
  calls.append(True)
  if len(calls)==2:
   write.set();assert done.wait(5)
 return original(source)
thread.start()
storage._PreviewScope._source_state=staticmethod(interleaved)
try:
 with storage._preview_reads():
  after=core.discover(document)[0]
  notes=notes_adapters()[0].discover(document)[0]
  assets=[adapter.discover(document)[0] for adapter in asset_adapters()[:2]]
  stable=core.discover(document)[0]
finally:
 storage._PreviewScope._source_state=staticmethod(original)
 _CoreAdapter.validate=original_validate
 write.set();thread.join(5)
 db.close_connection()
assert not thread.is_alive() and not errors,errors
summary={'core_status':after.status,'target_statuses':{item.owner:item.status for item in (notes,*assets)},'validation_issues':[list(value) for value in validations]}
(home/'dependency-discovery.json.log').write_text(json.dumps(summary,sort_keys=True))
assert validations[0]==('core_validation_unavailable',)
assert all(value==() for value in validations[1:])
assert after.status=='unavailable','failed core observation remained included'
assert after.dependencies==(storage_logical_id(document[DISCOVERY_CONTEXT_KEY],'config'),)
assert (after.owner,after.logical_id,after.path)==(before.owner,before.logical_id,before.path)
assert not classify_entries((after,notes,*assets)).complete
assert all(item.status=='unused' for item in (notes,*assets))
assert stable==before
with storage._preview_reads():fresh=core.discover(document)[0]
assert fresh==before,'fresh successful observation changed ordinary-note scope'
print('retired and reopened')
"""
)


def test_native_commit_during_first_snapshot_blocks_unknown_core_dependencies(tmp_path):
    _run(tmp_path, "native", "dependency-review", script=_SCRIPT, timeout=20)


_VALIDATION_BODY = r"""
import sys
case=sys.argv[1]
try:
 if case=='ordinary_note':
  db.add_note('Another ordinary note','No new optional owner')
 elif case=='invalid_schema':
  with db.transaction() as connection:connection.execute('CREATE TABLE unrecognized_fixture_table(value TEXT)')
 else:
  note=db.add_note('File note','Required external file')
  with db.transaction() as connection:
   connection.execute('UPDATE notes SET file_path_on_disk=?,version=version+1 WHERE id=?',(str(home/'missing.md'),note))
 with storage._preview_reads():
  observed=core.discover(document)[0]
  if case=='ordinary_note':assert observed==before
  elif case=='invalid_schema':
   assert core.validate(db.db_path)==('unsupported_schema',)
   assert observed.status=='unavailable'
   assert observed.dependencies==(storage_logical_id(document[DISCOVERY_CONTEXT_KEY],'config'),)
   assert not classify_entries((observed,)).complete
  else:
   assert core.validate(db.db_path)==()
   dependency=storage_logical_id(document[DISCOVERY_CONTEXT_KEY],'notes.file_notes')
   assert observed.status=='included' and dependency in observed.dependencies
   notes=notes_adapters()[0].discover(document)[0]
   assert notes.logical_id==dependency and notes.status=='missing_required'
   assert core.validate_dependencies(observed,db.db_path,{})==('dependency_unavailable',)
   assert not classify_entries((observed,notes)).complete
   assert not (home/'missing.md').exists()
finally:db.close_connection()
print('retired and reopened')
"""
_VALIDATION = _SCRIPT.split("validations=[]")[0] + _VALIDATION_BODY


@pytest.mark.parametrize("case", ["ordinary_note", "invalid_schema", "required_file"])
def test_native_core_discovery_keeps_real_scope_and_validation(tmp_path, case):
    _run(tmp_path, case, "core-discovery", script=_VALIDATION, timeout=20)
