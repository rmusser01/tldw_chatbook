"""Restore checks bind briefing rows to actual archived peers and selected roots."""

import pytest

from Tests.Backup_Recovery.test_briefing_audio_restore import (
    _PRIVATE,
    _captured_briefing,
)
from Tests.Backup_Recovery.test_complete_roundtrip import (
    _isolated_environment,
    _run_profile_child,
)

_CHECK = "\n".join(
    (
        _PRIVATE,
        r"""
import shutil,sqlite3,zipfile
from contextlib import closing
from dataclasses import replace
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery import staging
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
from tldw_chatbook.DB.recovery_operations import recovery_adapters,_briefing_relocation_authorizer
from tldw_chatbook.DB.private_sqlite import open_recovery_validation
from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
mode=os.environ['BRIEFING_DEPENDENCY_CASE']
receipt=json.loads((fixture/'restore-input.json').read_text())
seed=json.loads((fixture/'seed.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/(mode+'-acquired'),ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive)
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'recovered-briefing'})
asset=receipt['asset']
if mode=='byte_drift':
 from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
 original=staging.stage_restore
 observed=[]
 def changed(*args,**kwargs):
  stage=original(*args,**kwargs)
  descriptor=json.loads((stage/'candidate.json').read_text())
  record=next(row for row in descriptor['artifacts'] if row['logical_id']==asset)
  path=Path(record['candidate']);info=path.stat();payload=path.read_bytes()
  path.write_bytes(payload[:-1]+bytes([payload[-1]^1]))
  os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
  observed.append(str(path))
  return stage
 staging.stage_restore=changed
 try:
  restore_isolated(archive,plan,fixture/'drift-control',threading.Event())
 except ValueError as error:
  assert observed and str(error)=='candidate_receipt_changed',repr(error)
 else:raise AssertionError('Changed actual staged WAV was published')
 assert all(not path.exists() for _,path in plan.restore)
 assert Path(seed['audio']['file_path']).read_bytes().hex()==seed['hex']
else:
 stage=staging.stage_restore(archive,plan,fixture/(mode+'-stage'),threading.Event())
 descriptor=json.loads((stage/'candidate.json').read_text())
 try:
  items=staging._items(doc,plan)
  item=next(row for row in items.values() if row.owner=='db.subscriptions' and row.metadata.kind=='file')
  adapter=next(row for row in recovery_adapters() if row.owner_id=='db.subscriptions')
  candidates={row['logical_id']:Path(row['candidate']) for row in descriptor['artifacts']}
  topology={row.logical_id:(row.root_id,row.parent_id,row.relative_path,'directory' if row in doc.directories else 'file') for row in (*doc.directories,*doc.files)}
  database=candidates[item.logical_id]
  if mode.startswith('sql_'):
   if mode=='sql_trigger':
    with closing(sqlite3.connect(database)) as connection,connection:
     connection.execute("CREATE TRIGGER adversarial AFTER UPDATE OF file_path ON briefing_audio BEGIN UPDATE briefing_audio SET file_path='injected'; END")
   before=database.read_bytes()
   statements={
    'sql_other_column':"UPDATE briefing_audio SET status='failed'",
    'sql_delete':'DELETE FROM briefing_audio',
    'sql_schema':'CREATE TABLE injected(value TEXT)',
    'sql_attach':"ATTACH DATABASE ':memory:' AS injected",
    'sql_trigger':"UPDATE briefing_audio SET file_path='attempted'",
   }
   with open_recovery_validation(adapter.owner_id,database,writable=True) as connection:
    restrictions=_Restrictions(connection)
    connection.set_authorizer(_briefing_relocation_authorizer(restrictions.authorize))
    try:connection.execute(statements[mode])
    except sqlite3.DatabaseError:pass
    else:raise AssertionError('Unrelated or trigger-owned SQL accepted')
   assert database.read_bytes()==before
  elif mode.startswith('relocate_'):
   payload=next(row.payload for row in doc.files if row.logical_id==item.logical_id)
   with zipfile.ZipFile(archive.path) as stream:database.write_bytes(stream.read(payload))
   before=database.read_bytes();mapping=dict(plan.restore)
   if mode=='relocate_missing':mapping.pop(asset)
   else:item=replace(item,dependencies=tuple(key.replace('profile:'+receipt['source_profile']+':','profile:foreign:') if key==asset else key for key in item.dependencies))
   with _preview_reads():
    try:adapter.relocate_restore(item,database,mapping)
    except ValueError as error:assert str(error)=='briefing_audio_mapping_required',repr(error)
    else:raise AssertionError('Unbound relocation accepted')
   assert database.read_bytes()==before
  else:
   if mode=='missing':candidates.pop(asset)
   elif mode=='foreign':
    other=asset.replace('profile:'+receipt['source_profile']+':','profile:foreign:')
    item=replace(item,dependencies=tuple(other if key==asset else key for key in item.dependencies))
    candidates[other]=candidates.pop(asset);topology[other]=topology.pop(asset)
   elif mode=='ambiguous':
    other=asset+'-other'
    item=replace(item,dependencies=item.dependencies+(other,))
    candidates[other]=candidates[asset];topology[other]=topology[asset]
   elif mode=='nonregular':
    candidates[asset].unlink();candidates[asset].symlink_to(Path(seed['audio']['file_path']))
   elif mode in {'wrong_root','unsafe','changed_reference'}:
    expected=dict(plan.restore)[asset]
    value=(fixture/'unrelated'/expected.name if mode=='wrong_root' else expected.parent/'..'/expected.name if mode=='unsafe' else expected.parent/'other.wav')
    with open_recovery_validation(adapter.owner_id,database,writable=True) as connection,connection:
     restrictions=_Restrictions(connection)
     connection.set_authorizer(_briefing_relocation_authorizer(restrictions.authorize))
     connection.execute('BEGIN IMMEDIATE')
     connection.execute('UPDATE briefing_audio SET file_path=? WHERE id=?',(str(value),seed['audio']['id']))
   with _preview_reads():issues=adapter.validate_restore_dependencies(item,database,candidates,topology=topology)
   assert issues==('dependency_unavailable',),(mode,issues)
 finally:
  for name in descriptor['private_roots']:
   path=Path(name);assert fixture in path.parents;shutil.rmtree(path)
  shutil.rmtree(stage)
assert not blocked_attempts(),blocked_attempts()
print('BRIEFING_DEPENDENCY_REFUSED',mode)
""",
    )
)


@pytest.fixture(scope="module")
def captured_briefing(tmp_path_factory):
    root, environment = _captured_briefing(
        tmp_path_factory.mktemp("briefing-dependencies")
    )
    return root, _isolated_environment(root, environment)


@pytest.mark.parametrize(
    "mode",
    [
        "relocate_missing",
        "relocate_foreign",
        "missing",
        "foreign",
        "ambiguous",
        "wrong_root",
        "unsafe",
        "changed_reference",
        "nonregular",
        "sql_other_column",
        "sql_delete",
        "sql_schema",
        "sql_attach",
        "sql_trigger",
        "byte_drift",
    ],
)
def test_native_briefing_dependency_refusal(captured_briefing, mode):
    root, environment = captured_briefing
    _run_profile_child(
        root, mode, _CHECK, dict(environment, BRIEFING_DEPENDENCY_CASE=mode)
    )
