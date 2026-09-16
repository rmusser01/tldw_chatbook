"""Native rollback retires only a committed incoming nested Persona pack."""

import sys
import time

import pytest

from Tests.Backup_Recovery.test_config_sibling_capture import _SEED, _child

_SEED_PERSONA = _SEED.replace(
    "print('NATIVE_SEED_COMPLETE',flush=True)",
    """
from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
from tldw_chatbook.Persona_Visual.publication import _cleanup_marker, _repository_asset_row, _validate_snapshot
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Backup_Recovery.native_files import create_private_directory, create_private_file
from tldw_chatbook.Utils.platform_files import os as native_os
from tldw_chatbook.config import get_user_data_dir
from dataclasses import replace
from uuid import uuid4
import hashlib
source=Path.home()/'persona-input';source.mkdir(mode=0o700)
# Seed legitimate existing artwork in the publisher's format. Its publication
# API is POSIX-only; backup must also handle such existing data on Windows.
snapshot=replace(_snapshot(source),persona_id='persona-'+Path.home().name)
manifest,context,assets=_validate_snapshot(snapshot)
profile=get_user_data_dir()
pack=uuid4().hex;version=uuid4().hex
relative=f'persona_visual/packs/{pack}/versions/{version}'
directory=profile
for part in relative.split('/')+['assets']:
 directory=directory/part
 if not directory.exists():create_private_directory(directory)
version_path=profile/relative
identity=native_os.stat(version_path,follow_symlinks=False)
payloads={
 'manifest.json':snapshot.manifest_json.encode('utf-8'),
 '.persona-visual-cleanup':_cleanup_marker(os.urandom(32).hex(),relative,(identity.st_dev,identity.st_ino)),
}
asset_rows=[]
for index,(source_key,metadata) in enumerate(assets):
 leaf=f'assets/{index:03d}.png'
 payload=(source/source_key).read_bytes()
 assert metadata.mime_type=='image/png'
 assert len(payload)==metadata.byte_count and hashlib.sha256(payload).hexdigest()==metadata.sha256
 payloads[leaf]=payload
 asset_rows.append(_repository_asset_row(metadata,storage_relpath=f'{relative}/{leaf}'))
for leaf,payload in payloads.items():
 with create_private_file(version_path/leaf) as descriptor:
  assert native_os.write(descriptor,payload)==len(payload)
 assert (version_path/leaf).read_bytes()==payload
repository=PersonaVisualRepository(app.chachanotes_db)
graph=repository.activate_new_pack(
 persona_id=snapshot.persona_id,title=snapshot.title,description=snapshot.description,
 source_kind=snapshot.source_kind,source_context=context,manifest=manifest,
 manifest_storage_relpath=f'{relative}/manifest.json',assets=asset_rows,
 expected_persona_revision=snapshot.persona_revision,authority_guard=lambda:True,
)
assert repository.get_active_persona_pack(snapshot.persona_id)==graph
assert graph.version.manifest_sha256==hashlib.sha256(payloads['manifest.json']).hexdigest()
app.chachanotes_db.add_note('Original '+Path.home().name,'Before replacement for '+Path.home().name)
print('NATIVE_PERSONA_SEED_COMPLETE',flush=True)
""",
)

_CAPTURE = r"""
import threading
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
storage._shutdown()
home=Path.home();options={'staging_parent':home}
preview=preview_capture((effective_config_path(),),options=options)
assert preview.complete,preview.issues
output=home.parent/'incoming.tldw-backup.zip'
result=capture((effective_config_path(),),preview.scope_digest,output,options=options,cancel=threading.Event())
write_archive(result,output,password=None,cancel=threading.Event())
assert result.inventory.complete
print('NATIVE_PERSONA_CAPTURE_COMPLETE',flush=True)
"""

_REPLACE = r"""
import hashlib,json,threading
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import archive_reader,bootstrap,replacement,service_storage,storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import required_rollback_dependencies
from tldw_chatbook.Backup_Recovery.staging import stage_restore
selector=bootstrap.effective_config_path();home=Path.home()
control=service_storage.default_control_root();work=service_storage.ensure_storage(control)
storage._shutdown()
print('PERSONA_REPLACEMENT_REVIEW',flush=True)
archive=archive_reader.acquire(home.parent/'incoming.tldw-backup.zip',work/'incoming',ArchiveLimits(),None,threading.Event())
document=archive_reader.verify_sealed(archive);profile=document.profile_ids[0]
target=preview_capture((selector,),options={})
assert target.complete,target.issues
original_persona=[row for row in target.items if row.owner=='persona.assets']
original_root=next(row.path for row in original_persona if row.metadata.parent_id is None)
(home/'original-persona.json').write_text(json.dumps({'files':{str(row.path.relative_to(original_root)):hashlib.sha256(row.path.read_bytes()).hexdigest() for row in original_persona if row.status=='included'},'graph':{row.logical_id:list(row.dependencies) for row in original_persona}}))
assert profile!=hashlib.sha256(str(selector).encode()).hexdigest()[:24]
setup=home/'files-needing-setup';setup.mkdir(mode=0o700)
choices=dict(mode='replace',profile_bases={},external_destinations={},profile_names={},target=target,target_configs={profile:selector},setup_parent=setup)
plan=resolve_destinations(archive,**choices)
plan=resolve_destinations(archive,**choices,safety_scope=required_rollback_dependencies(plan))
print('PERSONA_REPLACEMENT_STAGE',flush=True)
candidate=stage_restore(archive,plan,work/'candidate',threading.Event())
try:
 operation=replacement.replace(plan,candidate,control_root=control,rollback_password=b'test-only-persona',cancel=threading.Event())
except replacement.RollbackCredentialReviewRequired as review:
 print('PERSONA_CREDENTIAL_REVIEW_ABORT',flush=True)
 pending=bootstrap._records(bootstrap.default_bootstrap_root())[0]
 assert len(pending)==1
 assert replacement.recover_replacement(pending[0]['operation_id'],control_root=control,action='abort',rollback_password=None,cancel=threading.Event())=='aborted'
 choices['target']=preview_capture((selector,),options={})
 first=resolve_destinations(archive,**choices)
 plan=resolve_destinations(archive,**choices,safety_scope=required_rollback_dependencies(first),acknowledged_credential_issues=review.issues)
 candidate=stage_restore(archive,plan,work/'candidate-reviewed',threading.Event())
 operation=replacement.replace(plan,candidate,control_root=control,rollback_password=b'test-only-persona',cancel=threading.Event())
with Journal(control,operation)._locked(exclusive=False) as parent:rows=Journal(control,operation)._records(parent)
assert rows[-1].event=='committed'
prepared=next(row.evidence for row in rows if row.event=='prepared')
new=[row for row in prepared['artifacts'] if row['action']=='publish' and row['previous'] is None and ':persona.assets:' in row['logical_id']]
assert len(new)==1 and new[0]['candidate']['kind']=='directory'
pack=Path(new[0]['target'])
assert pack.parent.name=='packs' and pack.parent.is_dir()
(home/'operation.json').write_text(json.dumps({'operation':operation,'pack':str(pack)}))
print('NATIVE_NESTED_PERSONA_REPLACEMENT_COMPLETE',flush=True)
"""

_REOPEN = r"""
import asyncio,hashlib,json,sys
from pathlib import Path
from Tests.network_guard import install
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from Tests.Backup_Recovery.thread_diagnostics import observe_notes_write_failure
async def run():
 app=TldwCli()
 from Tests.Backup_Recovery.initial_screen_observation import initial_screen_observation
 async with initial_screen_observation(app), app.run_test(size=(120,40)):
  notes={row['title']:row['content'] for row in app.chachanotes_db.list_notes()}
  assert notes['Original source']=='Before replacement for source' and 'Original target' not in notes
  with observe_notes_write_failure(Path.home()/'persona-note-write.log',app.chachanotes_db):
   app.chachanotes_db.add_note('After nested pack replacement','New current data must enter the later safety copy.')
  pack=Path(json.loads((Path.home()/'operation.json').read_text())['pack'])
  files={path.relative_to(pack).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in pack.rglob('*') if path.is_file()}
  (Path.home()/'current-persona.json').write_text(json.dumps(files))
  app.exit()
asyncio.run(run())
print('NATIVE_PERSONA_ORDINARY_REOPEN_WRITE_COMPLETE',flush=True)
"""

_RESTORED_REOPEN = (
    _REOPEN[: _REOPEN.index("async def run():")]
    + r"""
async def run():
 app=TldwCli()
 from Tests.Backup_Recovery.initial_screen_observation import initial_screen_observation
 async with initial_screen_observation(app), app.run_test(size=(120,40)):
  notes={row['title']:row['content'] for row in app.chachanotes_db.list_notes()}
  assert notes['Original target']=='Before replacement for target'
  assert 'Original source' not in notes and 'After nested pack replacement' not in notes
  app.exit()
asyncio.run(run())
print('NATIVE_PERSONA_ORDINARY_RESTORED_REOPEN_COMPLETE',flush=True)
"""
)

_LATER = r"""
import hashlib,json,threading,zipfile
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap,later_rollback,recovery_copies,service_storage,storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
from tldw_chatbook.Backup_Recovery.journal import Journal
home=Path.home();saved=json.loads((home/'operation.json').read_text());pack=Path(saved['pack'])
control=service_storage.default_control_root();storage._shutdown()
current=preview_capture((bootstrap.effective_config_path(),),options={})
assert current.complete,current.issues
semantic={i.logical_id:i for i in current.items if i.owner=='persona.assets' and i.path in (pack.parent,pack.parent.parent)}
assert len(semantic)==2
core_path=next(row.path for row in current.items if row.owner=='db.chachanotes.primary')
def live_sqlite_state():
 result=[]
 for suffix in ('','-wal','-shm'):
  path=core_path.with_name(core_path.name+suffix)
  if not path.exists():result.append(None);continue
  info=path.stat()
  result.append((info.st_dev,info.st_ino,info.st_size,info.st_mode,info.st_mtime_ns,info.st_ctime_ns,hashlib.sha256(path.read_bytes()).hexdigest()))
 return tuple(result)
real_preview=later_rollback._preview
preview_count=0
def stable_preview(*args,**kwargs):
 global preview_count
 before=live_sqlite_state()
 result=real_preview(*args,**kwargs)
 assert live_sqlite_state()==before,'native review mutated live SQLite source'
 preview_count+=1
 return result
later_rollback._preview=stable_preview
print('PERSONA_LATER_REVIEW',flush=True)
plan=later_rollback.preview_rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',target=current,cancel=threading.Event())
assert later_rollback.preview_rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',target=plan.target,cancel=threading.Event())==plan
assert all(next(i for i in plan.target.items if i.logical_id==key)==item for key,item in semantic.items())
assert not set(semantic).intersection(dict(plan.retire))
assert pack in dict(plan.retire).values()
from tldw_chatbook.Backup_Recovery import replacement
print('PERSONA_LATER_EXECUTE',flush=True)
try:
 new=recovery_copies.rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',new_password=b'test-only-current-persona',cancel=threading.Event(),approved_plan=plan)
except replacement.RollbackCredentialReviewRequired as review:
 print('PERSONA_CREDENTIAL_REVIEW_ABORT',flush=True)
 pending=bootstrap._records(bootstrap.default_bootstrap_root())[0]
 assert len(pending)==1
 assert replacement.recover_replacement(pending[0]['operation_id'],control_root=control,action='abort',rollback_password=None,cancel=threading.Event())=='aborted'
 current=preview_capture((bootstrap.effective_config_path(),),options={})
 plan=later_rollback.preview_rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',target=current,cancel=threading.Event(),acknowledged_credential_issues=review.issues)
 new=recovery_copies.rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',new_password=b'test-only-current-persona',cancel=threading.Event(),approved_plan=plan)
assert not pack.exists()
assert preview_count>=4
later_rollback._preview=real_preview
with Journal(control,new)._locked(exclusive=False) as parent:rows=Journal(control,new)._records(parent)
assert rows[-1].event=='committed'
from tldw_chatbook.Backup_Recovery import archive_reader
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
entry=next(row for row in recovery_copies.list_recovery_copies(control) if row.operation_id==new)
assert entry.status=='verified' and not entry.pending_operation
checked=archive_reader.acquire(entry.path,home/'current-copy-readback',ArchiveLimits(),b'test-only-current-persona',threading.Event())
document=archive_reader.verify_sealed(checked)
expected=json.loads((home/'current-persona.json').read_text())
with zipfile.ZipFile(checked.path) as archive:
 for relative,digest in expected.items():
  matches=[row for row in document.files if row.owner_id=='persona.assets' and row.relative_path.endswith(pack.name+'/'+relative)]
  assert len(matches)==1
  assert hashlib.sha256(archive.read(matches[0].payload)).hexdigest()==digest
 core=next(row for row in document.files if row.owner_id=='db.chachanotes.primary')
 copied_core=home/'verified-current-notes.sqlite3'
 copied_core.write_bytes(archive.read(core.payload))
import sqlite3
with sqlite3.connect(copied_core.as_uri()+'?mode=ro',uri=True) as connection:
 assert connection.execute('SELECT content FROM notes WHERE title=?',('After nested pack replacement',)).fetchall()==[('New current data must enter the later safety copy.',)]
assert pack.parent.is_dir() and pack.parent.parent.is_dir()
original=json.loads((home/'original-persona.json').read_text())
assert original['files']
for relative,digest in original['files'].items():
 assert hashlib.sha256((pack.parent.parent/relative).read_bytes()).hexdigest()==digest
restored=preview_capture((bootstrap.effective_config_path(),),options={})
assert restored.complete,restored.issues
assert {row.logical_id:list(row.dependencies) for row in restored.items if row.owner=='persona.assets'}==original['graph']
print('NATIVE_NESTED_PERSONA_LATER_COMPLETE',flush=True)
"""


def _prepare_native(tmp_path):
    for label in ("source", "target"):
        home = tmp_path / label
        (home / ".config/tldw_cli").mkdir(parents=True, mode=0o700)
        _child(
            home,
            _SEED_PERSONA,
            "seed-persona",
            timeout=480 if sys.platform == "win32" else 70,
        )
    _child(
        tmp_path / "source",
        _CAPTURE,
        "persona-capture",
        timeout=480 if sys.platform == "win32" else 120,
    )
    _child(
        tmp_path / "target",
        _REPLACE,
        "persona-replacement",
        timeout=600 if sys.platform == "win32" else 150,
    )
    _child(
        tmp_path / "target",
        _REOPEN,
        "persona-reopen",
        timeout=300 if sys.platform == "win32" else 90,
    )
    return tmp_path / "target"


# Windows child ceilings reuse existing seed/F9/later/open observations;
# The enclosing bounds exceed all seed/capture/recovery/reopen child ceilings.
@pytest.mark.timeout(4200 if sys.platform == "win32" else 900)
def test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path):
    started = time.monotonic()
    home = _prepare_native(tmp_path)
    later_timeout = 900 if sys.platform == "win32" else 150
    if sys.platform == "linux":
        # Reuse unspent setup allowance within the original 860s child budget;
        # reserve the final 90s reopen and 120s capture observations.
        later_timeout = started + 860 - time.monotonic() - (90 + 120)
        if later_timeout <= 0:
            raise TimeoutError("persona_rollback_test_budget_exhausted")
    output = _child(home, _LATER, "persona-later", timeout=later_timeout)
    assert "NATIVE_NESTED_PERSONA_LATER_COMPLETE" in output
    output = _child(
        home,
        _RESTORED_REOPEN,
        "persona-restored-reopen",
        timeout=300 if sys.platform == "win32" else 90,
    )
    assert "NATIVE_PERSONA_ORDINARY_RESTORED_REOPEN_COMPLETE" in output
    output = _child(
        home,
        _CAPTURE.replace("incoming.tldw-backup.zip", "after-later.tldw-backup.zip"),
        "persona-restored-capture",
        timeout=480 if sys.platform == "win32" else 120,
    )
    assert "NATIVE_PERSONA_CAPTURE_COMPLETE" in output


_NEGATIVE = (
    _LATER[: _LATER.index("plan=later_rollback.preview_rollback")]
    + r"""
import sys
from dataclasses import replace
case=sys.argv[1]
root=pack.parent.parent
pack_item=next(row for row in current.items if row.path==pack)
parent=next(row for row in current.items if row.path==pack.parent)
config=next(row for row in current.items if row.owner=='config')
core=next(row for row in current.items if row.owner=='db.chachanotes.primary')
journal=Journal(control,saved['operation'])
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
from tldw_chatbook.Backup_Recovery.journal import _Prepared
from tldw_chatbook.Backup_Recovery.staging import _items
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
original=load_plan(journal)
with journal._locked(exclusive=False) as descriptor:records=journal._records(descriptor)
prepared=_Prepared.model_validate(next(row.evidence for row in records if row.event=='prepared'))
doc,_=later_rollback._created_manifest(journal,original,prepared,records)
incoming=_items(doc,original)
item=next(row for row in incoming.values() if row.path==pack)
owner=next(row for row in install_adapters() if row.owner_id=='persona.assets')
with storage.acquire_storage(pack):
 key,ancestors=later_rollback._created_persona_subtree(item,incoming,current,config,owner,prepared)
assert key==pack_item.logical_id and {row[1] for row in ancestors}=={root,pack.parent}
# Every malformed reviewed graph starts from a working native relation.
if case.startswith('source-'):
 if case=='source-parent':item=replace(item,metadata=replace(item.metadata,parent_id=config.logical_id))
 elif case=='source-root':item=replace(item,metadata=replace(item.metadata,root_id=config.logical_id))
 elif case=='source-core':
  semantic=incoming[item.metadata.root_id]
  incoming[semantic.logical_id]=replace(semantic,dependencies=tuple(k for k in semantic.dependencies if 'db.chachanotes.primary' not in k))
 else:raise AssertionError(case)
 call=lambda:later_rollback._created_persona_subtree(item,incoming,current,config,owner,prepared)
else:
 if case=='current-parent':changed=replace(pack_item,metadata=replace(pack_item.metadata,parent_id=config.logical_id))
 elif case=='current-root':changed=replace(pack_item,metadata=replace(pack_item.metadata,root_id=config.logical_id))
 elif case=='current-core':changed=replace(core,path=core.path.with_name('foreign.db'))
 elif case=='current-config':changed=replace(config,path=config.path.with_name('other.toml'))
 elif case in ('duplicate-id','foreign-owner'):
  changed=pack_item
 else:raise AssertionError(case)
 if case=='duplicate-id':items=(*current.items,pack_item)
 elif case=='foreign-owner':items=(*current.items,replace(pack_item,logical_id='foreign',owner='foreign'))
 else:items=tuple(changed if row.logical_id==changed.logical_id else row for row in current.items)
 target=replace(current,items=items)
 if case=='current-config':config=changed
 call=lambda:later_rollback._created_persona_subtree(item,incoming,target,config,owner,prepared)
before={str(path):(path.stat().st_ino,hashlib.sha256(path.read_bytes()).hexdigest()) for path in control.rglob('*') if path.is_file()}
try:
 with storage.acquire_storage(pack):call()
except ValueError as error:
 assert str(error) in ('local_snapshot_created_scope_unverified','local_snapshot_created_owner_conflict')
else:raise AssertionError('invalid native topology accepted')
after={str(path):(path.stat().st_ino,hashlib.sha256(path.read_bytes()).hexdigest()) for path in control.rglob('*') if path.is_file()}
assert before==after and not bootstrap._records(bootstrap.default_bootstrap_root())[0]
print('NATIVE_PERSONA_NEGATIVE_COMPLETE',case,flush=True)
"""
)


@pytest.fixture(scope="module")
def native_persona_negatives(tmp_path_factory):
    return _prepare_native(tmp_path_factory.mktemp("persona-negative"))


@pytest.mark.parametrize(
    "case",
    [
        "source-parent",
        "source-root",
        "source-core",
        "current-parent",
        "current-root",
        "current-core",
        "current-config",
        "duplicate-id",
        "foreign-owner",
    ],
)
@pytest.mark.timeout(3600 if sys.platform == "win32" else 600)
def test_native_persona_mapping_refuses_unproved_graph(native_persona_negatives, case):
    output = _child(native_persona_negatives, _NEGATIVE, case, timeout=60)
    assert "NATIVE_PERSONA_NEGATIVE_COMPLETE" in output


_OVERLAP = (
    _NEGATIVE[: _NEGATIVE.index("with storage.acquire_storage(pack):")]
    + r"""
from tldw_chatbook.Utils.platform_files import os
observed={}
real=later_rollback._created_destination_target
def remember(*args,**kwargs):
 result=real(*args,**kwargs);observed['result']=result;return result
later_rollback._created_destination_target=remember
try:
 plan=later_rollback.preview_rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',target=current,cancel=threading.Event())
finally:later_rollback._created_destination_target=real
created,ancestors=observed['result'][1:]
assert later_rollback._known_absences(plan,original,prepared,created,active_ancestors=ancestors)==plan
ancestor_key=next(key for key,path in plan.restore if path==pack.parent)
extra={}
if case=='mapping-alone':ancestors={}
elif case=='ancestor-container':plan=replace(plan,containers=(*plan.containers,(ancestor_key,pack.parent)))
elif case=='ancestor-retire':plan=replace(plan,retire=(*plan.retire,(ancestor_key,pack.parent)))
elif case=='ancestor-not-restored':plan=replace(plan,restore=tuple(pair for pair in plan.restore if pair[1]!=pack.parent))
elif case=='restore-inside':
 child=next(row for row in current.items if row.path is not None and pack in row.path.parents and row.status=='included')
 plan=replace(plan,restore=(*plan.restore,(child.logical_id,child.path)))
elif case in ('candidate-inode','stale-generation'):
 if case=='candidate-inode':
  artifacts=[row.model_copy(update={'candidate':row.candidate.model_copy(update={'inode':row.candidate.inode+1})}) if row.target==str(pack) else row for row in prepared.artifacts]
  damaged=prepared.model_copy(update={'artifacts':artifacts})
 else:damaged=prepared.model_copy(update={'generation':'different-generation'})
 call=lambda:real(journal,original,damaged,records,current)
elif case not in ('ancestor-file','ancestor-missing','ancestor-replaced'):raise AssertionError(case)
before={str(path):(os.stat(path).st_ino,hashlib.sha256(path.read_bytes()).hexdigest()) for path in control.rglob('*') if path.is_file()}
held=home/'held-packs';parent_info=os.stat(pack.parent);root_info=os.stat(root)
try:
 if case.startswith('ancestor-') and case in ('ancestor-file','ancestor-missing','ancestor-replaced'):
  pack.parent.rename(held)
  if case=='ancestor-file':pack.parent.write_bytes(b'not a directory')
  elif case=='ancestor-replaced':
   pack.parent.mkdir(mode=0o700)
   for child in held.iterdir():child.rename(pack.parent/child.name)
   os.utime(pack.parent,ns=(parent_info.st_atime_ns,parent_info.st_mtime_ns))
  os.utime(root,ns=(root_info.st_atime_ns,root_info.st_mtime_ns))
  call=lambda:later_rollback.preview_rollback(saved['operation'],control_root=control,old_password=b'test-only-persona',target=current,cancel=threading.Event())
 elif case not in ('candidate-inode','stale-generation'):
  call=lambda:later_rollback._known_absences(plan,original,prepared,created,active_ancestors=ancestors)
 try:call()
 except bootstrap.RecoveryRequired as error:
  assert case=='ancestor-file' and str(error)=='storage_admission_unavailable'
 except (ValueError,OSError):pass
 else:raise AssertionError('unproved retirement overlap accepted')
finally:
 if held.exists():
  if case=='ancestor-file':pack.parent.unlink()
  elif case=='ancestor-replaced':
   for child in pack.parent.iterdir():child.rename(held/child.name)
   pack.parent.rmdir()
  held.rename(pack.parent)
  os.utime(pack.parent,ns=(parent_info.st_atime_ns,parent_info.st_mtime_ns))
  os.utime(root,ns=(root_info.st_atime_ns,root_info.st_mtime_ns))
after={str(path):(os.stat(path).st_ino,hashlib.sha256(path.read_bytes()).hexdigest()) for path in control.rglob('*') if path.is_file()}
assert before==after and not bootstrap._records(bootstrap.default_bootstrap_root())[0]
assert os.stat(pack.parent).st_ino==parent_info.st_ino
print('NATIVE_PERSONA_OVERLAP_REFUSED',case,flush=True)
"""
)


@pytest.mark.parametrize(
    "case",
    [
        "mapping-alone",
        "ancestor-container",
        "ancestor-retire",
        "ancestor-not-restored",
        "restore-inside",
        "candidate-inode",
        "stale-generation",
        "ancestor-file",
        "ancestor-missing",
        "ancestor-replaced",
    ],
)
@pytest.mark.timeout(3600 if sys.platform == "win32" else 600)
def test_native_persona_retirement_keeps_ancestor_exclusion(
    native_persona_negatives, case
):
    output = _child(
        native_persona_negatives,
        _OVERLAP,
        case,
        timeout=180 if sys.platform == "win32" else 60,
    )
    assert "NATIVE_PERSONA_OVERLAP_REFUSED" in output
