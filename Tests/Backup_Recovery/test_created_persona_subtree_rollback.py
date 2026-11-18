"""Native rollback retires only a committed incoming nested Persona pack."""

import sys

import pytest

from Tests.Backup_Recovery.test_config_sibling_capture import _SEED, _child


def _seed_persona(db, source, profile, *, persona_id="persona-local-1"):
    """Seed existing publisher-format artwork through native private files."""
    import hashlib
    import os
    from dataclasses import replace
    from uuid import uuid4

    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_directory,
        create_private_file,
    )
    from tldw_chatbook.Persona_Visual.publication import (
        _cleanup_marker,
        _repository_asset_row,
        _validate_snapshot,
    )
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
    from tldw_chatbook.Utils.platform_files import os as native_os

    # Its publication API is POSIX-only; backup also handles existing Windows data.
    snapshot = replace(_snapshot(source), persona_id=persona_id)
    manifest, context, assets = _validate_snapshot(snapshot)
    relative = f"persona_visual/packs/{uuid4().hex}/versions/{uuid4().hex}"
    directory = profile
    for part in relative.split("/") + ["assets"]:
        directory = directory / part
        if not directory.exists():
            create_private_directory(directory)
    version_path = profile / relative
    identity = native_os.stat(version_path, follow_symlinks=False)
    payloads = {
        "manifest.json": snapshot.manifest_json.encode("utf-8"),
        ".persona-visual-cleanup": _cleanup_marker(
            os.urandom(32).hex(), relative, (identity.st_dev, identity.st_ino)
        ),
    }
    asset_rows = []
    for index, (source_key, metadata) in enumerate(assets):
        leaf = f"assets/{index:03d}.png"
        payload = (source / source_key).read_bytes()
        assert metadata.mime_type == "image/png"
        assert len(payload) == metadata.byte_count
        assert hashlib.sha256(payload).hexdigest() == metadata.sha256
        payloads[leaf] = payload
        asset_rows.append(
            _repository_asset_row(metadata, storage_relpath=f"{relative}/{leaf}")
        )
    for leaf, payload in payloads.items():
        with create_private_file(version_path / leaf) as descriptor:
            assert native_os.write(descriptor, payload) == len(payload)
        assert (version_path / leaf).read_bytes() == payload
    repository = PersonaVisualRepository(db)
    graph = repository.activate_new_pack(
        persona_id=snapshot.persona_id,
        title=snapshot.title,
        description=snapshot.description,
        source_kind=snapshot.source_kind,
        source_context=context,
        manifest=manifest,
        manifest_storage_relpath=f"{relative}/manifest.json",
        assets=asset_rows,
        expected_persona_revision=snapshot.persona_revision,
        authority_guard=lambda: True,
    )
    assert repository.get_active_persona_pack(snapshot.persona_id) == graph
    assert (
        graph.version.manifest_sha256
        == hashlib.sha256(payloads["manifest.json"]).hexdigest()
    )


_SEED_PERSONA = _SEED.replace(
    "print('NATIVE_SEED_COMPLETE',flush=True)",
    """
from Tests.Backup_Recovery.test_created_persona_subtree_rollback import _seed_persona
from tldw_chatbook.config import get_user_data_dir
source=Path.home()/'persona-input';source.mkdir(mode=0o700)
_seed_persona(app.chachanotes_db,source,get_user_data_dir(),persona_id='persona-'+Path.home().name)
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
import asyncio,hashlib,json,sys,threading,time
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
  started=time.monotonic()
  try:
   with observe_notes_write_failure(Path.home()/'persona-note-write.log',app.chachanotes_db):
    app.chachanotes_db.add_note('After nested pack replacement','New current data must enter the later safety copy.')
  except BaseException:
   # TASK-41115: how long the INSERT waited, and the thread names the record's idents belong to.
   (Path.home()/'persona-note-write-context.json').write_text(json.dumps({'waited':round(time.monotonic()-started,3),'names':{str(t.ident):t.name for t in threading.enumerate()}}))
   raise
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


#: A child that uses no CPU this long is hung, however loaded the host is.
_HANG_SECONDS = 60


def _note_write_summary(home):
    """TASK-41115: the failed add_note's error, its wait and every thread's
    top frames, compact enough that the failure report keeps all of it."""
    import json

    record = json.loads((home / "persona-note-write.log").read_text("utf-8"))
    context = home / "persona-note-write-context.json"
    context = json.loads(context.read_text("utf-8")) if context.is_file() else {}
    names = context.get("names", {})
    error = {key: value for key, value in record["error"].items() if key != "frames"}
    lines = [
        f"add_note waited {context.get('waited', '?')} s; error {error}; "
        f"transaction {record['transaction_entry']}"
    ]
    for thread in record["threads"]:
        frames = " < ".join(
            f"{frame['file']}:{frame['function']}:{frame['line']}"
            for frame in thread["frames"][:12]
        )
        lines.append(f"{names.get(str(thread['thread']), thread['thread'])}: {frames}")
    return "\n".join(lines)


def _prepare_native(tmp_path):
    # TASK-41115: at load 83-106 the capture child outlasted its fixed 120 s in
    # every fixture build while busy. Like the replacement (TASK-40306), each
    # fixture child now fails only once it uses no CPU, under a 600 s cap.
    for label in ("source", "target"):
        home = tmp_path / label
        (home / ".config/tldw_cli").mkdir(parents=True, mode=0o700)
        _child(home, _SEED_PERSONA, "seed-persona", timeout=600, idle=_HANG_SECONDS)
    _child(
        tmp_path / "source",
        _CAPTURE,
        "persona-capture",
        timeout=600,
        idle=_HANG_SECONDS,
    )
    # TASK-40306: two whole-profile replaces (the credential review aborts the
    # first), built by every xdist worker. On a loaded host the child outlasted
    # its old 150 s while busy, so it now fails only once it uses no CPU.
    _child(
        tmp_path / "target",
        _REPLACE,
        "persona-replacement",
        timeout=600,
        idle=_HANG_SECONDS,
    )
    try:
        _child(
            tmp_path / "target",
            _REOPEN,
            "persona-reopen",
            timeout=600,
            idle=_HANG_SECONDS,
        )
    except AssertionError as failure:
        # TASK-40306: one reopen failed `database is locked` on add_note under
        # load and its thread record was lost with the basetemp; keep it.
        if (tmp_path / "target" / "persona-note-write.log").is_file():
            failure.add_note(_note_write_summary(tmp_path / "target"))
        raise
    return tmp_path / "target"


#: TASK-41115: each child after the fixture fails only once idle, under a wall
#: backstop about twice its loaded run. At load 38-68 the later child outlasted
#: the 299 s left of the old shared 860 s budget while busy, and the capture
#: child its fixed 120 s at load 83-106 (2026-11-16).
_LATER_BACKSTOP, _RESTORED_REOPEN_BACKSTOP, _RESTORED_CAPTURE_BACKSTOP = 900, 300, 480


# Above the sum of every child backstop: the fixture's five (600 s each) and
# the three below.
@pytest.mark.timeout(4800)
def test_native_nested_persona_pack_survives_reopen_and_later_rollback(tmp_path):
    home = _prepare_native(tmp_path)
    output = _child(
        home, _LATER, "persona-later", timeout=_LATER_BACKSTOP, idle=_HANG_SECONDS
    )
    assert "NATIVE_NESTED_PERSONA_LATER_COMPLETE" in output
    output = _child(
        home,
        _RESTORED_REOPEN,
        "persona-restored-reopen",
        timeout=_RESTORED_REOPEN_BACKSTOP,
        idle=_HANG_SECONDS,
    )
    assert "NATIVE_PERSONA_ORDINARY_RESTORED_REOPEN_COMPLETE" in output
    output = _child(
        home,
        _CAPTURE.replace("incoming.tldw-backup.zip", "after-later.tldw-backup.zip"),
        "persona-restored-capture",
        timeout=_RESTORED_CAPTURE_BACKSTOP,
        idle=_HANG_SECONDS,
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
# TASK-41115: a node's timeout also covers the module fixture it builds. At
# load 75-90 that build outlasted 600 s while busy, so the cap now sits above
# the fixture's five child caps (600 s each) plus this node's own child.
@pytest.mark.timeout(3600)
def test_native_persona_mapping_refuses_unproved_graph(native_persona_negatives, case):
    # TASK-41638: fail on a hang (no CPU for 60 s), not on a slow loaded child.
    output = _child(native_persona_negatives, _NEGATIVE, case, timeout=600, idle=60)
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
@pytest.mark.timeout(3600)  # As above.
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



# TASK-40305 AC2: later rollbacks of restores that changed or created the Actor
# Pack and reaction-artwork trees. One orchestrating child per profile, as the
# selected-group service tests do; activations run in their own processes so the
# orchestrator never imports config and holds no lease across maintenance
# (TASK-40132 lesson). The route argument picks the scenario:
#   changed         back up, add a second pack, restore, roll the restore back
#   created-source  back up a profile holding one pack (for both targets below)
#   created-target  restore that backup into a profile that had neither tree
#                   nor a persona store, then roll the restore back
#   nested-target   restore that backup into a profile holding another pack,
#                   so the restore creates a pack folder inside each existing
#                   root, then roll the restore back (TASK-41113)
# The created-source profile also holds every single-file owner of
# later_rollback._CREATED_FILE_OWNERS besides the persona store: created-target
# has none, so the restore creates each, and nested-target holds its own copy
# of each, which the rollback must bring back (TASK-41521).
#   recreated-target  as created-target, but one created file is replaced
#                   after the restore (same bytes, new inode): the later
#                   rollback refuses to retire it (TASK-41521 review)
_ACTOR_TREES_ROLLBACK = r"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap,credentials
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
credential_store=KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store=lambda:credential_store
route=sys.argv[1]
source=Path(os.environ['TLDW_CONFIG_PATH'])
data=Path(os.environ['XDG_DATA_HOME'])/'fixture';data.mkdir(mode=0o700)
source.write_text('[general]\nusers_name="default_user"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n')
source.chmod(0o600)
admission_authority(bootstrap.default_bootstrap_root())
producer='''
import json,os,sys,tomllib
from pathlib import Path
from Tests import network_guard
network_guard.install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
selector=Path(os.environ['TLDW_CONFIG_PATH'])
if sys.argv[1]!='activate':
    from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
    _populate_required_dependencies(preview_capture((selector,),options={'allow_partial':True}))
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
root=get_user_data_dir()
db=CharactersRAGDB(database_path(tomllib.loads(selector.read_text()),'chachanotes_db_path'),'actor-pack-rollback')
if sys.argv[1]=='populate':
    # A profile with no persona and no Actor Pack: the restore creates the
    # persona store file as well as both trees (TASK-41113).
    db.close_connection()
    print(json.dumps({'root':str(root)}))
    raise SystemExit(0)
from Tests.Actor_Packs.test_actor_pack_activation import _persona_runtime_section,_shared_visual_section,_write_archive
from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
try:
    repository=ActorPackRepository(db);local=build_persona_service(db)
    importer=ActorPackImportService(repository,staging_root=root/'actor_pack_imports',profile_root=root,local_service=local)
    activation=ActorPackActivationService(db,local,repository,PersonaActorPackCoordinator(repository,local),importer)
    archive=_write_archive(Path.home()/(sys.argv[2]+'.tldw-actor-pack'),actor_kind='persona',portable_uuid=sys.argv[2],sections=(_shared_visual_section(),_persona_runtime_section()))
    review=importer.inspect_archive(archive)
    activation.activate(review,'create_new')
finally:
    db.close_connection()
print(json.dumps({'root':str(root),'digest':review.content_digest}))
'''

def produce(*args):
    result=subprocess.run([sys.executable,'-c',producer,*args],capture_output=True,text=True,timeout=900 if sys.platform=='win32' else 240)
    assert result.returncode==0,result.stderr[-4000:]+result.stdout[-1000:]
    return json.loads(result.stdout.strip().splitlines()[-1])

def trees(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ('actor_packs','visual_identities') for p in sorted((root/name).rglob('*')) if p.is_file()}

def succeeded(service,operation):
    result=service.wait(operation)
    assert result['state']=='succeeded',dict(result)
    return result

def reviewed(service,start,retry):
    result=service.wait(start())
    if result['state']=='recovery_required' and result['review_issues']:
        issues=tuple(result['review_issues'])
        assert all(code.startswith('credential_') for code in issues),issues
        succeeded(service,service.start_recovery(result['result']['journal_operation_id'],action='abort'))
        result=service.wait(retry(issues))
    assert result['state']=='succeeded',dict(result)
    return result

def backup(service,root,destination):
    preview=service.preview_backup((source,),options={'data_groups':groups})
    assert preview.complete,preview.issues
    owners={row.owner for row in preview.items if row.path is not None and row.status=='included'
            and any(row.path.is_relative_to(root/name) for name in ('actor_packs','visual_identities'))}
    assert owners=={'persona.actor_packs','persona.visual_identity'},owners
    succeeded(service,service.start_backup((source,),preview.scope_digest,destination,options={'data_groups':groups},password=None))

def restore(service,destination):
    inspection=service.start_inspection(destination,password=None)
    succeeded(service,inspection)
    summary=service.summary(inspection)
    assert {'Actor Pack persona artwork','Reaction artwork'}<=set(summary['owner_labels']),summary['owner_labels']
    profile=summary['profile_ids'][0]
    def plan(issues=()):
        return service.preview_restore(inspection,mode='replace',profile_bases={},target_configs={profile:source},
            external_destinations={},profile_names={},target=service.preview_backup((source,),options={}),
            data_groups=groups,acknowledged_credential_issues=issues)
    restored=reviewed(service,lambda:service.start_restore(inspection,plan(),rollback_password=b'rollback-test-password'),
        lambda issues:service.start_restore(inspection,plan(issues),rollback_password=b'rollback-test-password'))
    return restored['result']['journal_operation_id']

def roll_back(service,operation,root,expected):
    current=service.preview_backup((source,),options={})
    rollback=preview_rollback(operation,control_root=service.control_root,old_password=b'rollback-test-password',target=current,cancel=Event())
    assert rollback.effective_groups==groups
    def later(issues=()):
        approved=rollback if not issues else service.preview_rollback(operation,old_password=b'rollback-test-password',
            target=service.preview_backup((source,),options={}),acknowledged_credential_issues=issues)
        return service.start_rollback(operation,approved,old_password=b'rollback-test-password',new_password=b'second-safety-password')
    reviewed(service,later,later)
    after=trees(root)
    assert after==expected,{'missing':sorted(set(expected)-set(after)),'extra':sorted(set(after)-set(expected)),
        'differ':sorted(key for key in set(after)&set(expected) if after[key]!=expected[key])}
    final=service.preview_backup((source,),options={'data_groups':groups})
    assert final.complete,final.issues

# The populated workspace and agent-run stores link to the ChaChaNotes core.
# The single files' owners add Audio, Saved chatbooks and Prompts (TASK-41521).
groups=('conversations','workspaces') if route=='changed' else ('audio','chatbooks','conversations','prompts','workspaces')
SINGLE_FILES={
    'audio.history':'tldw_chatbook_audio_history.json',
    'chat.dictionary_history':'tldw_chatbook_chat_dictionary_history.json',
    'chat.grammars':'tldw_chatbook_chat_grammars.json',
    'chat.prompt_history':'prompt_history.jsonl',
    'chat.rag_context':'tldw_chatbook_chat_rag_context.json',
    'chatbooks.registry':'tldw_chatbook_chatbooks.json',
    'feedback':'tldw_chatbook_feedback.json',
    'notes.templates':'note_templates.json',
}

def single_files(root):
    # Note templates live beside the config file, the rest in the data folder.
    return {owner:(source.parent if owner=='notes.templates' else root)/leaf for owner,leaf in SINGLE_FILES.items()}

def read_single_files(root):
    return {owner:path.read_text() if path.exists() else None for owner,path in single_files(root).items()}

def write_single_files(root,tag):
    for owner,path in single_files(root).items():
        if owner=='chatbooks.registry':
            # One metadata-only entry, in the layout capture and restore
            # rewrite the registry to, so a restored copy reads back equal.
            text=json.dumps({'records':[{'title':tag}]},ensure_ascii=False,indent=2)
        else:
            text=json.dumps({'owner':owner,'from':tag})
        path.write_text(text);path.chmod(0o600)
    return read_single_files(root)

shared=Path.home().parent.parent/'created-source'/'home'
service=RecoveryService(Path.home()/'recovery-control')
try:
    if route=='changed':
        first=produce('prepare','123e4567-e89b-42d3-a456-426614174000')
        root=Path(first['root'])
        destination=Path.home()/'conversations.tldw-backup.zip'
        backup(service,root,destination)
        backed=trees(root)
        assert any(first['digest'] in key for key in backed),backed
        # After the backup, a second activation adds a pack to both trees.
        second=produce('activate','223e4567-e89b-42d3-a456-426614174000')
        changed=trees(root)
        added=set(changed)-set(backed)
        assert set(backed)<set(changed) and {key.split('/')[0] for key in added}=={'actor_packs','visual_identities'},added
        assert all(second['digest'] in key for key in added),added
        operation=restore(service,destination)
        # The restore retired the second pack from both trees.
        assert trees(root)==backed,'restore did not bring back the backed-up trees'
        # Both owners are ordinary restored tree owners: the review maps them
        # like any other and the rollback republishes their files.
        roll_back(service,operation,root,changed)
    elif route=='created-source':
        first=produce('prepare','123e4567-e89b-42d3-a456-426614174000')
        root=Path(first['root'])
        files=write_single_files(root,'source')
        backup(service,root,Path.home()/'created.tldw-backup.zip')
        (Path.home()/'created-trees.json').write_text(json.dumps(trees(root)))
        (Path.home()/'created-files.json').write_text(json.dumps(files))
    elif route=='created-target':
        root=Path(produce('populate')['root'])
        store=root/'tldw_chatbook_personas.json'
        assert not (root/'actor_packs').exists() and not (root/'visual_identities').exists()
        assert not store.exists()
        assert set(read_single_files(root).values())=={None},read_single_files(root)
        operation=restore(service,shared/'created.tldw-backup.zip')
        incoming=json.loads((shared/'created-trees.json').read_text())
        assert incoming and trees(root)==incoming,'restore did not create both trees'
        assert store.is_file(),'restore did not create the persona store'
        files=json.loads((shared/'created-files.json').read_text())
        assert read_single_files(root)==files,'restore did not create the single files'
        # The prior state had neither tree, nor a persona store, nor any of the
        # single files: the rollback retires what the restore created.
        roll_back(service,operation,root,{})
        assert not store.exists(),'rollback kept the created persona store'
        kept={owner for owner,text in read_single_files(root).items() if text is not None}
        assert not kept,'rollback kept created single files: '+repr(sorted(kept))
    elif route=='recreated-target':
        root=Path(produce('populate')['root'])
        operation=restore(service,shared/'created.tldw-backup.zip')
        files=json.loads((shared/'created-files.json').read_text())
        assert read_single_files(root)==files,'restore did not create the single files'
        # Same bytes under a new inode: only the created file's device/inode
        # check tells it from the file the restore published.
        path=single_files(root)['feedback']
        temp=path.with_name(path.name+'.replacement')
        temp.write_text(path.read_text());temp.chmod(0o600);os.replace(temp,path)
        try:
            preview_rollback(operation,control_root=service.control_root,old_password=b'rollback-test-password',
                target=service.preview_backup((source,),options={}),cancel=Event())
        except ValueError as error:
            assert str(error)=='local_snapshot_created_root_changed',error
        else:
            raise AssertionError('later rollback accepted a created file recreated after the restore')
        assert path.read_text()==files['feedback'],'refused rollback touched the recreated file'
    else:
        assert route=='nested-target',route
        root=Path(produce('prepare','223e4567-e89b-42d3-a456-426614174000')['root'])
        before=trees(root)
        # The target's own persona store: the restore replaces it, so only the
        # created-file rule's previous-is-None test keeps the rollback from
        # retiring it as restore-created (TASK-41113 review).
        store=root/'tldw_chatbook_personas.json'
        own=store.read_bytes()
        own_files=write_single_files(root,'target')
        incoming=json.loads((shared/'created-trees.json').read_text())
        assert before and not set(before)&set(incoming),(before,incoming)
        operation=restore(service,shared/'created.tldw-backup.zip')
        assert trees(root)==incoming,'restore did not replace the target pack'
        assert store.read_bytes()!=own,'restore did not replace the own persona store'
        assert read_single_files(root)==json.loads((shared/'created-files.json').read_text()),'restore did not replace the own single files'
        # The restore created the source's pack folder inside each existing
        # root and retired the target's own; the rollback reverses both.
        roll_back(service,operation,root,before)
        assert store.is_file() and store.read_bytes()==own,'rollback lost the own persona store'
        # Only a file the restore created is retired: each one it replaced
        # comes back (TASK-41521).
        assert read_single_files(root)==own_files,'rollback lost an own single file'
finally:
    service.close()
assert not network_guard.blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.timeout(4200 if sys.platform == "win32" else 1500)
def test_later_rollback_of_a_restore_that_changed_actor_pack_trees(tmp_path):
    """TASK-40305 AC2: restore a backup over a profile whose Actor Pack
    (``actor_packs/``) and reaction-artwork (``visual_identities/``) trees
    changed since, then roll that restore back later. Every file of both trees
    comes back byte for byte."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "changed",
        "rollback",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=4000 if sys.platform == "win32" else 1400,
    )


# Above the two children's summed budgets on every platform; the source
# child's budget holds its producer's (900 on win32).
@pytest.mark.timeout(5700 if sys.platform == "win32" else 1500)
def test_later_rollback_of_a_restore_that_created_actor_pack_trees(tmp_path):
    """TASK-40305 AC2: restore a backup holding both trees into a profile that
    never had either, then roll that restore back later. The rollback leaves
    neither tree holding a file, as before the restore, and retires the
    persona store and the single files the restore created (TASK-41113,
    TASK-41521)."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    for name in ("created-source", "created-target"):
        (tmp_path / name).mkdir(mode=0o700)
    _run(
        tmp_path / "created-source",
        "created-source",
        "backup",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "created-target",
        "created-target",
        "rollback",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=4000 if sys.platform == "win32" else 1100,
    )


# Above the two children's summed budgets on every platform.
@pytest.mark.timeout(5700 if sys.platform == "win32" else 1500)
def test_later_rollback_refuses_a_created_file_recreated_after_the_restore(tmp_path):
    """TASK-41521 review: replace one file the restore created with a copy of
    itself (a new inode) before the later rollback. The rollback refuses with
    ``local_snapshot_created_root_changed`` and leaves the file as it is."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    for name in ("created-source", "recreated-target"):
        (tmp_path / name).mkdir(mode=0o700)
    _run(
        tmp_path / "created-source",
        "created-source",
        "backup",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "recreated-target",
        "recreated-target",
        "refused",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=4000 if sys.platform == "win32" else 1100,
    )


# Above the two children's summed budgets on every platform.
@pytest.mark.timeout(5700 if sys.platform == "win32" else 1500)
def test_later_rollback_of_a_restore_that_created_nested_actor_pack_folders(tmp_path):
    """TASK-41113 AC2: restore a backup holding one pack into a profile that
    holds another, so the restore creates the incoming pack's folder inside
    both existing roots, then roll that restore back later. Both trees come
    back byte for byte, as do the persona store and the single files the
    restore replaced (TASK-41521)."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    for name in ("created-source", "nested-target"):
        (tmp_path / name).mkdir(mode=0o700)
    _run(
        tmp_path / "created-source",
        "created-source",
        "backup",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=1500 if sys.platform == "win32" else 300,
    )
    _run(
        tmp_path / "nested-target",
        "nested-target",
        "rollback",
        script=_ACTOR_TREES_ROLLBACK,
        timeout=4000 if sys.platform == "win32" else 1100,
    )
