"""A real Markdown import receipt and note placement survive isolated restore."""

import json
import os
from pathlib import Path

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _isolated_environment,
    _run_profile_child,
)

_PRIVATE = r"""
import asyncio,hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['NOTE_RECEIPT_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_SEED = (
    _PRIVATE
    + r"""
from dataclasses import asdict
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir,get_notes_sync_state_db_path
from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
from tldw_chatbook.Notes.note_import_parsers import parse_import_sources
from tldw_chatbook.Notes.note_import_planner import classify_import_batch
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds
from tldw_chatbook.Notes.note_import_execution_models import approve_note_import_plan,ImportSessionState
from tldw_chatbook.Notes.note_import_executor import NoteImportExecutor,LocalNoteImportTarget
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
async def main():
 app=TldwCli()
 try:
  markdown=fixture/'selected.md';markdown.write_text('# Retained imported note\n\nNative Markdown body.\n')
  bounds=ImportBounds(max_files=4,max_file_bytes=4096,max_total_bytes=8192,max_depth=4)
  discovery=discover_import_sources((markdown,),bounds)
  parsed=parse_import_sources(discovery,bounds,destination_folder_segments=('Imported','Retained'))
  plan=classify_import_batch(parsed,bounds)
  approved=approve_note_import_plan(plan)
  folders=LocalNoteFolderRepository(app.chachanotes_db)
  receipts=NoteImportReceiptRepository(get_notes_sync_state_db_path())
  executor=NoteImportExecutor(target=LocalNoteImportTarget(db=app.chachanotes_db,folder_repository=folders),receipt_repository=receipts)
  receipt=await executor.execute_async(approved)
  assert receipt.state is ImportSessionState.COMPLETED and receipt.imported==1 and receipt.failed==0
  snapshot=receipts.load_session_snapshot(approved.approval_id)
  assert snapshot.state is ImportSessionState.COMPLETED and len(snapshot.items)==1
  assert len(snapshot.payload_effects)==1 and len(snapshot.folder_effects)==2 and len(snapshot.membership_effects)==1
  note_id=snapshot.payload_effects[0].target_note_id;assert note_id
  note=app.chachanotes_db.get_note_by_id(note_id)
  assert note['title']=='Retained imported note' and note['content']==markdown.read_text()
  folder=folders.get_folder_by_path(('Imported','Retained'));assert folder is not None
  memberships=folders.list_memberships(note_ids=(note_id,),include_inactive=True)
  assert len(memberships)==1 and memberships[0].folder_id==folder.folder_id
  assert receipts.aggregate_receipt(approved.approval_id)==receipt
  user_root=get_user_data_dir();assert receipts.db_path==user_root/'tldw_chatbook_notes_sync_state.db'
  (fixture/'seed.json').write_text(json.dumps({'approval':approved.approval_id,'snapshot':asdict(snapshot),'receipt':asdict(receipt),'note':note,'folder':asdict(folder),'memberships':[asdict(m) for m in memberships],'markdown':str(markdown),'markdown_hex':markdown.read_bytes().hex(),'user_root':str(user_root),'receipt_path':str(receipts.db_path),'receipt_sha256':hashlib.sha256(receipts.db_path.read_bytes()).hexdigest(),'config_hex':selector.read_bytes().hex()},indent=2,default=str))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())

"""
)

_CAPTURE = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.config import get_user_data_dir
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seed=json.loads((fixture/'seed.json').read_text());source_root=get_user_data_dir()
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues,'blocking':[{'owner':item.owner,'logical_id':item.logical_id,'path':str(item.path),'status':item.status} for item in preview.items if item.owner=='unknown' or item.status in {'unsupported','unavailable','missing_required'}]},indent=2))
  assert preview.complete,preview.issues
  destination=fixture/'note-import.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);source={item.logical_id:item for item in captured.inventory.items}
  receipts=[row for row in manifest['files'] if row['owner_id']=='notes.sync_state']
  assert len(receipts)==1 and source[receipts[0]['logical_id']].path==Path(seed['receipt_path'])
  assert any(key.endswith(':db.chachanotes.primary') for key in source[receipts[0]['logical_id']].dependencies)
  markdown=Path(seed['markdown'])
  assert not any(item.path==markdown and item.status=='included' for item in captured.inventory.items)
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  (fixture/'capture.json').write_text(json.dumps({'manifest':json.loads(captured.manifest_bytes),'archive_sha256':written.digest,'receipt_file':receipts[0]},indent=2))
  mapping={};selected=fixture/'restore-destinations'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-import'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'mapping':mapping,'source_profile':profile,'receipt_id':receipts[0]['logical_id']},indent=2))
  assert selector.read_bytes().hex()==seed['config_hex']
  assert hashlib.sha256(Path(seed['receipt_path']).read_bytes()).hexdigest()==seed['receipt_sha256']
  assert Path(seed['markdown']).read_bytes().hex()==seed['markdown_hex']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)

_RESTORE = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
receipt=json.loads((fixture/'restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive);assert doc.consistency=='coherent'
import zipfile
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  if row.owner_id=='notes.sync_state':assert hashlib.sha256(zipped.read(row.payload)).hexdigest()==row.sha256
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-import'})
try:profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 (fixture/'restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str));raise
requirements=profile_requirements(profile,fixture/'control')
assert requirements['requirements_checked'] and 'notes.sync_state' in requirements['pending_owners']
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'state':'restoration_validated','requirements':requirements}))
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
restored=json.loads((fixture/'restored.json').read_text());select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from dataclasses import asdict
from tldw_chatbook.config import get_notes_sync_state_db_path
from tldw_chatbook.Backup_Recovery.isolated_restore import profile_requirements
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'seed.json').read_text());user_root=get_user_data_dir()
  assert user_root!=Path(seed['user_root'])
  receipts=NoteImportReceiptRepository(get_notes_sync_state_db_path())
  assert receipts.db_path==user_root/'tldw_chatbook_notes_sync_state.db'
  before=hashlib.sha256(receipts.db_path.read_bytes()).hexdigest()
  snapshot=receipts.load_session_snapshot(seed['approval'])
  receipt=receipts.aggregate_receipt(seed['approval'])
  normalized=lambda value:json.loads(json.dumps(value,default=str))
  assert normalized(asdict(snapshot))==seed['snapshot']
  assert normalized(asdict(receipt))==seed['receipt']
  note_id=snapshot.payload_effects[0].target_note_id
  assert normalized(app.chachanotes_db.get_note_by_id(note_id))==seed['note']
  folders=LocalNoteFolderRepository(app.chachanotes_db)
  folder=folders.get_folder_by_path(('Imported','Retained'))
  assert normalized(asdict(folder))==seed['folder']
  memberships=folders.list_memberships(note_ids=(note_id,),include_inactive=True)
  assert normalized([asdict(m) for m in memberships])==seed['memberships']
  assert hashlib.sha256(receipts.db_path.read_bytes()).hexdigest()==before
  requirements=profile_requirements(restored['profile'],fixture/'control')
  assert requirements['requirements_checked'] and 'notes.sync_state' in requirements['pending_owners']
  assert hashlib.sha256(Path(seed['receipt_path']).read_bytes()).hexdigest()==seed['receipt_sha256']
  assert Path(seed['markdown']).read_bytes().hex()==seed['markdown_hex']
  (fixture/'readback.json').write_text(json.dumps({'approval':receipt.approval_id,'session':snapshot.session_id,'note':note_id,'folder':folder.folder_id,'state':receipt.state,'imported':receipt.imported,'source_preserved':True,'receipt_bytes_preserved':True,'requirements':requirements,'blocked_network_attempts':len(blocked_attempts())},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())

"""
)


def test_native_note_import_receipt_restores_with_fresh_passive_reads(tmp_path):
    root = tmp_path.resolve()
    for name in ("home", "xdg-config", "xdg-data", "cache", "tmp", "profile"):
        (root / name).mkdir(mode=0o700)
    profile = root / "profile"
    (profile / "custom").mkdir(mode=0o700)
    (profile / "data").mkdir(mode=0o700)
    selector = profile / "config.toml"
    selector.write_text(
        '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
        "[AppRAGSearchConfig.rag.indexing]\nenabled=false\n"
        f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
        + "".join(
            f"{key}={json.dumps(str(profile / 'custom' / leaf))}\n"
            for key, leaf in (
                ("chachanotes_db_path", "notes.db"),
                ("media_db_path", "media.db"),
                ("research_db_path", "research.db"),
                ("prompts_db_path", "prompts.db"),
            )
        )
    )
    selector.chmod(0o600)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
        if key in os.environ
    }
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "xdg-config"),
        XDG_DATA_HOME=str(root / "xdg-data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        NOTE_RECEIPT_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    _run_profile_child(root, "seed", _SEED, environment)
    _run_profile_child(root, "capture", _CAPTURE, environment)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "restore", _RESTORE, restored_environment)
    _run_profile_child(root, "read", _READ, restored_environment)
    evidence = json.loads((root / "readback.json").read_text())
    assert evidence["source_preserved"] and evidence["blocked_network_attempts"] == 0
    seed = json.loads((root / "seed.json").read_text())
    assert selector.read_bytes().hex() == seed["config_hex"]
