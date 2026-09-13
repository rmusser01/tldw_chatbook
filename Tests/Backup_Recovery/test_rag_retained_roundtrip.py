"""Native RAG definitions, tracking history and whole projections survive restore.

Supplied vectors qualify retained native data, not an embedding model or readiness.
A refusal of the native default profile selector must fail this public roundtrip.
"""

import json
import os
from pathlib import Path

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _isolated_environment,
    _run_profile_child,
)

_PRIVATE = r"""
import asyncio,hashlib,json,os,stat,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['RAG_RETAINED_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
def normalized(value):return json.loads(json.dumps(value,default=str))
def tree(root):
 return {'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},'directories':{str(p.relative_to(root)):{'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns} for p in (root,*root.rglob('*')) if p.is_dir()}}
def originals(seed):
 assert Path(seed['selector']).read_bytes().hex()==seed['config_hex']
 assert Path(seed['profile_path']).read_bytes().hex()==seed['profile_hex']
 assert hashlib.sha256(Path(seed['indexing_path']).read_bytes()).hexdigest()==seed['indexing_sha256']
 assert tree(Path(seed['projection_root']))==seed['projection']
"""

_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir,get_rag_indexing_db_path
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
async def main():
 app=TldwCli()
 try:
  note_id=app.chachanotes_db.add_note('Retained RAG note','Native note for retained supplied vectors.')
  note=app.chachanotes_db.get_note_by_id(note_id);assert note['id']==note_id
  user_root=get_user_data_dir();projection_root=user_root/'chromadb'
  manager=ConfigProfileManager();profile=manager.create_custom_profile('Retained local RAG',base_profile='hybrid_basic')
  definition=normalized(profile.to_dict());profile_path=manager.profiles_dir/(profile.id+'.json')
  assert definition['rag_config']['vector_store']['type']=='chroma'
  assert Path(definition['rag_config']['vector_store']['persist_directory'])==projection_root
  assert projection_root.is_absolute()
  store=ChromaVectorStore(projection_root,collection_name='retained_notes')
  try:
   store.add([note_id+':'+str(i) for i in range(1100)],[[1.,0.]]*1100,[note['content']]*1100,[{'doc_id':note_id,'doc_title':note['title'],'chunk_index':i} for i in range(1100)])
  finally:store.close()
  assert tuple(projection_root.glob('*/header.bin'))
  indexing_path=get_rag_indexing_db_path();db=RAGIndexingDB(indexing_path)
  try:
   db.mark_item_indexed(note_id,'note',note['last_modified'],chunk_count=1100,metadata={'collection':'retained_notes','fixture':'supplied-vectors'})
   db.update_collection_state('retained_notes',1,1,metadata={'source_note':note_id})
   tracking={'item':db.get_indexed_item_info(note_id,'note'),'collection':db.get_collection_state('retained_notes'),'stats':db.get_indexing_stats()}
   assert tracking['item']['chunk_count']==1100 and tracking['stats']['total_indexed']==1
  finally:db.close()
  seed={'selector':str(selector),'config_hex':selector.read_bytes().hex(),'user_root':str(user_root),'note':note,'profile_id':profile.id,'profile':definition,'profile_path':str(profile_path),'profile_hex':profile_path.read_bytes().hex(),'indexing_path':str(indexing_path),'indexing_sha256':hashlib.sha256(indexing_path.read_bytes()).hexdigest(),'tracking':tracking,'projection_root':str(projection_root),'projection':tree(projection_root)}
  (fixture/'seed.json').write_text(json.dumps(seed,indent=2,default=str));originals(normalized(seed))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
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
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seed=json.loads((fixture/'seed.json').read_text());source_root=Path(seed['user_root']);options={'staging_parent':fixture}
  preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues,'blocking':[{'owner':i.owner,'path':str(i.path),'status':i.status} for i in preview.items if i.status in {'unsupported','unavailable','missing_required'}]},indent=2))
  assert preview.complete,preview.issues
  destination=fixture/'retained-rag.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent'
  source={item.logical_id:item for item in captured.inventory.items}
  files=[row for row in manifest['files'] if row['owner_id']=='rag.projections']
  assert {row['relative_path']:hashlib.sha256((captured.root/row['payload']).read_bytes()).hexdigest() for row in files}==seed['projection']['files']
  root_id=files[0]['root_id']
  directories={row['relative_path'] or '.':{'mode':row['metadata']['mode'],'mtime_ns':row['metadata']['mtime_ns']} for row in manifest['directories'] if row['root_id']==root_id}
  assert directories==seed['projection']['directories']
  definitions=[row for row in manifest['files'] if row['owner_id']=='rag.definitions']
  assert len(definitions)==1 and source[definitions[0]['logical_id']].path==Path(seed['profile_path'])
  # Managed-secret exclusion removes the native empty API-key field.
  expected_definition=normalized(seed['profile'])
  assert expected_definition['rag_config']['embedding'].pop('api_key') is None
  assert json.loads((captured.root/definitions[0]['payload']).read_text())==expected_definition
  indexing=[row for row in manifest['files'] if row['owner_id']=='db.rag_indexing']
  assert len(indexing)==1 and source[indexing[0]['logical_id']].path==Path(seed['indexing_path'])
  # SQLite backup rewrites header counters; compare actual tracking on a copy.
  from shutil import copyfile
  from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
  snapshot=fixture/'indexing-readback.db';copyfile(captured.root/indexing[0]['payload'],snapshot)
  tracking_db=RAGIndexingDB(snapshot)
  try:
   tracking=normalized({'item':tracking_db.get_indexed_item_info(seed['note']['id'],'note'),'collection':tracking_db.get_collection_state('retained_notes'),'stats':tracking_db.get_indexing_stats()})
   assert tracking==seed['tracking']
  finally:tracking_db.close()
  projection_dependencies={source[key].owner for key in source[root_id].dependencies}
  assert {'rag.definitions','db.rag_indexing','db.media.primary','db.chachanotes.primary','db.prompts.primary'}<=projection_dependencies
  tracking_dependencies={source[key].owner for key in source[indexing[0]['logical_id']].dependencies}
  assert {'db.media.primary','db.chachanotes.primary','db.prompts.primary'}<=tracking_dependencies
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  (fixture/'capture.json').write_text(json.dumps({'manifest':json.loads(captured.manifest_bytes),'archive_sha256':written.digest,'projection_dependencies':sorted(projection_dependencies),'tracking_dependencies':sorted(tracking_dependencies)},indent=2))
  mapping={};selected=fixture/'restore-destinations'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-rag'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'mapping':mapping,'source_profile':profile},indent=2))
  originals(seed);assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)

_RESTORE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
receipt=json.loads((fixture/'restore-input.json').read_text());seed=json.loads((fixture/'seed.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive);assert doc.consistency=='coherent'
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  data=zipped.read(row.payload);assert len(data)==row.size and hashlib.sha256(data).hexdigest()==row.sha256
 assert {row.relative_path:row.sha256 for row in doc.files if row.owner_id=='rag.projections'}==seed['projection']['files']
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-rag'})
try:profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 originals(seed)
 (fixture/'restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args,'source_preserved':True},default=str));raise
requirements=profile_requirements(profile,fixture/'control')
assert requirements['requirements_checked'] and {'rag.definitions','db.rag_indexing','rag.projections'}<=set(requirements['pending_owners'])
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements}))
originals(seed);assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'restored.json').read_text());select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.config import get_user_data_dir,get_rag_indexing_db_path,get_chachanotes_db_path
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
seed=json.loads((fixture/'seed.json').read_text());root=get_user_data_dir();assert root!=Path(seed['user_root'])
requirements=profile_requirements(restored['profile'],fixture/'control')
assert requirements['requirements_checked'] and {'rag.definitions','db.rag_indexing','rag.projections'}<=set(requirements['pending_owners'])
profile_path=root/'rag_profiles'/(seed['profile_id']+'.json');before=profile_path.read_bytes()
profile=normalized(ConfigProfileManager().get_profile(seed['profile_id']).to_dict())
expected=seed['profile'];expected['rag_config']['vector_store']['persist_directory']=str(root/'chromadb')
assert profile==expected
assert profile_path.read_bytes()==before
indexing_path=get_rag_indexing_db_path();before_db=hashlib.sha256(indexing_path.read_bytes()).hexdigest();db=RAGIndexingDB(indexing_path)
try:
 tracking=normalized({'item':db.get_indexed_item_info(seed['note']['id'],'note'),'collection':db.get_collection_state('retained_notes'),'stats':db.get_indexing_stats()})
 assert tracking==seed['tracking']
finally:db.close()
notes=CharactersRAGDB(get_chachanotes_db_path(),'retained-reader')
try:assert normalized(notes.get_note_by_id(seed['note']['id']))==seed['note']
finally:notes.close_connection()
with acquire_storage(root/'chromadb'):
 retained=tree(root/'chromadb')
 assert retained['files']==seed['projection']['files']
 assert retained['directories']=={key:{'mode':0o700,'mtime_ns':value['mtime_ns']} for key,value in seed['projection']['directories'].items()}
assert profile_requirements(restored['profile'],fixture/'control')==requirements
originals(seed);assert not blocked_attempts(),blocked_attempts()
(fixture/'readback.json').write_text(json.dumps({'source_preserved':True,'projection_retained':True,'requirements':requirements,'indexing_before':before_db,'indexing_after':hashlib.sha256(indexing_path.read_bytes()).hexdigest(),'blocked_network_attempts':len(blocked_attempts())},indent=2))
"""
)


def test_native_rag_retained_roundtrip_with_fresh_passive_reads(tmp_path):
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
        RAG_RETAINED_FIXTURE=str(root),
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
    assert evidence["source_preserved"] and evidence["projection_retained"]
    assert evidence["blocked_network_attempts"] == 0
