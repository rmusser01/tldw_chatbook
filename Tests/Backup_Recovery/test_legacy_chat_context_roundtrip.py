"""An explicit version-1 chat context sidecar survives isolated recovery.

The sidecar in this test is a historical compatibility fixture.  Current app
code creates the related conversation and messages, but does not write this
legacy representation.
"""

import hashlib
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
fixture=Path(os.environ['LEGACY_CHAT_CONTEXT_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(path):
 info=path.stat(follow_symlinks=False)
 return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns]
def snapshot(paths):return {str(path):{'sha256':digest(path),'identity':identity(path)} for path in paths}
def unchanged(receipt):
 for raw,evidence in receipt.items():
  path=Path(raw);assert digest(path)==evidence['sha256'] and identity(path)==evidence['identity']
"""


_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_chachanotes_db_path,get_user_data_dir
async def main():
 app=TldwCli()
 try:
  service=app.local_chat_conversation_service;assert service is not None
  migration=app.citation_legacy_migration_service;assert migration is service.citation_legacy_migration
  assert migration.writes_enabled is False and migration.ready is False
  conversation_id=service.create_conversation(title='Retained legacy citation context',source='local')
  user_id=app.chachanotes_db.add_message({'conversation_id':conversation_id,'sender':'user','content':'What does the retained local note say?'})
  answer='The retained local note says the archive keeps its citation [1].'
  assistant_id=app.chachanotes_db.add_message({'conversation_id':conversation_id,'sender':'assistant','content':answer})
  assert all(isinstance(value,str) and value for value in (conversation_id,user_id,assistant_id))
  sidecar=get_user_data_dir()/'tldw_chatbook_chat_rag_context.json'
  record={
   'conversation_id':conversation_id,
   'message_id':assistant_id,
   'rag_context':{
    'evidence_bundle':{
     'bundle_id':'legacy-retained-bundle',
     'query':'retained local note',
     'references':[{
      'evidence_id':'1','source_id':'retained-note-1','source_type':'note',
      'title':'Retained local note','snippet':'Archive keeps its citation.',
      'authority_label':'Local Library','content_ref':'retained-local-note-1',
     }],
    },
    'citation_validation':{'valid':True},
   },
   'citations':[{'evidence_id':'1','source_id':'retained-note-1'}],
   'answer_body':answer,
   'last_modified':'2026-09-11T12:00:00Z',
  }
  fixture_document={'version':1,'conversations':{conversation_id:{assistant_id:record}}}
  sidecar.write_text(json.dumps(fixture_document,indent=2,sort_keys=True)+'\n',encoding='utf-8')
  sidecar.chmod(0o600)
  messages=service.get_messages_with_context(conversation_id)
  assert [row['id'] for row in messages]==[user_id,assistant_id]
  retained=messages[1]
  assert retained['content']==answer and retained['rag_context']==record['rag_context']
  assert retained['citations']==record['citations'] and retained['citation_provenance_state']=='legacy_fallback'
  assert messages[0]['rag_context'] is None and messages[0]['citations']==[]
  assert migration.get_journal(conversation_id) is None
  database=get_chachanotes_db_path();assert database==selector.parent/'custom'/'notes.db'
  sources=snapshot((selector,sidecar))
  seed={'selector':str(selector),'database':str(database),'sidecar':str(sidecar),'user_root':str(get_user_data_dir()),
        'conversation_id':conversation_id,'user_message_id':user_id,'assistant_message_id':assistant_id,
        'answer':answer,'record':record,'sidecar_hex':sidecar.read_bytes().hex(),'sources':sources,
        'database_initial_sha256':digest(database),
        'migration':{'writes_enabled':migration.writes_enabled,'ready':migration.ready,'journal':None,'read_state':'legacy_fallback'}}
  (fixture/'seed.json').write_text(json.dumps(seed,indent=2))
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
 seed=json.loads((fixture/'seed.json').read_text());unchanged(seed['sources'])
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  service=app.local_chat_conversation_service;migration=app.citation_legacy_migration_service
  assert migration is service.citation_legacy_migration and migration.writes_enabled is False and migration.ready is False
  assert migration.get_journal(seed['conversation_id']) is None
  messages=service.get_messages_with_context(seed['conversation_id'])
  assert [row['id'] for row in messages]==[seed['user_message_id'],seed['assistant_message_id']]
  assert messages[1]['rag_context']==seed['record']['rag_context'] and messages[1]['citations']==seed['record']['citations']
  assert messages[1]['citation_provenance_state']=='legacy_fallback'
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues},indent=2));assert preview.complete,preview.issues
  destination=fixture/'legacy-chat-context.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent';source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id']=='chat.rag_context'];assert len(rows)==1
  row=rows[0];payload=captured.root/row['payload'];source_path=source[row['logical_id']].path
  assert source_path==Path(seed['sidecar']) and payload.read_bytes()==bytes.fromhex(seed['sidecar_hex'])==source_path.read_bytes()
  assert digest(payload)==row['sha256']
  receipt={'logical_id':row['logical_id'],'owner_id':row['owner_id'],'relative_path':row['relative_path'],'payload':row['payload'],'sha256':row['sha256'],'size':row['size']}
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  selected=fixture/'restore-destinations';mapping={}
  source_root=Path(seed['user_root'])
  for directory in manifest['directories']:
   if directory['parent_id'] is not None:continue
   key=directory['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if directory.get('synthetic') else source[key].owner
   if original==source_root or source_root in original.parents:target=selected/'data'/'restored-legacy-chat'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'capture.json').write_text(json.dumps({'archive':str(destination),'archive_sha256':written.digest,
    'manifest_sha256':hashlib.sha256(captured.manifest_bytes).hexdigest(),'sidecar_receipt':receipt,
    'mapping':mapping,'source_profile':profile},indent=2))
  unchanged(seed['sources']);assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
seed=json.loads((fixture/'seed.json').read_text());receipt=json.loads((fixture/'capture.json').read_text())
receipt['source_snapshot']=snapshot((selector,Path(seed['database']),Path(seed['sidecar'])))
receipt['database_post_close_sha256']=receipt['source_snapshot'][seed['database']]['sha256']
assert Path(seed['sidecar']).read_bytes()==bytes.fromhex(seed['sidecar_hex'])
(fixture/'capture.json').write_text(json.dumps(receipt,indent=2))
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
seed=json.loads((fixture/'seed.json').read_text());receipt=json.loads((fixture/'capture.json').read_text());unchanged(receipt['source_snapshot'])
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event());doc=verify_sealed(archive)
assert doc.consistency=='coherent' and archive.digest==receipt['archive_sha256']
with zipfile.ZipFile(archive.path) as zipped:
 assert hashlib.sha256(zipped.read('manifest.json')).hexdigest()==receipt['manifest_sha256']
 for row in doc.files:
  payload=zipped.read(row.payload);assert len(payload)==row.size and hashlib.sha256(payload).hexdigest()==row.sha256
 sidecar=receipt['sidecar_receipt'];row=next(row for row in doc.files if row.logical_id==sidecar['logical_id'])
 assert row.owner_id=='chat.rag_context' and row.sha256==sidecar['sha256'] and row.size==sidecar['size']
 assert zipped.read(row.payload)==bytes.fromhex(seed['sidecar_hex'])
destinations={key:Path(path) for key,path in receipt['mapping'].items()}
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={receipt['source_profile']:'restored-legacy-chat'})
profile=restore_isolated(archive,plan,fixture/'control',threading.Event());requirements=json.loads(json.dumps(profile_requirements(profile,fixture/'control')))
assert requirements['requirements_checked'] and requirements['needs_setup'] and 'chat.rag_context' in requirements['pending_owners']
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements},indent=2))
unchanged(receipt['source_snapshot']);assert not blocked_attempts(),blocked_attempts()
"""
)


_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'restored.json').read_text());select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_chachanotes_db_path,get_user_data_dir
async def main():
 seed=json.loads((fixture/'seed.json').read_text());capture=json.loads((fixture/'capture.json').read_text());unchanged(capture['source_snapshot'])
 root=get_user_data_dir();assert root!=Path(seed['user_root'])
 sidecar=root/'tldw_chatbook_chat_rag_context.json';database=get_chachanotes_db_path()
 assert sidecar.read_bytes()==bytes.fromhex(seed['sidecar_hex'])
 restored_sidecar_before=snapshot((sidecar,))
 app=TldwCli()
 try:
  service=app.local_chat_conversation_service;migration=app.citation_legacy_migration_service
  assert service.rag_context_store_path==sidecar and migration is service.citation_legacy_migration
  assert migration.writes_enabled is False and migration.ready is False and migration.get_journal(seed['conversation_id']) is None
  messages=service.get_messages_with_context(seed['conversation_id'])
  assert [row['id'] for row in messages]==[seed['user_message_id'],seed['assistant_message_id']]
  assert [row['content'] for row in messages]==['What does the retained local note say?',seed['answer']]
  retained=messages[1]
  assert retained['rag_context']==seed['record']['rag_context'] and retained['citations']==seed['record']['citations']
  assert retained['citation_provenance_state']=='legacy_fallback'
  assert messages[0]['rag_context'] is None and messages[0]['citations']==[]
  assert snapshot((sidecar,))==restored_sidecar_before
  requirements=json.loads(json.dumps(profile_requirements(restored['profile'],fixture/'control')))
  assert requirements==restored['requirements'] and 'chat.rag_context' in requirements['pending_owners']
  unchanged(capture['source_snapshot']);assert not blocked_attempts(),blocked_attempts()
  (fixture/'readback.json').write_text(json.dumps({'source_preserved':True,'restored_sidecar_preserved':True,
    'restored_database':str(database),'restored_sidecar':str(sidecar),'conversation_id':seed['conversation_id'],
    'message_ids':[seed['user_message_id'],seed['assistant_message_id']],
    'citation_provenance_state':retained['citation_provenance_state'],
    'migration':{'writes_enabled':migration.writes_enabled,'ready':migration.ready,'journal':None},
    'pending_owner':'chat.rag_context','blocked_network_attempts':len(blocked_attempts())},indent=2))
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def test_native_version1_chat_context_roundtrip_uses_real_message_ids(tmp_path):
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
        ),
        encoding="utf-8",
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
        LEGACY_CHAT_CONTEXT_FIXTURE=str(root),
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
    assert evidence["source_preserved"] and evidence["restored_sidecar_preserved"]
    assert evidence["blocked_network_attempts"] == 0
    seed = json.loads((root / "seed.json").read_text())
    captured = json.loads((root / "capture.json").read_text())
    journals = sorted(str(path) for path in (root / "control").glob("operation-*"))
    assert len(journals) == 1
    print(
        "LEGACY_CHAT_CONTEXT_EVIDENCE",
        json.dumps(
            {
                "fixture_root": str(root),
                "test_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "child_paths": {
                    name: {
                        "source": str(root / f"{name}.py"),
                        "log": str(root / f"{name}.log"),
                    }
                    for name in ("seed", "capture", "restore", "read")
                },
                "source_paths": list(captured["source_snapshot"]),
                "source_sha256": {
                    Path(path).name: receipt["sha256"]
                    for path, receipt in captured["source_snapshot"].items()
                },
                "database_initial_sha256": seed["database_initial_sha256"],
                "database_post_close_sha256": captured["database_post_close_sha256"],
                "archive_path": captured["archive"],
                "archive_sha256": captured["archive_sha256"],
                "manifest_sha256": captured["manifest_sha256"],
                "payload_receipt": captured["sidecar_receipt"],
                "journal_paths": journals,
                "conversation_id": evidence["conversation_id"],
                "message_ids": evidence["message_ids"],
                "restored_database": evidence["restored_database"],
                "restored_sidecar": evidence["restored_sidecar"],
                "citation_provenance_state": evidence["citation_provenance_state"],
                "migration": evidence["migration"],
                "pending_owner": evidence["pending_owner"],
                "blocked_network_attempts": evidence["blocked_network_attempts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
