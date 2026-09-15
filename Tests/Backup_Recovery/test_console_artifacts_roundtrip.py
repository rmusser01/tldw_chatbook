"""Accepted Console records and passive artifacts survive isolated restore."""

import hashlib
import json
import os
from pathlib import Path

from Tests.Backup_Recovery.test_complete_roundtrip import _run_profile_child

_PRIVATE = r"""
import asyncio,hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['CONSOLE_ARTIFACT_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_CAPTURE = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture,preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Utils.custom_tokenizers import get_tokenizer_manager
from tldw_chatbook.Agents.run_log_format import iter_records
from Tests.Chat.test_console_agent_bridge import _ChunkGateway,_test_resolution
async def main():
 app=TldwCli();entered,release=threading.Event(),threading.Event()
 class Gateway(_ChunkGateway):
  async def stream_chat(self,*args,**kwargs):
   entered.set()
   while not release.is_set():await asyncio.sleep(.005)
   async for chunk in super().stream_chat(*args,**kwargs):yield chunk
 gateway=Gateway([['warmup answer'],['captured terminal answer']])
 runtime=app.console_runtime;store=runtime.ensure_chat_store()
 bridge=runtime.ensure_agent_bridge(store_factory=lambda:store,provider_gateway_factory=lambda:gateway)
 controller=runtime.ensure_chat_controller(store=store,provider_gateway=gateway,agent_bridge=bridge)
 session=store.ensure_session(title='Retained Console artifacts')
 question=store.append_message(session.id,role=ConsoleMessageRole.USER,content='saved console question')
 assistant=store.append_message(session.id,role=ConsoleMessageRole.ASSISTANT,content='')
 release.set()
 await controller._run_maintenance_agent_call(bridge.run_reply,conversation_id='warm-console',session_id=session.id,resolution=_test_resolution(),assistant_message_id=assistant.id,model='test-model',session_system_prompt='',agent_messages=[{'role':'user','content':'warmup question'}],should_cancel=lambda:False)
 # Install a real valid local tokenizer document, without encoding text or
 # assigning it to the model used by either accepted Console turn.
 from tokenizers import Tokenizer,models
 source=fixture/'retained-tokenizer.json'
 source.write_text(Tokenizer(models.WordLevel({'[UNK]':0,'retained':1},unk_token='[UNK]')).to_str())
 tokenizer=get_tokenizer_manager()
 assert tokenizer.install_tokenizer(str(source),'retained-artifact')
 tokenizer.add_mapping('unexecuted-retained-model','retained-artifact')
 tokenizer_root=Path(tokenizer.tokenizers_dir)
 assert tokenizer_root==Path.home()/'.config'/'tldw_cli'/'tokenizers'
 tokenizer_bytes={p.name:p.read_bytes().hex() for p in tokenizer_root.iterdir() if p.is_file()}
 release.clear();entered.clear()
 assistant=store.append_message(session.id,role=ConsoleMessageRole.ASSISTANT,content='')
 accepted=asyncio.create_task(controller._run_maintenance_agent_call(bridge.run_reply,conversation_id='captured-console',session_id=session.id,resolution=_test_resolution(),assistant_message_id=assistant.id,model='test-model',session_system_prompt='',agent_messages=[{'role':'user','content':'saved console question'}],should_cancel=lambda:False))
 monitoring=None;finishing=None;watchdog=None;cancel=threading.Event()
 try:
  async with asyncio.timeout(15):
   while not entered.is_set():
    if accepted.done():raise AssertionError(('returned before provider',accepted.result()))
    await asyncio.sleep(.01)
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  assert preview.complete,(preview.issues,[(i.owner,str(i.path),i.status) for i in preview.items if i.status in ('unsupported','unavailable','missing_required')])
  monitoring=asyncio.create_task(monitor_app(app))
  async def finish():
   async with asyncio.timeout(10):
    while not controller._maintenance_paused:await asyncio.sleep(.01)
   assert not accepted.done();release.set()
  finishing=asyncio.create_task(finish())
  watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
  destination=fixture/'console-artifacts.tldw-backup.zip'
  result=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  run_id,outcome=await accepted;await finishing
  assert outcome.final_text=='captured terminal answer'
  async with asyncio.timeout(10):
   while storage._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  run=bridge._db.get_run(run_id)
  assert run['status']=='done' and run['result']=='captured terminal answer'
  after=store.append_message(session.id,role=ConsoleMessageRole.USER,content='after capture')
  manifest=json.loads(result.manifest_bytes);source_items={item.logical_id:item for item in result.inventory.items}
  assert result.inventory.complete and manifest['consistency']=='coherent'
  rows=[row for row in manifest['files'] if row['owner_id'] in {'db.agent_runs','agents.history','tokenizers.custom'}]
  assert {row['owner_id'] for row in rows}=={'db.agent_runs','agents.history','tokenizers.custom'}
  artifacts={row['logical_id']:{'owner':row['owner_id'],'hex':(result.root/row['payload']).read_bytes().hex(),'source':str(source_items[row['logical_id']].path)} for row in rows if row['owner_id']!='db.agent_runs'}
  logs=[bytes.fromhex(row['hex']) for row in artifacts.values() if row['owner']=='agents.history']
  assert any(record.run_id==run_id and 'captured terminal answer' in record.content for body in logs for record in iter_records(body))
  assert {Path(row['source']).name:row['hex'] for row in artifacts.values() if row['owner']=='tokenizers.custom'}==tokenizer_bytes
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination
  selected=fixture/'destinations';source_root=get_user_data_dir();mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source_items[key].path if key in source_items else source_items[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source_items[key].owner
   if owner=='tokenizers.custom':target=fixture/'restore-home'/'.config'/'tldw_cli'/'tokenizers'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-console'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  receipt={'archive':str(destination),'archive_sha256':sealed.digest,'manifest':manifest,'mapping':mapping,'source_profile':profile,'run_id':run_id,'run':run,'session':session.id,'question':question.id,'assistant':assistant.id,'after':after.id,'artifacts':artifacts,'tokenizer_bytes':tokenizer_bytes,'source_paths':[str(selector),*[str(source_items[row['logical_id']].path) for row in rows]],'source_config_hex':selector.read_bytes().hex()}
  (fixture/'capture.json').write_text(json.dumps(receipt,indent=2,default=str))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  release.set();await accepted
  if watchdog:watchdog.cancel()
  cancel.set();tasks=[t for t in (finishing,monitoring) if t is not None]
  for task in tasks:task.cancel()
  await asyncio.gather(*tasks,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
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
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
receipt=json.loads((fixture/'capture.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive);assert doc.consistency=='coherent'
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  if row.logical_id in receipt['artifacts']:assert zipped.read(row.payload).hex()==receipt['artifacts'][row.logical_id]['hex']
plan=plan_restore(archive,mode='isolated',destinations={key:Path(value) for key,value in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-console'})
(fixture/'plan.json').write_text(json.dumps({'issues':plan.issues,'restore':[(key,str(path)) for key,path in plan.restore]}))
try:profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 (fixture/'refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str));raise
tokenizer_root=Path.home()/'.config'/'tldw_cli'/'tokenizers'
before=json.loads((fixture/'tokenizer-root-before.json').read_text())
info=tokenizer_root.stat()
assert (info.st_dev,info.st_ino)==tuple(before)
root_key=next(key for key,path in plan.destinations if path==tokenizer_root)
assert dict(plan.restore)[root_key]==tokenizer_root
applied=next(value for key,_,value in plan.metadata if key==root_key)
assert (info.st_mode&0o777,info.st_mtime_ns)==(applied.mode,applied.mtime_ns)
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'tokenizer_root_inode':info.st_ino}))
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
selected=json.loads((fixture/'restored.json').read_text());select_profile(selected['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.run_log_format import iter_records
from tldw_chatbook.Utils.custom_tokenizers import get_tokenizer_manager
async def main():
 app=TldwCli();runs=None
 try:
  receipt=json.loads((fixture/'capture.json').read_text());plan=json.loads((fixture/'plan.json').read_text());paths=dict(plan['restore'])
  runs=AgentRunsDB(Path(app.chachanotes_db.db_path).parent/'agent_runs.db')
  restored_run=runs.get_run(receipt['run_id'])
  assert restored_run==receipt['run']
  assert restored_run['status']=='done' and restored_run['result']=='captured terminal answer'
  for key,row in receipt['artifacts'].items():
   assert Path(paths[key]).read_bytes().hex()==row['hex']
  logs=[Path(paths[key]).read_bytes() for key,row in receipt['artifacts'].items() if row['owner']=='agents.history']
  assert any(record.run_id==receipt['run_id'] and 'captured terminal answer' in record.content for body in logs for record in iter_records(body))
  tokenizer=get_tokenizer_manager();root=Path(tokenizer.tokenizers_dir)
  assert root==Path.home()/'.config'/'tldw_cli'/'tokenizers'
  assert 'retained-artifact' in tokenizer.list_available_tokenizers() and tokenizer.has_tokenizers()
  assert json.loads((root/'mappings.json').read_text())['unexecuted-retained-model']=='retained-artifact'
  assert {p.name:p.read_bytes().hex() for p in root.iterdir() if p.is_file()}==receipt['tokenizer_bytes']
  assert tokenizer._tokenizers=={}
  assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in json.loads((fixture/'source-hashes.json').read_text()).items())
  assert not blocked_attempts(),blocked_attempts()
  (fixture/'readback.json').write_text(json.dumps({'run_id':receipt['run_id'],'terminal':True,'tokenizer_executed':False,'source_preserved':True}))
 finally:
  if runs is not None:runs.close()
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def test_accepted_console_artifacts_restore_with_fresh_passive_reads(tmp_path):
    root = tmp_path.resolve()
    for name in ("home", "config", "data", "cache", "tmp", "profile", "restore-home"):
        (root / name).mkdir(mode=0o700)
    profile = root / "profile"
    (profile / "custom").mkdir(mode=0o700)
    (profile / "data").mkdir(mode=0o700)
    selector = profile / "config.toml"
    selector.write_text(
        '[general]\nusers_name="default_user"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[AppRAGSearchConfig.rag.indexing]\nenabled=false\n'
        + f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
        + "".join(
            f"{key}={json.dumps(str(profile / 'custom' / leaf))}\n"
            for key, leaf in [
                ("chachanotes_db_path", "notes.db"),
                ("media_db_path", "media.db"),
                ("research_db_path", "research.db"),
                ("prompts_db_path", "prompts.db"),
            ]
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
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        TLDW_CONFIG_PATH=str(selector),
        CONSOLE_ARTIFACT_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    _run_profile_child(root, "capture", _CAPTURE, environment, timeout=120)
    receipt = json.loads((root / "capture.json").read_text())
    (root / "source-hashes.json").write_text(
        json.dumps(
            {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in receipt["source_paths"]
            }
        )
    )
    (root / "destinations").mkdir(mode=0o700)
    # Explicitly prepare the exact private empty ordinary owner root.
    tokenizer_root = root / "restore-home" / ".config" / "tldw_cli" / "tokenizers"
    tokenizer_root.mkdir(parents=True, mode=0o700)
    info = tokenizer_root.stat()
    (root / "tokenizer-root-before.json").write_text(
        json.dumps([info.st_dev, info.st_ino])
    )
    fresh = dict(
        environment,
        HOME=str(root / "restore-home"),
        USERPROFILE=str(root / "restore-home"),
    )
    _run_profile_child(root, "restore", _RESTORE, fresh)
    _run_profile_child(root, "read", _READ, fresh)
