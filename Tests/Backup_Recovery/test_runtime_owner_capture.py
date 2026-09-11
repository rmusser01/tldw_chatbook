"""Public live capture of installed finite owners and ordinary postcapture use."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_MODEL = r"""
import asyncio,json,os,sys,threading
from dataclasses import replace
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n');selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture,preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Model_Artifacts import ArtifactRef,ArtifactRole
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService
from tldw_chatbook.Model_Artifacts.recovery import managed_artifact_root
from tldw_chatbook.config import get_user_data_dir
from Tests.Model_Artifacts.test_provision_install import _descriptor
async def main():
 app=TldwCli()
 service=ModelArtifactService(managed_artifact_root(get_user_data_dir()))
 dependency=ArtifactRef('support','revision','fp32');root=ArtifactRef('primary','revision','fp32')
 descriptors=[replace(_descriptor(dependency,role=ArtifactRole.DEPENDENCY,files_body=b'dependency model bytes'),model_id='fixture/dependency'),replace(_descriptor(root,dependencies=(dependency,),files_body=b'primary model bytes'),model_id='fixture/primary')]
 originals={}
 for descriptor,body in zip(descriptors,(b'dependency model bytes',b'primary model bytes')):
  source=home/descriptor.reference.artifact_id;source.mkdir(mode=0o700)
  (source/'model.bin').write_bytes(body)
  service.install(descriptor,source)
  directory=service.artifact_path(descriptor.reference)
  originals[directory/'model.bin']=body
  originals[directory/'manifest.json']=(directory/'manifest.json').read_bytes()
 options={'staging_parent':home,'model_ids':('fixture/primary',)}
 preview=preview_capture((selector,),options=options)
 bad=[(i.owner,i.logical_id,i.status) for i in preview.items if i.status in ('unsupported','unavailable','missing_required')]
 assert preview.complete,(preview.issues,bad)
 # This exact native model handle remains live until the real app monitor pauses.
 handle=service.acquire_installed_root(root)
 monitoring=asyncio.create_task(monitor_app(app))
 async def finish_reader():
  for _ in range(500):
   if storage._pause is not None:break
   await asyncio.sleep(.01)
  assert storage._pause is not None
  await asyncio.sleep(.05)
  assert not capturing.done()
  handle.close()
 finishing=asyncio.create_task(finish_reader())
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(60,cancel.set)
 destination=home/'models.tldw-backup.zip'
 try:
  capturing=asyncio.create_task(asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel))
  result=await capturing
  await finishing
  for _ in range(500):
   if storage._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage._pause is None and app._backup_runtime_maintenance is None
  # Ordinary mutation and native handle acquisition both resume before packaging.
  later=replace(_descriptor(ArtifactRef('later','revision','fp32'),files_body=b'later'),model_id='fixture/later')
  source=home/'later';source.mkdir();(source/'model.bin').write_bytes(b'later')
  service.install(later,source)
  with service.acquire_installed_root(root):pass
  manifest=json.loads(result.manifest_bytes)
  assert result.inventory.complete and manifest['consistency']=='coherent'
  paths={item.logical_id:item.path for item in result.inventory.items}
  captured={paths[row['logical_id']]:result.root/row['payload'] for row in manifest['files'] if row['owner_id']=='models.artifacts'}
  assert all(captured[path].read_bytes()==body for path,body in originals.items())
  assert not any('later' in str(path) for path in captured)
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination and destination.is_file()
  assert not blocked_attempts()
 except BaseException:
  print('MODEL CAPTURE STATE',repr(getattr(app,'_backup_maintenance_error',None)),repr(storage._pause),repr(app._backup_runtime_maintenance), 'reader', finishing.done(), repr(finishing.exception()) if finishing.done() and not finishing.cancelled() else None,flush=True)
  raise
 finally:
  handle.close();watchdog.cancel();cancel.set();finishing.cancel();monitoring.cancel()
  await asyncio.gather(finishing,monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
asyncio.run(main())
print('retired and reopened')
"""


def test_actual_model_dependency_capture_and_native_handle_resume(tmp_path):
    _run(tmp_path, "models", "capture", script=_MODEL, timeout=120)


_LOGS = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
root=get_user_data_dir();logs=root/'tool_sandbox'/'.agent-runs';logs.mkdir(parents=True)
(logs/'saved.log').write_bytes(b'accepted terminal log bytes')
runs=AgentRunsDB(root/'agent_runs.db','fixture');runs.close()
workspaces=WorkspaceDB(root/'tldw_chatbook_workspaces.db','fixture');workspaces.close()
route=sys.argv[1]
if route=='unknown':(logs.parent/'unowned.txt').write_text('unclassified')
if route=='file':
 import shutil
 shutil.rmtree(logs.parent);logs.parent.write_bytes(b'not a directory')
if route=='alias':
 logs.rename(root/'external-logs');logs.symlink_to(root/'external-logs',target_is_directory=True)
p=preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True})
items=[i for i in p.items if i.owner=='agents.history']
if route=='logs':
 assert not any(i.owner=='unknown' and i.path==logs.parent for i in p.items)
 assert 'overlapping_owner_roots' not in p.issues,p.issues
 assert any(i.path==logs.parent and i.status=='included_directory' for i in items)
 assert any(i.path==logs/'saved.log' and i.status=='included' for i in items)
else:
 assert not p.complete
 assert any(i.status in ('unsupported','unavailable') for i in items),items
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["logs", "unknown", "alias", "file"])
def test_default_run_log_container_has_declared_topology(tmp_path, route):
    _run(tmp_path, route, "capture", script=_LOGS)


_CONSOLE = r"""
import asyncio,json,os,sqlite3,sys,threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH']);voices=home/'voices'
selector.write_text('[general]\nusers_name="default_user"\n[app_tts]\nCHATTERBOX_VOICE_DIR='+json.dumps(str(voices))+'\n');selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture,preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from Tests.TTS.test_profile_reference_repository import _canonical
from Tests.Chat.test_console_agent_bridge import _ChunkGateway,_test_resolution
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
async def main():
 app=TldwCli();manager=ChatterboxVoiceManager(voices)
 wav=_canonical(sample=9,frames=800).wav_bytes;source=home/'reference.wav';source.write_bytes(wav)
 assert manager.create_profile('captured',str(source))[0]
 entered,release=threading.Event(),threading.Event()
 class Gateway(_ChunkGateway):
  async def stream_chat(self,*args,**kwargs):
   entered.set()
   while not release.is_set():await asyncio.sleep(.005)
   self.fixture_turns=getattr(self,'fixture_turns',0)+1
   assert manager.update_profile('captured',description='warmup asset' if self.fixture_turns==1 else 'accepted operation asset')[0]
   async for chunk in super().stream_chat(*args,**kwargs):yield chunk
 gateway=Gateway([['warmup answer'],['captured terminal answer']])
 runtime=app.console_runtime;store=runtime.ensure_chat_store()
 bridge=runtime.ensure_agent_bridge(store_factory=lambda:store,provider_gateway_factory=lambda:gateway)
 controller=runtime.ensure_chat_controller(store=store,provider_gateway=gateway,agent_bridge=bridge)
 session=store.ensure_session(title='Captured Console')
 store.append_message(session.id,role=ConsoleMessageRole.USER,content='saved console question')
 assistant=store.append_message(session.id,role=ConsoleMessageRole.ASSISTANT,content='')
 # An ordinary completed turn initializes the actual lazy tokenizer/source roots.
 # New owner roots appearing after review still correctly require a fresh preview.
 release.set()
 await controller._run_maintenance_agent_call(bridge.run_reply,conversation_id='warm-console',session_id=session.id,resolution=_test_resolution(),assistant_message_id=assistant.id,model='test-model',session_system_prompt='',agent_messages=[{'role':'user','content':'warmup question'}],should_cancel=lambda:False)
 release.clear();entered.clear()
 assistant=store.append_message(session.id,role=ConsoleMessageRole.ASSISTANT,content='')
 accepted=asyncio.create_task(controller._run_maintenance_agent_call(bridge.run_reply,conversation_id='captured-console',session_id=session.id,resolution=_test_resolution(),assistant_message_id=assistant.id,model='test-model',session_system_prompt='',agent_messages=[{'role':'user','content':'saved console question'}],should_cancel=lambda:False))
 monitoring=None;finishing=None;watchdog=None;cancel=threading.Event()
 try:
  # Native run/tool preparation precedes the provider and may take several
  # seconds on the full app. Bound preparation separately from capture.
  preparation_deadline=asyncio.get_running_loop().time()+15
  while not entered.is_set():
   if accepted.done():raise AssertionError(('returned before provider',accepted.result()))
   assert asyncio.get_running_loop().time()<preparation_deadline,'provider preparation timeout'
   await asyncio.sleep(.01)
  options={'staging_parent':home}
  preview=preview_capture((selector,),options=options)
  assert preview.complete,(preview.issues,[(i.owner,str(i.path),i.status) for i in preview.items if i.status in ('unsupported','unavailable','missing_required')])
  monitoring=asyncio.create_task(monitor_app(app))
  async def finish():
   for _ in range(500):
    if controller._maintenance_paused:break
    await asyncio.sleep(.01)
   assert controller._maintenance_paused and not accepted.done()
   release.set()
  finishing=asyncio.create_task(finish())
  watchdog=asyncio.get_running_loop().call_later(60,cancel.set)
  destination=home/'console.tldw-backup.zip'
  result=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  run_id,outcome=await accepted;await finishing
  assert outcome.final_text=='captured terminal answer'
  for _ in range(500):
   if storage._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage._pause is None and app._backup_runtime_maintenance is None
  # Both ordinary source families resume before archive construction.
  assert manager.update_profile('captured',description='after capture')[0]
  store.append_message(session.id,role=ConsoleMessageRole.USER,content='after capture')
  manifest=json.loads(result.manifest_bytes)
  assert result.inventory.complete and manifest['consistency']=='coherent'
  paths={item.logical_id:item.path for item in result.inventory.items}
  files={paths[row['logical_id']]:result.root/row['payload'] for row in manifest['files']}
  catalog=json.loads(files[voices/'chatterbox_profiles.json'].read_text())
  assert catalog['captured']['description']=='accepted operation asset'
  assert files[Path(catalog['captured']['reference_audio'])].read_bytes()==wav
  member=next(row for row in manifest['files'] if row['owner_id']=='db.agent_runs')
  with closing(sqlite3.connect(result.root/member['payload'])) as connection:
   assert connection.execute('SELECT status,result FROM agent_runs WHERE id=?',(run_id,)).fetchone()==('done','captured terminal answer')
  logs=[result.root/row['payload'] for row in manifest['files'] if row['owner_id']=='agents.history']
  assert logs and any(b'captured terminal answer' in path.read_bytes() for path in logs)
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination and destination.is_file()
  assert not blocked_attempts()
 finally:
  release.set()
  await accepted
  if watchdog:watchdog.cancel()
  cancel.set()
  tasks=[task for task in (finishing,monitoring) if task is not None]
  for task in tasks:task.cancel()
  await asyncio.gather(*tasks,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
asyncio.run(main())
print('retired and reopened')
"""


def test_actual_console_terminal_and_asset_capture_resume(tmp_path):
    _run(tmp_path, "console", "capture", script=_CONSOLE, timeout=120)
