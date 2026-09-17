"""Actual live source coverage and private capture; no synthetic settlement owners."""

import pytest

from Tests.TTS.test_loose_voice_finite_operations import _child


def test_registry_preparation_does_not_request_runtime_retirement(tmp_path):
    from tldw_chatbook.Backup_Recovery.admission import Admission, fcntl

    source = tmp_path / "source"
    source.write_bytes(b"retained")
    authority = Admission(tmp_path / "control")
    authority.register("live", (source,))
    with (
        authority.normal(("live",)),
        authority._directory() as parent,
        authority._lock(parent, "registry.lock", fcntl.LOCK_EX),
    ):
        assert not authority.pause_requested(("live",))


def test_known_constructor_selected_voice_root_requires_declaration(tmp_path):
    _child(
        tmp_path,
        r"""
import gc
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
manager=ChatterboxVoiceManager(root)
assert manager.create_profile('saved',str(source))[0]
del manager
gc.collect()
preview=preview_capture((config,),options={'allow_partial':True})
assert any(item.owner=='tts.voices' and item.path==root
           and item.status=='unsupported'
           and ':uncovered_live_source-' in item.logical_id
           for item in preview.items), 'known selected persistent root was silently omitted'
config.write_text('[general]\nusers_name="default_user"\n[app_tts]\nCHATTERBOX_VOICE_DIR='+json.dumps(str(root))+'\n')
preview=preview_capture((config,),options={'allow_partial':True})
assert any(item.owner=='tts.voices' and item.path==root/'chatterbox_profiles.json'
           and item.status=='included' for item in preview.items)
assert not any(item.owner=='tts.voices' and item.status=='unsupported' for item in preview.items)
settled()
""",
    )


_LIVE = r"""
import asyncio, json, os, sqlite3, sys, threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice','pyaudio'): sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home=Path.home(); config=Path(os.environ['TLDW_CONFIG_PATH'])
selected=home/'outside-profile-voices'
config.write_text('[general]\nusers_name="default_user"\n[app_tts]\nCHATTERBOX_VOICE_DIR='+json.dumps(str(selected))+'\n')
config.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
from Tests.TTS.test_profile_reference_repository import _canonical, _audio_cpp_draft, _requirement

async def main():
 app=TldwCli()
 # Populate an initialized profile through actual owners. These dependencies
 # remain explicit; this fixture does not change fresh-profile missing policy.
 from tldw_chatbook.config import get_user_data_dir
 from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
 from tldw_chatbook.Agents.agent_models import AgentDefinition
 from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
 from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
 from hashlib import sha256
 data=get_user_data_dir()
 runs=AgentRunsDB(data/'agent_runs.db',client_id='capture-fixture')
 runs.create_agent_definition(AgentDefinition(name='research',description='saved',instructions='Read the local notes.'))
 runs.close()
 note=b'# Saved note\nPrivate capture fixture.\n'
 replica=FileNotesReplica(data/'file_notes.sqlite')
 replica.upsert_file(str(home/'notes'),'saved.md',note,content_hash=sha256(note).hexdigest(),decoded_text=note.decode(),size=len(note),mtime_ns=1)
 replica.close()
 skills=LocalSkillsService(store_dir=data/'skills')
 await skills.create_skill(name='saved-skill',content='---\nname: saved-skill\ndescription: Saved research instructions\n---\nRead the local notes.\n')
 reference=_canonical(sample=7,frames=800)
 source=home/'reference.wav'; source.write_bytes(reference.wav_bytes); source.chmod(0o600)
 manager=ChatterboxVoiceManager(selected)
 assert manager.create_profile('configured',str(source))[0]
 shared=home/'.config'/'tldw_cli'/'higgs_voices'
 shared_manager=HiggsVoiceProfileManager(shared)
 assert shared_manager.create_profile('shared',str(source))[0]
 repo=await app._ensure_tts_profile_repository(); assert repo is not None
 created=await repo.create_profile(_audio_cpp_draft('private reference'))
 attached=await repo.set_reference(created.value.profile_id,reference,_requirement(),
    expected_revision=created.value.revision,expected_generation=created.generation)
 voice_metadata={}
 for voice_root,voice_name in ((selected,'configured'),(shared,'shared')):
  catalog=voice_root/('chatterbox_profiles.json' if voice_root==selected else 'voice_profiles.json')
  audio=Path(json.loads(catalog.read_text())[voice_name]['reference_audio'])
  for path in (catalog,audio):
   info=path.stat();voice_metadata[path]=(info.st_mode & 0o777,info.st_mtime_ns)
 options={'allow_partial':True,'staging_parent':home}
 preview=preview_capture((config,),options=options)
 # Keep the fresh application's genuine missing/unavailable sources visible.
 assert not preview.complete
 assert not set(preview.issues)-{'unsupported','unavailable','missing_required','unsupported_owner'}, preview.issues
 destination=home/'snapshot.tldw-backup.zip'
 monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event()
 watchdog=asyncio.get_running_loop().call_later(20,cancel.set)
 try:
  result=await asyncio.to_thread(capture,(config,),preview.scope_digest,destination,
     options=options,cancel=cancel)
  for _ in range(500):
   if storage._pause is None and app._backup_runtime_maintenance is None: break
   await asyncio.sleep(.01)
  assert storage._pause is None and app._backup_runtime_maintenance is None
  # Ordinary writers run successfully before packaging any archive bytes.
  assert manager.create_profile('after_capture',str(source))[0]
  resumed=await repo.create_profile(_audio_cpp_draft('after capture'))
  assert resumed.value.display_name=='after capture'
  assert app.chachanotes_db.get_connection().execute('SELECT 1').fetchone()[0]==1
  manifest=json.loads(result.manifest_bytes)
  assert manifest['consistency']=='partial'
  paths={item.logical_id:item.path for item in result.inventory.items}
  files={paths[f['logical_id']]:result.root/f['payload'] for f in manifest['files']}
  for member in manifest['files']:
   path=paths[member['logical_id']]
   if path in voice_metadata:
    assert (member['metadata']['mode'],member['metadata']['mtime_ns'])==voice_metadata[path]
  for root,name in ((selected,'configured'),(shared,'shared')):
   catalog=root/('chatterbox_profiles.json' if root==selected else 'voice_profiles.json')
   saved=json.loads(files[catalog].read_text())
   audio=Path(saved[name]['reference_audio'])
   assert files[audio].read_bytes()==reference.wav_bytes
   assert 'after_capture' not in saved
  captured_profile=next(result.root/f['payload'] for f in manifest['files'] if f['owner_id']=='tts.profile_store')
  with closing(sqlite3.connect(captured_profile)) as connection:
   assert connection.execute('SELECT wav_bytes,reference_text,sha256 FROM tts_profile_clone_references').fetchone()==(reference.wav_bytes,reference.reference_text,reference.sha256)
   assert connection.execute('SELECT display_name FROM tts_generation_profiles').fetchall()==[('private reference',)]
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination and destination.is_file()
  assert not blocked_attempts()
 finally:
  watchdog.cancel(); cancel.set(); monitoring.cancel()
  try: await monitoring
  except asyncio.CancelledError: pass
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
asyncio.run(main())
print('retired and reopened')
"""


def test_actual_app_captures_voice_and_reference_bytes_before_packaging(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "capture", "ordinary", script=_LIVE)


@pytest.mark.parametrize("preexisting", [False, True])
def test_controller_worker_retires_owned_handles_after_cancelled_waiter(
    tmp_path, preexisting
):
    from Tests.Backup_Recovery.test_activation_agents import _BRIDGE_WORKER
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = (
        _BRIDGE_WORKER.split("thread=threading.Thread")[0]
        + r"""
import asyncio,time
from concurrent.futures import ThreadPoolExecutor
from Tests.Chat.test_console_prompt_queue_coordinator import _arm_controller, SequencedGateway
async def main():
 loop=asyncio.get_running_loop()
 loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
 controller,_,_=_arm_controller(SequencedGateway())
 controller._agent_bridge=bridge
 entered,release=threading.Event(),threading.Event()
 if PREEXISTING:
  def seed():
   bridge_db.list_runs('conv-1')
   with registry.db.connection() as connection: connection.execute('SELECT 1')
  await asyncio.to_thread(seed)
 def actual_reply():
  reply()
  entered.set(); assert release.wait(5)
 waiter=asyncio.create_task(controller._run_maintenance_agent_call(actual_reply))
 while not entered.is_set():
  if waiter.done(): waiter.result()
  await asyncio.sleep(.005)
 waiter.cancel()
 try: await waiter
 except asyncio.CancelledError: pass
 controller.maintenance_close_admission()
 assert not await controller.maintenance_drain(time.monotonic()+.02)
 release.set()
 assert await controller.maintenance_drain(time.monotonic()+3)
 assert outcomes[0].status==RUN_DONE
 def check_worker():
  for connection in worker_connections:
   if PREEXISTING:
    assert connection.execute('SELECT 1').fetchone()[0]==1
   else:
    try: sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError as exc: assert 'closed' in str(exc)
    else: raise AssertionError('controller worker retained owned native connection')
  if PREEXISTING:
   bridge_db.close();registry.db.close()
 await asyncio.to_thread(check_worker)
 for connection in caller_connections: assert connection.execute('SELECT 1').fetchone()[0]==1
 assert bridge_db.list_runs('conv-1')[0]['status']==RUN_DONE
 bridge_db.close();registry.db.close()
 pause=storage._begin_local_pause()
 try: assert pause.drain(time.monotonic()+1)
 finally: pause.resume()
 controller.maintenance_resume()
 assert not blocked_attempts()
asyncio.run(main())
print('retired and reopened')
"""
    )
    script = script.replace("PREEXISTING", repr(preexisting))
    _run(tmp_path, "controller", "approved", script=script)
