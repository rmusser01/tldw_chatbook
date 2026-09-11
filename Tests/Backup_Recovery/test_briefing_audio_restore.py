"""Native briefing audio keeps its database reference through isolated restore."""

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
fixture=Path(os.environ['BRIEFING_RESTORE_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_SEED = (
    _PRIVATE
    + r"""
from io import BytesIO
import wave
from tldw_chatbook.app import TldwCli
from tldw_chatbook.TTS.profile_types import TTSProfileDraft
from tldw_chatbook.Subscriptions.briefing_audio import generate_script_audio,audio_file_path_is_safe
async def main():
 app=TldwCli()
 try:
  repository=await app._ensure_tts_profile_repository()
  assert repository is not None
  profile=(await repository.create_profile(TTSProfileDraft(display_name='Retained briefing voice',provider_id='openai',model_id='tts-1',voice_id='alloy',response_format='wav',speed=1.0,options={}))).value
  profile_service=await app._ensure_tts_profile_service()
  db=app.subscriptions_db
  watchlist=app.watchlist_bundle_service.create(name='Retained briefing')['id']
  briefing=db.insert_briefing(watchlist)
  roster=[{'name':'Host','voice_profile_id':str(profile.profile_id)}]
  turns=[{'speaker':'Host','text':'Retained local briefing.'}]
  script=db.insert_briefing_script(briefing,preset_id=None,preset_name='Single voice',roster_snapshot_json=json.dumps(roster))
  db.update_briefing_script(script,status='complete',turns_json=json.dumps(turns))
  calls=[]
  async def synthesize(service,selection,text,*,turn_index):
   assert service is app.tts_service and selection.profile_id==str(profile.profile_id)
   assert text==turns[turn_index]['text'] and selection.profile_revision==profile.revision
   calls.append(turn_index)
   output=BytesIO()
   with wave.open(output,'wb') as audio:
    audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(22050)
    audio.writeframes(b'\x00\x00'*2205)
   return output.getvalue()
  row=await generate_script_audio(db,script,tts_service=app.tts_service,profile_service=profile_service,synthesize=synthesize)
  assert row['status']=='complete',row
  assert calls==[0] and audio_file_path_is_safe(row['file_path'])
  payload=Path(row['file_path']).read_bytes()
  assert payload[:4]==b'RIFF'
  failed=db.create_briefing_audio(script,voice_snapshot_json='[]')
  db.update_briefing_audio(failed,status='failed',error='Retained unsuccessful generation',file_path='/historical/unavailable.wav')
  seed={'audio':row,'failed':db.get_briefing_audio(failed),'script':db.get_briefing_script(script),'profile':str(profile.profile_id),'profile_revision':profile.revision,'hex':payload.hex(),'sha256':hashlib.sha256(payload).hexdigest(),'config_hex':selector.read_bytes().hex()}
  (fixture/'seed.json').write_text(json.dumps(seed,indent=2))
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
  seed=json.loads((fixture/'seed.json').read_text())
  source_root=get_user_data_dir()
  options={'staging_parent':fixture}
  preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues,'blocking':[{'owner':item.owner,'logical_id':item.logical_id,'path':str(item.path),'status':item.status} for item in preview.items if item.owner=='unknown' or item.status in {'unsupported','unavailable','missing_required'}]},indent=2))
  assert preview.complete,preview.issues
  destination=fixture/'briefing.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes)
  source={item.logical_id:item for item in captured.inventory.items}
  assets=[row for row in manifest['files'] if row['owner_id']=='subscriptions.assets']
  assert len(assets)==1 and source[assets[0]['logical_id']].path==Path(seed['audio']['file_path'])
  assert (captured.root/assets[0]['payload']).read_bytes().hex()==seed['hex']
  assert {'db.subscriptions','tts.profile_store'}<={row['owner_id'] for row in manifest['files']}
  await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  mapping={};selected=fixture/'restore-destinations'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id']
   member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'recovered-briefing'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0]
  mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'mapping':mapping,'source_profile':profile,'asset':assets[0]['logical_id']},indent=2))
  assert Path(seed['audio']['file_path']).read_bytes().hex()==seed['hex']
  assert selector.read_bytes().hex()==seed['config_hex']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel()
  await asyncio.gather(monitoring,return_exceptions=True)
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
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
receipt=json.loads((fixture/'restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
assert verify_sealed(archive).consistency=='coherent'
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'recovered-briefing'})
try:
 profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 (fixture/'restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str))
 raise
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'asset':str(dict(plan.restore)[receipt['asset']])}))
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from uuid import UUID
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
restored=json.loads((fixture/'restored.json').read_text())
source_absent=os.environ.get('BRIEFING_SOURCE_ABSENT')=='1'
select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Subscriptions.briefing_audio import audio_file_path_is_safe
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'seed.json').read_text())
  row=app.subscriptions_db.get_briefing_audio(seed['audio']['id'])
  assert row['status']=='complete' and row['script_id']==seed['script']['id']
  assert {key:value for key,value in row.items() if key!='file_path'}=={key:value for key,value in seed['audio'].items() if key!='file_path'}
  assert app.subscriptions_db.get_briefing_audio(seed['failed']['id'])==seed['failed']
  assert row['file_path']==restored['asset'] and row['file_path']!=seed['audio']['file_path']
  assert audio_file_path_is_safe(row['file_path'])
  assert Path(row['file_path']).read_bytes().hex()==seed['hex']
  assert app.subscriptions_db.get_briefing_script(seed['script']['id'])==seed['script']
  repository=await app._ensure_tts_profile_repository()
  profile=(await repository.get_profile(UUID(seed['profile']))).value
  assert profile.revision==seed['profile_revision'] and profile.voice_id=='alloy'
  if source_absent:assert not Path(seed['audio']['file_path']).exists()
  else:assert Path(seed['audio']['file_path']).read_bytes().hex()==seed['hex']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def _captured_briefing(tmp_path):
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
        BRIEFING_RESTORE_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    dependency_path = os.environ.get("BRIEFING_TEST_DEPENDENCIES")
    if dependency_path:
        environment["PYTHONPATH"] += os.pathsep + dependency_path
    _run_profile_child(root, "seed", _SEED, environment)
    _run_profile_child(root, "capture", _CAPTURE, environment)
    return root, environment


def test_native_briefing_audio_restores_with_fresh_passive_reference(tmp_path):
    root, environment = _captured_briefing(tmp_path)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "restore", _RESTORE, restored_environment)
    _run_profile_child(root, "read", _READ, restored_environment)
    seed = json.loads((root / "seed.json").read_text())
    source = Path(seed["audio"]["file_path"])
    held = root / "source-audio-held.wav"
    source.rename(held)
    try:
        _run_profile_child(
            root,
            "read-source-absent",
            _READ,
            dict(restored_environment, BRIEFING_SOURCE_ABSENT="1"),
        )
    finally:
        held.rename(source)
    assert source.read_bytes().hex() == seed["hex"]
