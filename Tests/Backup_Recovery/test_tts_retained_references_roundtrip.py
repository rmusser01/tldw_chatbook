"""Retained TTS references survive Complete archive and isolated publication.

These cases qualify passive persistence only.  They do not install a model,
approve a deferred loose-voice owner, or attempt synthesis.
"""

import json
import os
from pathlib import Path

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _CHILD_ENVIRONMENT_KEYS,
    _isolated_environment,
    _run_profile_child,
)

_PRIVATE = r"""
import asyncio,hashlib,json,os,stat,sys,threading,wave
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['TTS_RETAINED_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
def identity(path):
 info=path.stat(follow_symlinks=False)
 return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tree(root):
 return {
  'files':{str(path.relative_to(root)):{'sha256':digest(path),'mode':stat.S_IMODE(path.stat().st_mode),'mtime_ns':path.stat().st_mtime_ns} for path in root.rglob('*') if path.is_file()},
  'directories':{str(path.relative_to(root)):{'mode':stat.S_IMODE(path.stat().st_mode),'mtime_ns':path.stat().st_mtime_ns} for path in (root,*root.rglob('*')) if path.is_dir()},
 }
"""


_BLOB_SEED = (
    _PRIVATE
    + r"""
from uuid import uuid4
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_tts_profiles_db_path,get_user_data_dir
from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY
from tldw_chatbook.TTS.profile_reference_audio import canonicalize_reference_wav
from tldw_chatbook.TTS.profile_reference_types import TTSCloneRecipeRequirement
from tldw_chatbook.TTS.profile_types import TTSProfileDraft
def projection(profile,reference):
 summary=profile.reference
 requirement=reference.recipe_requirement
 return {
  'profile_id':str(profile.profile_id),'display_name':profile.display_name,'normalized_name':profile.normalized_name,
  'provider_id':profile.provider_id,'model_id':profile.model_id,'voice_id':profile.voice_id,
  'response_format':profile.response_format,'speed':profile.speed,'options':dict(profile.options),
  'revision':profile.revision,'created_at':profile.created_at.isoformat(),'updated_at':profile.updated_at.isoformat(),
  'reference_id':str(summary.reference_id),'reference_created_at':summary.created_at.isoformat(),'reference_updated_at':summary.updated_at.isoformat(),
  'byte_length':summary.byte_length,'duration_ms':summary.duration_ms,'sample_rate_hz':summary.sample_rate_hz,
  'channels':summary.channels,'sample_encoding':summary.sample_encoding,'reference_text':reference.reference_text,
  'sha256':reference.sha256,'wav_hex':reference.wav_bytes.hex(),
  'recipe_requirement':{'recipe_id':requirement.recipe_id,'recipe_revision':requirement.recipe_revision,'model_id':requirement.model_id},
 }
async def main():
 app=TldwCli()
 try:
  # This fixture explicitly selects a Research path; initialize its lazy store.
  assert app.local_research_service.list_sessions()==[]
  source=fixture/'canonical-source.wav'
  with wave.open(str(source),'wb') as output:
   output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000);output.writeframes(b'\x17\x00'*1600)
  source.chmod(0o600)
  with wave.open(str(source),'rb') as parsed:
   assert (parsed.getnchannels(),parsed.getsampwidth(),parsed.getframerate(),parsed.getnframes())==(1,2,16000,1600)
  canonical=canonicalize_reference_wav(source,'A private synthetic reference.')
  recipe=AUDIO_CPP_RECIPE_REGISTRY.for_package('pocket_tts_english_q8_0')
  requirement=TTSCloneRecipeRequirement(recipe_id=recipe.recipe_id,recipe_revision=recipe.recipe_revision,model_id=recipe.default_public_model_id)
  draft=TTSProfileDraft(display_name='Retained reference',provider_id='audio_cpp',model_id=recipe.default_public_model_id,voice_id=None,response_format='wav',speed=1.0,options={})
  repository=await app._ensure_tts_profile_repository();assert repository is not None
  created=await repository.create_profile_with_reference(draft,uuid4(),canonical,requirement,expected_generation=repository.generation)
  exact=await repository.get_reference(created.value.profile_id,expected_revision=created.value.revision,expected_generation=repository.generation)
  database=get_tts_profiles_db_path();assert database==selector.parent/'custom'/'tts-profiles.db'
  seed={'source_wav':str(source),'source_wav_sha256':digest(source),'source_wav_identity':identity(source),
        'database':str(database),'database_sha256':digest(database),'database_identity':identity(database),
        'selector':str(selector),'config_hex':selector.read_bytes().hex(),'user_root':str(get_user_data_dir()),
        'recipe':{'recipe_id':recipe.recipe_id,'recipe_revision':recipe.recipe_revision,'model_id':recipe.default_public_model_id},
        'profile':projection(created.value,exact.value),'source_generation':repository.generation}
  (fixture/'blob-seed.json').write_text(json.dumps(seed,indent=2))
  assert hashlib.sha256(bytes.fromhex(seed['profile']['wav_hex'])).hexdigest()==seed['profile']['sha256']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_BLOB_CAPTURE = (
    _PRIVATE
    + r"""
import sqlite3
from contextlib import closing
from uuid import UUID
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seed=json.loads((fixture/'blob-seed.json').read_text());database=Path(seed['database'])
  assert selector.read_bytes().hex()==seed['config_hex']
  assert digest(database)==seed['database_sha256'] and identity(database)==seed['database_identity']
  assert digest(Path(seed['source_wav']))==seed['source_wav_sha256'] and identity(Path(seed['source_wav']))==seed['source_wav_identity']
  repository=await app._ensure_tts_profile_repository();profile=await repository.get_profile(UUID(seed['profile']['profile_id']))
  assert profile.value.revision==seed['profile']['revision']
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'blob-preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues},indent=2))
  assert preview.complete,(preview.issues,[(item.owner,item.status,str(item.path)) for item in preview.items if item.status in {'missing_required','unsupported','unavailable'}])
  destination=fixture/'tts-blob.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent'
  source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id'] in {'tts.profile_store','tts.references'}]
  assert len(rows)==2 and {row['owner_id'] for row in rows}=={'tts.profile_store','tts.references'}
  assert all(source[row['logical_id']].path==database for row in rows)
  groups={source[row['logical_id']].shared_group for row in rows};assert len(groups)==1 and None not in groups
  dependencies={row['group_id']:set(row['members']) for row in manifest['dependency_groups']}
  reference=next(row for row in rows if row['owner_id']=='tts.references');store=next(row for row in rows if row['owner_id']=='tts.profile_store')
  assert store['logical_id'] in source[reference['logical_id']].dependencies
  assert any({reference['logical_id'],store['logical_id']}<=members for members in dependencies.values())
  payloads={}
  wanted=seed['profile']
  for row in rows:
   payload=captured.root/row['payload'];payload_hash=digest(payload);assert payload_hash==row['sha256']
   with closing(sqlite3.connect(payload.as_uri()+'?mode=ro',uri=True)) as connection:
    profile_row=connection.execute('SELECT profile_id,display_name,normalized_name,provider_id,model_id,voice_id,response_format,speed,options_json,revision FROM tts_generation_profiles').fetchone()
    reference_row=connection.execute('SELECT reference_id,wav_bytes,reference_text,sha256,byte_length,duration_ms,sample_rate_hz,channels,sample_encoding,recipe_id,recipe_revision FROM tts_profile_clone_references').fetchone()
   assert profile_row==(wanted['profile_id'],wanted['display_name'],wanted['normalized_name'],wanted['provider_id'],wanted['model_id'],wanted['voice_id'],wanted['response_format'],wanted['speed'],'{}',wanted['revision'])
   assert reference_row==(wanted['reference_id'],bytes.fromhex(wanted['wav_hex']),wanted['reference_text'],wanted['sha256'],wanted['byte_length'],wanted['duration_ms'],wanted['sample_rate_hz'],wanted['channels'],wanted['sample_encoding'],wanted['recipe_requirement']['recipe_id'],wanted['recipe_requirement']['recipe_revision'])
   payloads[row['owner_id']]={'logical_id':row['logical_id'],'root_id':row['root_id'],'sha256':payload_hash,'size':row['size']}
  assert len({entry['sha256'] for entry in payloads.values()})==1
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  selected=fixture/'blob-restore-destinations';mapping={};shared_target=selected/'custom'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner in {'tts.profile_store','tts.references'}:target=shared_target
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==Path(seed['user_root']) or Path(seed['user_root']) in original.parents:target=selected/'data'/'restored-tts-reference'/original.relative_to(Path(seed['user_root']))
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert {mapping[entry['root_id']] for entry in payloads.values()}=={str(shared_target)}
  profile_id=manifest['profile_ids'][0];mapping['profile:'+profile_id+':paths.data_dir']=str(selected/'data')
  (fixture/'blob-capture.json').write_text(json.dumps({'archive':str(destination),'archive_sha256':written.digest,'manifest_sha256':hashlib.sha256(captured.manifest_bytes).hexdigest(),'payloads':payloads,'mapping':mapping,'source_profile':profile_id,'restored_database':str(shared_target/database.name),'shared_group':groups.pop()},indent=2))
  assert identity(database)[:2]==seed['database_identity'][:2]
  assert selector.read_bytes().hex()==seed['config_hex']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
seed=json.loads((fixture/'blob-seed.json').read_text());receipt=json.loads((fixture/'blob-capture.json').read_text());database=Path(seed['database'])
assert identity(database)[:2]==seed['database_identity'][:2]
receipt['source_post_close_sha256']=digest(database);receipt['source_post_close_identity']=identity(database)
(fixture/'blob-capture.json').write_text(json.dumps(receipt,indent=2))
"""
)


_BLOB_RESTORE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.Utils.platform_files import os as native_os
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
seed=json.loads((fixture/'blob-seed.json').read_text());receipt=json.loads((fixture/'blob-capture.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'blob-acquired',ArchiveLimits(),None,threading.Event());doc=verify_sealed(archive)
assert doc.consistency=='coherent' and archive.digest==receipt['archive_sha256']
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  payload=zipped.read(row.payload);assert len(payload)==row.size and hashlib.sha256(payload).hexdigest()==row.sha256
for owner,evidence in receipt['payloads'].items():
  row=next(row for row in doc.files if row.logical_id==evidence['logical_id']);assert row.sha256==evidence['sha256']
(fixture/'blob-restore-destinations').mkdir(mode=0o700,exist_ok=True)
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-tts-reference'})
profile=restore_isolated(archive,plan,fixture/'blob-control',threading.Event())
requirements=profile_requirements(profile,fixture/'blob-control')
assert requirements['requirements_checked'] and requirements['needs_setup']
assert {'tts.profile_store','tts.references'}<=set(requirements['pending_owners'])
restored_database=Path(receipt['restored_database']);assert restored_database.is_file() and stat.S_IMODE(native_os.stat(restored_database).st_mode)==0o600
assert digest(restored_database) in {entry['sha256'] for entry in receipt['payloads'].values()}
(fixture/'blob-restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements,'restored_database':str(restored_database)}))
assert digest(Path(seed['database']))==receipt['source_post_close_sha256'] and identity(Path(seed['database']))==receipt['source_post_close_identity']
assert Path(seed['selector']).read_bytes().hex()==seed['config_hex']
assert not blocked_attempts(),blocked_attempts()
"""
)


_BLOB_READ = (
    _PRIVATE
    + r"""
from uuid import UUID
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'blob-restored.json').read_text());select_profile(restored['profile'],fixture/'blob-control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_tts_profiles_db_path,get_user_data_dir
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'blob-seed.json').read_text());wanted=seed['profile'];database=get_tts_profiles_db_path()
  assert database==Path(restored['restored_database']) and database.is_file()
  repository=await app._ensure_tts_profile_repository();assert repository is not None
  profile=await repository.get_profile(UUID(wanted['profile_id']))
  reference=await repository.get_reference(UUID(wanted['profile_id']),expected_revision=profile.value.revision,expected_generation=repository.generation)
  assert profile.generation==reference.generation==repository.generation
  value=profile.value;exact=reference.value;summary=value.reference;requirement=exact.recipe_requirement
  assert (str(value.profile_id),value.display_name,value.normalized_name,value.provider_id,value.model_id,value.voice_id,value.response_format,value.speed,dict(value.options),value.revision)==(wanted['profile_id'],wanted['display_name'],wanted['normalized_name'],wanted['provider_id'],wanted['model_id'],wanted['voice_id'],wanted['response_format'],wanted['speed'],wanted['options'],wanted['revision'])
  assert (str(summary.reference_id),summary.byte_length,summary.duration_ms,summary.sample_rate_hz,summary.channels,summary.sample_encoding)==(wanted['reference_id'],wanted['byte_length'],wanted['duration_ms'],wanted['sample_rate_hz'],wanted['channels'],wanted['sample_encoding'])
  assert (summary.created_at.isoformat(),summary.updated_at.isoformat(),value.created_at.isoformat(),value.updated_at.isoformat())==(wanted['reference_created_at'],wanted['reference_updated_at'],wanted['created_at'],wanted['updated_at'])
  assert (exact.reference_text,exact.sha256,exact.wav_bytes.hex())==(wanted['reference_text'],wanted['sha256'],wanted['wav_hex'])
  assert {'recipe_id':requirement.recipe_id,'recipe_revision':requirement.recipe_revision,'model_id':requirement.model_id}==wanted['recipe_requirement']==seed['recipe']
  requirements=profile_requirements(restored['profile'],fixture/'blob-control')
  assert json.loads(json.dumps(requirements))==restored['requirements'] and {'tts.profile_store','tts.references'}<=set(requirements['pending_owners'])
  assert not any(path.is_file() for root in (Path.home()/'models',get_user_data_dir()/'models') if root.exists() for path in root.rglob('*'))
  receipt=json.loads((fixture/'blob-capture.json').read_text())
  assert digest(Path(seed['database']))==receipt['source_post_close_sha256'] and identity(Path(seed['database']))==receipt['source_post_close_identity']
  assert Path(seed['selector']).read_bytes().hex()==seed['config_hex']
  (fixture/'blob-readback.json').write_text(json.dumps({'profile_id':wanted['profile_id'],'reference_id':wanted['reference_id'],'profile_revision':value.revision,'fresh_generation':repository.generation,'canonical_sha256':exact.sha256,'restored_database_sha256':digest(database),'requirements':requirements,'blocked_network_attempts':len(blocked_attempts())},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_VOICE_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
async def main():
 app=TldwCli()
 try:
  source=fixture/'loose-source.wav'
  with wave.open(str(source),'wb') as output:
   output.setnchannels(1);output.setsampwidth(2);output.setframerate(8000);output.writeframes(b'\x29\x00'*800)
  source.chmod(0o600)
  with wave.open(str(source),'rb') as parsed:
   audio={'channels':parsed.getnchannels(),'sample_width':parsed.getsampwidth(),'sample_rate':parsed.getframerate(),'frames':parsed.getnframes()}
  assert audio=={'channels':1,'sample_width':2,'sample_rate':8000,'frames':800}
  root=selector.parent/'retained-voice-source';manager=ChatterboxVoiceManager(root)
  ok,message=manager.create_profile('retained_voice',str(source),display_name='Retained voice',language='en',description='Private fixture',tags=['retained'])
  assert ok,message
  catalog=root/'chatterbox_profiles.json';copied=root/'retained_voice'/'reference.wav'
  catalog.chmod(0o600);copied.chmod(0o600);root.chmod(0o700);copied.parent.chmod(0o700)
  profile=manager.get_profile('retained_voice');assert profile is not None and profile['reference_audio']==str(copied)
  seed={'root':str(root),'source':str(source),'source_sha256':digest(source),'source_identity':identity(source),
        'user_root':str(get_user_data_dir()),'selector':str(selector),'config_hex':selector.read_bytes().hex(),'profile':profile,'audio':audio,
        'tree':tree(root),'catalog_identity':identity(catalog),'copied_identity':identity(copied)}
  (fixture/'voice-seed.json').write_text(json.dumps(seed,indent=2))
  assert seed['tree']['files']['retained_voice/reference.wav']['sha256']==seed['source_sha256']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_VOICE_CAPTURE = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seed=json.loads((fixture/'voice-seed.json').read_text());source_root=Path(seed['root']);assert tree(source_root)==seed['tree']
  assert selector.read_bytes().hex()==seed['config_hex']
  manager=ChatterboxVoiceManager(source_root);assert manager.get_profile('retained_voice')==seed['profile']
  options={'staging_parent':fixture};preview=preview_capture((selector,),options=options)
  (fixture/'voice-preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues},indent=2));assert preview.complete,preview.issues
  destination=fixture/'tts-voice.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent';source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id']=='tts.voices' and source[row['logical_id']].path.is_relative_to(source_root)]
  assert {source[row['logical_id']].path.relative_to(source_root).as_posix() for row in rows}=={'chatterbox_profiles.json','retained_voice/reference.wav'}
  payloads={}
  for row in rows:
   relative=source[row['logical_id']].path.relative_to(source_root).as_posix();payload=captured.root/row['payload']
   assert digest(payload)==row['sha256']==seed['tree']['files'][relative]['sha256'];payloads[relative]={'logical_id':row['logical_id'],'sha256':row['sha256'],'size':row['size']}
  roots={row['root_id'] for row in rows};assert len(roots)==1;voice_root_id=roots.pop()
  directories={row['relative_path'] or '.':row for row in manifest['directories'] if row['root_id']==voice_root_id}
  assert set(directories)==set(seed['tree']['directories'])
  for relative,row in directories.items():
   assert row['metadata']['mode']==seed['tree']['directories'][relative]['mode'] and row['metadata']['mtime_ns']==seed['tree']['directories'][relative]['mtime_ns']
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  selected=fixture/'voice-restore-destinations';retained_root=selected/'retained-inert-voices';mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if key==voice_root_id:target=retained_root
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==Path(seed['user_root']) or Path(seed['user_root']) in original.parents:target=selected/'data'/'restored-retained-voice'/original.relative_to(Path(seed['user_root']))
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert not retained_root.exists()
  profile_id=manifest['profile_ids'][0];mapping['profile:'+profile_id+':paths.data_dir']=str(selected/'data')
  (fixture/'voice-capture.json').write_text(json.dumps({'archive':str(destination),'archive_sha256':written.digest,'manifest_sha256':hashlib.sha256(captured.manifest_bytes).hexdigest(),'payloads':payloads,'mapping':mapping,'source_profile':profile_id,'voice_root_id':voice_root_id,'retained_root':str(retained_root)},indent=2))
  assert tree(source_root)==seed['tree'] and selector.read_bytes().hex()==seed['config_hex']
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_VOICE_RESTORE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
seed=json.loads((fixture/'voice-seed.json').read_text());receipt=json.loads((fixture/'voice-capture.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'voice-acquired',ArchiveLimits(),None,threading.Event());doc=verify_sealed(archive)
assert doc.consistency=='coherent' and archive.digest==receipt['archive_sha256']
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  payload=zipped.read(row.payload);assert len(payload)==row.size and hashlib.sha256(payload).hexdigest()==row.sha256
 for relative,evidence in receipt['payloads'].items():
  row=next(row for row in doc.files if row.logical_id==evidence['logical_id']);assert row.sha256==seed['tree']['files'][relative]['sha256']
destinations={key:Path(path) for key,path in receipt['mapping'].items()};assert not Path(receipt['retained_root']).exists()
(fixture/'voice-restore-destinations').mkdir(mode=0o700,exist_ok=True)
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={receipt['source_profile']:'restored-retained-voice'})
assert 'owner_setup_required:tts.voices' in plan.issues
profile=restore_isolated(archive,plan,fixture/'voice-control',threading.Event());requirements=profile_requirements(profile,fixture/'voice-control')
assert requirements['requirements_checked'] and requirements['needs_setup'] and 'tts.voices' in requirements['pending_owners']
retained_root=Path(receipt['retained_root']);retained=tree(retained_root)
assert {key:value['sha256'] for key,value in retained['files'].items()}=={key:value['sha256'] for key,value in seed['tree']['files'].items()}
assert all(value['mode']==0o600 for value in retained['files'].values()) and all(value['mode']==0o700 for value in retained['directories'].values())
assert {key:value['mtime_ns'] for key,value in retained['files'].items()}=={key:value['mtime_ns'] for key,value in seed['tree']['files'].items()}
assert {key:value['mtime_ns'] for key,value in retained['directories'].items()}=={key:value['mtime_ns'] for key,value in seed['tree']['directories'].items()}
(fixture/'voice-restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements,'retained_root':str(retained_root),'plan_issues':plan.issues},indent=2))
assert tree(Path(seed['root']))==seed['tree'] and Path(seed['selector']).read_bytes().hex()==seed['config_hex']
assert not blocked_attempts(),blocked_attempts()
"""
)


_VOICE_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'voice-restored.json').read_text());select_profile(restored['profile'],fixture/'voice-control')
from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
seed=json.loads((fixture/'voice-seed.json').read_text());retained_root=Path(restored['retained_root'])
manager=ChatterboxVoiceManager(retained_root);profiles=manager.load_profiles();profile=manager.get_profile('retained_voice')
catalog=json.loads((retained_root/'chatterbox_profiles.json').read_text());stored=catalog['retained_voice']
assert profiles['retained_voice']==seed['profile'] and profile==seed['profile'] and stored==seed['profile']
historical=stored['reference_audio'];restored_wav=retained_root/'retained_voice'/'reference.wav'
assert historical==seed['profile']['reference_audio'] and Path(historical)!=restored_wav
assert digest(restored_wav)==seed['source_sha256']
with wave.open(str(restored_wav),'rb') as parsed:
 audio={'channels':parsed.getnchannels(),'sample_width':parsed.getsampwidth(),'sample_rate':parsed.getframerate(),'frames':parsed.getnframes()}
assert audio==seed['audio']
requirements=profile_requirements(restored['profile'],fixture/'voice-control')
assert json.loads(json.dumps(requirements))==restored['requirements'] and requirements['needs_setup'] and 'tts.voices' in requirements['pending_owners']
assert tree(Path(seed['root']))==seed['tree'] and Path(seed['selector']).read_bytes().hex()==seed['config_hex']
(fixture/'voice-readback.json').write_text(json.dumps({'native_reader':'readable','refusal':None,'historical_reference_audio':historical,'retained_wav':str(restored_wav),'retained_wav_sha256':digest(restored_wav),'requirements':requirements,'blocked_network_attempts':len(blocked_attempts())},indent=2))
assert not blocked_attempts(),blocked_attempts()
"""
)


def _environment(root: Path, *, with_voice_root: bool) -> dict[str, str]:
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
                (
                    ("chachanotes_db_path", "notes.db"),
                    ("media_db_path", "media.db"),
                    ("research_db_path", "research.db"),
                    ("prompts_db_path", "prompts.db"),
                )
                + (
                    ()
                    if with_voice_root
                    else (("tts_profiles_db_path", "tts-profiles.db"),)
                )
            )
        )
        + (
            "[app_tts]\nCHATTERBOX_VOICE_DIR="
            + json.dumps(str(profile / "retained-voice-source"))
            + "\n"
            if with_voice_root
            else ""
        )
    )
    selector.chmod(0o600)
    environment = {
        key: os.environ[key]
        for key in _CHILD_ENVIRONMENT_KEYS
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
        TTS_RETAINED_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    return environment


def test_canonical_tts_reference_blob_restores_with_fresh_native_getters(tmp_path):
    root = tmp_path.resolve()
    environment = _environment(root, with_voice_root=False)
    _run_profile_child(root, "blob-seed", _BLOB_SEED, environment)
    _run_profile_child(root, "blob-capture", _BLOB_CAPTURE, environment)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "blob-restore", _BLOB_RESTORE, restored_environment)
    _run_profile_child(root, "blob-read", _BLOB_READ, restored_environment)
    evidence = json.loads((root / "blob-readback.json").read_text())
    assert evidence["canonical_sha256"]
    assert evidence["blocked_network_attempts"] == 0


def test_deferred_loose_voice_retains_catalog_and_wav_without_binding(tmp_path):
    root = tmp_path.resolve()
    environment = _environment(root, with_voice_root=True)
    _run_profile_child(root, "voice-seed", _VOICE_SEED, environment)
    _run_profile_child(root, "voice-capture", _VOICE_CAPTURE, environment)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "voice-restore", _VOICE_RESTORE, restored_environment)
    _run_profile_child(root, "voice-read", _VOICE_READ, restored_environment)
    evidence = json.loads((root / "voice-readback.json").read_text())
    assert evidence["native_reader"] == "readable"
    assert evidence["refusal"] is None
    assert evidence["blocked_network_attempts"] == 0
