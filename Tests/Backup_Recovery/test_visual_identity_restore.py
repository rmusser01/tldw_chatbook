"""A native custom character visual identity survives an isolated restore."""

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
fixture=Path(os.environ['VISUAL_RESTORE_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_SEED = (
    _PRIVATE
    + r"""
from io import BytesIO
from PIL import Image
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Character_Chat.visual_identity import create_visual_identity_candidate,publish_visual_identity_candidate,resolve_visual_identity,parse_visual_identity_manifest_json,load_visual_identity_asset
async def main():
 app=TldwCli()
 try:
  db=app.chachanotes_db
  cards=[card for card in db.list_character_cards() if card.get('extensions',{}).get('tldw/builtin_id')=='samira']
  assert len(cards)==1,cards
  actor=cards[0]['id'];repository=VisualIdentityRepository(db)
  previous=repository.get_active_actor_pack('character',actor)
  assert previous is not None and previous['pack']['source_kind']=='builtin'
  output=BytesIO();Image.new('RGB',(16,16),(11,29,47)).save(output,format='PNG');payload=output.getvalue()
  candidate=create_visual_identity_candidate(db,actor_kind='character',actor_id=actor)
  candidate.stage_replacement('thinking',payload,source='upload')
  assert repository.get_active_actor_pack('character',actor)==previous
  published=publish_visual_identity_candidate(db,candidate,user_data_dir=get_user_data_dir())
  graph=repository.get_active_actor_pack('character',actor)
  assert graph['pack']['source_kind']=='manual' and graph['pack']['id']==published.new_pack_id
  assert graph['version']['id']==published.new_version_id and published.new_pack_id!=published.old_pack_id
  resolved=resolve_visual_identity(db,actor_kind='character',actor_id=actor,requested_state='thinking',user_data_dir=get_user_data_dir())
  assert resolved.storage_source=='manual' and resolved.resolved_expression_key=='thinking' and resolved.image_bytes==payload
  manifest=parse_visual_identity_manifest_json(graph['version']['manifest_json'])
  selected=next(asset for asset in manifest.assets if asset.expression_key=='thinking')
  assert load_visual_identity_asset(selected,source_kind='manual',user_data_dir=get_user_data_dir()).data==payload
  user_root=get_user_data_dir();files={str(path.relative_to(user_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (user_root/'visual_identities').rglob('*') if path.is_file()}
  assert files and selected.sha256==hashlib.sha256(payload).hexdigest()
  (fixture/'seed.json').write_text(json.dumps({'actor':actor,'graph':graph,'previous_binding':previous['binding'],'hex':payload.hex(),'sha256':selected.sha256,'asset_id':resolved.asset_id,'relpath':selected.storage_relpath,'user_root':str(user_root),'files':files,'config_hex':selector.read_bytes().hex()},indent=2))
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
  destination=fixture/'visual.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);source={item.logical_id:item for item in captured.inventory.items}
  assets=[row for row in manifest['files'] if row['owner_id']=='persona.visual_identity']
  assert len(assets)==len(seed['files']) and assets
  for row in assets:
   relative=str(source[row['logical_id']].path.relative_to(source_root))
   assert hashlib.sha256((captured.root/row['payload']).read_bytes()).hexdigest()==seed['files'][relative]
  assert 'db.chachanotes.primary' in {row['owner_id'] for row in manifest['files']}
  await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  mapping={};selected=fixture/'restore-destinations'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-visual'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'mapping':mapping,'source_profile':profile,'asset_count':len(assets)},indent=2))
  assert selector.read_bytes().hex()==seed['config_hex']
  assert {rel:hashlib.sha256((source_root/rel).read_bytes()).hexdigest() for rel in seed['files']}==seed['files']
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
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
receipt=json.loads((fixture/'restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
assert verify_sealed(archive).consistency=='coherent'
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-visual'})
try:profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 (fixture/'restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str));raise
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'state':'restoration_validated'}))
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
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Character_Chat.visual_identity import resolve_visual_identity,parse_visual_identity_manifest_json,load_visual_identity_asset
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'seed.json').read_text());user_root=get_user_data_dir()
  assert user_root!=Path(seed['user_root'])
  graph=VisualIdentityRepository(app.chachanotes_db).get_active_actor_pack('character',seed['actor'])
  assert graph==seed['graph']
  resolved=resolve_visual_identity(app.chachanotes_db,actor_kind='character',actor_id=seed['actor'],requested_state='thinking',user_data_dir=user_root)
  assert resolved.storage_source=='manual' and resolved.resolved_expression_key=='thinking'
  assert resolved.asset_id==seed['asset_id'] and resolved.pack_id==graph['pack']['id'] and resolved.pack_version_id==graph['version']['id']
  assert resolved.storage_relpath==seed['relpath'] and resolved.image_bytes.hex()==seed['hex']
  manifest=parse_visual_identity_manifest_json(graph['version']['manifest_json'])
  selected=next(asset for asset in manifest.assets if asset.expression_key=='thinking')
  loaded=load_visual_identity_asset(selected,source_kind='manual',user_data_dir=user_root)
  assert loaded.data.hex()==seed['hex'] and hashlib.sha256(loaded.data).hexdigest()==seed['sha256']
  assert {rel:hashlib.sha256((user_root/rel).read_bytes()).hexdigest() for rel in seed['files']}==seed['files']
  assert {rel:hashlib.sha256((Path(seed['user_root'])/rel).read_bytes()).hexdigest() for rel in seed['files']}==seed['files']
  (fixture/'readback.json').write_text(json.dumps({'actor':seed['actor'],'pack':resolved.pack_id,'version':resolved.pack_version_id,'asset':resolved.asset_id,'sha256':selected.sha256,'restored_root':str(user_root),'source_preserved':True,'blocked_network_attempts':len(blocked_attempts())},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def test_native_character_visual_identity_restores_with_fresh_passive_read(tmp_path):
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
        VISUAL_RESTORE_FIXTURE=str(root),
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
