"""Native workspace shadow commits survive capture and isolated passive reads."""

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
fixture=Path(os.environ['SHADOW_RESTORE_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_SEED = (
    _PRIVATE
    + r"""
from dataclasses import asdict
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
async def main():
 app=TldwCli()
 try:
  external=fixture/'workspace';external.mkdir(mode=0o700)
  file=external/'a.txt';file.write_bytes(b'first retained version\n')
  registry=app.workspace_registry_service
  workspace=registry.create_workspace(workspace_id='retained-shadow',name='Retained shadow')
  before=set(threading.enumerate())
  binding=registry.add_folder_binding(workspace.workspace_id,external,allow_write=False)
  # Registration starts the existing real initial-snapshot worker. Settle it
  # before editing, without replacing its implementation or creating a receipt.
  for thread in threading.enumerate():
   if thread not in before and thread.name=='change-review-initial-snapshot':
    thread.join(timeout=10);assert not thread.is_alive(),'initial snapshot did not settle'
  assert binding.metadata.get('access')=='ro'
  repo=ShadowRepoService().repo_for_root(external)
  first=repo.snapshot('first retained snapshot')
  file.write_bytes(b'second retained version\n')
  second=repo.snapshot('second retained snapshot')
  assert first!=second and repo.tip()==second
  assert repo.has_snapshot(first) and repo.has_snapshot(second)
  assert repo.file_bytes(first,'a.txt')==b'first retained version\n'
  assert repo.file_bytes(second,'a.txt')==file.read_bytes()
  user_root=get_user_data_dir();owner_root=user_root/'change_review'
  assert repo.git_dir.is_relative_to(owner_root)
  files={str(p.relative_to(owner_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in owner_root.rglob('*') if p.is_file()}
  directories=sorted(str(p.relative_to(owner_root)) for p in owner_root.rglob('*') if p.is_dir())
  assert files and not any(p.is_symlink() for p in owner_root.rglob('*'))
  (fixture/'seed.json').write_text(json.dumps({'workspace':asdict(workspace),'binding':asdict(binding),'first':first,'second':second,'external':str(external),'external_hex':file.read_bytes().hex(),'files':files,'directories':directories,'user_root':str(user_root),'config_hex':selector.read_bytes().hex()},indent=2))
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
  destination=fixture/'shadow.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);source={item.logical_id:item for item in captured.inventory.items}
  assets=[row for row in manifest['files'] if row['owner_id']=='workspaces.change_tracking']
  assert len(assets)==len(seed['files']) and assets
  for row in assets:
   relative=str(source[row['logical_id']].path.relative_to(source_root/'change_review'))
   assert hashlib.sha256((captured.root/row['payload']).read_bytes()).hexdigest()==seed['files'][relative]
  external=Path(seed['external'])
  assert not any(item.path is not None and (item.path==external or external in item.path.parents) and item.status in {'included','included_directory'} for item in captured.inventory.items)
  assert 'db.workspaces' in {row['owner_id'] for row in manifest['files']}
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  (fixture/'capture.json').write_text(json.dumps({'manifest':json.loads(captured.manifest_bytes),'archive_sha256':written.digest,'shadow_files':len(assets)},indent=2))
  mapping={};selected=fixture/'restore-destinations'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-shadow'/original.relative_to(source_root)
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
  assert {rel:hashlib.sha256((source_root/'change_review'/rel).read_bytes()).hexdigest() for rel in seed['files']}==seed['files']
  assert (Path(seed['external'])/'a.txt').read_bytes().hex()==seed['external_hex']
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
doc=verify_sealed(archive);assert doc.consistency=='coherent'
import zipfile
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  if row.owner_id=='workspaces.change_tracking':assert hashlib.sha256(zipped.read(row.payload)).hexdigest()==row.sha256
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-shadow'})
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
from dataclasses import asdict
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'seed.json').read_text());user_root=get_user_data_dir()
  assert user_root!=Path(seed['user_root'])
  owner_root=user_root/'change_review'
  def observed(root):
   return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
  assert observed(owner_root)==seed['files']
  assert sorted(str(p.relative_to(owner_root)) for p in owner_root.rglob('*') if p.is_dir())==seed['directories']
  registry=app.workspace_registry_service
  workspace=registry.get_workspace(seed['workspace']['workspace_id'])
  binding=registry.get_runtime_binding(seed['binding']['binding_id'])
  assert asdict(workspace)==seed['workspace'] and asdict(binding)==seed['binding']
  assert binding.metadata.get('access')=='ro'
  # The fresh reader issues only explicit native read-only Git operations.
  # It never snapshots, initializes, checks out, restores paths or grants access.
  repo=ShadowRepoService().repo_for_root(binding.locator)
  assert repo.git_dir.is_relative_to(owner_root)
  assert repo.tip()==seed['second']
  assert repo.has_snapshot(seed['first']) and repo.has_snapshot(seed['second'])
  assert repo.file_bytes(seed['first'],'a.txt')==b'first retained version\n'
  assert repo.file_bytes(seed['second'],'a.txt')==b'second retained version\n'
  assert observed(owner_root)==seed['files']
  assert sorted(str(p.relative_to(owner_root)) for p in owner_root.rglob('*') if p.is_dir())==seed['directories']
  assert observed(Path(seed['user_root'])/'change_review')==seed['files']
  assert (Path(seed['external'])/'a.txt').read_bytes().hex()==seed['external_hex']
  (fixture/'readback.json').write_text(json.dumps({'first':seed['first'],'second':seed['second'],'restored_root':str(owner_root),'source_preserved':True,'restored_bytes_preserved':True,'blocked_network_attempts':len(blocked_attempts())},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())

"""
)


def test_native_workspace_shadow_history_restores_with_fresh_passive_reads(tmp_path):
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
        SHADOW_RESTORE_FIXTURE=str(root),
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
