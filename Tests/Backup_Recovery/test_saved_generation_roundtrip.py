"""Saved image generations survive Complete archive and isolated restore."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import (
    _isolated_environment,
    _run_profile_child,
)


def _generated_target_case(tmp_path, relative_path, *, synthetic=False, renamed=False):
    from tldw_chatbook.Backup_Recovery.config_adapter import _Generated
    from tldw_chatbook.Backup_Recovery.staging import _config_targets

    owner = _Generated("generation.assets")
    data = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "restored"},
    }
    expected = tmp_path / "data" / "restored" / "generated_images"
    destination = expected / relative_path
    if renamed:
        destination = expected / "saved" / "renamed.png"
    logical_id = "profile:p:generation.assets:payload"
    payload = SimpleNamespace(
        logical_id=logical_id,
        owner_id=owner.owner_id,
        root_id="generation-root",
        relative_path=relative_path,
    )
    root = SimpleNamespace(
        logical_id="generation-root",
        root_id="generation-root",
        parent_id=None,
        relative_path="",
        synthetic=synthetic,
    )
    doc = SimpleNamespace(files=(payload,), directories=(root,))
    plan = SimpleNamespace(
        restore=((logical_id, destination),),
        destinations=((root.logical_id, expected),),
        issues=(),
    )
    _config_targets(
        data,
        "p",
        tmp_path / "config.toml",
        doc,
        plan,
        {owner.owner_id: owner},
    )
    return destination


def test_generated_saved_member_uses_selected_native_root(tmp_path):
    destination = _generated_target_case(tmp_path, "saved/retained.png")
    assert destination == (
        tmp_path / "data" / "restored" / "generated_images/saved/retained.png"
    )
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("relative_path", "synthetic", "renamed"),
    [
        ("temp/disposable.png", False, False),
        ("message/temporary.mp4", False, False),
        ("saved/retained.png", True, False),
        ("saved/retained.png", False, True),
    ],
)
def test_generated_relocation_rejects_non_saved_or_unbound_members(
    tmp_path, relative_path, synthetic, renamed
):
    with pytest.raises(
        ValueError, match="owner_relocation_unverified:generation.assets"
    ):
        _generated_target_case(
            tmp_path, relative_path, synthetic=synthetic, renamed=renamed
        )
    assert not (tmp_path / "data").exists()


_PRIVATE = r"""
import asyncio,hashlib,json,os,stat,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['SAVED_GENERATION_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
def identity(path):
 info=path.stat(follow_symlinks=False)
 return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
"""


_SEED = (
    _PRIVATE
    + r"""
from PIL import Image
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Media_Creation.image_generation_service import ImageGenerationService,GenerationResult
async def main():
 app=TldwCli()
 try:
  service=ImageGenerationService();assert service.client is None
  temporary=service.output_dir/'temp'/'retained-source.png'
  Image.new('RGB',(3,2),(17,34,51)).save(temporary,format='PNG',optimize=False)
  with Image.open(temporary) as image:
   image.load();assert (image.format,image.mode,image.size,image.getpixel((1,1)))==('PNG','RGB',(3,2),(17,34,51))
  result=GenerationResult(success=True,images=[str(temporary)],prompt='Private retained fixture',negative_prompt='',parameters={})
  returned=await service.save_generation(result,name='retained')
  retained=service.output_dir/'saved'/'retained.png'
  assert returned==[str(retained)] and retained.is_file() and not temporary.exists()
  disposable=service.output_dir/'temp'/'disposable.png'
  Image.new('RGB',(1,1),(5,6,7)).save(disposable,format='PNG',optimize=False)
  seed={'selector':str(selector),'config_hex':selector.read_bytes().hex(),'user_root':str(get_user_data_dir()),
        'output_dir':str(service.output_dir),'returned_paths':returned,'retained':str(retained),
        'retained_sha256':digest(retained),'retained_identity':identity(retained),'retained_size':retained.stat().st_size,
        'disposable':str(disposable),'disposable_sha256':digest(disposable),'disposable_identity':identity(disposable),
        'decoded':{'format':'PNG','mode':'RGB','size':[3,2],'pixel':[17,34,51]}}
  (fixture/'saved-generation-seed.json').write_text(json.dumps(seed,indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


_CAPTURE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Media_Creation.image_generation_service import ImageGenerationService
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seed=json.loads((fixture/'saved-generation-seed.json').read_text());retained=Path(seed['retained']);disposable=Path(seed['disposable'])
  service=ImageGenerationService();assert service.client is None and service.output_dir==Path(seed['output_dir'])
  assert digest(retained)==seed['retained_sha256'] and identity(retained)==seed['retained_identity']
  assert digest(disposable)==seed['disposable_sha256'] and identity(disposable)==seed['disposable_identity']
  options={'staging_parent':fixture,'temporary_media':False};preview=preview_capture((selector,),options=options)
  generation=[item for item in preview.items if item.owner=='generation.assets']
  kept=[item for item in generation if item.path==retained];temporary=[item for item in generation if item.path==disposable]
  assert len(kept)==1 and kept[0].status=='included',[(str(item.path),item.status) for item in kept]
  assert len(temporary)==1 and temporary[0].status=='intentionally_excluded',[(str(item.path),item.status) for item in temporary]
  bad=[(item.owner,str(item.path),item.status) for item in preview.items if item.status not in ('included','included_directory','unused','intentionally_excluded','intentionally_deleted')]
  assert preview.complete,(preview.issues,bad)
  destination=fixture/'saved-generation.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes);assert manifest['consistency']=='coherent';source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id']=='generation.assets']
  assert len(rows)==1,[(row['relative_path'],row['sha256']) for row in rows]
  row=rows[0];assert row['relative_path']=='saved/retained.png' and source[row['logical_id']].path==retained
  payload=captured.root/row['payload'];assert digest(payload)==row['sha256']==seed['retained_sha256'] and payload.stat().st_size==seed['retained_size']
  assert all(entry['relative_path']!='temp/disposable.png' for entry in manifest['files'])
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  assert written.digest==digest(destination)
  with zipfile.ZipFile(destination) as zipped:
   archived=zipped.read(row['payload']);assert hashlib.sha256(archived).hexdigest()==seed['retained_sha256'] and archived==retained.read_bytes()
  selected=fixture/'saved-generation-restore-destinations';mapping={};source_root=Path(seed['user_root'])
  generation_root_id=row['root_id']
  for directory in manifest['directories']:
   if directory['parent_id'] is not None:continue
   key=directory['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if directory.get('synthetic') else source[key].owner
   if key==generation_root_id:target=selected/'data'/'restored-saved-generation'/'generated_images'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-saved-generation'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert mapping[generation_root_id]==str(selected/'data'/'restored-saved-generation'/'generated_images')
  assert len(manifest['profile_ids'])==1;profile=manifest['profile_ids'][0]
  mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  receipt={'archive':str(destination),'archive_sha256':written.digest,
           'archive_identity':identity(destination),'manifest_sha256':hashlib.sha256(captured.manifest_bytes).hexdigest(),
           'payload':{'logical_id':row['logical_id'],'root_id':generation_root_id,'archive_member':row['payload'],'relative_path':row['relative_path'],'sha256':row['sha256'],'size':row['size']},
           'mapping':mapping,'source_profile':profile,'restored_path':str(Path(mapping[generation_root_id])/'saved'/'retained.png')}
  (fixture/'saved-generation-capture.json').write_text(json.dumps(receipt,indent=2))
  assert digest(retained)==seed['retained_sha256'] and identity(retained)==seed['retained_identity']
  assert digest(disposable)==seed['disposable_sha256'] and identity(disposable)==seed['disposable_identity']
  assert selector.read_bytes().hex()==seed['config_hex'] and not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
seed=json.loads((fixture/'saved-generation-seed.json').read_text());receipt=json.loads((fixture/'saved-generation-capture.json').read_text())
retained=Path(seed['retained']);disposable=Path(seed['disposable'])
assert digest(retained)==seed['retained_sha256'] and identity(retained)==seed['retained_identity']
assert digest(disposable)==seed['disposable_sha256'] and identity(disposable)==seed['disposable_identity']
receipt['source_post_close_sha256']=digest(retained);receipt['source_post_close_identity']=identity(retained)
(fixture/'saved-generation-capture.json').write_text(json.dumps(receipt,indent=2))
"""
)


_RESTORE = (
    _PRIVATE
    + r"""
import zipfile
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements,_launch_state
from tldw_chatbook.Backup_Recovery.journal import Journal
seed=json.loads((fixture/'saved-generation-seed.json').read_text());receipt=json.loads((fixture/'saved-generation-capture.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'saved-generation-acquired',ArchiveLimits(),None,threading.Event());doc=verify_sealed(archive)
assert doc.consistency=='coherent' and archive.digest==receipt['archive_sha256']
with zipfile.ZipFile(archive.path) as zipped:
 manifest_member=zipped.read('manifest.json');assert hashlib.sha256(manifest_member).hexdigest()==receipt['manifest_sha256']
 for row in doc.files:
  payload=zipped.read(row.payload);assert len(payload)==row.size and hashlib.sha256(payload).hexdigest()==row.sha256
 row=next(row for row in doc.files if row.logical_id==receipt['payload']['logical_id'])
 payload=zipped.read(row.payload);assert row.relative_path=='saved/retained.png' and hashlib.sha256(payload).hexdigest()==seed['retained_sha256']
selected=fixture/'saved-generation-restore-destinations';selected.mkdir(mode=0o700,exist_ok=True)
destinations={key:Path(path) for key,path in receipt['mapping'].items()}
plan=plan_restore(archive,mode='isolated',destinations=destinations,target=None,profile_names={receipt['source_profile']:'restored-saved-generation'})
assert 'owner_setup_required:generation.assets' not in plan.issues
try:profile=restore_isolated(archive,plan,fixture/'saved-generation-control',threading.Event())
except Exception as error:
 (fixture/'saved-generation-restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args,'archive_path':str(archive.path),'archive_identity':identity(archive.path),'archive_sha256':digest(archive.path),'manifest_sha256':hashlib.sha256(manifest_member).hexdigest(),'source_path':seed['retained'],'source_sha256':digest(Path(seed['retained'])),'planned_restored_path':receipt['restored_path']},default=str,indent=2))
 raise
requirements=profile_requirements(profile,fixture/'saved-generation-control')
assert requirements['requirements_checked'] and requirements['needs_setup']
assert 'generation.assets' in requirements['required_owners'] and 'generation.assets' in requirements['pending_owners']
entry,witness=_launch_state(profile,fixture/'saved-generation-control');journal=Journal(fixture/'saved-generation-control',witness['operation_id'])
with journal._locked(exclusive=False) as parent:records=journal._records(parent)
assert records and records[-1].event=='committed'
restored=Path(receipt['restored_path']);assert digest(restored)==seed['retained_sha256'] and restored.stat().st_size==seed['retained_size']
assert digest(Path(seed['retained']))==receipt['source_post_close_sha256'] and identity(Path(seed['retained']))==receipt['source_post_close_identity']
(fixture/'saved-generation-restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements,'entry':entry.model_dump(),'journal_operation_id':witness['operation_id'],'journal_events':[record.event for record in records],'journal_path':str(journal.root),'restored_path':str(restored),'restored_sha256':digest(restored),'acquired_path':str(archive.path),'acquired_identity':identity(archive.path),'acquired_sha256':digest(archive.path)},indent=2))
assert Path(seed['selector']).read_bytes().hex()==seed['config_hex'] and not blocked_attempts(),blocked_attempts()
"""
)


_READ = (
    _PRIVATE
    + r"""
from PIL import Image
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'saved-generation-restored.json').read_text());select_profile(restored['profile'],fixture/'saved-generation-control')
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Media_Creation.image_generation_service import ImageGenerationService
seed=json.loads((fixture/'saved-generation-seed.json').read_text());receipt=json.loads((fixture/'saved-generation-capture.json').read_text())
service=ImageGenerationService();path=service.output_dir/'saved'/'retained.png'
assert service.client is None and service.output_dir==get_user_data_dir()/'generated_images' and path==Path(receipt['restored_path'])
assert digest(path)==seed['retained_sha256'] and path.stat().st_size==seed['retained_size']
with Image.open(path) as image:
 image.load();decoded={'format':image.format,'mode':image.mode,'size':list(image.size),'pixel':list(image.getpixel((1,1)))}
assert decoded==seed['decoded']
requirements=profile_requirements(restored['profile'],fixture/'saved-generation-control')
assert json.loads(json.dumps(requirements))==restored['requirements'] and requirements['needs_setup'] and 'generation.assets' in requirements['pending_owners']
source=Path(seed['retained']);assert digest(source)==receipt['source_post_close_sha256'] and identity(source)==receipt['source_post_close_identity']
evidence={'native_service_output_dir':str(service.output_dir),'native_saved_path':str(path),'native_saved_identity':identity(path),'native_saved_sha256':digest(path),'decoded_png':decoded,
          'source_path':str(source),'source_identity':identity(source),'source_sha256':digest(source),'source_preserved':True,
          'archive_path':receipt['archive'],'archive_sha256':receipt['archive_sha256'],'manifest_sha256':receipt['manifest_sha256'],'payload':receipt['payload'],
          'acquired_path':restored['acquired_path'],'acquired_identity':restored['acquired_identity'],'acquired_sha256':restored['acquired_sha256'],
          'journal_path':restored['journal_path'],'journal_operation_id':restored['journal_operation_id'],'journal_events':restored['journal_events'],
          'requirements':requirements,'blocked_network_attempts':len(blocked_attempts())}
(fixture/'saved-generation-readback.json').write_text(json.dumps(evidence,indent=2))
assert not blocked_attempts(),blocked_attempts()
"""
)


def _environment(root: Path) -> dict[str, str]:
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
        SAVED_GENERATION_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    return environment


def test_saved_generation_restores_to_fresh_native_service_path(tmp_path):
    root = tmp_path.resolve()
    environment = _environment(root)
    _run_profile_child(root, "saved-generation-seed", _SEED, environment)
    _run_profile_child(root, "saved-generation-capture", _CAPTURE, environment)
    restored_environment = _isolated_environment(root, environment)
    _run_profile_child(root, "saved-generation-restore", _RESTORE, restored_environment)
    _run_profile_child(root, "saved-generation-read", _READ, restored_environment)
    evidence = json.loads((root / "saved-generation-readback.json").read_text())
    assert evidence["native_saved_sha256"] == evidence["source_sha256"]
    assert evidence["source_preserved"] and evidence["blocked_network_attempts"] == 0
    assert evidence["journal_events"][-1] == "committed"
