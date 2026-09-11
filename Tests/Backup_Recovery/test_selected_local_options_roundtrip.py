"""Selected inert model closure, external files and native diagnostics roundtrip."""

import json
import os
from pathlib import Path

import pytest

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
fixture=Path(os.environ['SELECTED_OPTIONS_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
"""

_SEED = (
    _PRIVATE
    + r"""
from dataclasses import replace
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir,get_cli_log_file_path
from tldw_chatbook.Model_Artifacts import ArtifactRef,ArtifactRole
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService
from tldw_chatbook.Model_Artifacts.recovery import managed_artifact_root
from Tests.Model_Artifacts.test_provision_install import _descriptor
from tldw_chatbook.Logging_Config import configure_application_logging,PrivateRotatingFileHandler
import logging
async def main():
 app=TldwCli()
 try:
  root=get_user_data_dir();service=ModelArtifactService(managed_artifact_root(root))
  dependency=ArtifactRef('support','revision','fp32');primary=ArtifactRef('primary','revision','fp32')
  bodies=(b'dependency model bytes',b'primary model bytes',b'unselected model bytes')
  descriptors=(replace(_descriptor(dependency,role=ArtifactRole.DEPENDENCY,files_body=bodies[0]),model_id='fixture/dependency'),replace(_descriptor(primary,dependencies=(dependency,),files_body=bodies[1]),model_id='fixture/primary'),replace(_descriptor(ArtifactRef('unselected','revision','fp32'),files_body=bodies[2]),model_id='fixture/unselected'))
  model_files={}
  for descriptor,body in zip(descriptors,bodies):
   incoming=fixture/('incoming-'+descriptor.reference.artifact_id);incoming.mkdir(mode=0o700);(incoming/'model.bin').write_bytes(body)
   service.install(descriptor,incoming)
   directory=service.artifact_path(descriptor.reference)
   for leaf in ('manifest.json','model.bin'):
    path=directory/leaf;model_files[str(path.relative_to(root))]=path.read_bytes().hex()
  external=fixture/'external-selected';external.mkdir(mode=0o700);(external/'empty').mkdir(mode=0o700)
  (external/'unsanitized.txt').write_text('Explicit external fixture; API_KEY=synthetic-retained-value\n')
  (external/'binary.dat').write_bytes(bytes(range(32)))
  configure_application_logging(app)
  log_path=get_cli_log_file_path()
  handlers=[handler for handler in logging.getLogger().handlers if isinstance(handler,PrivateRotatingFileHandler)]
  assert len(handlers)==1 and Path(handlers[0].baseFilename)==log_path
  for handler in handlers:handler.flush();logging.getLogger().removeHandler(handler);handler.close()
  diagnostic=log_path.read_bytes();assert b'persistent_sink_installed' in diagnostic
  (fixture/'seed.json').write_text(json.dumps({'user_root':str(root),'models':model_files,'external':str(external),'external_files':{path.name:path.read_bytes().hex() for path in external.iterdir() if path.is_file()},'diagnostic':str(log_path),'diagnostic_hex':diagnostic.hex(),'config_hex':selector.read_bytes().hex()},indent=2))
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
  seed=json.loads((fixture/'seed.json').read_text());source_root=Path(seed['user_root']);external=Path(seed['external'])
  default=preview_capture((selector,),options={'staging_parent':fixture})
  assert default.complete,default.issues
  assert not any(item.owner=='external.files' for item in default.items)
  assert all(item.status=='intentionally_excluded' for item in default.items if item.owner=='diagnostics.logs')
  assert not any(item.status=='included' and item.path.name=='model.bin' for item in default.items if item.owner=='models.artifacts')
  options={'staging_parent':fixture,'model_ids':('fixture/primary',),'external_roots':(external,),'diagnostics':True,'allow_partial':True}
  preview=preview_capture((selector,),options=options)
  (fixture/'preview.json').write_text(json.dumps({'complete':preview.complete,'issues':preview.issues,'rows':[{'owner':item.owner,'path':str(item.path),'status':item.status} for item in preview.items if item.owner in ('models.artifacts','external.files','diagnostics.logs') or item.status in ('unsupported','unavailable','missing_required')]},indent=2))
  assert preview.complete,preview.issues
  destination=fixture/'selected-options.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  manifest=json.loads(captured.manifest_bytes);source={item.logical_id:item for item in captured.inventory.items}
  assert manifest['consistency']=='partial'
  assert 'Arbitrary diagnostic and external content is not credential-sanitized.' in manifest['report']['lines']
  assert 'External files have per-file stability checks, not folder-wide consistency.' in manifest['report']['lines']
  payloads={source[row['logical_id']].path:captured.root/row['payload'] for row in manifest['files']}
  for relative,body in seed['models'].items():
   path=source_root/relative
   if path.name=='manifest.json' or '/unselected/' not in relative:assert payloads[path].read_bytes().hex()==body
   else:assert path not in payloads
  for leaf,body in seed['external_files'].items():assert payloads[external/leaf].read_bytes().hex()==body
  assert payloads[Path(seed['diagnostic'])].read_bytes().hex()==seed['diagnostic_hex']
  written=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  (fixture/'capture.json').write_text(json.dumps({'manifest':manifest,'archive_sha256':written.digest},indent=2))
  selected=fixture/'restore-destinations';mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   if owner=='external.files':target=selected/'external'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==source_root or source_root in original.parents:target=selected/'data'/'restored-options'/original.relative_to(source_root)
   elif original==selector.parent:target=selected/'config'
   elif original==selector.parent/'custom':target=selected/'custom'
   else:
    assert owner=='persona.visual_identity_builtin',(key,owner,str(original));target=selected/'inactive-builtin'
   mapping[key]=str(target)
  assert len(manifest['profile_ids'])==1
  profile=manifest['profile_ids'][0];mapping['profile:'+profile+':paths.data_dir']=str(selected/'data')
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'mapping':mapping,'source_profile':profile,'external':str(selected/'external')},indent=2))
  assert selector.read_bytes().hex()==seed['config_hex']
  assert all((source_root/relative).read_bytes().hex()==body for relative,body in seed['models'].items())
  assert Path(seed['diagnostic']).read_bytes().hex()==seed['diagnostic_hex']
  assert all((external/leaf).read_bytes().hex()==body for leaf,body in seed['external_files'].items())
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)

_RESTORE = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,profile_requirements
import zipfile
receipt=json.loads((fixture/'restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive);assert doc.consistency=='partial'
with zipfile.ZipFile(archive.path) as zipped:
 for row in doc.files:
  data=zipped.read(row.payload);assert len(data)==row.size and hashlib.sha256(data).hexdigest()==row.sha256
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={receipt['source_profile']:'restored-options'})
try:profile=restore_isolated(archive,plan,fixture/'control',threading.Event())
except Exception as error:
 (fixture/'restore-refusal.json').write_text(json.dumps({'type':type(error).__name__,'args':error.args},default=str));raise
requirements=profile_requirements(profile,fixture/'control')
assert requirements['requirements_checked'] and 'models.artifacts' in requirements['pending_owners']
(fixture/'restored.json').write_text(json.dumps({'profile':profile,'requirements':requirements}))
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,profile_requirements
restored=json.loads((fixture/'restored.json').read_text());select_profile(restored['profile'],fixture/'control')
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_user_data_dir,get_cli_log_file_path
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService
from tldw_chatbook.Model_Artifacts.recovery import managed_artifact_root
async def main():
 app=TldwCli()
 try:
  seed=json.loads((fixture/'seed.json').read_text());receipt=json.loads((fixture/'restore-input.json').read_text());root=get_user_data_dir()
  service=ModelArtifactService(managed_artifact_root(root));installed=service.list_installed()
  descriptors={item.descriptor.model_id:item.descriptor for item in installed if item.descriptor is not None}
  assert set(descriptors)=={'fixture/primary','fixture/dependency','fixture/unselected'}
  assert descriptors['fixture/primary'].dependencies==(descriptors['fixture/dependency'].reference,)
  assert not any(item.active or item.ready for item in installed)
  for relative,body in seed['models'].items():
   target=root/relative
   if target.name=='manifest.json' or '/unselected/' not in relative:assert target.read_bytes().hex()==body
   else:assert not target.exists()
  external=Path(receipt['external'])
  assert (external/'empty').is_dir() and not tuple((external/'empty').iterdir())
  assert {path.name:path.read_bytes().hex() for path in external.iterdir() if path.is_file()}==seed['external_files']
  assert get_cli_log_file_path().read_bytes().hex()==seed['diagnostic_hex']
  assert all((Path(seed['user_root'])/relative).read_bytes().hex()==body for relative,body in seed['models'].items())
  assert Path(seed['diagnostic']).read_bytes().hex()==seed['diagnostic_hex']
  assert all((Path(seed['external'])/leaf).read_bytes().hex()==body for leaf,body in seed['external_files'].items())
  requirements=profile_requirements(restored['profile'],fixture/'control');assert 'models.artifacts' in requirements['pending_owners']
  assert not blocked_attempts(),blocked_attempts()
  (fixture/'readback.json').write_text(json.dumps({'source_preserved':True,'model_ids':sorted(descriptors),'requirements':requirements,'blocked_network_attempts':len(blocked_attempts())},indent=2))
 finally:
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)


def test_selected_local_options_restore_with_passive_native_reads(tmp_path):
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
        SELECTED_OPTIONS_FIXTURE=str(root),
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


@pytest.mark.parametrize(
    "configured,leaf,valid",
    [
        (None, "tldw_cli_app.log", True),
        ("custom.log", "custom.log", True),
        ("custom.log", "custom.log.1", True),
        ("custom.log", "custom.log.123", True),
        ("custom.log", "foreign.log", False),
        ("custom.log", "nested/custom.log", False),
        ("custom.log", "../custom.log", False),
        ("custom.log", "custom.log.bak", False),
        ("custom.log", "custom.log.١", False),
        ("custom.log", "custom.log.", False),
        ("custom.log", "custom.log.-1", False),
        ("../custom.log", "custom.log", False),
        ("", "custom.log", False),
        (123, "custom.log", False),
    ],
)
def test_diagnostic_restore_path_accepts_only_installed_family(
    tmp_path, configured, leaf, valid
):
    from tldw_chatbook.Backup_Recovery.config_adapter import _Diagnostics

    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Local"},
    }
    if configured is not None:
        config["logging"] = {"log_filename": configured}
    owner = _Diagnostics("diagnostics.logs")
    if valid:
        assert owner._restore_path(config, leaf) == tmp_path / "data" / "Local" / leaf
    else:
        with pytest.raises(ValueError):
            owner._restore_path(config, leaf)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("mode", ["exact", "wrong_parent", "file_container"])
def test_diagnostic_staging_requires_exact_selected_parent(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.config_adapter import _Diagnostics
    from tldw_chatbook.Backup_Recovery.staging import _config_targets

    owner = _Diagnostics("diagnostics.logs")

    def forbidden(*args):
        raise AssertionError("restore cannot discover target logs")

    monkeypatch.setattr(_Diagnostics, "discover", forbidden)
    data = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Local"},
        "logging": {"log_filename": "custom.log"},
    }
    parent = (
        tmp_path / "wrong" if mode == "wrong_parent" else tmp_path / "data" / "Local"
    )
    if mode == "file_container":
        parent /= "custom.log.2"
    logical = "profile:p:diagnostics.logs:retained"
    payload = SimpleNamespace(
        logical_id=logical,
        owner_id=owner.owner_id,
        root_id="root",
        relative_path="custom.log.2",
    )
    root = SimpleNamespace(logical_id="root", synthetic=mode != "file_container")
    doc = SimpleNamespace(files=(payload,), directories=(root,))
    plan = SimpleNamespace(
        restore=((logical, parent / payload.relative_path),),
        destinations=(("root", parent),),
        issues=(),
    )
    if mode != "exact":
        with pytest.raises(
            ValueError, match="owner_relocation_unverified:diagnostics.logs"
        ):
            _config_targets(
                data, "p", tmp_path / "config.toml", doc, plan, {owner.owner_id: owner}
            )
    else:
        _config_targets(
            data, "p", tmp_path / "config.toml", doc, plan, {owner.owner_id: owner}
        )
    assert not parent.exists()
