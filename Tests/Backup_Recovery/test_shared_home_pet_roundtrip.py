"""Native shared-HOME pet state and automatic backups across isolated restore."""

import json
import os
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import _run_profile_child

_PRIVATE = r"""
import asyncio,hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for module in ('sounddevice','pyaudio'):sys.modules[module]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['PET_FIXTURE'])
"""

_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
async def main():
 app=TldwCli()
 try:
  label=Path(os.environ['TLDW_CONFIG_PATH']).parent.name
  store=ConfigFileStorage()
  assert store.filepath==Path.home()/'.config'/'tldw_chatbook'/'tamagotchi_pets.json'
  state={'name':label+' native pet','happiness':72,'hunger':31,'energy':84,'health':93,'age':4}
  if label=='beta':
   alpha=json.loads((fixture/'alpha-pet.json').read_text())
   assert store.load('alpha')==alpha
  assert store.save(label,state)
  initial=store.load(label)
  assert all(initial[key]==value for key,value in state.items()) and initial['last_saved']
  if label=='beta':
   state['happiness']=81
   assert store.save(label,state)
   assert store.load(label)['happiness']==81
   files={p.name:p.read_bytes().hex() for p in store.filepath.parent.iterdir()}
   backups={name:value for name,value in files.items() if '.backup_' in name}
   assert backups
   assert any(json.loads(bytes.fromhex(value))=={'alpha':alpha,'beta':initial} for value in backups.values())
   (fixture/'pets-before.json').write_text(json.dumps({'parent':str(store.filepath.parent),'files':files,'values':{key:store.load(key) for key in store.list_pets()}}))
  (fixture/(label+'-pet.json')).write_text(json.dumps(store.load(label)))
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
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  before=json.loads((fixture/'pets-before.json').read_text());parent=Path(before['parent'])
  selectors=tuple(fixture/name/'config.toml' for name in ('alpha','beta'))
  options={'staging_parent':fixture}
  preview=preview_capture(selectors,options=options)
  aliases=[item for item in preview.items if item.owner=='tamagotchi.config' and item.status=='included']
  evidence={'complete':preview.complete,'issues':preview.issues,'aliases':[{'id':item.logical_id,'path':str(item.path),'dependencies':item.dependencies,'shared_group':item.shared_group} for item in aliases]}
  (fixture/'pet-preview.json').write_text(json.dumps(evidence,indent=2))
  assert preview.complete,evidence
  for name in before['files']:
   rows=[item for item in aliases if item.path==parent/name]
   assert len(rows)==2 and rows[0].shared_group and rows[0].shared_group==rows[1].shared_group
   assert len({row.logical_id.split(':')[1] for row in rows})==2
   assert all('profile:'+row.logical_id.split(':')[1]+':config' in row.dependencies for row in rows)
  assert len(aliases)==2*len(before['files'])
  archive=fixture/'pets.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,selectors,preview.scope_digest,archive,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes)
  assert len(manifest['profile_ids'])==2 and manifest['consistency']=='coherent'
  source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id']=='tamagotchi.config']
  assert len(rows)==len(aliases)
  for row in rows:
   name=source[row['logical_id']].path.name
   assert (captured.root/row['payload']).read_bytes()==bytes.fromhex(before['files'][name])
  assert {p.name:p.read_bytes().hex() for p in parent.iterdir()}==before['files']
  sealed=await asyncio.to_thread(write_archive,captured,archive,password=None,cancel=cancel)
  assert sealed.path==archive
  (fixture/'pet-capture.json').write_text(json.dumps({'manifest':manifest,'pet_rows':rows,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
  labels={row['logical_id'].split(':')[1]:source[row['logical_id']].path.parent.name for row in manifest['files'] if row['owner_id']=='config'}
  selected=fixture/'restored';new_home=fixture/'restore-home';mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   profile=(member['logical_id'] if row.get('synthetic') else key).split(':')[1]
   label=labels[profile];old=fixture/label;new=selected/label;data=old/'data'/'default_user'
   if owner=='tamagotchi.config':
    assert original==parent
    target=new_home/'.config'/'tldw_chatbook'
   elif owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==data or data in original.parents:target=new/'data'/('recovered-'+label)/original.relative_to(data)
   elif original==old:target=new/'config'
   elif original==old/'custom':target=new/'custom'
   elif original==fixture/'shared':target=selected/'shared'
   else:
    assert owner=='persona.visual_identity_builtin',(key,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  for profile,label in labels.items():mapping['profile:'+profile+':paths.data_dir']=str(selected/label/'data')
  originals=[*selectors,*(parent/name for name in before['files'])]
  (fixture/'pet-restore-input.json').write_text(json.dumps({'archive':str(archive),'labels':labels,'mapping':mapping,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}}))
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
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
receipt=json.loads((fixture/'pet-restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive)
import zipfile
before=json.loads((fixture/'pets-before.json').read_text())
with zipfile.ZipFile(archive.path) as verified:
 for row in doc.files:
  if row.owner_id=='tamagotchi.config':
   assert verified.read(row.payload)==bytes.fromhex(before['files'][Path(row.relative_path).name])
assert Path.home()==fixture/'restore-home'
(fixture/'restored').mkdir(mode=0o700)
parent=Path.home()/'.config'/'tldw_chatbook';parent.mkdir(parents=True,mode=0o700)
assert not list(parent.iterdir())
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={key:'recovered-'+label for key,label in receipt['labels'].items()})
(fixture/'pet-plan.json').write_text(json.dumps({'issues':plan.issues,'restore':[(key,str(path)) for key,path in plan.restore]}))
control=fixture/'restored-control'
first=restore_isolated(archive,plan,control,threading.Event())
service=RecoveryService(control)
try:
 rows=service.profiles();assert len(rows)==2 and first in {row['profile_id'] for row in rows}
 profiles=[]
 for row in rows:
  entry=_launch_descriptor(row['profile_id'],control)
  assert row['status']=='restoration_validated' and 'tamagotchi.config' in row['pending_owners']
  profiles.append({'profile_id':entry.profile_id,'label':receipt['labels'][entry.source_profile]})
 (fixture/'pet-restored-profiles.json').write_text(json.dumps(profiles))
finally:service.close()
assert {p.name:p.read_bytes().hex() for p in parent.iterdir()}==before['files']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in receipt['source_hashes'].items())
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
selected_profile=os.environ['PET_PROFILE']
select_profile(selected_profile,fixture/'restored-control')
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
before=json.loads((fixture/'pets-before.json').read_text())
parent=Path.home()/'.config'/'tldw_chatbook'
assert parent!=Path(before['parent'])
assert {p.name:p.read_bytes().hex() for p in parent.iterdir()}==before['files']
store=ConfigFileStorage()
assert store.filepath==parent/'tamagotchi_pets.json'
assert set(store.list_pets())==set(before['values'])
assert {key:store.load(key) for key in store.list_pets()}==before['values']
assert {p.name:p.read_bytes().hex() for p in parent.iterdir()}==before['files']
receipt=json.loads((fixture/'pet-restore-input.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in receipt['source_hashes'].items())
assert not blocked_attempts(),blocked_attempts()
print('FRESH_DEFAULT_PETS_READ',selected_profile)
"""
)


def test_native_shared_home_pets_and_backups_restore_for_both_profiles(tmp_path):
    root = tmp_path.resolve()
    for name in (
        "home",
        "xdg-config",
        "xdg-data",
        "cache",
        "tmp",
        "shared",
        "restore-home",
        "restore-config",
        "restore-data",
        "restore-cache",
        "restore-tmp",
    ):
        (root / name).mkdir(mode=0o700)
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
        if key in os.environ
    }
    env.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "xdg-config"),
        XDG_DATA_HOME=str(root / "xdg-data"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        PET_FIXTURE=str(root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    for label in ("alpha", "beta"):
        profile = root / label
        (profile / "custom").mkdir(parents=True, mode=0o700)
        (profile / "data").mkdir(mode=0o700)
        selector = profile / "config.toml"
        selector.write_text(
            '[general]\nusers_name="default_user"\ndefault_tab="settings"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n[AppRAGSearchConfig.rag.indexing]\nenabled=false\n'
            + f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
            + "".join(
                f"{key}={json.dumps(str(path))}\n"
                for key, path in (
                    ("chachanotes_db_path", profile / "custom" / "notes.db"),
                    ("media_db_path", profile / "custom" / "media.db"),
                    ("research_db_path", profile / "custom" / "research.db"),
                    ("prompts_db_path", root / "shared" / "prompts.db"),
                )
            )
        )
        selector.chmod(0o600)
        _run_profile_child(
            root, "seed-" + label, _SEED, dict(env, TLDW_CONFIG_PATH=str(selector))
        )
    _run_profile_child(
        root,
        "capture",
        _CAPTURE,
        dict(env, TLDW_CONFIG_PATH=str(root / "alpha" / "config.toml")),
    )
    fresh = dict(
        env,
        HOME=str(root / "restore-home"),
        USERPROFILE=str(root / "restore-home"),
        XDG_CONFIG_HOME=str(root / "restore-config"),
        XDG_DATA_HOME=str(root / "restore-data"),
        XDG_CACHE_HOME=str(root / "restore-cache"),
        TMPDIR=str(root / "restore-tmp"),
        TLDW_CONFIG_PATH=str(root / "alpha" / "config.toml"),
    )
    _run_profile_child(root, "restore", _RESTORE, fresh)
    for row in json.loads((root / "pet-restored-profiles.json").read_text()):
        _run_profile_child(
            root,
            "read-" + row["label"],
            _READ,
            dict(fresh, PET_PROFILE=row["profile_id"]),
        )


def _pet_records(backup=False):
    from tldw_chatbook.Backup_Recovery.archive_models import (
        Directory,
        Metadata,
        Payload,
    )

    leaf = (
        "tamagotchi_pets.backup_20260101_121314.json"
        if backup
        else "tamagotchi_pets.json"
    )
    key = "profile:actual:tamagotchi.config" + (":" + leaf if backup else "")
    root = Directory(
        logical_id="root",
        root_id="root",
        parent_id=None,
        relative_path="",
        synthetic=True,
        metadata=Metadata(version=1, mode=0o700, mtime_ns=0),
    )
    payload = Payload(
        logical_id=key,
        root_id="root",
        parent_id="root",
        relative_path=leaf,
        owner_id="tamagotchi.config",
        payload="payload/pet",
        size=1,
        sha256="0" * 64,
    )
    return root, payload


@pytest.mark.parametrize("backup", [False, True])
def test_pet_owner_accepts_only_its_existing_default_locator(
    tmp_path, monkeypatch, backup
):
    from tldw_chatbook.Widgets.Tamagotchi.recovery import recovery_adapters

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    root, payload = _pet_records(backup)
    parent = (
        tmp_path / "appdata" if os.name == "nt" else tmp_path / ".config"
    ) / "tldw_chatbook"
    assert (
        recovery_adapters()[0]._restore_path("actual", payload, root)
        == parent / payload.relative_path
    )
    assert not parent.exists()


@pytest.mark.parametrize(
    "mode",
    [
        "owner",
        "profile",
        "canonical_id",
        "backup_id",
        "backup_suffix",
        "nested",
        "concrete",
        "parent",
        "root",
        "nested_root",
    ],
)
def test_pet_owner_rejects_unrecognized_payload_mapping(mode):
    from tldw_chatbook.Widgets.Tamagotchi.recovery import recovery_adapters

    root, payload = _pet_records()
    if mode == "owner":
        payload = payload.model_copy(update={"owner_id": "runtime.source_state"})
    elif mode == "profile":
        payload = payload.model_copy(
            update={"logical_id": "profile:foreign:tamagotchi.config"}
        )
    elif mode == "canonical_id":
        payload = payload.model_copy(
            update={"logical_id": payload.logical_id + ":extra"}
        )
    elif mode == "backup_id":
        payload = payload.model_copy(
            update={"relative_path": "tamagotchi_pets.backup_20260101_121314.json"}
        )
    elif mode == "backup_suffix":
        leaf = "tamagotchi_pets.backup_wrong.json"
        payload = payload.model_copy(
            update={
                "relative_path": leaf,
                "logical_id": payload.logical_id + ":" + leaf,
            }
        )
    elif mode == "nested":
        payload = payload.model_copy(
            update={"relative_path": "nested/tamagotchi_pets.json"}
        )
    elif mode == "concrete":
        root = root.model_copy(update={"synthetic": False})
    elif mode == "parent":
        payload = payload.model_copy(update={"parent_id": "other"})
    elif mode == "root":
        root = root.model_copy(update={"root_id": "other"})
    else:
        root = root.model_copy(update={"parent_id": "ancestor"})
    with pytest.raises(
        ValueError, match="owner_relocation_unverified:tamagotchi.config"
    ):
        recovery_adapters()[0]._restore_path("actual", payload, root)


@pytest.mark.parametrize("backup", [False, True])
@pytest.mark.parametrize("wrong_parent", [False, True])
def test_pet_staging_requires_exact_default_destination(
    tmp_path, monkeypatch, backup, wrong_parent
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.staging import _config_targets
    from tldw_chatbook.Widgets.Tamagotchi.recovery import recovery_adapters

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    root, payload = _pet_records(backup)
    parent = (
        tmp_path / "appdata" if os.name == "nt" else tmp_path / ".config"
    ) / "tldw_chatbook"
    destination = (
        parent / "foreign" if wrong_parent else parent
    ) / payload.relative_path
    owner = recovery_adapters()[0]
    doc = SimpleNamespace(directories=(root,), files=(payload,))
    plan = SimpleNamespace(
        restore=((payload.logical_id, destination),),
        destinations=((root.logical_id, destination.parent),),
        issues=(),
    )
    if wrong_parent:
        with pytest.raises(
            ValueError, match="owner_relocation_unverified:tamagotchi.config"
        ):
            _config_targets(
                {},
                "actual",
                tmp_path / "config.toml",
                doc,
                plan,
                {owner.owner_id: owner},
            )
    else:
        _config_targets(
            {}, "actual", tmp_path / "config.toml", doc, plan, {owner.owner_id: owner}
        )
    assert not parent.exists()
