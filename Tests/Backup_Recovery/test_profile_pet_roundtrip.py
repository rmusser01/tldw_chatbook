"""Per-profile pet state and its dated backups across backup and isolated restore.

TASK-42000: each profile keeps its own pets beside its config. Both profiles
start from one copy of the old shared pet file, which never changes; a change
under one profile is not seen by the other; a backup holds each profile's own
pet rows, and an isolated restore puts them beside each restored config.
"""

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
shared=Path.home()/'.config'/'tldw_chatbook'
def files(folder):
 return {p.name:p.read_bytes().hex() for p in folder.iterdir() if p.name.startswith('tamagotchi_pets')}
LEGACY=json.loads((fixture/'legacy.json').read_text())
"""

_SEED = (
    _PRIVATE
    + r"""
selector=Path(os.environ['TLDW_CONFIG_PATH']);label=selector.parent.name
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
# The stores this profile's config requires (its custom research database
# too): before this, the backup preview stopped at missing_required.
preview=preview_capture((selector,),options={'allow_partial':True})
_populate_required_dependencies(preview)
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
research=LocalResearchService(next(item.path for item in preview.items if item.logical_id.endswith(':research.local')))
if hasattr(research,'_ensure_schema'):research._ensure_schema()
for closer in ('close','close_connection'):
 if hasattr(research,closer):getattr(research,closer)();break
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
async def main():
 app=TldwCli()
 try:
  store=ConfigFileStorage()
  assert store.filepath==selector.parent/'tamagotchi_pets.json',str(store.filepath)
  # Both profiles start from one copy of the old shared pets (AC#4), and
  # beta never sees the pet alpha saved (AC#1).
  assert {key:store.load(key) for key in store.list_pets()}=={'legacy':LEGACY}
  state={'name':label+' native pet','happiness':72,'hunger':31,'energy':84,'health':93,'age':4}
  assert store.save(label,state)
  initial=store.load(label)
  assert all(initial[key]==value for key,value in state.items()) and initial['last_saved']
  state['happiness']=81
  assert store.save(label,state)
  assert store.load(label)['happiness']==81
  before=files(selector.parent)
  assert 'tamagotchi_pets.seeded' in before and any('.backup_' in name for name in before),sorted(before)
  (fixture/(label+'-pets.json')).write_text(json.dumps({'files':before,'values':{key:store.load(key) for key in store.list_pets()}}))
  assert files(shared)==json.loads((fixture/'shared-before.json').read_text())
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
  labels=('alpha','beta')
  before={label:json.loads((fixture/(label+'-pets.json')).read_text()) for label in labels}
  selectors=tuple(fixture/name/'config.toml' for name in labels)
  options={'staging_parent':fixture}
  preview=preview_capture(selectors,options=options)
  pets=[item for item in preview.items if item.owner=='tamagotchi.config' and item.status=='included']
  evidence={'complete':preview.complete,'issues':preview.issues,'missing':[(i.owner,i.status) for i in preview.items if i.status in {'missing_required','unsupported','unavailable'}],
   'pets':[{'id':item.logical_id,'path':str(item.path),'shared_group':item.shared_group} for item in pets]}
  (fixture/'pet-preview.json').write_text(json.dumps(evidence,indent=2))
  assert preview.complete,evidence
  # Each profile holds its own pet rows, never an alias of the other's.
  assert not any(item.shared_group for item in pets),evidence
  assert {(item.path.parent.name,item.path.name) for item in pets}=={(label,name) for label in labels for name in before[label]['files']},evidence
  archive=fixture/'pets.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,selectors,preview.scope_digest,archive,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes)
  assert len(manifest['profile_ids'])==2 and manifest['consistency']=='coherent'
  source={item.logical_id:item for item in captured.inventory.items}
  rows=[row for row in manifest['files'] if row['owner_id']=='tamagotchi.config']
  assert len(rows)==len(pets)
  for row in rows:
   path=source[row['logical_id']].path
   assert (captured.root/row['payload']).read_bytes()==bytes.fromhex(before[path.parent.name]['files'][path.name])
  sealed=await asyncio.to_thread(write_archive,captured,archive,password=None,cancel=cancel)
  assert sealed.path==archive
  labels={row['logical_id'].split(':')[1]:source[row['logical_id']].path.parent.name for row in manifest['files'] if row['owner_id']=='config'}
  selected=fixture/'restored';mapping={}
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id'];member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   profile=(member['logical_id'] if row.get('synthetic') else key).split(':')[1]
   label=labels[profile];old=fixture/label;new=selected/label;data=old/'data'/'default_user'
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original==data or data in original.parents:target=new/'data'/('recovered-'+label)/original.relative_to(data)
   elif original==old:target=new/'config'
   elif original==old/'custom':target=new/'custom'
   elif original==fixture/'shared':target=selected/'shared'
   else:
    assert owner=='persona.visual_identity_builtin',(key,str(original))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  for profile,label in labels.items():mapping['profile:'+profile+':paths.data_dir']=str(selected/label/'data')
  (fixture/'pet-restore-input.json').write_text(json.dumps({'archive':str(archive),'labels':labels,'mapping':mapping}))
  assert files(shared)==json.loads((fixture/'shared-before.json').read_text())
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
assert Path.home()==fixture/'restore-home'
(fixture/'restored').mkdir(mode=0o700)
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
  label=receipt['labels'][entry.source_profile]
  # Each profile's pets restore beside its own restored config (AC#3).
  assert Path(entry.config).parent==fixture/'restored'/label/'config',entry.config
  assert files(Path(entry.config).parent)==json.loads((fixture/(label+'-pets.json')).read_text())['files']
  profiles.append({'profile_id':entry.profile_id,'label':label})
 (fixture/'pet-restored-profiles.json').write_text(json.dumps(profiles))
finally:service.close()
# The restore never writes an old shared pet folder.
assert not shared.exists()
assert not blocked_attempts(),blocked_attempts()
"""
)

_READ = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
selected_profile,label=os.environ['PET_PROFILE'],os.environ['PET_LABEL']
select_profile(selected_profile,fixture/'restored-control')
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
before=json.loads((fixture/(label+'-pets.json')).read_text())
folder=fixture/'restored'/label/'config'
store=ConfigFileStorage()
assert store.filepath==folder/'tamagotchi_pets.json',str(store.filepath)
assert {key:store.load(key) for key in store.list_pets()}==before['values']
assert files(folder)==before['files']
assert not shared.exists()
assert not blocked_attempts(),blocked_attempts()
print('FRESH_PROFILE_PETS_READ',selected_profile)
"""
)

_LEGACY = {"name": "Old shared pet", "happiness": 55, "hunger": 40, "energy": 60, "health": 88, "age": 9}


@pytest.mark.timeout(1800 if os.name == "nt" else 900)
def test_each_profile_keeps_its_own_pets_through_backup_and_isolated_restore(tmp_path):
    """TASK-42000 AC#1, AC#3 (isolated restore) and AC#4."""
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
    # The old shared pet folder every profile used before TASK-42000.
    shared = root / "home" / ".config" / "tldw_chatbook"
    shared.mkdir(parents=True, mode=0o700)
    (shared / "tamagotchi_pets.json").write_text(json.dumps({"legacy": _LEGACY}))
    (shared / "tamagotchi_pets.backup_20250101_000000.json").write_text("{}")
    (root / "legacy.json").write_text(json.dumps(_LEGACY))

    def snapshot():
        return {
            path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in shared.iterdir()
        }

    before_shared = snapshot()
    (root / "shared-before.json").write_text(
        json.dumps({name: data.hex() for name, (data, _, _) in before_shared.items()})
    )
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
            dict(fresh, PET_PROFILE=row["profile_id"], PET_LABEL=row["label"]),
        )
    assert snapshot() == before_shared


_LEAVES = {
    "current": ("tamagotchi_pets.json", ""),
    "dated": (
        "tamagotchi_pets.backup_20260101_121314.json",
        ":tamagotchi_pets.backup_20260101_121314.json",
    ),
    "marker": ("tamagotchi_pets.seeded", ":seeded"),
}


def _pet_records(kind="current", root_id="root"):
    from tldw_chatbook.Backup_Recovery.archive_models import (
        Directory,
        Metadata,
        Payload,
    )

    leaf, suffix = _LEAVES[kind]
    root = Directory(
        logical_id=root_id,
        root_id=root_id,
        parent_id=None,
        relative_path="",
        synthetic=True,
        metadata=Metadata(version=1, mode=0o700, mtime_ns=0),
    )
    payload = Payload(
        logical_id="profile:actual:tamagotchi.config" + suffix,
        root_id=root_id,
        parent_id=root_id,
        relative_path=leaf,
        owner_id="tamagotchi.config",
        payload="payload/pet",
        size=1,
        sha256="0" * 64,
    )
    return root, payload


@pytest.mark.parametrize(
    "mode",
    [
        "owner",
        "profile",
        "canonical_id",
        "backup_id",
        "backup_suffix",
        "marker_id",
        "nested",
        "concrete",
        "parent",
        "root",
        "nested_root",
    ],
)
def test_pet_owner_rejects_unrecognized_payload_mapping(tmp_path, mode):
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
    elif mode == "marker_id":
        payload = payload.model_copy(
            update={"relative_path": "tamagotchi_pets.seeded"}
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
        recovery_adapters()[0]._restore_path(
            "actual", payload, root, tmp_path / "config.toml"
        )


def _staged(tmp_path, records, destinations, *, derive):
    """Run staging's owner check on archived rows mapped to their roots."""
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.staging import _config_targets
    from tldw_chatbook.Widgets.Tamagotchi.recovery import (
        recovery_adapters as pet_adapters,
    )

    roots = {root.logical_id: root for root, _ in records}
    payloads = tuple(payload for _, payload in records)
    doc = SimpleNamespace(
        directories=tuple(roots.values()), files=payloads, producer_inventory=()
    )
    plan = SimpleNamespace(
        restore=tuple(
            (row.logical_id, destinations[row.root_id] / row.relative_path)
            for row in payloads
        ),
        destinations=tuple(destinations.items()),
        issues=(),
    )
    owners = {
        owner.owner_id: owner for owner in (*recovery_adapters(), *pet_adapters())
    }
    data = {
        "general": {"users_name": "u"},
        "paths": {"data_dir": str(tmp_path / "data")},
    }
    try:
        return _config_targets(
            data,
            "actual",
            tmp_path / "profile" / "config.toml",
            doc,
            plan,
            owners,
            derive=derive,
        )
    except ValueError as error:
        return str(error)


@pytest.mark.parametrize("derive", [False, True])
@pytest.mark.parametrize("kind", sorted(_LEAVES))
@pytest.mark.parametrize("wrong_parent", [False, True])
def test_pet_staging_requires_the_profiles_config_folder(
    tmp_path, derive, kind, wrong_parent
):
    """TASK-42000: staging accepts a pet file only beside the target config."""
    folder = tmp_path / "profile"
    root = folder / "foreign" if wrong_parent else folder
    result = _staged(tmp_path, [_pet_records(kind)], {"root": root}, derive=derive)
    if derive:
        # Deriving computes the root's folder; it reads no reviewed destination.
        assert result == {"root": folder}
    else:
        assert result == (
            "owner_relocation_unverified:tamagotchi.config" if wrong_parent else {}
        )
    assert not folder.exists()


def _co_rooted(other):
    """A pet file sharing one synthetic root with another owner's file."""
    from tldw_chatbook.Backup_Recovery.archive_models import Payload

    root, pet = _pet_records()
    leaf, owner, key = {
        "config": ("config.toml", "config", "profile:actual:config"),
        "data": ("tldw_chatbook_personas.json", "personas", "profile:actual:personas"),
    }[other]
    neighbour = Payload(
        logical_id=key,
        root_id=root.logical_id,
        parent_id=root.logical_id,
        relative_path=leaf,
        owner_id=owner,
        payload="payload/other",
        size=1,
        sha256="0" * 64,
    )
    return [(root, pet), (root, neighbour)]


@pytest.mark.parametrize("derive", [False, True])
def test_co_rooted_pet_and_config_restore_together_beside_the_config(
    tmp_path, derive
):
    """TASK-42000 (review point 2): a crafted archive whose pet file shares a
    root with the profile's config restores both beside the config."""
    result = _staged(
        tmp_path, _co_rooted("config"), {"root": tmp_path / "profile"}, derive=derive
    )
    assert result == ({"root": tmp_path / "profile"} if derive else {})


@pytest.mark.parametrize("derive", [False, True])
@pytest.mark.parametrize("folder", ["config", "data"])
def test_co_rooted_pet_and_data_file_refuse_by_name(tmp_path, derive, folder):
    """TASK-42000 (review point 2): a pet file sharing a root with a file
    from another folder is never dropped; the restore refuses by name."""
    destination = (
        tmp_path / "profile" if folder == "config" else tmp_path / "data" / "u"
    )
    refusal = _staged(tmp_path, _co_rooted("data"), {"root": destination}, derive=derive)
    assert isinstance(refusal, str) and refusal.startswith(
        "owner_relocation_unverified:"
    ), refusal
