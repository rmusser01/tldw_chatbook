"""Actual two-profile capture of a shared registry and profile-local ZIPs."""

import json
import os
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_complete_roundtrip import _run_profile_child as _child
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_PRIVATE = r"""
import asyncio,hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['SHARED_CHATBOOK_FIXTURE'])
selector=Path(os.environ['TLDW_CONFIG_PATH'])
label=selector.parent.name
from tldw_chatbook.app import TldwCli
"""

_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.Chatbooks.database_paths import get_private_chatbooks_dir
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
async def main():
 app=TldwCli()
 try:
  books=app.local_chatbook_service
  assert app.prompts_db.db_path==fixture/'shared'/'prompts.db'
  assert books.registry_path==fixture/'shared'/'tldw_chatbook_chatbooks.json'
  note=app.chachanotes_db.add_note(label+' retained note',label+' exact native note bytes')
  archive=get_private_chatbooks_dir()/(label+'.zip')
  exported=await books.export_chatbook({'name':label+' retained book','output_path':str(archive),'content_selections':{ContentType.NOTE:[str(note)]}})
  assert exported['success'],exported
  record=await books.create_chatbook(name=label+' retained book',file_path=archive,metadata={'profile_label':label})
  preview=await books.preview_chatbook(archive)
  assert preview['success'],preview
  (fixture/(label+'-seed.json')).write_text(json.dumps({'note':note,'archive':str(archive),'digest':hashlib.sha256(archive.read_bytes()).hexdigest(),'registry':str(books.registry_path),'record':record,'manifest':preview['manifest']}))
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
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(50,cancel.set)
 try:
  seeds=[json.loads((fixture/(name+'-seed.json')).read_text()) for name in ('alpha','beta')]
  registry=app.local_chatbook_service.registry_path
  original=registry.read_bytes()
  records=await app.local_chatbook_service.list_chatbooks()
  assert {row['id'] for row in records}=={seed['record']['id'] for seed in seeds}
  assert len({seed['registry'] for seed in seeds})==1
  selectors=tuple(fixture/name/'config.toml' for name in ('alpha','beta'))
  options={'staging_parent':fixture}
  preview=preview_capture(selectors,options=options)
  aliases=[item for item in preview.items if item.owner=='chatbooks.registry']
  before={'complete':preview.complete,'issues':preview.issues,'registry_aliases':[{'id':item.logical_id,'path':str(item.path),'dependencies':item.dependencies,'shared_group':item.shared_group} for item in aliases]}
  (fixture/'shared-registry-preview.json').write_text(json.dumps(before,indent=2))
  assert preview.complete,before
  assert len(aliases)==2 and all(item.path==registry for item in aliases)
  assert aliases[0].shared_group and aliases[0].shared_group==aliases[1].shared_group
  destination=fixture/'shared-registry.tldw-backup.zip'
  captured=await asyncio.to_thread(capture,selectors,preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert captured.inventory.complete
  manifest=json.loads(captured.manifest_bytes)
  assert len(manifest['profile_ids'])==2 and manifest['consistency']=='coherent'
  source={item.logical_id:item for item in captured.inventory.items}
  files={row['logical_id']:row for row in manifest['files']}
  archive_paths={Path(seed['archive']) for seed in seeds}
  archives={key for key,row in files.items() if row['owner_id']=='chatbooks.archives' and source[key].path in archive_paths}
  assert len(archives)==2
  for seed in seeds:
   key=next(key for key in archives if source[key].path==Path(seed['archive']))
   assert hashlib.sha256((captured.root/files[key]['payload']).read_bytes()).hexdigest()==seed['digest']
  documents={key:json.loads((captured.root/row['payload']).read_text()) for key,row in files.items() if row['owner_id']=='chatbooks.registry'}
  (fixture/'shared-registry-captured.json').write_text(json.dumps({'documents':documents,'archives':sorted(archives)},indent=2))
  assert len(documents)==2
  for key,document in documents.items():
   rows=document['records']
   assert {row['id'] for row in rows}=={seed['record']['id'] for seed in seeds}
   for seed in seeds:
    row=next(row for row in rows if row['id']==seed['record']['id'])
    assert row['name']==seed['record']['name'] and row['metadata']==seed['record']['metadata']
    reference=row['__chatbook_archive_reference']
    assert set(reference)=={'logical_id'},(key,row,archives)
    assert reference['logical_id'] in archives and reference['logical_id'] in source[key].dependencies
    assert source[reference['logical_id']].path==Path(seed['archive'])
  assert registry.read_bytes()==original
  assert all(hashlib.sha256(Path(seed['archive']).read_bytes()).hexdigest()==seed['digest'] for seed in seeds)
  assert not destination.exists()
  from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
  sealed=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=cancel)
  assert sealed.path==destination
  labels={key.split(':')[1]:source[key].path.parent.name for key,row in files.items() if row['owner_id']=='config'}
  mapping={};selected=fixture/'restored'
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   key=row['logical_id']
   member=next((entry for entry in manifest['files'] if entry['root_id']==key),None)
   original_path=source[key].path if key in source else source[member['logical_id']].path.parent
   owner=member['owner_id'] if row.get('synthetic') else source[key].owner
   profile=(member['logical_id'] if row.get('synthetic') else key).split(':')[1]
   name=labels[profile];old=fixture/name;new=selected/name
   data=old/'data'/'default_user'
   if owner=='eval.definitions':target=selected/'inactive-eval'
   elif original_path==data or data in original_path.parents:
    target=new/'data'/('recovered-'+name)/original_path.relative_to(data)
   elif original_path==old:target=new/'config'
   elif original_path==old/'custom':target=new/'custom'
   elif original_path==fixture/'shared':target=selected/'shared'
   else:
    assert owner=='persona.visual_identity_builtin',(key,str(original_path))
    target=selected/'inactive-builtin'
   mapping[key]=str(target)
  for profile,name in labels.items():mapping['profile:'+profile+':paths.data_dir']=str(selected/name/'data')
  original_paths=[registry,*archive_paths,*selectors]
  (fixture/'restore-input.json').write_text(json.dumps({'archive':str(destination),'labels':labels,'mapping':mapping,'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths}},indent=2))
  assert not blocked_attempts(),blocked_attempts()
 finally:
  cancel.set();watchdog.cancel();monitoring.cancel()
  await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)

_RESTORE = r"""
import hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
fixture=Path(os.environ['SHARED_CHATBOOK_FIXTURE'])
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
receipt=json.loads((fixture/'restore-input.json').read_text())
archive=acquire(Path(receipt['archive']),fixture/'acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive)
(fixture/'restored').mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={key:Path(path) for key,path in receipt['mapping'].items()},target=None,profile_names={key:'recovered-'+name for key,name in receipt['labels'].items()})
control=fixture/'restored-control'
first=restore_isolated(archive,plan,control,threading.Event())
service=RecoveryService(control)
try:
 rows=service.profiles()
 assert len(rows)==2 and first in {row['profile_id'] for row in rows}
 profiles=[]
 for row in rows:
  entry=_launch_descriptor(row['profile_id'],control)
  assert row['status']=='restoration_validated'
  profiles.append({'profile_id':entry.profile_id,'label':receipt['labels'][entry.source_profile]})
 (fixture/'restored-profiles.json').write_text(json.dumps(profiles))
finally:service.close()
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in receipt['source_hashes'].items())
assert not blocked_attempts(),blocked_attempts()
"""

_READ = r"""
import asyncio,hashlib,json,os,sys,zipfile
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
fixture=Path(os.environ['SHARED_CHATBOOK_FIXTURE'])
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(os.environ['SHARED_CHATBOOK_PROFILE'],fixture/'restored-control')
from tldw_chatbook.app import TldwCli
async def main():
 app=TldwCli()
 try:
  books=app.local_chatbook_service
  assert books.registry_path==fixture/'restored'/'shared'/'tldw_chatbook_chatbooks.json'
  before=books.registry_path.read_bytes()
  records=await books.list_chatbooks()
  assert len(records)==2
  for name in ('alpha','beta'):
   seed=json.loads((fixture/(name+'-seed.json')).read_text())
   record=await books.get_chatbook(seed['record']['id'])
   expected=fixture/'restored'/name/'data'/('recovered-'+name)/'chatbooks'/(name+'.zip')
   assert record['file_path']==str(expected) and '__chatbook_archive_reference' not in record
   for field in ('id','name','metadata','created_at','updated_at'):
    assert record[field]==seed['record'][field]
   assert hashlib.sha256(expected.read_bytes()).hexdigest()==seed['digest']
   preview=await books.preview_chatbook(expected)
   assert preview['success'] and preview['manifest']==seed['manifest'],preview
   content=next(item for item in preview['manifest']['content_items'] if item['id']==str(seed['note']))
   with zipfile.ZipFile(expected) as archive:
    assert name+' exact native note bytes' in archive.read(content['file_path']).decode()
  assert books.registry_path.read_bytes()==before
  receipt=json.loads((fixture/'restore-input.json').read_text())
  assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in receipt['source_hashes'].items())
  assert not blocked_attempts(),blocked_attempts()
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""


def test_actual_shared_registry_capture_preserves_both_profile_zip_references(tmp_path):
    root = tmp_path.resolve()
    for name in ("home", "xdg-config", "xdg-data", "cache", "tmp", "shared"):
        (root / name).mkdir(mode=0o700)
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
        SHARED_CHATBOOK_FIXTURE=str(root),
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
            '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
            "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
            f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n[database]\n"
            + "".join(
                f"{key}={json.dumps(str(path))}\n"
                for key, path in (
                    ("chachanotes_db_path", profile / "custom" / "notes.db"),
                    ("media_db_path", profile / "custom" / "media.db"),
                    ("research_db_path", profile / "custom" / "research.db"),
                    ("prompts_db_path", root / "shared" / "prompts.db"),
                )
            ),
            encoding="utf-8",
        )
        selector.chmod(0o600)
        _child(
            root,
            "seed-" + label,
            _SEED,
            dict(environment, TLDW_CONFIG_PATH=str(selector)),
        )
    _child(
        root,
        "capture",
        _CAPTURE,
        dict(environment, TLDW_CONFIG_PATH=str(root / "alpha" / "config.toml")),
    )
    _child(root, "restore", _RESTORE, environment)
    for profile in json.loads((root / "restored-profiles.json").read_text()):
        _child(
            root,
            "read-" + profile["label"],
            _READ,
            dict(environment, SHARED_CHATBOOK_PROFILE=profile["profile_id"]),
        )


_POLICY = r"""
import json,sys
from dataclasses import replace
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.config_adapter import _ChatbookRegistry
from tldw_chatbook.Backup_Recovery.models import StorageItem
root=Path.home();candidate=root/'candidate.json';route=sys.argv[1]
key='profile:q:chatbooks.archives:zip'
item=StorageItem('chatbooks.registry','profile:p:chatbooks.registry',root/'registry.json','included',(key,),'shared:registry')
peer=replace(item,logical_id='profile:q:chatbooks.registry')
archive=StorageItem('chatbooks.archives',key,root/'selected.zip','included',())
if route=='different_path':peer=replace(peer,path=root/'other.json')
if route=='different_group':peer=replace(peer,shared_group='shared:other')
if route=='wrong_peer_owner':peer=replace(peer,owner='skills')
if route=='wrong_archive_owner':archive=replace(archive,owner='skills')
if route=='missing_dependency':peer=replace(peer,dependencies=())
if route=='wrong_profile':peer=replace(peer,logical_id='profile:r:chatbooks.registry')
if route=='peer_unselected':peer=replace(peer,status='unused')
sources=(item,peer,archive)
if route=='missing_peer':sources=(item,archive)
if route=='missing_zip':sources=(item,peer)
if route=='no_context':sources=()
mapping={key:archive.path}
if route=='wrong_mapping':mapping[key]=root/'different.zip'
if route=='missing_mapping':mapping={}
record={'id':'book','file_path':None,'__chatbook_archive_reference':{'logical_id':key}}
if route=='external':record['__chatbook_archive_reference']={'status':'unresolved'}
document=json.dumps({'records':[record]});candidate.write_text(document);candidate.chmod(0o600)
owner=_ChatbookRegistry('chatbooks.registry')
valid=route in ('positive','external','wrong_mapping','missing_mapping')
try:owner.validate_restore_reference_owners(item,candidate,{key:archive.owner},sources)
except ValueError:
 assert not valid,route
else:assert valid,route
assert candidate.read_text()==document
try:owner.relocate_restore(item,candidate,mapping,sources)
except ValueError:
 assert route not in ('positive','external'),route
 assert candidate.read_text()==document
else:
 assert route in ('positive','external'),route
 result=json.loads(candidate.read_text())['records'][0]
 if route=='positive':assert result['file_path']==str(archive.path) and '__chatbook_archive_reference' not in result
 else:assert result==record
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    (
        "positive",
        "external",
        "different_path",
        "different_group",
        "wrong_peer_owner",
        "wrong_archive_owner",
        "missing_dependency",
        "wrong_profile",
        "peer_unselected",
        "missing_peer",
        "missing_zip",
        "no_context",
        "wrong_mapping",
        "missing_mapping",
    ),
)
def test_shared_reference_requires_exact_selected_peer_and_archive(tmp_path, route):
    _run(tmp_path, route, "", script=_POLICY)


_ORDER = r"""
import json,sys
from dataclasses import replace
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.config_adapter import _ChatbookRegistry
from tldw_chatbook.Backup_Recovery.models import StorageItem
root=Path.home();archive_path=root/'shared.zip';archive_path.write_bytes(b'inert policy input')
left_key='profile:p:chatbooks.archives:zip';right_key='profile:q:chatbooks.archives:zip'
left=StorageItem('chatbooks.registry','profile:p:chatbooks.registry',root/'registry.json','included',(left_key,),'shared:registry')
right=replace(left,logical_id='profile:q:chatbooks.registry',dependencies=(right_key,))
left_zip=StorageItem('chatbooks.archives',left_key,archive_path,'included',(),'shared:zip')
right_zip=replace(left_zip,logical_id=right_key)
if sys.argv[1]=='different_path':right=replace(right,path=root/'foreign.json')
owner=_ChatbookRegistry('chatbooks.registry')
merged=owner.shared_dependencies((left,right,left_zip,right_zip))
if sys.argv[1]=='different_path':
 assert merged==(left,right,left_zip,right_zip)
else:
 assert all(set(item.dependencies)=={left_key,right_key} for item in merged[:2])
 observed=[]
 for index,items in enumerate((merged,tuple(reversed(merged)))):
  for item in merged[:2]:
   candidate=root/('candidate-'+str(index)+item.logical_id.split(':')[1]+'.json')
   candidate.write_text(json.dumps({'records':[{'id':'book','file_path':str(archive_path)}]}));candidate.chmod(0o600)
   owner.prepare_capture(item,candidate,items)
   observed.append(candidate.read_bytes())
 assert len(set(observed))==1
 assert json.loads(observed[0])['records'][0]['__chatbook_archive_reference']=={'logical_id':left_key}
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ("deterministic", "different_path"))
def test_shared_dependency_mapping_is_finite_and_order_independent(tmp_path, route):
    _run(tmp_path, route, "", script=_ORDER)
