"""Typed Chatbook archive references survive actual backup and native recovery."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_skills_chatbooks_capture import _LIVE

_READ_RESTORED = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(sys.argv[1],Path(sys.argv[2]))
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook import config
registry=Path(sys.argv[3]);old=Path(sys.argv[4])
assert not old.exists()
async def read():
 service=LocalChatbookService(db_paths={},registry_path=registry)
 records=await service.list_chatbooks()
 record=next(row for row in records if row['name']=='captured')
 assert record['file_path'] != str(old) and '__chatbook_archive_reference' not in record
 result=await service.preview_chatbook(Path(record['file_path']))
 assert result['success'],result
 unresolved=next(row for row in records if row['name']=='external')
 assert unresolved['file_path'] is None and unresolved['__chatbook_archive_reference']=={'status':'unresolved'}
 await service.update_chatbook(unresolved['id'],description='native save preserves inert reference')
 unresolved=next(row for row in await service.list_chatbooks() if row['name']=='external')
 assert unresolved['__chatbook_archive_reference']=={'status':'unresolved'}
 await service.update_chatbook(unresolved['id'],file_path=record['file_path'])
 relinked=next(row for row in await service.list_chatbooks() if row['name']=='external')
 assert '__chatbook_archive_reference' not in relinked and relinked['file_path']==record['file_path']
asyncio.run(read())
assert not blocked_attempts()
"""

_ROUNDTRIP = r"""
  from tldw_chatbook.Backup_Recovery.archive_reader import acquire
  from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
  acquired=acquire(destination,home/'acquired',ArchiveLimits(),None,threading.Event())
  profile=manifest['profile_ids'][0]
  destinations={}
  isolated=home/'isolated';isolated.mkdir(mode=0o700)
  from tldw_chatbook.config import get_user_data_dir
  source_data=get_user_data_dir()
  for row in manifest['directories']:
   if row['parent_id'] is not None:continue
   source=paths.get(row['logical_id'])
   if source is None:
    member=next(payload for payload in manifest['files'] if payload['root_id']==row['logical_id'])
    source=paths[member['logical_id']].parent
   if source==source_data or source_data in source.parents:
    target=isolated/'data'/'recovered'/source.relative_to(source_data)
   elif source==selector.parent:
    target=isolated/'config'
   else:
    target=isolated/'setup'/row['logical_id'].replace(':','_')
   destinations[row['logical_id']]=target
  destinations['profile:'+profile+':paths.data_dir']=isolated/'data'
  (home/'capture-receipt.json').write_text(json.dumps({'archive':str(destination),'profile':profile,'destinations':{key:str(value) for key,value in destinations.items()},'registry_id':next(row['logical_id'] for row in manifest['files'] if row['owner_id']=='chatbooks.registry'),'old_archive':str(archive)}))

"""


_RESTORE_ARCHIVE = r"""
import json,subprocess,sys,os
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
receipt=json.loads(Path(RECEIPT).read_text());home=Path.home()
acquired=acquire(Path(receipt['archive']),home/'acquired',ArchiveLimits(),None,Event())
plan=plan_restore(acquired,mode='isolated',destinations={key:Path(value) for key,value in receipt['destinations'].items()},target=None,profile_names={receipt['profile']:'recovered'})
control=home/'restore-control'
restored=restore_isolated(acquired,plan,control,Event())
selected_registry=dict(plan.restore)[receipt['registry_id']]
old_archive=Path(receipt['old_archive']);old_archive.unlink()
result=subprocess.run([sys.executable,'-c',READ_RESTORED,restored,str(control),str(selected_registry),str(old_archive)],capture_output=True,text=True,timeout=30)
assert result.returncode==0,result.stderr[-5000:]
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("source_form", ["absolute", "relative"])
def test_public_capture_restores_typed_reference_for_fresh_native_zip_reader(
    tmp_path, source_form
):
    script = (
        "READ_RESTORED = "
        + repr(_READ_RESTORED)
        + "\n"
        + _LIVE.replace(
            " live_registry=books.registry_path.read_bytes()",
            " await books.create_chatbook(name='external',file_path=home/'external-user-choice.zip')\n"
            " await books.create_chatbook(name='metadata only')\n"
            " live_registry=books.registry_path.read_bytes()",
        ).replace(
            "  assert sealed.path==destination and destination.is_file()",
            "  assert sealed.path==destination and destination.is_file()\n"
            + _ROUNDTRIP,
        )
    )
    if source_form == "relative":
        script = script.replace(
            " await books.create_chatbook(name='captured',file_path=archive)",
            " os.chdir(home)\n"
            " relative=archive.relative_to(home)\n"
            " await books.create_chatbook(name='captured',file_path=relative)\n"
            " assert (await books.list_chatbooks())[0]['file_path']==str(relative)\n"
            " assert (await books.preview_chatbook(relative))['success']",
        ).replace(
            "file_path=home/'external-user-choice.zip'",
            "file_path=Path('external-user-choice.zip')",
        )
    _run(tmp_path, "content", "complete", script=script, timeout=120)
    restore_root = tmp_path / "restore-process"
    restore_root.mkdir(mode=0o700)
    script = (
        "RECEIPT = "
        + repr(str(tmp_path / "home" / "capture-receipt.json"))
        + "\nREAD_RESTORED = "
        + repr(_READ_RESTORED)
        + "\n"
        + _RESTORE_ARCHIVE
    )
    _run(restore_root, "restore", "native reader", script=script, timeout=60)


_POLICY = r"""
import json,os,sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.config_adapter import _ChatbookRegistry
from tldw_chatbook.Backup_Recovery.models import StorageItem
root=Path.home();candidate=root/'registry.json'
key='__chatbook_archive_reference';archive_id='profile:p:chatbooks.archives:member'
item=StorageItem('chatbooks.registry','profile:p:chatbooks.registry',candidate,'included',(archive_id,))
owner=_ChatbookRegistry('chatbooks.registry')
route=sys.argv[1]
record={'id':'one','file_path':None,key:{'logical_id':archive_id}}
mapping={archive_id:root/'selected.zip'};owners={archive_id:'chatbooks.archives'}
if route=='unknown':record[key]['logical_id']='profile:p:chatbooks.archives:unknown'
if route=='wrong_owner':owners[archive_id]='skills'
if route=='other_profile':record[key]['logical_id']='profile:q:chatbooks.archives:member'
if route=='missing_mapping':mapping={}
if route=='path_marker':record['file_path']=str(root/'old.zip')
if route=='unsafe_mapping':mapping[archive_id]=root/'..'/'outside.zip'
if route=='relative_mapping':mapping[archive_id]=Path('selected.zip')
if route=='relative_import':record.pop(key);record['file_path']='old.zip'
if route=='legacy_path':record.pop(key);record['file_path']=str(root/'old.zip')
if route=='unresolved':record[key]={'status':'unresolved'}
candidate.write_text(json.dumps({'records':[record]}));candidate.chmod(0o600)
before=candidate.read_bytes()
try:
 owner.validate_restore_reference_owners(item,candidate,owners)
 owner.relocate_restore(item,candidate,mapping)
except ValueError as error:
 assert route not in ('positive','unresolved'),error
 assert candidate.read_bytes()==before
else:
 assert route in ('positive','unresolved'),route
 result=json.loads(candidate.read_text())['records'][0]
 if route=='positive':assert key not in result and result['file_path']==str(mapping[archive_id])
 else:
  assert result==record
  owner.prepare_capture(item,candidate,())
  assert json.loads(candidate.read_text())['records'][0]==record
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "positive",
        "unresolved",
        "unknown",
        "wrong_owner",
        "other_profile",
        "missing_mapping",
        "path_marker",
        "unsafe_mapping",
        "relative_mapping",
        "relative_import",
        "legacy_path",
    ],
)
def test_typed_reference_requires_declared_archive_owner_and_selected_path(
    tmp_path, route
):
    _run(tmp_path, route, "policy", script=_POLICY)


@pytest.mark.parametrize(
    "route", ["inert", "injected", "path_marker", "relative", "traversal", "duplicate"]
)
def test_live_discovery_accepts_only_inert_unresolved_marker(tmp_path, route):
    from Tests.Backup_Recovery.test_skills_chatbooks_capture import _BOOK_REFERENCES

    mutation = r"""
import json
body=json.loads(service.registry_path.read_text())
record=body['records'][0]
if route in ('inert','injected','path_marker'):
 record['__chatbook_archive_reference']={'status':'unresolved'} if route=='inert' else {'logical_id':'profile:p:chatbooks.archives:invented'}
 if route!='path_marker':record['file_path']=None
if route=='relative':
 os.chdir(archives)
 asyncio.run(service.update_chatbook(record['id'],file_path=Path('saved.zip')))
 body=json.loads(service.registry_path.read_text())
if route=='traversal':record['file_path']=str(Path.home()/'..'/'saved.zip')
encoded=json.dumps(body)
if route=='duplicate':encoded=encoded.replace('"file_path":', '"file_path": null, "file_path":',1)
service.registry_path.write_text(encoded)
"""
    script = (
        _BOOK_REFERENCES.replace(
            "p=preview_capture", mutation + "\np=preview_capture", 1
        )
        .replace(
            "if route in ('saved','external','metadata'):",
            "if route in ('inert','relative'):",
        )
        .replace(
            " assert archive.logical_id in catalog.dependencies",
            " assert (archive.logical_id in catalog.dependencies)==(route=='relative')",
        )
    )
    _run(tmp_path, route, "live references", script=script)
