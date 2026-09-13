"""Populated installed Skills/Chatbooks capture and ordinary prepackaging use."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_LIVE = r"""
import asyncio,json,os,sqlite3,sys,threading
from contextlib import closing
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n');selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant
from tldw_chatbook.Backup_Recovery.capture_service import capture,preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunLimits
from tldw_chatbook.Chatbooks.database_paths import get_private_chatbooks_dir
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
async def main():
 app=TldwCli();skills=app.local_skills_service;trust=app.local_skill_trust_service;books=app.local_chatbook_service
 assert isinstance(trust.trust_store.marker_store,FileSkillTrustGenerationMarkerStore)
 await skills.create_skill(name='capture-skill',content='---\nname: capture-skill\ndescription: captured bundle\n---\nPrivate captured skill body.',supporting_files={'scripts/produce.py':"from pathlib import Path\nPath('result.txt').write_text('retained actual script output')\n"})
 trust.bootstrap_trust('private-test-passphrase',salt=b'9'*32)
 trust.grant_script_execution('capture-skill')
 script=await skills.run_skill_script('capture-skill','scripts/produce.py',[],limits=ScriptRunLimits(wall_clock_seconds=5))
 assert script.output_dir,script
 output=Path(script.output_dir)/'result.txt';assert output.read_text()=='retained actual script output'
 note=app.chachanotes_db.add_note('Captured native note','Private note inside native DB and archive.')
 archive=get_private_chatbooks_dir()/'captured.zip'
 result=await books.export_chatbook({'name':'captured','output_path':str(archive),'content_selections':{ContentType.NOTE:[str(note)]}})
 assert result['success'],result
 await books.create_chatbook(name='captured',file_path=archive)
 parsed=await books.preview_chatbook(archive);assert parsed.get('success'),parsed
 live_registry=books.registry_path.read_bytes()
 options={'staging_parent':home}
 preview=preview_capture((selector,),options=options)
 bad=[(i.owner,str(i.path),i.status) for i in preview.items if i.status in ('unsupported','unavailable','missing_required')]
 assert preview.complete,(preview.issues,bad)
 entered,release=threading.Event(),threading.Event();original=SkillTrustStore.save_manifest
 def held_save(self,*args,**kwargs):
  entered.set();assert release.wait(15);return original(self,*args,**kwargs)
 SkillTrustStore.save_manifest=held_save
 accepted=asyncio.create_task(asyncio.to_thread(trust.trust_current_skill,'capture-skill'))
 while not entered.is_set():
  if accepted.done():accepted.result()
  await asyncio.sleep(.005)
 monitoring=asyncio.create_task(monitor_app(app));cancel=threading.Event()
 async def finish():
  for _ in range(1000):
   if participant.producer.closed:break
   await asyncio.sleep(.005)
  assert participant.producer.closed and not accepted.done()
  assert storage._pause is None,'core paused before accepted content finished'
  release.set()
 finishing=asyncio.create_task(finish());watchdog=asyncio.get_running_loop().call_later(60,cancel.set)
 destination=home/'content.tldw-backup.zip'
 try:
  captured=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  await accepted;await finishing;SkillTrustStore.save_manifest=original
  for _ in range(500):
   if storage._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage._pause is None and app._backup_runtime_maintenance is None
  manifest=json.loads(captured.manifest_bytes)
  assert captured.inventory.complete and manifest['consistency']=='coherent'
  paths={i.logical_id:i.path for i in captured.inventory.items}
  files={paths[row['logical_id']]:captured.root/row['payload'] for row in manifest['files']}
  originals={p:p.read_bytes() for p in skills.store_dir.rglob('*') if p.is_file()}
  originals.update({archive:archive.read_bytes(),output:output.read_bytes()})
  assert books.registry_path.read_bytes()==live_registry
  staged_registry=files[books.registry_path].read_bytes()
  record=json.loads(staged_registry)['records'][0]
  archive_id=next(i.logical_id for i in captured.inventory.items if i.path==archive)
  assert record['file_path'] is None
  assert record['__chatbook_archive_reference']=={'logical_id':archive_id}
  assert str(archive).encode() not in staged_registry
  assert all(files[path].read_bytes()==body for path,body in originals.items())
  historical=json.loads(files[trust.trust_store.manifest_path].read_text())
  assert historical['manifest']['generation']==2
  marker=trust.trust_store.marker_store.marker_path
  assert json.loads(files[marker].read_text())['generation']==2
  member=next(row for row in manifest['files'] if row['owner_id']=='db.chachanotes.primary')
  with closing(sqlite3.connect(captured.root/member['payload'])) as db:
   assert db.execute('SELECT content FROM notes WHERE id=?',(note,)).fetchone()[0]=='Private note inside native DB and archive.'
  await skills.create_skill(name='resumed-skill',content='---\nname: resumed-skill\ndescription: resumed\n---\nAfter capture.')
  trust.trust_current_skill('capture-skill')
  await books.create_chatbook(name='after capture')
  app.chachanotes_db.add_note('After capture','Resumed native writer.')
  assert all(files[path].read_bytes()==body for path,body in originals.items())
  assert files[books.registry_path].read_bytes()==staged_registry
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,captured,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination and destination.is_file()
  assert not blocked_attempts()
 finally:
  SkillTrustStore.save_manifest=original;release.set();cancel.set();watchdog.cancel()
  await accepted
  for task in (finishing,monitoring):task.cancel()
  await asyncio.gather(finishing,monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles();await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
print('retired and reopened')
"""


def test_populated_skills_chatbooks_complete_capture_resumes_before_packaging(tmp_path):
    _run(tmp_path, "content", "complete", script=_LIVE, timeout=120)


_OUTPUT = r"""
import json,os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
route=sys.argv[1];selector=Path(os.environ['TLDW_CONFIG_PATH'])
text='[general]\nusers_name="fixture"\n'
if route in ('scratch','sandbox'):
 key='[skills]\nscript_scratch_root' if route=='scratch' else '[tools]\nfile_sandbox_root'
 text+=key+'='+json.dumps(str(Path.home()/'unsupported-custom'))+'\n'
if route=='canonical':
 from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
 canonical=user_data_dir({'general':{'users_name':'fixture'}})/'tool_sandbox'
 text+='[tools]\nfile_sandbox_root='+json.dumps(str(canonical))+'\n[skills]\nscript_scratch_root='+json.dumps(str(canonical/'skill_script_output'))+'\n'
selector.write_text(text);selector.chmod(0o600)
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
root=get_user_data_dir();workspace=WorkspaceDB(root/'tldw_chatbook_workspaces.db','fixture');workspace.close()
sandbox=root/'tool_sandbox';output=sandbox/'skill_script_output';output.mkdir(parents=True)
(output/'saved.txt').write_bytes(b'retained script-owned sandbox output')
route=sys.argv[1]
if route=='unknown':(sandbox/'unowned.txt').write_bytes(b'unknown sibling')
if route=='alias':output.rename(root/'elsewhere');output.symlink_to(root/'elsewhere',target_is_directory=True)
preview=preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True})
items=[item for item in preview.items if item.owner in ('skills','agents.history')]
if route in ('output','canonical'):
 assert not any(i.status in ('unsupported','unavailable','missing_required') for i in items),[(str(i.path),i.status) for i in items]
 assert 'overlapping_owner_roots' not in preview.issues
 assert not (root/'agent_runs.db').exists()
 leaf=next(i for i in items if i.path==output/'saved.txt')
 assert leaf.owner=='agents.history' and leaf.status=='included'
 assert leaf.metadata.relative_path=='skill_script_output/saved.txt'
else:
 assert not preview.complete
 assert any(i.status in ('unsupported','unavailable') for i in items)
 if route in ('scratch','sandbox'):assert any(i.logical_id.endswith(':script_output_unqualified') for i in items)
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["output", "canonical", "unknown", "alias", "scratch", "sandbox"]
)
def test_exact_script_output_topology_and_unsupported_config(tmp_path, route):
    _run(tmp_path, route, "output", script=_OUTPUT)


_DIRECTORY = r"""
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Chatbooks.database_paths import get_private_chatbooks_dir,secure_chatbook_directory
from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.config import get_user_data_dir
root=get_user_data_dir();explicit=Path.home()/'explicit-chatbooks';canonical=root/'chatbooks'
participant._maintenance_close_admission()
try:
 for operation in (get_private_chatbooks_dir,lambda:secure_chatbook_directory(explicit)):
  try:operation()
  except RecoveryRequired:pass
  else:raise AssertionError('directory created while paused')
 assert not canonical.exists() and not explicit.exists()
finally:participant._maintenance_resume()
assert get_private_chatbooks_dir()==canonical
assert secure_chatbook_directory(explicit)==explicit
assert canonical.is_dir() and explicit.is_dir()
assert not blocked_attempts()
print('retired and reopened')
"""


def test_chatbook_directory_helpers_refuse_before_creation_and_resume(tmp_path):
    _run(tmp_path, "directories", "resume", script=_DIRECTORY)


_BOOK_REFERENCES = r"""
import asyncio,os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.config import get_user_data_dir,get_prompts_db_path
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
root=get_user_data_dir();db=PromptsDatabase(get_prompts_db_path(),'fixture');db.close_connection()
service=LocalChatbookService({'Prompts':str(db.db_path)})
archives=root/'chatbooks';archives.mkdir();source=archives/'saved.zip';source.write_bytes(b'retained archive bytes')
route=sys.argv[1]
asyncio.run(service.create_chatbook(name='saved',file_path=source))
if route=='missing':source.unlink()
if route=='alias':source.unlink();source.symlink_to(Path.home()/'external')
if route=='malformed':service.registry_path.write_bytes(b'{bad JSON')
if route=='external':asyncio.run(service.create_chatbook(name='external',file_path=Path.home()/'missing-external.zip'))
if route=='metadata':asyncio.run(service.create_chatbook(name='metadata only'))
p=preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True})
items=[item for item in p.items if item.owner in ('chatbooks.registry','chatbooks.archives')]
if route in ('saved','external','metadata'):
 assert not any(item.status in ('unsupported','unavailable','missing_required') for item in items),[(item.path,item.status) for item in items]
 catalog=next(i for i in items if i.path==service.registry_path);archive=next(i for i in items if i.path==source)
 assert archive.logical_id in catalog.dependencies
 assert not any(i.path==Path.home()/'missing-external.zip' for i in p.items)
else:
 assert any(i.owner=='chatbooks.registry' and i.status in ('unsupported','unavailable','missing_required') and not i.logical_id.endswith(':participant_pending') for i in items),[(i.path,i.status) for i in items]
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["saved", "missing", "alias", "malformed", "external", "metadata"]
)
def test_chatbook_catalog_preserves_owned_reference_evidence(tmp_path, route):
    _run(tmp_path, route, "references", script=_BOOK_REFERENCES)


_SKILL_REFERENCES = r"""
import asyncio,json,os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
root=get_user_data_dir()/'skills';trustroot=root/'trust'
service=LocalSkillsService(store_dir=root)
asyncio.run(service.create_skill(name='retained',content='---\nname: retained\ndescription: reference\n---\nSaved skill bytes.'))
store=SkillTrustStore(trustroot,FileSkillTrustGenerationMarkerStore(trustroot/'generation_marker.json',store_dir=trustroot))
trust=SkillTrustService(skills_dir=root/'skills',trust_store=store)
trust.bootstrap_trust('pw',salt=b'9'*32)
route=sys.argv[1]
if route=='bundle':(root/'skills'/'retained'/'SKILL.md').unlink()
if route=='snapshot':next(store.snapshots_dir.glob('*.json')).unlink()
if route=='index':service.index_path.write_text('{"skills":[]}')
if route=='escape':
 payload=json.loads(store.manifest_path.read_text());payload['manifest']['skills']['retained']['snapshot_id']='../elsewhere';store.manifest_path.write_text(json.dumps(payload))
p=preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True})
items=[i for i in p.items if i.owner=='skills']
if route=='retained':assert not any(i.status in ('unsupported','unavailable','missing_required') for i in items),[(i.path,i.status) for i in items]
elif route in ('bundle','snapshot'):assert any(i.status=='missing_required' for i in items),[(i.path,i.status) for i in items]
else:assert any(i.status=='unsupported' for i in items),[(i.path,i.status) for i in items]
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["retained", "bundle", "snapshot", "index", "escape"])
def test_skill_saved_bundle_and_snapshot_references_refuse_missing_bytes(
    tmp_path, route
):
    _run(tmp_path, route, "references", script=_SKILL_REFERENCES)
