"""Installed optional owner absence and reference-driven inventory declarations."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import os,sys,tomllib
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n')
selector.chmod(0o600)
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY,DiscoveryContext
from tldw_chatbook.Backup_Recovery.profile_paths import database_path,user_data_dir
config=tomllib.loads(selector.read_text())
config[DISCOVERY_CONTEXT_KEY]=DiscoveryContext(selector,'unused')
data=user_data_dir(config);data.mkdir(parents=True,mode=0o700,exist_ok=True)
route=sys.argv[1]
if route.startswith('graph'):
 from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
 from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
 from tldw_chatbook.Library.library_collections_service import LocalLibraryCollectionsService
 from tldw_chatbook.DB.recovery_core import core_adapters
 notes=CharactersRAGDB(database_path(config,'chachanotes_db_path'),'unused-test')
 collections=LibraryCollectionsDB(database_path(config,'library_collections_db_path'),'unused-test')
 if route=='graph_referenced':
  note=notes.add_note('File note','Saved note bytes')
  with notes.transaction() as cursor:
   cursor.execute('UPDATE notes SET file_path_on_disk=?,version=version+1 WHERE id=?',(str(data/'missing.md'),note))
  service=LocalLibraryCollectionsService(collections)
  collection=service.create_collection('Saved references')
  service.add_item_to_collection(collection.collection_id,source_type='note',source_id=note)
 notes.close();collections.close()
 adapters={a.owner_id:a for a in core_adapters()}
 item=adapters['db.chachanotes.primary'].discover(config)[0]
 assert ('profile:unused:notes.file_notes' in item.dependencies)==(route=='graph_referenced'),item
 item=adapters['db.library_collections'].discover(config)[0]
 expected=('profile:unused:config',)+(('profile:unused:db.chachanotes.primary',) if route=='graph_referenced' else ())
 assert item.dependencies==expected,item
 if route=='graph_referenced':
  assert adapters[item.owner].validate_dependencies(item,item.path,{})==('dependency_unavailable',)
elif route.startswith('video'):
 from tldw_chatbook.Video_Generation.video_store import VideoStore
 from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
 store=VideoStore()
 with store._root_lease(): pass
 path=store._lease_path
 if route=='video_payload': path.write_bytes(b'not a capacity lease')
 if route=='video_link':
  path.unlink();target=data/'target';target.write_bytes(b'');path.symlink_to(target)
 adapter=next(a for a in recovery_adapters() if a.owner_id=='generation.assets')
 item=next((i for i in adapter.discover(config) if i.path==path),None)
 assert item is not None,'actual default capacity lock was not declared'
 assert (item.status=='intentionally_excluded')==(route=='video_empty'),item
 assert not blocked_attempts()
elif route.startswith('pets'):
 from tldw_chatbook.Widgets.Tamagotchi.recovery import recovery_adapters
 parent=Path.home()/'.config'/'tldw_chatbook'
 if route!='pets_container_absent':parent.parent.mkdir(mode=0o700,exist_ok=True)
 if route=='pets_link': parent.symlink_to(data,target_is_directory=True)
 if route=='pets_file': parent.write_bytes(b'not a pet directory')
 items=recovery_adapters()[0].discover(config)
 assert items[0].path==parent/'tamagotchi_pets.json'
 assert (items[0].status=='unused')==(route in ('pets_absent','pets_container_absent')),items
 if route in ('pets_absent','pets_container_absent'):assert not parent.exists()
 else:assert items[0].status in {'unavailable','unsupported'},items
else: raise AssertionError(route)
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "graph_empty",
        "graph_referenced",
        "video_empty",
        "video_payload",
        "video_link",
        "pets_absent",
        "pets_container_absent",
        "pets_link",
        "pets_file",
    ],
)
def test_installed_owner_inventory_contract(tmp_path, route):
    _run(tmp_path, route, "unused", script=_SCRIPT)


_LAZY = (
    _SCRIPT.split("route=sys.argv[1]")[0]
    + r"""
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.DB.recovery_operations import recovery_adapters as operations
from tldw_chatbook.Notes.recovery import recovery_adapters as notes_adapters
from tldw_chatbook.TTS.recovery import recovery_adapters as tts_adapters
notes=CharactersRAGDB(database_path(config,'chachanotes_db_path'),'unused-test')
notes.close()
adapters={a.owner_id:a for a in (*core_adapters(),*operations(),*notes_adapters(),*tts_adapters())}
for name in ('db.library_ingest_jobs','db.agent_runs','notes.file_notes','notes.sync_state','tts.profile_store','tts.references'):
 adapter=adapters[name]
 item=adapter.discover(config)[0]
 assert item.status=='unused',(name,item)
 assert not item.path.exists()
 for suffix in ('-wal','-shm','-journal'):
  sidecar=Path(str(item.path)+suffix);sidecar.write_bytes(b'used owner evidence')
  assert adapter.discover(config)[0].status!='unused',(name,suffix)
  sidecar.unlink()
for name in ('db.library_ingest_jobs','tts.profile_store'):
 adapter=adapters[name]
 config['database']={adapter.setting_name:str(data/'explicitly-required.db')}
 assert adapter.discover(config)[0].status=='missing_required'
config.pop('database')
config['file_notes']={'root':str(data/'selected-notes')}
assert adapters['notes.file_notes'].discover(config)[0].status=='missing_required'
config.pop('file_notes')
notes=CharactersRAGDB(database_path(config,'chachanotes_db_path'),'unused-test')
note=notes.add_note('Saved file note','Saved content')
with notes.transaction() as cursor:
 cursor.execute('UPDATE notes SET sync_root_folder=?,version=version+1 WHERE id=?',(str(data/'selected-notes'),note))
notes.close()
assert adapters['notes.file_notes'].discover(config)[0].status=='missing_required'
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_exact_lazy_defaults_preserve_required_missing_evidence(tmp_path):
    _run(tmp_path, "lazy", "unused", script=_LAZY)


_FRESH = r"""
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
from tldw_chatbook.Backup_Recovery.capture_service import capture,preview_capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
async def main():
 app=TldwCli()
 app.chachanotes_db.add_note('Saved fresh profile note','Private note bytes before capture.')
 options={'staging_parent':home}
 preview=preview_capture((selector,),options=options)
 bad=[(i.owner,str(i.path),i.status,i.dependencies) for i in preview.items if i.status not in ('included','included_directory','unused','intentionally_excluded','intentionally_deleted')]
 assert preview.complete,(preview.issues,bad)
 destination=home/'complete.tldw-backup.zip'
 monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(20,cancel.set)
 try:
  result=await asyncio.to_thread(capture,(selector,),preview.scope_digest,destination,options=options,cancel=cancel)
  for _ in range(500):
   if storage._pause is None and app._backup_runtime_maintenance is None:break
   await asyncio.sleep(.01)
  assert storage._pause is None and app._backup_runtime_maintenance is None
  app.chachanotes_db.add_note('After capture','Resumed writer bytes')
  manifest=json.loads(result.manifest_bytes)
  assert result.inventory.complete
  assert manifest['consistency']=='coherent'
  member=next(f for f in manifest['files'] if f['owner_id']=='db.chachanotes.primary')
  with closing(sqlite3.connect(result.root/member['payload'])) as connection:
   assert connection.execute('SELECT title,content FROM notes').fetchall()==[('Saved fresh profile note','Private note bytes before capture.')]
  assert not destination.exists()
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=threading.Event())
  assert sealed.path==destination and destination.is_file()
  assert not blocked_attempts()
 finally:
  watchdog.cancel();cancel.set();monitoring.cancel()
  try:await monitoring
  except asyncio.CancelledError:pass
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
asyncio.run(main())
print('retired and reopened')
"""


def test_fresh_app_complete_public_capture_preserves_real_bytes_and_resumes(tmp_path):
    _run(tmp_path, "fresh", "complete", script=_FRESH)


_AGENTS = (
    _SCRIPT.split("route=sys.argv[1]")[0]
    + r"""
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Agents.recovery import recovery_adapters
workspaces=WorkspaceDB(database_path(config,'workspaces_db_path'))
workspaces.close()
route=sys.argv[1]
if route=='configured':
 root=data/'selected-tools';root.mkdir()
 config['tools']={'file_sandbox_root':str(root)}
elif route=='retained':
 root=data/'tool_sandbox'/'.agent-runs';root.mkdir(parents=True)
 (root/'retained.jsonl').write_text('{"run_id":"retained"}\n')
items=recovery_adapters()[0].discover(config)
assert any(i.status=='missing_required' for i in items),items
assert not (data/'agent_runs.db').exists()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["configured", "retained"])
def test_missing_run_history_with_selected_or_retained_logs_refuses(tmp_path, route):
    _run(tmp_path, route, "unused", script=_AGENTS)


_INGEST = (
    _SCRIPT.split("route=sys.argv[1]")[0]
    + r"""
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob,IngestJobState
from tldw_chatbook.DB.recovery_core import core_adapters
store=LibraryIngestJobsDB(database_path(config,'library_ingest_jobs_db_path'),'unused-test')
local=sys.argv[1]=='local'
store.upsert_job(LibraryIngestJob('ingest-job-1',str(data/'original.pdf'),state=IngestJobState.DONE,origin='local' if local else 'server',media_id=42 if local else None,remote_media_id=None if local else '42'))
store.close()
adapter=next(a for a in core_adapters() if a.owner_id=='db.library_ingest_jobs')
item=adapter.discover(config)[0]
assert item.status=='included'
assert ('profile:unused:db.media.primary' in item.dependencies)==local,item
assert adapter.validate(item.path)==()
assert adapter.validate_dependencies(item,item.path,{})==(('dependency_unavailable',) if local else ())
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["local", "server"])
def test_ingest_history_requires_only_referenced_local_media(tmp_path, route):
    _run(tmp_path, route, "unused", script=_INGEST)
