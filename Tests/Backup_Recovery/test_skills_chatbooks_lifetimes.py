"""Installed local content operations retain complete native work for maintenance."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_runtime_startup_handoff import _SCRIPT as _RUNTIME


@pytest.mark.parametrize("owner", ["skills", "chatbooks"])
def test_actual_app_refuses_new_content_during_pause(tmp_path, owner):
    check = r"""
        from pathlib import Path
        if sys.argv[1] == 'skills':
            source = app.local_skills_service
            call = lambda: source.create_skill(name='paused-content', content='---\nname: paused-content\ndescription: maintenance proof\n---\nContent must not be published during pause.')
            path = source.index_path
        else:
            source = app.local_chatbook_service
            call = lambda: source.create_chatbook(name='paused-content')
            path = source.registry_path
        before = path.read_bytes() if path.exists() else None
        with __import__('pytest').raises(RecoveryRequired):
            await call()
        assert (path.read_bytes() if path.exists() else None) == before
"""
    # Instantiate the current lazy owner while ordinary storage is admitted.
    script = _RUNTIME.replace(
        "    app = TldwCli()\n",
        "    app = TldwCli()\n"
        "    if sys.argv[1] == 'skills':\n"
        "        app.local_skills_service\n",
    ).replace(
        "        startup = next(iter(storage._startups.values()))",
        check + "\n        startup = next(iter(storage._startups.values()))",
    ).replace(
        "        assert not errors\n",
        "        await call()\n        assert path.exists()\n        assert not errors\n",
    )
    _run(tmp_path, owner, "resume", script=script)


_NATIVE = r"""
import asyncio, json, os, sys, threading, time
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant, operation, run_async
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
route=sys.argv[1];root=Path.home();entered=threading.Event();release=threading.Event();finished=threading.Event()
if route in ('skills','chatbooks'):
 owner=LocalSkillsService(store_dir=root/'outside-skills') if route=='skills' else LocalChatbookService(registry_path=root/'outside-chatbooks.json')
 method='_save_index' if route=='skills' else '_save_registry';original=getattr(owner,method)
 def save(value):
  entered.set();assert release.wait(10);return original(value)
 setattr(owner,method,save)
 def invoke(name):
  return owner.create_skill(name=name,content='---\nname: '+name+'\ndescription: retained\n---\nAccepted content.') if route=='skills' else owner.create_chatbook(name=name)
 def worker():
  try:return asyncio.run(invoke('accepted'))
  finally:finished.set()
 async def main():
  task=asyncio.create_task(asyncio.to_thread(worker))
  while not entered.is_set():
   if task.done():task.result()
   await asyncio.sleep(.001)
  participant._maintenance_close_admission()
  try:
   assert not await participant._maintenance_drain(time.monotonic()+.02)
   task.cancel()
   try:await task
   except asyncio.CancelledError:pass
   assert not finished.is_set()
   try:await invoke('paused')
   except RecoveryRequired:pass
   else:raise AssertionError('new work admitted')
   release.set()
   assert await participant._maintenance_drain(time.monotonic()+5)
   assert finished.is_set()
  finally:release.set();participant._maintenance_resume()
  setattr(owner,method,original)
  await invoke('resumed')
 asyncio.run(main())
 path=owner.index_path if route=='skills' else owner.registry_path
 assert 'accepted' in path.read_text() and 'resumed' in path.read_text() and 'paused' not in path.read_text()
elif route=='constructors':
 participant._maintenance_close_admission()
 try:
  for cls in (ChatbookCreator,ChatbookImporter):
   try:cls({})
   except RecoveryRequired:pass
   else:raise AssertionError('constructor created temp roots during pause')
 finally:participant._maintenance_resume()
 assert not (root/'.local/share/tldw_cli/temp').exists()
elif route in ('export','import','library'):
 from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
 from tldw_chatbook.RAG_Search.ingestion_indexing import suppress_ingestion_indexing
 db=MediaDatabase(root/'media.db',client_id='content-native')
 with suppress_ingestion_indexing():
  db.add_media_with_keywords(url='https://example.invalid/source',title='Exported source',media_type='document',content='Native content must survive export and import.',overwrite=True)
 media_id=str(db.execute_query('SELECT id FROM Media').fetchone()[0]);db.close_connection()
 archive=root/'outside.zip';creator=ChatbookCreator({'Media':str(db.db_path)})
 before=set(storage._live_leases)
 if route=='library':
  from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
  owner=LocalChatbookService({'Media':str(db.db_path)},registry_path=root/'registry.json')
  result=LibraryScreen._run_library_export_via_service(owner,{'name':'native','output_path':str(archive),'content_selections':{ContentType.MEDIA:[media_id]},'include_media':True},name='native',description='')
  assert result['success'] and result['registry_recorded'],result
 else:
  result=creator.create_chatbook('native','',{ContentType.MEDIA:[media_id]},archive,include_media=True)
  assert result[0],result
 assert archive.exists()
 assert not (set(storage._live_leases)-before),'export leaked operation-owned native handles'
 importer=ChatbookImporter({'Media':str(root/'imported.db')})
 manifest,error=importer.preview_chatbook(archive);assert manifest and not error
 if route=='import':
  result=importer.import_chatbook(archive);assert result[0],result
  assert not (set(storage._live_leases)-before),'import leaked operation-owned native handles'
  imported=MediaDatabase(root/'imported.db',client_id='verify')
  assert imported.execute_query('SELECT content FROM Media').fetchone()[0]=='Native content must survive export and import.'
  imported.close_connection()
 assert not list(importer.temp_dir.glob('preview_*'))
 assert not list(importer.temp_dir.glob('import_*'))
elif route=='copied_context':
 async def main():
  with operation((root/'skills',)):
   async def copied():
    with operation((root/'skills',)):pass
   try:await asyncio.create_task(copied())
   except RecoveryRequired:pass
   else:raise AssertionError('inherited child gained source authority')
 asyncio.run(main())
else:raise AssertionError(route)
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "skills",
        "chatbooks",
        "constructors",
        "export",
        "import",
        "library",
        "copied_context",
    ],
)
def test_actual_content_native_lifetimes(tmp_path, route):
    _run(tmp_path, route, "native", script=_NATIVE)


_CHAIN = r"""
import asyncio, json, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
route=sys.argv[1];root=Path.home();entered=threading.Event();release=threading.Event();finished=threading.Event()
if route=='trust':
 from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
 from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
 from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
 marker=FileSkillTrustGenerationMarkerStore(root/'trust'/'marker.json',store_dir=root/'trust')
 store=SkillTrustStore(root/'trust',marker)
 trust=SkillTrustService(skills_dir=root/'skills'/'skills',trust_store=store)
 skills=LocalSkillsService(store_dir=root/'skills')
 asyncio.run(skills.create_skill(name='retained',content='---\nname: retained\ndescription: retained trust\n---\nNative trust content.'))
 original=FileSkillTrustGenerationMarkerStore.save_marker
 def blocked_marker(self,**kwargs):
  entered.set();assert release.wait(10);return original(self,**kwargs)
 FileSkillTrustGenerationMarkerStore.save_marker=blocked_marker
 invoke=lambda:trust.bootstrap_trust('test-passphrase',salt=b'3'*32)
else:
 archive=root/'native.zip'
 assert ChatbookCreator({}).create_chatbook('empty','',{},archive)[0]
 importer=ChatbookImporter({});original=importer._extract_private_archive
 def blocked_extract(*args):
  result=original(*args);entered.set();assert release.wait(10);return result
 importer._extract_private_archive=blocked_extract
 invoke=lambda:importer.preview_chatbook(archive)
def worker():
 try:return invoke()
 finally:finished.set()
async def main():
 work=asyncio.create_task(asyncio.to_thread(worker))
 while not entered.is_set():
  if work.done():work.result()
  await asyncio.sleep(.001)
 participant._maintenance_close_admission()
 try:
  assert not await participant._maintenance_drain(time.monotonic()+.02)
  work.cancel()
  try:await work
  except asyncio.CancelledError:pass
  assert not finished.is_set()
  release.set();assert await participant._maintenance_drain(time.monotonic()+10)
  assert finished.is_set()
 finally:release.set();participant._maintenance_resume()
asyncio.run(main())
if route=='trust':
 FileSkillTrustGenerationMarkerStore.save_marker=original
 manifest=store.load_manifest(trust._keys)
 assert manifest['generation']==1 and 'retained' in manifest['skills']
 trust.trust_current_skill('retained')
 assert store.load_manifest(trust._keys)['generation']==2
else:
 importer._extract_private_archive=original
 assert not list(importer.temp_dir.iterdir())
 assert importer.preview_chatbook(archive)[0].name=='empty'
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["trust", "preview"])
def test_actual_native_completion_keeps_coupled_content(tmp_path, route):
    _run(tmp_path, route, "native", script=_CHAIN)


def test_library_export_finishes_registry_after_admission_closes(tmp_path):
    script = _NATIVE.replace(
        "name='native',description='')",
        "name='native',description='',progress_callback=lambda event: participant._maintenance_close_admission())",
    ).replace(
        "assert result['success'] and result['registry_recorded'],result",
        "assert result['success'] and result['registry_recorded'],result\n  assert asyncio.run(participant._maintenance_drain(time.monotonic()+1))\n  participant._maintenance_resume()",
    )
    _run(tmp_path, "library", "native", script=script)


_SCRIPT_CANCELLATION = r"""
import asyncio, time
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant
from tldw_chatbook.Skills_Interop.skill_trust_store import SkillTrustStore,FileSkillTrustGenerationMarkerStore
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunLimits
root=Path.home();store=root/'skills';trustroot=root/'trust'
marker=FileSkillTrustGenerationMarkerStore(trustroot/'marker.json',store_dir=trustroot)
trust=SkillTrustService(skills_dir=store/'skills',trust_store=SkillTrustStore(trustroot,marker))
service=LocalSkillsService(store_dir=store,trust_service=trust)
asyncio.run(service.create_skill(name='demo-skill',content='---\nname: demo-skill\ndescription: proof\n---\nBody',supporting_files={'scripts/slow.py':"import pathlib,time\npathlib.Path('started').write_text('accepted')\ntime.sleep(.6)\npathlib.Path('finished').write_text('native completion')\n"}))
trust.bootstrap_trust('pw',salt=b'3'*32)
output=service._script_output_root()
async def main():
 work=asyncio.create_task(service.run_skill_script('demo-skill','scripts/slow.py',[],limits=ScriptRunLimits(wall_clock_seconds=5)))
 started=None
 for _ in range(500):
  started=next(output.glob('*/started'),None)
  if started:break
  if work.done():raise AssertionError(work.result())
  await asyncio.sleep(.001)
 assert started is not None
 participant._maintenance_close_admission()
 try:
  work.cancel()
  try:await work
  except asyncio.CancelledError:pass
  assert not (started.parent/'finished').exists(),'cancelled waiter blocked on native finish'
  assert not await participant._maintenance_drain(time.monotonic()+.02)
  assert await participant._maintenance_drain(time.monotonic()+5)
  assert (started.parent/'finished').read_text()=='native completion'
 finally:participant._maintenance_resume()
 result=await service.run_skill_script('demo-skill','scripts/slow.py',[],limits=ScriptRunLimits(wall_clock_seconds=5))
 assert result.output_dir and (Path(result.output_dir)/'finished').exists()
asyncio.run(main())
assert not blocked_attempts()
print('retired and reopened')
"""


def test_actual_script_waiter_detaches_while_native_output_drains(tmp_path):
    _run(tmp_path, "script", "cancel", script=_SCRIPT_CANCELLATION)


def test_library_worker_closes_only_its_new_source_handles(tmp_path):
    old = "  result=LibraryScreen._run_library_export_via_service(owner,{'name':'native','output_path':str(archive),'content_selections':{ContentType.MEDIA:[media_id]},'include_media':True},name='native',description='')"
    new = r"""
  from types import SimpleNamespace
  from tldw_chatbook.Library.library_export_scope import ExportScope
  caller=db.get_connection();before.update(storage._live_leases)
  observed=[];errors=[];outcomes=[];original_ids=db.get_all_active_media_ids
  def ids(*args,**kwargs):
   result=original_ids(*args,**kwargs);observed.append(db.get_connection());return result
  db.get_all_active_media_ids=ids
  screen=SimpleNamespace(app_instance=SimpleNamespace(local_chatbook_service=owner),
   _build_library_export_payload=LibraryScreen._build_library_export_payload,
   _run_library_export_via_service=LibraryScreen._run_library_export_via_service,
   _marshal_library_export_failure=lambda *args:errors.append(args),
   _marshal_library_export_success=lambda *args,**kwargs:outcomes.append((args,kwargs)),
   _marshal_library_export_cancelled=lambda *args:errors.append(args),
   app=SimpleNamespace(call_from_thread=lambda *args:None))
  def actual_worker():
   try:LibraryScreen._run_library_export_worker.__wrapped__(screen,run_id=1,scope=ExportScope(kind='media'),name='native',description='',media_quality='thumbnail',destination=str(archive),media_db=db,chachanotes_db=None,prompts_db=None,preresolved_selections=None,cancel_event=None)
   except BaseException as error:errors.append(error)
  worker=threading.Thread(target=actual_worker);worker.start();worker.join(10)
  assert not worker.is_alive() and not errors and outcomes,(errors,outcomes)
  assert observed
  import sqlite3
  for connection in observed:
   try:connection.execute('SELECT 1')
   except sqlite3.ProgrammingError:pass
   else:raise AssertionError('Library worker retained native source handle')
  assert caller.execute('SELECT 1').fetchone()
  result={'success':True,'registry_recorded':True}
"""
    if old not in _NATIVE:
        raise AssertionError("native Library fixture anchor missing")
    _run(tmp_path, "library", "native", script=_NATIVE.replace(old, new))


def test_actual_app_settles_accepted_content_before_core_pause(tmp_path):
    setup = r"""
    from tldw_chatbook.Backup_Recovery.local_content_lifetime import participant as content
    owner=app.local_chatbook_service
    accepted_entered=threading.Event();accepted_release=threading.Event()
    original_save=owner._save_registry
    def blocked_save(value):
        accepted_entered.set();assert accepted_release.wait(10);return original_save(value)
    owner._save_registry=blocked_save
    accepted=asyncio.create_task(asyncio.to_thread(lambda:asyncio.run(owner.create_chatbook(name='accepted-before-core-pause'))))
    while not accepted_entered.is_set():
        if accepted.done():accepted.result()
        await asyncio.sleep(.001)
    async def finish_accepted():
        while not content.producer.closed:await asyncio.sleep(.001)
        assert storage._pause is None
        accepted_release.set()
        result=await accepted
        assert result['name']=='accepted-before-core-pause'
    finishing=asyncio.create_task(finish_accepted())
"""
    script = (
        _RUNTIME.replace(
            "    runtime = RuntimeMaintenance(app)",
            setup + "\n    runtime = RuntimeMaintenance(app)",
        )
        .replace(
            "        runtime.retire_local_caches()",
            "        await finishing\n        assert 'accepted-before-core-pause' in owner.registry_path.read_text()\n        owner._save_registry=original_save\n        runtime.retire_local_caches()",
        )
        .replace(
            "        assert not errors\n",
            "        await owner.create_chatbook(name='resumed-after-core-pause')\n        assert not errors\n",
        )
    )
    _run(tmp_path, "content", "resume", script=script)


_ROOT_ADMISSION = r"""
import json, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
root=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH']);route=sys.argv[1]
data=root/'selected-data';store=root/'skills';custom=root/'custom-output';sandbox=root/'custom-sandbox'
text='[general]\nusers_name="fixture"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n'
if route=='scratch':text+='[skills]\nscript_scratch_root='+json.dumps(str(custom))+'\n'
if route=='sandbox':text+='[tools]\nfile_sandbox_root='+json.dumps(str(sandbox))+'\n'
selector.write_text(text);selector.chmod(0o600)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile
control=bootstrap.default_bootstrap_root();authority=admission_authority(control)
store.mkdir(mode=0o700);(data/'fixture').mkdir(parents=True,mode=0o700)
authority.register('selected',(selector.parent,store,data/'fixture'))
bind_profile(control,selector,('selected',),control/'admission')
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
service=LocalSkillsService(store_dir=store)
expected=custom if route=='scratch' else (sandbox if route=='sandbox' else data/'fixture'/'tool_sandbox')
assert not expected.exists()
pause=None
if route=='default':
 from tldw_chatbook.Tools import file_operation_tools as files
 from tldw_chatbook.Backup_Recovery import storage_admission as storage
 original=files._resolve_sandbox_config
 def selected_then_pause():
  global pause
  value=original();pause=storage._begin_local_pause();return value
 files._resolve_sandbox_config=selected_then_pause
try:
 try:service._script_output_root()
 except bootstrap.RecoveryRequired as error:
  if route!='default':assert str(error)=='storage_scope_not_enrolled',str(error)
 else:raise AssertionError('unselected output source was admitted')
finally:
 if pause is not None:pause.resume()
assert not expected.exists(),'root was created before native source refusal'
assert not (data/'fixture'/'tool_sandbox').exists(),'denied custom root fell back to another output root'
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["scratch", "sandbox", "default"])
def test_script_output_source_refusal_precedes_first_directory_creation(
    tmp_path, route
):
    _run(tmp_path, route, "denied", script=_ROOT_ADMISSION)


_ROOT_SELECTION = r"""
import json, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
root=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH']);route=sys.argv[1]
store=root/'skills';custom=root/'custom-output';data=root/'selected-data'
data.mkdir(mode=0o700)
if route=='unsafe':custom=store/'nested-output'
if route=='uncreatable':custom.write_text('existing regular file')
selector.write_text('[general]\nusers_name="fixture"\n[paths]\ndata_dir='+json.dumps(str(data))+'\n[skills]\nscript_scratch_root='+json.dumps(str(custom))+'\n');selector.chmod(0o600)
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
service=LocalSkillsService(store_dir=store)
if route=='config-denied':
 pause=storage._begin_local_pause()
 try:
  try:service._script_scratch_root()
  except RecoveryRequired:pass
  else:raise AssertionError('config source refusal converted to fallback')
 finally:pause.resume()
 assert not custom.exists()
else:
 selected=service._script_scratch_root()
 assert selected==(None if route=='unsafe' else str(custom))
 if route!='uncreatable':assert not custom.exists(),'pure selection created output root'
 output=service._script_output_root()
 assert output.is_dir()
 if route=='configured':assert output==custom
 else:assert output==data/'fixture'/'tool_sandbox'/'skill_script_output'
 if route=='uncreatable':assert custom.read_text()=='existing regular file'
 if route=='unsafe':assert not custom.exists()
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["configured", "uncreatable", "unsafe", "config-denied"]
)
def test_script_root_selection_preserves_ordinary_fallback_and_refusal(tmp_path, route):
    _run(tmp_path, route, "selection", script=_ROOT_SELECTION)
