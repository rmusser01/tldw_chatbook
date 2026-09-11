"""Complete release gates are separate from individual native qualification."""

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path


def test_complete_capability_requires_every_release_gate():
    from tldw_chatbook.Backup_Recovery.qualification import release_capability

    gates = {
        "helper": True,
        "owner_coverage": True,
        "admission": True,
        "archive": True,
        "native_publish": True,
        "restore": True,
        "product_flow": True,
    }
    assert release_capability(**gates) is True
    for name in gates:
        assert release_capability(**{**gates, name: False}) is False, name


_PRIVATE = r"""
import asyncio,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector=Path(os.environ['TLDW_CONFIG_PATH'])
fixture=Path(os.environ['ROUNDTRIP_FIXTURE'])
name=selector.parent.name
"""

_SEED = (
    _PRIVATE
    + r"""
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chatbooks.database_paths import get_private_chatbooks_dir
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJob,IngestJobState
from tldw_chatbook.config import get_library_ingest_jobs_db_path
async def main():
 app=TldwCli()
 try:
  assert app.chachanotes_db.db_path==selector.parent/'custom'/'notes.db'
  assert app.media_db.db_path==selector.parent/'custom'/'media.db'
  assert app.prompts_db.db_path==fixture/'shared'/'prompts.db'
  core=app.chachanotes_db
  note=core.add_note(name+' retained note',name+' native note bytes')
  deleted=core.add_note(name+' deleted note','Retained soft deletion '+name)
  assert core.soft_delete_note(deleted,1)
  conversation=core.add_conversation({'title':name+' conversation'})
  message=core.add_message({'conversation_id':conversation,'sender':'User','content':name+' message bytes'})
  media,_,_=app.media_db.add_media_with_keywords(title=name+' media',media_type='text',content=name+' media bytes',keywords=['retained'])
  prompt,_,_=app.prompts_db.add_prompt(name+' prompt','fixture',name+' details',user_prompt=name+' prompt bytes')
  assert media and prompt
  research=app.local_research_service
  session=research.create_session(title=name+' research',query=name+' research query')
  deleted_session=research.create_session(title=name+' historical research',query='Preserved history')
  assert research.delete_session(deleted_session['id'])
  empty=get_private_chatbooks_dir()
  assert not list(empty.iterdir())
  jobs=LibraryIngestJobsDB(get_library_ingest_jobs_db_path(),'fixture')
  try:
   jobs.upsert_job(LibraryIngestJob('ingest-job-1',str(fixture/(name+'-input.txt')),state=IngestJobState.QUEUED))
  finally:jobs.close()
  (fixture/(name+'-seed.json')).write_text(json.dumps(dict(note=note,deleted=deleted,conversation=conversation,message=message,media=media,prompt=prompt,research=session['id'],deleted_research=deleted_session['id'],empty=str(empty))))
  assert not blocked_attempts(),blocked_attempts()
  print('SEEDED',name,flush=True)
 finally:
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
"""
)

# Register only actual rooted paths after both independent seed processes exit.
# Native profile records provide known selectors; installed discovery still owns
# every item and all dependency/completeness decisions.
_BIND = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile,UNBOUND_NAMESPACE
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,_capture_names
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
authority.register('fixture.shared',(fixture/'shared',))
for label in ('alpha','beta'):
 profile=fixture/label
 authority.register('fixture.'+label,(profile,))
for label in ('alpha','beta'):
 profile=fixture/label
 inventory=preview_capture((profile/'config.toml',),options={'staging_parent':fixture})
 assert inventory.complete,inventory.issues
 names=tuple(key for key in _capture_names(authority,inventory) if key!=UNBOUND_NAMESPACE)
 bind_profile(root,profile/'config.toml',names,root/'admission')
assert not blocked_attempts()
print('BOUND_BOTH',flush=True)
"""
)

_RESUME = (
    _PRIVATE
    + r"""
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
seed=json.loads((fixture/(name+'-seed.json')).read_text())
db=CharactersRAGDB(selector.parent/'custom'/'notes.db','fixture-resume')
try:
 assert db.get_note_by_id(seed['note'])['content']==name+' native note bytes'
 resumed=db.add_note('After capture '+name,'Ordinary resumed writer '+name)
 assert db.get_note_by_id(resumed)['content']=='Ordinary resumed writer '+name
 (fixture/(name+'-resumed.json')).write_text(json.dumps({'note':resumed}))
finally:db.close()
assert not blocked_attempts()
print('RESUMED',name,flush=True)
"""
)

# Join two fixed Python programs; SQL below uses fixed text and bound values.
_CAPTURE = "".join(
    (
        _PRIVATE,
        r"""
import hashlib,sqlite3,subprocess,zipfile
from contextlib import closing
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app
from tldw_chatbook.Backup_Recovery import storage_admission,archive_reader
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
async def main():
 app=TldwCli();monitoring=asyncio.create_task(monitor_app(app))
 cancel=threading.Event();watchdog=asyncio.get_running_loop().call_later(55,cancel.set)
 try:
  selectors=tuple(fixture/label/'config.toml' for label in ('alpha','beta'))
  options={'staging_parent':fixture}
  explicit=preview_capture(selectors,options=options)
  preview=preview_capture((),options=options)
  assert explicit.scope_digest==preview.scope_digest
  assert {item.path for item in preview.items if item.owner=='config'}==set(selectors)
  owners={adapter.owner_id:[] for adapter in install_adapters()}
  inventory_only={}
  for item in preview.items:
   target=owners if item.owner in owners else inventory_only
   target.setdefault(item.owner,[]).append(item.status)
  (fixture/'owner-accounting.json').write_text(json.dumps({'installed':owners,'inventory_only':inventory_only},sort_keys=True,indent=2))
  assert set(inventory_only)<= {'recovery.control','server.data','sqlite.transient'},inventory_only
  assert all(status=='intentionally_excluded' for states in inventory_only.values() for status in states),inventory_only
  bad=[(item.owner,str(item.path),item.status,item.dependencies) for item in preview.items if item.status not in ('included','included_directory','unused','intentionally_excluded','intentionally_deleted')]
  assert preview.complete,(preview.issues,bad)
  shared=[item for item in preview.items if item.owner=='db.prompts.primary']
  assert len(shared)==2 and shared[0].path==shared[1].path==fixture/'shared'/'prompts.db'
  assert shared[0].shared_group and shared[0].shared_group==shared[1].shared_group
  destination=fixture/'two-profiles.tldw-backup.zip'
  result=await asyncio.to_thread(capture,(),preview.scope_digest,destination,options=options,cancel=cancel)
  async with asyncio.timeout(10):
   while storage_admission._pause is not None or app._backup_runtime_maintenance is not None:await asyncio.sleep(.01)
  assert not destination.exists()
  after=app.chachanotes_db.add_note('After capture alpha','Ordinary resumed writer alpha')
  env=os.environ.copy();env['TLDW_CONFIG_PATH']=str(selectors[1])
  with (fixture/'beta-resume.log').open('w') as output:
   resumed=await asyncio.to_thread(subprocess.run,[sys.executable,str(fixture/'resume.py')],cwd=Path.cwd(),env=env,stdout=output,stderr=subprocess.STDOUT,timeout=20,check=False)
  assert resumed.returncode==0,(fixture/'beta-resume.log').read_text()
  after_ids={'alpha':after,'beta':json.loads((fixture/'beta-resumed.json').read_text())['note']}
  assert result.inventory.complete
  manifest=json.loads(result.manifest_bytes)
  assert manifest['consistency']=='coherent' and len(manifest['profile_ids'])==2
  source={item.logical_id:item for item in result.inventory.items}
  represented={row['logical_id'] for row in (*manifest['files'],*manifest['directories'])}
  groups={row['group_id']:set(row['members']) for row in manifest['dependency_groups']}
  for item in source.values():
   if item.shared_group and item.logical_id in represented:
    peers={peer.logical_id for peer in source.values() if peer.shared_group==item.shared_group and peer.logical_id in represented}
    group='group:'+hashlib.sha256(item.logical_id.encode()).hexdigest()
    assert peers<=groups[group],(item.logical_id,peers,groups[group])
  for row in manifest['directories']:
   item=source.get(row['logical_id'])
   if item and item.shared_group:
    meta=item.metadata
    assert row['root_id']==meta.root_id and row['parent_id']==meta.parent_id
    assert row['relative_path']==meta.relative_path
    assert row['metadata']=={'version':1,'mode':meta.mode,'mtime_ns':meta.mtime_ns}
  for label in ('alpha','beta'):
   seed=json.loads((fixture/(label+'-seed.json')).read_text())
   for owner,leaf in (('db.chachanotes.primary','notes.db'),('db.media.primary','media.db'),('research.local','research.db')):
    row=next(row for row in manifest['files'] if row['owner_id']==owner and source[row['logical_id']].path==fixture/label/'custom'/leaf)
    with closing(sqlite3.connect(result.root/row['payload'])) as db:
     if owner=='db.chachanotes.primary':
      assert db.execute('SELECT content,deleted FROM notes WHERE id=?',(seed['note'],)).fetchone()==(label+' native note bytes',0)
      assert db.execute('SELECT content,deleted FROM notes WHERE id=?',(seed['deleted'],)).fetchone()==('Retained soft deletion '+label,1)
      assert db.execute('SELECT content FROM messages WHERE id=? AND conversation_id=?',(seed['message'],seed['conversation'])).fetchone()==(label+' message bytes',)
      assert db.execute('SELECT 1 FROM notes WHERE id=?',(after_ids[label],)).fetchone() is None
     elif owner=='db.media.primary':
      assert db.execute('SELECT content FROM Media WHERE id=?',(seed['media'],)).fetchone()==(label+' media bytes',)
     else:
      assert db.execute('SELECT query FROM research_sessions WHERE id=?',(seed['research'],)).fetchone()==(label+' research query',)
      assert db.execute('SELECT deleted FROM research_sessions WHERE id=?',(seed['deleted_research'],)).fetchone()==(1,)
   assert any(row['logical_id'] in source and source[row['logical_id']].owner=='chatbooks.archives' and source[row['logical_id']].path==Path(seed['empty']) for row in manifest['directories'])
  prompt_rows=[row for row in manifest['files'] if row['owner_id']=='db.prompts.primary']
  assert len(prompt_rows)==2 and len({row['sha256'] for row in prompt_rows})==1
  with closing(sqlite3.connect(result.root/prompt_rows[0]['payload'])) as db:
   assert db.execute("SELECT name,user_prompt FROM Prompts WHERE name IN ('alpha prompt','beta prompt') ORDER BY name").fetchall()==[('alpha prompt','alpha prompt bytes'),('beta prompt','beta prompt bytes')]
  jobs=[row for row in manifest['files'] if row['owner_id']=='db.library_ingest_jobs']
  assert len(jobs)==2
  for row in jobs:
   with closing(sqlite3.connect(result.root/row['payload'])) as db:
    assert db.execute('SELECT state FROM ingest_jobs').fetchall()==[('queued',)]
  sealed=await asyncio.to_thread(write_archive,result,destination,password=None,cancel=cancel)
  acquired=await asyncio.to_thread(archive_reader.acquire,sealed.path,fixture/'readback',ArchiveLimits(),None,cancel)
  assert acquired.manifest_bytes==sealed.manifest_bytes
  with zipfile.ZipFile(acquired.path) as archive:
   assert archive.read('manifest.json')==result.manifest_bytes
   for row in manifest['files']:
    assert hashlib.sha256(archive.read(row['payload'])).hexdigest()==row['sha256']
  assert not blocked_attempts(),blocked_attempts()
  print('TWO_PROFILE_COMPLETE_CAPTURE_AND_RESUME',flush=True)
 finally:
  watchdog.cancel();cancel.set();monitoring.cancel();await asyncio.gather(monitoring,return_exceptions=True)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close();await app.tts_service.wait_closed()
asyncio.run(main())
""",
    )
)


def _run_profile_child(root, name, script, environment, *, timeout=60):
    """Run one fixed private fixture program and preserve its full native output."""
    program = root / (name + ".py")
    program.write_text(script, encoding="utf-8")
    output_path = root / (name + ".log")
    with output_path.open("w", encoding="utf-8") as output:
        # Fixed interpreter and fixture-owned source; no shell or external command.
        result = subprocess.run(  # nosec B603
            [sys.executable, str(program)],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    assert result.returncode == 0, output_path.read_text(encoding="utf-8")[-14000:]
    return output_path.read_text(encoding="utf-8")


def test_two_known_profiles_complete_capture_preserves_native_state_and_resumes(
    tmp_path,
):
    """Stage1: A is live, B is closed at capture; this is not restore qualification."""
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
        ROUNDTRIP_FIXTURE=str(root),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    for name in ("alpha", "beta"):
        profile = root / name
        (profile / "custom").mkdir(parents=True, mode=0o700)
        (profile / "data").mkdir(mode=0o700)
        selector = profile / "config.toml"
        selector.write_text(
            '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
            "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
            f"[paths]\ndata_dir={json.dumps(str(profile / 'data'))}\n"
            "[database]\n"
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
        (root / (name + "-input.txt")).write_text("Queued input " + name)
        selected = dict(environment, TLDW_CONFIG_PATH=str(selector))
        assert "SEEDED " + name in _run_profile_child(
            root, "seed-" + name, _SEED, selected
        )
    environment["TLDW_CONFIG_PATH"] = str(root / "alpha" / "config.toml")
    assert "BOUND_BOTH" in _run_profile_child(root, "bind", _BIND, environment)
    (root / "resume.py").write_text(_RESUME, encoding="utf-8")
    assert "TWO_PROFILE_COMPLETE_CAPTURE_AND_RESUME" in _run_profile_child(
        root, "capture", _CAPTURE, environment
    )
