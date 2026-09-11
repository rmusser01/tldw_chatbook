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
from tldw_chatbook.config import get_library_ingest_jobs_db_path,get_writing_db_path,get_notifications_db_path
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
  writing=app.local_writing_service
  assert writing is not None and not writing.is_memory_db and writing.db_path==get_writing_db_path()
  project=writing.create_project(title=name+' writing')
  chapter=writing.create_chapter(project['id'],title=name+' chapter')
  scene=writing.create_scene(chapter['id'],title=name+' scene',content_markdown=name+' original scene')
  version=writing.create_version('scene',scene['id'],label='before edit')
  writing.update_scene(scene['id'],content_markdown=name+' revised scene')
  assert writing.get_scene(scene['id'])['content_markdown']==name+' revised scene'
  assert writing.get_version('scene',scene['id'],version['version_number'])['payload']['content_markdown']==name+' original scene'
  study=app.local_study_service;quiz_service=app.local_quiz_service
  assert study.db is core and quiz_service.db is core
  deck=study.create_deck(name=name+' study',description='Retained local deck')
  card=study.create_flashcard(deck_id=deck['id'],front=name+' question',back=name+' answer',tags=['retained',name],notes=name+' study note')
  assert study.get_deck(deck['id'])['card_count']==1
  assert study.get_flashcard(card['id'])['back']==name+' answer'
  quiz=quiz_service.create_quiz(name=name+' quiz',description='Stored completed quiz')
  question=quiz_service.create_question(quiz['id'],question_type='fill_blank',question_text=name+' retained answer?',correct_answer=name,points=2)
  started=quiz_service.start_attempt(quiz['id'])
  attempt=quiz_service.submit_attempt(started['id'],answers=[{'question_id':question['id'],'user_answer':name}])
  assert attempt['score']==2 and attempt['total_possible']==2 and attempt['completed_at']
  assert attempt['answers'][0]['question_id']==question['id'] and attempt['answers'][0]['is_correct'] is True
  notifications=app.client_notifications_db
  assert not notifications.is_memory_db and Path(notifications.db_path)==get_notifications_db_path()
  notification=notifications.insert_notification(category='study',title=name+' completed quiz',message=name+' durable completion',source_backend='local',source_entity_kind='study_quiz_attempt',source_entity_id=attempt['id'],payload={'quiz_id':quiz['id'],'attempt_id':attempt['id']})
  notification=app.client_notifications_service.update_notification(notification['id'],is_read=True,is_dismissed=True)
  assert notification['is_read'] and notification['is_dismissed']
  assert notification['payload']=={'quiz_id':quiz['id'],'attempt_id':attempt['id']}
  domains=dict(writing=dict(path=str(writing.db_path),project=project['id'],chapter=chapter['id'],scene=scene['id'],version=version['id'],version_number=version['version_number']),study=dict(deck=deck['id'],card=card['id']),quiz=dict(quiz=quiz['id'],question=question['id'],attempt=attempt['id'],completed_at=attempt['completed_at']),notification=dict(path=str(notifications.db_path),id=notification['id'],read_at=notification['read_at'],dismissed_at=notification['dismissed_at']))
  from tldw_chatbook.Chat.prompt_history import PromptHistory,default_prompt_history_path
  from tldw_chatbook.Widgets.emoji_picker import save_recent_emoji,load_recent_emojis
  history=PromptHistory(default_prompt_history_path())
  assert await history.append(name+' retained input')
  emoji='★' if name=='alpha' else '✓'
  save_recent_emoji(emoji)
  assert load_recent_emojis()==[emoji]
  state=selector.parent/'ui_state.toml'
  state.write_text('[sidebar]\nsearch_query='+json.dumps(name+' retained search')+'\n')
  theme=selector.parent/'themes'/(name+'.toml')
  theme.parent.mkdir(mode=0o700,exist_ok=True)
  theme.write_text('[theme]\nname='+json.dumps(name)+'\ndark=true\n[colors]\nbackground="#112233"\n')
  from tldw_chatbook.Chunking.chunking_templates import ChunkingTemplate,ChunkingTemplateManager
  persona_service=app.local_character_persona_service
  persona=persona_service.create_persona_profile({'name':name+' persona','system_prompt':name+' retained persona prompt','is_active':False})
  dictionary_service=app.local_chat_dictionary_service
  dictionary=dictionary_service.create_dictionary({'name':name+' dictionary','description':name+' retained dictionary history'})
  dictionary_version=dictionary_service.get_version(dictionary['id'],dictionary['version'])
  grammar=await app.local_chat_grammars_service.create_grammar(name=name+' grammar',grammar_text='root ::= "'+name+'"')
  feedback=await app.local_feedback_service.submit_feedback(conversation_id=conversation,message_id=message,feedback_type='helpful',helpful=True,user_notes=name+' retained feedback')
  audio_service=app.local_audio_services_service
  # Only the external generator is substituted; history uses the actual owner.
  audio_service.tts_audio_generator=lambda **kwargs:(name+' retained audio bytes').encode()
  audio=await audio_service.create_audio_speech({'input':name+' spoken input','response_format':'wav'})
  await audio_service.update_tts_history_favorite(audio['history_id'],{'favorite':True})
  templates=ChunkingTemplateManager()
  template=ChunkingTemplate(name=name+'-retained',description=name+' chunking template',pipeline=[])
  templates.save_template(template)
  durable_api=dict(persona=persona['id'],dictionary=dictionary['id'],dictionary_revision=dictionary_version['revision'],grammar=grammar['id'],feedback=feedback['feedback_id'],audio=audio['history_id'],template=template.name)
  durable_files={owner:dict(path=str(path),hex=path.read_bytes().hex()) for owner,path in (('chat.prompt_history',history.path),('ui.emoji_recents',selector.parent/'recent_emojis.json'),('ui.state',state),('ui.themes',theme),('personas',persona_service.persona_store_path),('chat.dictionary_history',dictionary_service.history_store_path),('chat.grammars',app.local_chat_grammars_service.store_path),('feedback',app.local_feedback_service.store_path),('audio.history',audio_service.history_store_path),('chunking.templates',templates.user_templates_dir/(template.name+'.json')))}
  empty=get_private_chatbooks_dir()
  assert not list(empty.iterdir())
  jobs=LibraryIngestJobsDB(get_library_ingest_jobs_db_path(),'fixture')
  try:
   jobs.upsert_job(LibraryIngestJob('ingest-job-1',str(fixture/(name+'-input.txt')),state=IngestJobState.QUEUED))
  finally:jobs.close()
  (fixture/(name+'-seed.json')).write_text(json.dumps(dict(note=note,deleted=deleted,conversation=conversation,message=message,media=media,prompt=prompt,research=session['id'],deleted_research=deleted_session['id'],empty=str(empty),domains=domains,durable_files=durable_files,durable_api=durable_api)))
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
   for owner,expected in seed['durable_files'].items():
    row=next(row for row in manifest['files'] if row['owner_id']==owner and source[row['logical_id']].path==Path(expected['path']))
    assert source[row['logical_id']].status=='included'
    assert (result.root/row['payload']).read_bytes()==bytes.fromhex(expected['hex'])
   for owner,leaf in (('db.chachanotes.primary','notes.db'),('db.media.primary','media.db'),('research.local','research.db')):
    row=next(row for row in manifest['files'] if row['owner_id']==owner and source[row['logical_id']].path==fixture/label/'custom'/leaf)
    with closing(sqlite3.connect(result.root/row['payload'])) as db:
     if owner=='db.chachanotes.primary':
      assert db.execute('SELECT content,deleted FROM notes WHERE id=?',(seed['note'],)).fetchone()==(label+' native note bytes',0)
      assert db.execute('SELECT content,deleted FROM notes WHERE id=?',(seed['deleted'],)).fetchone()==('Retained soft deletion '+label,1)
      assert db.execute('SELECT content FROM messages WHERE id=? AND conversation_id=?',(seed['message'],seed['conversation'])).fetchone()==(label+' message bytes',)
      assert db.execute('SELECT 1 FROM notes WHERE id=?',(after_ids[label],)).fetchone() is None
      study=seed['domains']['study'];quiz=seed['domains']['quiz']
      assert db.execute('SELECT name,description,card_count,is_deleted FROM decks WHERE id=?',(study['deck'],)).fetchone()==(label+' study','Retained local deck',1,0)
      card=db.execute('SELECT deck_id,front,back,tags,type,metadata,is_deleted FROM flashcards WHERE id=?',(study['card'],)).fetchone()
      assert card[:5]==(study['deck'],label+' question',label+' answer','retained '+label,'basic')
      assert json.loads(card[5])=={'notes':label+' study note'} and card[6]==0
      assert db.execute('SELECT name,description,total_questions,deleted FROM quizzes WHERE id=?',(quiz['quiz'],)).fetchone()==(label+' quiz','Stored completed quiz',1,0)
      question=db.execute('SELECT quiz_id,question_type,question_text,correct_answer,points,deleted FROM quiz_questions WHERE id=?',(quiz['question'],)).fetchone()
      assert question[:3]==(quiz['quiz'],'fill_blank',label+' retained answer?')
      assert question[3]==label and question[4:]==(2,0)
      attempt=db.execute('SELECT quiz_id,completed_at,score,total_possible,questions_snapshot,answers FROM quiz_attempts WHERE id=?',(quiz['attempt'],)).fetchone()
      assert attempt[:4]==(quiz['quiz'],quiz['completed_at'],2,2)
      questions=json.loads(attempt[4]);answers=json.loads(attempt[5])
      assert len(questions)==len(answers)==1 and questions[0]['id']==quiz['question']
      assert questions[0]['question_text']==label+' retained answer?' and questions[0]['correct_answer']==label
      assert answers[0]['question_id']==quiz['question'] and answers[0]['user_answer']==label
      assert answers[0]['is_correct'] is True and answers[0]['points_awarded']==2
      for alias_owner in ('study.local','quiz.local'):
       alias=next(peer for peer in manifest['files'] if peer['owner_id']==alias_owner and source[peer['logical_id']].path==source[row['logical_id']].path)
       assert row['logical_id'] in source[alias['logical_id']].dependencies
       assert alias['sha256']==row['sha256']
     elif owner=='db.media.primary':
      assert db.execute('SELECT content FROM Media WHERE id=?',(seed['media'],)).fetchone()==(label+' media bytes',)
     else:
      assert db.execute('SELECT query FROM research_sessions WHERE id=?',(seed['research'],)).fetchone()==(label+' research query',)
      assert db.execute('SELECT deleted FROM research_sessions WHERE id=?',(seed['deleted_research'],)).fetchone()==(1,)
   assert any(row['logical_id'] in source and source[row['logical_id']].owner=='chatbooks.archives' and source[row['logical_id']].path==Path(seed['empty']) for row in manifest['directories'])
   for owner,domain in (('writing.local','writing'),('notifications.client','notification')):
    expected=seed['domains'][domain]
    row=next(row for row in manifest['files'] if row['owner_id']==owner and source[row['logical_id']].path==Path(expected['path']))
    with closing(sqlite3.connect((result.root/row['payload']).as_uri()+'?mode=ro',uri=True)) as db:
     if domain=='writing':
      assert db.execute('SELECT title FROM writing_projects WHERE id=?',(expected['project'],)).fetchone()==(label+' writing',)
      assert db.execute('SELECT project_id,title FROM writing_chapters WHERE id=?',(expected['chapter'],)).fetchone()==(expected['project'],label+' chapter')
      assert db.execute('SELECT chapter_id,project_id,content_markdown FROM writing_scenes WHERE id=?',(expected['scene'],)).fetchone()==(expected['chapter'],expected['project'],label+' revised scene')
      version=db.execute('SELECT entity_type,entity_id,version_number,label,payload_json FROM writing_versions WHERE id=?',(expected['version'],)).fetchone()
      assert version[:4]==('scene',expected['scene'],expected['version_number'],'before edit')
      history=json.loads(version[4])
      assert history['content_markdown']==label+' original scene'
      assert history['chapter_id']==expected['chapter'] and history['project_id']==expected['project']
     else:
      notification=db.execute('SELECT category,title,message,source_backend,source_entity_kind,source_entity_id,payload,is_read,is_dismissed,read_at,dismissed_at FROM client_notifications WHERE id=?',(expected['id'],)).fetchone()
      assert notification[:6]==('study',label+' completed quiz',label+' durable completion','local','study_quiz_attempt',seed['domains']['quiz']['attempt'])
      assert json.loads(notification[6])=={'quiz_id':seed['domains']['quiz']['quiz'],'attempt_id':seed['domains']['quiz']['attempt']}
      assert notification[7:]==(1,1,expected['read_at'],expected['dismissed_at'])
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


def _capture_two_profiles(tmp_path):
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
    return root, environment


def test_two_known_profiles_complete_capture_preserves_native_state_and_resumes(
    tmp_path,
):
    _capture_two_profiles(tmp_path)


_PLAN_ISOLATED = (
    _PRIVATE
    + r"""
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
assert 'tldw_chatbook.config' not in sys.modules
archive=acquire(fixture/'two-profiles.tldw-backup.zip',fixture/'stage2-acquired',ArchiveLimits(),None,threading.Event())
doc=verify_sealed(archive)
assert doc.consistency=='coherent' and len(doc.profile_ids)==2
destination=fixture/'restore-destinations'
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
producer={row.logical_id:row for row in doc.producer_inventory}
profiles={profile:('first','second')[index] for index,profile in enumerate(doc.profile_ids)}
names={profile:'recovered-'+label for profile,label in profiles.items()}
mapping={}
# Exact installed destinations for this finite captured cohort; new owners must
# be reviewed explicitly instead of silently landing in a miscellaneous folder.
custom={'db.chachanotes.primary','chat.attachments','notes.sync_bindings','quiz.local','study.local','db.media.primary','research.local'}
ordinary={'db.evals','db.library_collections','db.library_ingest_jobs','db.scheduled_tasks','db.subscriptions','db.workspaces','kanban.local','mcp.targets','notifications.client','runtime.event_state','runtime.sync_state','writing.local','chat.prompt_history','personas','chat.dictionary_history','chat.grammars','feedback','audio.history'}
trees={'chat.dictionaries':'chat_dicts','chatbooks.archives':'chatbooks','rag.definitions':'rag_profiles','chunking.templates':'chunking_templates'}
for key,row in roots.items():
 owner=producer[key].owner_id
 if row.synthetic:
  members=[item for item in doc.files if item.root_id==key]
  assert len(members)==1,(key,members)
  member=members[0]
  assert member.owner_id==owner and member.logical_id.startswith('profile:')
  profile=member.logical_id.split(':')[1]
 else:
  assert key.startswith('profile:'),key
  profile=key.split(':')[1]
 home=destination/profiles[profile]
 data=home/'data'/names[profile]
 if row.synthetic:
  if owner in {'config','config.history','runtime.source_state','ui.state','ui.emoji_recents'}:target=home/'config'
  elif owner in custom:target=home/'custom'
  elif owner in {'db.prompts.primary','chatbooks.registry'}:target=destination/'shared-prompts'
  elif owner=='eval.definitions':target=destination/'inactive-eval'
  else:
   assert owner in ordinary,(key,owner)
   target=data
 else:
  if owner=='persona.visual_identity_builtin':target=destination/'inactive-builtin'
  elif owner=='ui.themes':target=home/'config'/'themes'
  else:
   assert owner in trees,(key,owner)
   target=data/trees[owner]
 mapping[key]=target
for profile,label in profiles.items():mapping[f'profile:{profile}:paths.data_dir']=destination/label/'data'
builtin=[key for key in roots if producer[key].owner_id=='persona.visual_identity_builtin']
assert len(builtin)==2 and all(not roots[key].synthetic for key in builtin)
assert producer[builtin[0]].shared_group and producer[builtin[0]].shared_group==producer[builtin[1]].shared_group
assert mapping[builtin[0]]==mapping[builtin[1]] and not mapping[builtin[0]].exists()
assert set(roots)<=mapping.keys()
(fixture/'stage2-manifest.json').write_bytes(archive.manifest_bytes)
evidence={'archive':str(archive.path),'archive_digest':archive.digest,'destinations':{key:str(value) for key,value in mapping.items()},'profile_names':names,'shared_builtin_roots':builtin,'shared_builtin_group':producer[builtin[0]].shared_group}
(fixture/'stage2-mapping.json').write_text(json.dumps(evidence,sort_keys=True,indent=2))
try:
 plan=plan_restore(archive,mode='isolated',destinations=mapping,target=None,profile_names=names)
except ValueError as error:
 (fixture/'stage2-refusal.json').write_text(json.dumps({'type':type(error).__name__,'reason':str(error)}))
 raise
assert dict(plan.destinations)=={key:mapping[key] for key in roots}
assert {row.logical_id for row in doc.files}<=set(dict(plan.restore))
assert {row.logical_id for row in doc.directories if not row.synthetic}<=set(dict(plan.restore))
assert {dict(plan.restore)[key] for key in builtin}=={destination/'inactive-builtin'}
assert not any(path.exists() for path in mapping.values())
assert not blocked_attempts(),blocked_attempts()
print('TWO_PROFILE_ISOLATED_PLAN_VALIDATED',flush=True)
"""
)


def _isolated_environment(root, environment):
    """Give restoration its own private selectors and native control storage."""
    for name in (
        "restore-home",
        "restore-config",
        "restore-data",
        "restore-cache",
        "restore-tmp",
        "restore-destinations",
    ):
        (root / name).mkdir(mode=0o700)
    selector = root / "restore-config" / "config.toml"
    selector.write_text('[general]\nusers_name="restore-parent"\n', encoding="utf-8")
    selector.chmod(0o600)
    return dict(
        environment,
        HOME=str(root / "restore-home"),
        USERPROFILE=str(root / "restore-home"),
        XDG_CONFIG_HOME=str(root / "restore-config"),
        XDG_DATA_HOME=str(root / "restore-data"),
        XDG_CACHE_HOME=str(root / "restore-cache"),
        TMPDIR=str(root / "restore-tmp"),
        TLDW_CONFIG_PATH=str(selector),
    )


def test_two_captured_profiles_plan_shared_concrete_roots_for_isolated_restore(
    tmp_path,
):
    """Stage2a plans every earned root; it does not publish or approve owners."""
    root, environment = _capture_two_profiles(tmp_path)
    environment = _isolated_environment(root, environment)
    assert "TWO_PROFILE_ISOLATED_PLAN_VALIDATED" in _run_profile_child(
        root, "stage2-plan", _PLAN_ISOLATED, environment
    )


_OPEN_RESTORED = r"""
import asyncio,json,sys
from datetime import datetime
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
fixture=Path(sys.argv[-1]);args=sys.argv[1:-1]
profile=args[args.index('--recovery-profile')+1]
control=Path(args[args.index('--recovery-control-root')+1])
attempt=args[args.index('--recovery-launch-attempt')+1]
expected=json.loads((fixture/'stage2-installed.json').read_text())[profile]
seed=json.loads((fixture/(expected['label']+'-seed.json')).read_text())
sys.argv=['tldw-chatbook',*args,'--help']
from tldw_chatbook.cli import main_cli_runner
try:main_cli_runner()
except SystemExit as error:assert error.code in (None,0),error
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.profile_open import opened_receipt
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
assert opened_receipt(profile,control,attempt) is None
async def main():
 app=TldwCli()
 async with app.run_test(size=(100,36)):
  async with asyncio.timeout(20):
   while not getattr(app,'_recovery_open_checked',False):await asyncio.sleep(.03)
  receipt=opened_receipt(profile,control,attempt)
  assert receipt is not None
  assert app._initial_screen_pushed and app._ui_ready
  assert receipt.installation_id==app.client_id==expected['installation_id']
  assert str(app.chachanotes_db.db_path)==expected['core']
  assert str(app.media_db.db_path)==expected['media']
  assert str(app.prompts_db.db_path)==expected['prompts']
  label=expected['label'];core=app.chachanotes_db
  from tldw_chatbook.Chat.prompt_history import PromptHistory,default_prompt_history_path
  from tldw_chatbook.Widgets.emoji_picker import load_recent_emojis
  history=PromptHistory(default_prompt_history_path());await history.load()
  assert history.size==1 and history.complete(label)==label+' retained input'
  assert load_recent_emojis()==['★' if label=='alpha' else '✓']
  durable=seed['durable_api']
  persona=app.local_character_persona_service.get_persona_profile(durable['persona'])
  assert persona['system_prompt']==label+' retained persona prompt' and not persona['is_active']
  version=app.local_chat_dictionary_service.get_version(durable['dictionary'],durable['dictionary_revision'])
  assert version['snapshot']['description']==label+' retained dictionary history'
  grammar=await app.local_chat_grammars_service.get_grammar(durable['grammar'])
  assert grammar['grammar_text']=='root ::= "'+label+'"' and grammar['validation_status']=='unchecked'
  feedback=await app.local_feedback_service.get_feedback(durable['feedback'])
  assert feedback['conversation_id']==seed['conversation'] and feedback['message_id']==seed['message']
  assert feedback['helpful'] is True and feedback['user_notes']==label+' retained feedback'
  audio=await app.local_audio_services_service.get_tts_history_entry(durable['audio'])
  assert audio['text']==label+' spoken input' and audio['content']==(label+' retained audio bytes').encode() and audio['favorite']
  from tldw_chatbook.Chunking.chunking_templates import ChunkingTemplateManager
  template=ChunkingTemplateManager().load_template(durable['template'])
  assert template is not None and template.description==label+' chunking template' and template.pipeline==[]
  assert core.get_note_by_id(seed['note'])['content']==label+' native note bytes'
  assert core.get_note_by_id(seed['deleted']) is None
  assert core.get_message_by_id(seed['message'])['content']==label+' message bytes'
  assert core.get_note_by_title('After capture '+label) is None
  other='beta' if label=='alpha' else 'alpha'
  assert core.get_note_by_title(other+' retained note') is None
  assert app.media_db.get_media_by_id(seed['media'])['content']==label+' media bytes'
  assert app.local_research_service.get_session(seed['research'])['query']==label+' research query'
  assert app.local_research_service.get_session(seed['deleted_research']) is None
  domains=seed['domains'];writing=app.local_writing_service
  assert writing is not None and writing.db_path!=Path(domains['writing']['path'])
  project=writing.get_project(domains['writing']['project'])
  chapter=writing.get_chapter(domains['writing']['chapter'])
  scene=writing.get_scene(domains['writing']['scene'])
  version=writing.get_version('scene',scene['id'],domains['writing']['version_number'])
  assert project['title']==label+' writing' and chapter['project_id']==project['id']
  assert scene['chapter_id']==chapter['id'] and scene['project_id']==project['id']
  assert scene['content_markdown']==label+' revised scene'
  assert version['id']==domains['writing']['version'] and version['label']=='before edit'
  assert version['payload']['content_markdown']==label+' original scene'
  assert version['payload']['chapter_id']==chapter['id'] and version['payload']['project_id']==project['id']
  study=app.local_study_service;quiz_service=app.local_quiz_service
  assert study.db is core and quiz_service.db is core
  deck=study.get_deck(domains['study']['deck']);card=study.get_flashcard(domains['study']['card'])
  assert deck['name']==label+' study' and deck['card_count']==1
  assert card['deck_id']==deck['id'] and card['front']==label+' question' and card['back']==label+' answer'
  assert card['tags']=='retained '+label and json.loads(card['metadata'])=={'notes':label+' study note'}
  quiz=quiz_service.get_quiz(domains['quiz']['quiz'])
  completed=quiz_service.get_attempt(domains['quiz']['attempt'],include_questions=True,include_answers=True)
  assert quiz['name']==label+' quiz' and quiz['total_questions']==1
  assert completed['quiz_id']==quiz['id'] and completed['completed_at']==datetime.fromisoformat(domains['quiz']['completed_at'])
  assert completed['score']==completed['total_possible']==2
  assert len(completed['questions'])==len(completed['answers'])==1
  question=completed['questions'][0];answer=completed['answers'][0]
  assert question['id']==answer['question_id']==domains['quiz']['question']
  assert question['question_text']==label+' retained answer?' and question['correct_answer']==label
  assert answer['user_answer']==label and answer['is_correct'] is True and answer['points_awarded']==2
  assert Path(app.client_notifications_db.db_path)!=Path(domains['notification']['path'])
  notification=app.client_notifications_db.get_notification(domains['notification']['id'])
  assert notification['title']==label+' completed quiz' and notification['message']==label+' durable completion'
  assert notification['source_backend']=='local' and notification['source_entity_kind']=='study_quiz_attempt'
  assert notification['source_entity_id']==completed['id']
  assert notification['payload']=={'quiz_id':quiz['id'],'attempt_id':completed['id']}
  assert notification['is_read'] and notification['is_dismissed']
  assert notification['read_at']==domains['notification']['read_at'] and notification['dismissed_at']==domains['notification']['dismissed_at']
  for original in ('alpha','beta'):
   identity=json.loads((fixture/(original+'-seed.json')).read_text())
   assert app.prompts_db.get_prompt_by_id(identity['prompt'])['user_prompt']==original+' prompt bytes'
  job=app.library_ingest_jobs.get_job('ingest-job-1')
  assert job is not None and job.state==IngestJobState.FAILED
  assert job.error=='Interrupted by app restart' and not job.permanent
  assert job.source_path==str(fixture/(label+'-input.txt'))
  created=core.add_note('Restored ordinary write '+label,'Fresh installation '+label)
  row=core.get_note_by_id(created)
  assert row['content']=='Fresh installation '+label and row['client_id']==app.client_id
 assert not blocked_attempts(),blocked_attempts()
 print('RESTORED_PROFILE_OPENED',label,flush=True)
asyncio.run(main())
"""


_RESTORE_ISOLATED = "".join(
    (
        _PLAN_ISOLATED,
        r"""
import hashlib,sqlite3,stat,subprocess
from contextlib import closing
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
selected=dict(plan.restore)
source_paths=[fixture/label/'config.toml' for label in ('alpha','beta')]
source_paths += [fixture/label/'custom'/leaf for label in ('alpha','beta') for leaf in ('notes.db','media.db','research.db')]
source_paths.append(fixture/'shared'/'prompts.db')
source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
control=fixture/'restore-home'/'control'
first=restore_isolated(archive,plan,control,threading.Event())
assert all(path.is_file() for key,path in plan.restore if key in {row.logical_id for row in doc.files})
planned_metadata={key:(desired,applied) for key,desired,applied in plan.metadata}
for row in doc.directories:
 if row.synthetic:continue
 desired,applied=planned_metadata[row.logical_id]
 assert desired==row.metadata
 assert applied.mode==0o700 and applied.mtime_ns==desired.mtime_ns
 assert ('metadata_normalized:'+row.logical_id in plan.issues)==(desired!=applied)
 info=selected[row.logical_id].stat()
 assert stat.S_ISDIR(info.st_mode)
 assert stat.S_IMODE(info.st_mode)==applied.mode
 assert info.st_mtime_ns==desired.mtime_ns,(row.logical_id,info.st_mtime_ns,desired.mtime_ns)
service=RecoveryService(control)
try:
 rows=service.profiles()
 assert len(rows)==2 and first in {row['profile_id'] for row in rows}
 assert all(row['status']=='restoration_validated' and row['needs_setup'] and row['requirements_checked'] for row in rows)
 assert len({row['generation'] for row in rows})==1
 installed={}
 for row in rows:
  entry=_launch_descriptor(row['profile_id'],control)
  profile=entry.source_profile
  def path_for(owner):
   matches=[item for item in doc.files if item.owner_id==owner and item.logical_id.startswith('profile:'+profile+':')]
   assert len(matches)==1,(owner,matches)
   return selected[matches[0].logical_id]
  core=path_for('db.chachanotes.primary')
  with closing(sqlite3.connect(core.as_uri()+'?mode=ro',uri=True)) as db:
   labels=[label for label in ('alpha','beta') if db.execute('SELECT 1 FROM notes WHERE id=?',(json.loads((fixture/(label+'-seed.json')).read_text())['note'],)).fetchone()]
   assert len(labels)==1,labels
   label=labels[0];seed=json.loads((fixture/(label+'-seed.json')).read_text())
   for owner,expected_file in seed['durable_files'].items():
    payload=next(item for item in doc.files if item.owner_id==owner and item.logical_id.startswith('profile:'+profile+':'))
    assert selected[payload.logical_id].read_bytes()==bytes.fromhex(expected_file['hex'])
   assert db.execute('SELECT content,deleted FROM notes WHERE id=?',(seed['deleted'],)).fetchone()==('Retained soft deletion '+label,1)
   assert db.execute('SELECT 1 FROM notes WHERE title=?',('After capture '+label,)).fetchone() is None
  with closing(sqlite3.connect(path_for('research.local').as_uri()+'?mode=ro',uri=True)) as db:
   assert db.execute('SELECT deleted FROM research_sessions WHERE id=?',(seed['deleted_research'],)).fetchone()==(1,)
  with closing(sqlite3.connect(path_for('db.library_ingest_jobs').as_uri()+'?mode=ro',uri=True)) as db:
   assert db.execute('SELECT job_id,state FROM ingest_jobs').fetchall()==[('ingest-job-1','queued')]
  empty=selected[f'profile:{profile}:chatbooks.archives']
  assert empty.is_dir() and not list(empty.iterdir())
  installed[entry.profile_id]={'label':label,'installation_id':entry.installation_id,'config':entry.config,'core':str(core),'media':str(path_for('db.media.primary')),'prompts':str(path_for('db.prompts.primary'))}
 assert len({row['installation_id'] for row in installed.values()})==2
 assert len({row['core'] for row in installed.values()})==2
 assert len({row['prompts'] for row in installed.values()})==1
 prompts=Path(next(iter(installed.values()))['prompts'])
 assert prompts.stat().st_ino!=(fixture/'shared'/'prompts.db').stat().st_ino
 (fixture/'stage2-installed.json').write_text(json.dumps(installed,indent=2))
 print('TWO_PROFILE_RESTORATION_VALIDATED',flush=True)
 original_call=subprocess.call
 def headless(argv,**kwargs):
  profile=argv[argv.index('--recovery-profile')+1]
  output_path=fixture/('stage2-open-'+installed[profile]['label']+'.log')
  with output_path.open('w') as output:
   result=subprocess.run([sys.executable,'-c',OPEN_CHILD,*argv[4:],str(fixture)],**kwargs,stdout=output,stderr=subprocess.STDOUT,text=True,timeout=45)
  assert result.returncode==0,output_path.read_text()[-14000:]
  return result.returncode
 subprocess.call=headless
 try:
  for row in rows:
   operation=service.start_open_profile(row['profile_id'])
   state=service.wait(operation,timeout=50)
   assert state['state']=='succeeded',dict(state)
   assert state['result']['opened_successfully'] and state['result']['needs_setup'],dict(state)
 finally:subprocess.call=original_call
 assert all(row['needs_setup'] and row['status']=='restoration_validated' for row in service.profiles())
finally:service.close()
assert source_hashes=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
for label in ('alpha','beta'):
 seed=json.loads((fixture/(label+'-seed.json')).read_text())
 assert all(Path(row['path']).read_bytes()==bytes.fromhex(row['hex']) for row in seed['durable_files'].values())
assert not blocked_attempts(),blocked_attempts()
print('TWO_PROFILE_RESTORED_AND_OPENED',flush=True)
""",
    )
)


def test_two_captured_profiles_restore_and_open_with_native_content(tmp_path):
    """Stage2b executes and opens the finite cohort without approving owners."""
    root, environment = _capture_two_profiles(tmp_path)
    environment = _isolated_environment(root, environment)
    script = "OPEN_CHILD=" + repr(_OPEN_RESTORED) + "\n" + _RESTORE_ISOLATED
    assert "TWO_PROFILE_RESTORED_AND_OPENED" in _run_profile_child(
        root, "stage2-restore", script, environment, timeout=75
    )
