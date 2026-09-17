"""Actual chat tombstones release only the bound profile's recovered references."""

import json
import os
import subprocess  # nosec B404 - fixed offline native fixture program
import sys
from pathlib import Path

import pytest

_PROGRAM = r"""
import os,sys
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import recovered_media as owner
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
root=Path(os.environ['FIXTURE_ROOT'])
db=config.get_chachanotes_db_lazy()
conversation=db.add_conversation({'title':'Reference lifecycle'})
def message(text,parent=None):
 return db.add_message({'conversation_id':conversation,'sender':'User','content':text,'parent_message_id':parent})
first=message('first');child=message('child',first);keep=message('keep')
if sys.argv[1]=='no_catalog':
 service=ConsoleRuntime(SimpleNamespace(chachanotes_db=db)).ensure_chat_store().persistence
 assert not service.recovered_media_cleanup_pending
 service.delete_message_subtree(message_id=first)
 assert not service.recovered_media_cleanup_pending
 assert not (root/'data'/'test'/'recovered_media').exists()
 db.close();assert not blocked_attempts();sys.exit(0)
source=root/'opaque';source.write_bytes(b'unchanged retained media');source.chmod(0o600)
media=owner.RecoveredMedia(root/'data'/'test'/'recovered_media')
profile=owner.current_profile_id()
asset=media.retain(source,profile=profile,message=first,slug='one',media_type='image/png')
for p,m,s in ((profile,first,'two'),(profile,child,'child'),(profile,keep,'keep'),('historical',first,'one')):
 media.add_reference(asset,profile=p,message=m,slug=s,media_type='image/png')
service=ConsoleRuntime(SimpleNamespace(chachanotes_db=db)).ensure_chat_store().persistence
assert isinstance(service,ChatPersistenceService)
mode=sys.argv[1]
def refs():return owner.review_recovered_asset(media.root,asset).references
def gone():return all(p!='historical' and m not in (first,child) or p=='historical' for p,m,s,t in refs())
try:
 if mode=='subtree':
  rows=service.delete_message_subtree(message_id=first)
  assert {r['message_id'] for r in rows}=={first,child}
  assert gone() and len(refs())==2,refs()
  assert not service.recovered_media_cleanup_pending
 elif mode=='native_failure':
  original=db.soft_delete_message_subtree
  def fail(*args,**kwargs):raise RuntimeError('native write refused')
  db.soft_delete_message_subtree=fail
  before=refs()
  try:service.delete_message_subtree(message_id=first)
  except RuntimeError:pass
  else:raise AssertionError('chat mutation succeeded')
  assert refs()==before and not db.get_message_tombstones([first,child])
  db.soft_delete_message_subtree=original
 elif mode in ('owner_failure','process_failure'):
  original=owner.release_message_references
  def fail(*args,**kwargs):raise RuntimeError('owner unavailable')
  owner.release_message_references=fail
  rows=service.delete_message_subtree(message_id=first)
  assert {r['message_id'] for r in rows}=={first,child}
  assert service.recovered_media_cleanup_pending and not gone()
  assert service.recovered_media_cleanup_warning=='Message deleted; recovered-media reference cleanup is pending.'
  owner.release_message_references=original
  if mode=='process_failure':
   import json
   media.add_reference(asset,profile=profile,message='missing-id',slug='missing',media_type='image/png')
   (root/'committed.json').write_text(json.dumps({'asset':asset,'first':first,'child':child,'keep':keep,'profile':profile,'references':refs()}))
  else:
   recomposed=ConsoleRuntime(SimpleNamespace(chachanotes_db=db)).ensure_chat_store().persistence
   assert gone() and not recomposed.recovered_media_cleanup_pending
   assert recomposed.retry_recovered_media_references()
 elif mode=='retry':
  db.soft_delete_message_subtree(first,expected_version=1)
  media.add_reference(asset,profile=profile,message='missing-id',slug='missing',media_type='image/png')
  restarted=ConsoleRuntime(SimpleNamespace(chachanotes_db=db)).ensure_chat_store().persistence
  assert gone() and len(refs())==3,refs()
  assert any(r[1]=='missing-id' for r in refs()) and any(r[1]==keep for r in refs())
  assert restarted.retry_recovered_media_references()
 elif mode=='tombstone':
  media.delete(asset)
  service.delete_message_subtree(message_id=first)
  assert gone() and len(refs())==2
  assert media.resolve(asset)==('deleted',None)
 elif mode=='config_drift':
  selected=Path(os.environ['TLDW_CONFIG_PATH']);selected.write_text(selected.read_text()+'\n# edited\n')
  service.delete_message_subtree(message_id=first)
  assert service.recovered_media_cleanup_pending and not gone()
 elif mode=='varied_refs':
  extra=[]
  for slug,media_type in (('clip','video/webm'),('photo','image/jpeg')):
   new=media.retain(source,profile=profile,message=first,slug=slug,media_type=media_type)
   media.add_reference(new,profile=profile,message=first,slug=slug+'-alias',media_type=media_type)
   media.add_reference(new,profile='historical',message=first,slug=slug,media_type=media_type)
   extra.append(new)
  service.delete_message_subtree(message_id=first)
  assert gone() and len(refs())==2
  for other in extra:
   review=owner.review_recovered_asset(media.root,other)
   assert len(review.references)==1 and review.references[0][0]=='historical',review
   assert media.resolve(other)[1].read_bytes()==source.read_bytes()
  assert not service.recovered_media_cleanup_pending
 elif mode=='save_history':
  assert service.save_history(conversation_id=conversation,chatbot_history=[{'id':keep,'role':'User','content':'keep'}])==1
  assert gone() and len(refs())==2
  assert not service.recovered_media_cleanup_pending
 elif mode in ('root_drift','catalog_drift'):
  import shutil
  if mode=='root_drift':
   moved=media.root.with_name('previous-root');media.root.rename(moved)
   shutil.copytree(moved,media.root)
  else:
   replacement=media.root/'replacement.sqlite3';shutil.copyfile(media.db_path,replacement);replacement.chmod(0o600);replacement.replace(media.db_path)
  service.delete_message_subtree(message_id=first)
  assert service.recovered_media_cleanup_pending and not gone()
 elif mode=='native_pause':
  from tldw_chatbook.Backup_Recovery import storage_admission
  original=db.soft_delete_message_subtree;pauses=[]
  def committed_then_pause(*args,**kwargs):
   result=original(*args,**kwargs);pauses.append(storage_admission._begin_local_pause());return result
  db.soft_delete_message_subtree=committed_then_pause
  try:
   rows=service.delete_message_subtree(message_id=first)
   assert {r['message_id'] for r in rows}=={first,child}
   assert service.recovered_media_cleanup_pending
  finally:
   db.soft_delete_message_subtree=original
   for pause in pauses:pause.resume()
  assert not gone()
  assert service.retry_recovered_media_references() and gone()
 elif mode=='ui_warning':
  import asyncio
  from textual.widgets import Button
  from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
  from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
  from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage,ConsoleMessageRole
  from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
  store=ConsoleChatStore(persistence=service)
  nodes=[ConsoleChatMessage(role=ConsoleMessageRole.USER,content=row['content'],persisted_message_id=row['id'],parent_message_id=row['parent_message_id']) for row in db.get_messages_for_conversation(conversation)]
  session=store.restore_persisted_session(title='Reference lifecycle',workspace_id=None,persisted_conversation_id=conversation,all_nodes=nodes,active_leaf_persisted_id=child)
  selected=next(node for node in store.messages_for_session(session.id) if node.persisted_message_id==first)
  notices=[]
  async def sync():pass
  host=SimpleNamespace(_console_speech_states={},_ensure_console_chat_store=lambda:store,_console_message_presentation=lambda message:message,_console_message_action_service=ConsoleMessageActionService(),_pending_console_delete_message_id=selected.id,_ensure_console_chat_controller=lambda:SimpleNamespace(clear_original_attempts_for_session=lambda session:None),_console_original_attempt_previews={},_invalidate_console_persisted_rows_cache=lambda:None,_sync_native_console_chat_ui=sync,app_instance=SimpleNamespace(notify=lambda message,**kwargs:notices.append((message,kwargs))))
  button=Button('Delete');button.console_action_id='delete';button.console_message_id=selected.id
  original=owner.release_message_references
  def fail(*args,**kwargs):raise RuntimeError('owner unavailable')
  owner.release_message_references=fail
  try:assert asyncio.run(ConsoleMessageController.handle_console_message_action(host,Button.Pressed(button)))
  finally:owner.release_message_references=original
  assert {row['message_id'] for row in db.get_message_tombstones([first,child])}=={first,child}
  assert not gone() and service.recovered_media_cleanup_pending
  assert any(text==service.recovered_media_cleanup_warning and options['severity']=='warning' for text,options in notices),notices
  try:store.get_message(selected.id)
  except KeyError:pass
  else:raise AssertionError('committed deletion remained visible')
 elif mode=='foreign_db':
  from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
  other=CharactersRAGDB(root/'foreign.sqlite',client_id='fixture')
  try:
   cid=other.add_conversation({'title':'other'})
   other.add_message({'id':first,'conversation_id':cid,'sender':'User','content':'foreign'})
   foreign=ConsoleRuntime(SimpleNamespace(chachanotes_db=other)).ensure_chat_store().persistence
   foreign.delete_message_subtree(message_id=first)
   assert foreign.recovered_media_cleanup_pending and not gone()
  finally:other.close()
 else:raise AssertionError(mode)
 if mode!='tombstone':assert media.resolve(asset)[1].read_bytes()==source.read_bytes()
 assert not blocked_attempts()
finally:db.close()
"""


_RESUME = r"""
import json,os
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import recovered_media as owner
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
root=Path(os.environ['FIXTURE_ROOT']);saved=json.loads((root/'committed.json').read_text())
media_root=root/'data'/'test'/'recovered_media'
db=config.get_chachanotes_db_lazy()
try:
 before=owner.review_recovered_asset(media_root,saved['asset'])
 assert [list(row) for row in before.references]==saved['references']
 assert {row['message_id'] for row in db.get_message_tombstones([saved['first'],saved['child'],'missing-id',saved['keep']])}=={saved['first'],saved['child']}
 service=ConsoleRuntime(SimpleNamespace(chachanotes_db=db)).ensure_chat_store().persistence
 assert not service.recovered_media_cleanup_pending
 after=owner.review_recovered_asset(media_root,saved['asset'])
 assert len(after.references)==3,after
 assert set((p,m) for p,m,s,t in after.references)=={('historical',saved['first']),(saved['profile'],saved['keep']),(saved['profile'],'missing-id')}
 assert (media_root/(saved['asset']+'.payload')).read_bytes()==(root/'opaque').read_bytes()
 assert service.retry_recovered_media_references()
 assert owner.review_recovered_asset(media_root,saved['asset'])==after
 assert not blocked_attempts()
finally:db.close()
"""


@pytest.mark.parametrize(
    "mode",
    [
        "subtree",
        "native_failure",
        "owner_failure",
        "retry",
        "tombstone",
        "config_drift",
        "foreign_db",
        "save_history",
        "root_drift",
        "catalog_drift",
        "native_pause",
        "ui_warning",
        "no_catalog",
        "varied_refs",
        "process_failure",
    ],
)
def test_bound_message_reference_lifecycle(tmp_path, mode):
    for name in ("home", "config", "data", "cache", "tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    selected = tmp_path / "config" / "config.toml"
    selected.write_text(
        '[general]\nusers_name="test"\n[paths]\ndata_dir='
        + json.dumps(str(tmp_path / "data"))
        + "\n"
    )
    selected.chmod(0o600)
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "TMPDIR": str(tmp_path / "tmp"),
        "TLDW_CONFIG_PATH": str(selected),
        "FIXTURE_ROOT": str(tmp_path),
        "TLDW_TEST_MODE": "1",
        "TLDW_DISABLE_CONFIG_WATCH": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONPATH": str(Path.cwd()),
    }
    log = tmp_path / "child.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", _PROGRAM, mode],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )  # nosec B603 - fixed program and private fixture
    assert result.returncode == 0, log.read_text()[-10000:]
    if mode == "process_failure":
        # First process has closed its DB and exited before the next native open.
        resume_log = tmp_path / "resume.log"
        with resume_log.open("w") as output:
            resumed = subprocess.run(  # nosec B603 - fixed offline continuation
                [sys.executable, "-c", _RESUME],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        assert resumed.returncode == 0, resume_log.read_text()[-10000:]
