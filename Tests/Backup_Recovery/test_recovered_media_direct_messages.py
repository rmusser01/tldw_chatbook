"""Actual direct-message deletion releases only positively tombstoned references."""

import json
import os
import subprocess  # nosec B404 - fixed offline native fixture
import sys
from pathlib import Path

import pytest

_PROGRAM = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import recovered_media as owner
from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
from tldw_chatbook.Character_Chat.Character_Chat_Lib import remove_message_from_conversation
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
from loguru import logger
root=Path(os.environ['FIXTURE_ROOT'])
db=config.get_chachanotes_db_lazy()
character=db.add_character_card({'name':'Direct references'})
conversation=ChatConversationService(db).create_conversation(title='Direct references',character_id=character,assistant_kind='character',discovery_owner='ccp_character')
first=db.add_message({'conversation_id':conversation,'sender':'User','content':'first'})
keep=db.add_message({'conversation_id':conversation,'sender':'User','content':'keep'})
source=root/'opaque';source.write_bytes(b'unchanged retained media');source.chmod(0o600)
media=owner.RecoveredMedia(root/'data'/'test'/'recovered_media')
profile=owner.current_profile_id()
asset=media.retain(source,profile=profile,message=first,slug='one',media_type='image/png')
for p,m,s,t in ((profile,first,'two','image/png'),(profile,keep,'keep','image/png'),('historical',first,'one','image/png')):
 media.add_reference(asset,profile=p,message=m,slug=s,media_type=t)
service=build_persona_service(db)
route,mode=sys.argv[1:]
def refs():return owner.review_recovered_asset(media.root,asset).references
before=refs()
def delete(version=1):
 if route=='library':return remove_message_from_conversation(db,first,version)
 if route=='alias':return service.delete_character_message(first,expected_version=version)
 return service.delete_character_chat_message(first,expected_version=version)
try:
 if mode=='native_failure':
  try:result=delete(999)
  except ConflictError:
   assert route!='library'
  else:assert route=='library' and result is False
  assert refs()==before and not db.get_message_tombstones([first])
 elif mode=='conversation_only':
  assert ChatConversationService(db).delete_conversation(conversation,expected_version=1)
  assert refs()==before and not db.get_message_tombstones([first,keep])
 elif mode=='foreign_db':
  from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
  other=CharactersRAGDB(root/'foreign.sqlite',client_id='fixture')
  try:
   actor=other.add_character_card({'name':'Foreign character'})
   cid=ChatConversationService(other).create_conversation(title='Foreign character',character_id=actor,assistant_kind='character',discovery_owner='ccp_character')
   other.add_message({'id':first,'conversation_id':cid,'sender':'User','content':'foreign'})
   if route=='library':
    assert remove_message_from_conversation(other,first,1) is True
    assert {row['message_id'] for row in other.get_message_tombstones([first])}=={first}
   else:
    foreign=build_persona_service(other)
    assert foreign.delete_character_chat_message(first)=={'status':'deleted','message_id':first}
    assert foreign.recovered_media_cleanup_warning
    assert {row['message_id'] for row in other.get_message_tombstones([first])}=={first}
    # Ordinary foreign construction stays supported. Retargeting an installed
    # selected service must still refuse its native guarded operation.
    service.db=other
    try:service.delete_character_chat_message(first)
    except Exception as error:assert 'chat_source_selection_changed' in str(error),repr(error)
    else:raise AssertionError('retargeted installed source accepted')
   assert refs()==before and not db.get_message_tombstones([first])
  finally:other.close()
 elif mode=='config_drift':
  selected=Path(os.environ['TLDW_CONFIG_PATH'])
  def edit():selected.write_text(selected.read_text()+'\n# changed selected source\n')
  if route=='library':
   original=db.soft_delete_message
   def committed_then_edit(*args,**kwargs):
    result=original(*args,**kwargs);edit();return result
   db.soft_delete_message=committed_then_edit
  else:edit()
  try:result=delete()
  except Exception as error:
   assert route!='library' and ('source' in str(error) or 'scope' in str(error)),repr(error)
  else:
   assert result is True if route=='library' else result['status']=='deleted'
   if route!='library':assert service.recovered_media_cleanup_warning
  assert refs()==before
 else:
  messages=[]
  sink=logger.add(lambda entry:messages.append(str(entry)))
  if mode=='owner_failure':
   def fail(*args,**kwargs):raise RuntimeError('owner unavailable')
   owner.release_message_references=fail
  result=delete()
  assert result is True if route=='library' else result=={'status':'deleted','message_id':first}
  assert {row['message_id'] for row in db.get_message_tombstones([first])}=={first}
  if mode=='owner_failure':
   assert refs()==before
   if route!='library':assert service.recovered_media_cleanup_warning=='Message deleted; recovered-media reference cleanup is pending.'
   assert any('Message deleted; recovered-media reference cleanup is pending.' in text for text in messages)
  else:
   assert refs()==tuple(ref for ref in before if ref[0]!=profile or ref[1]!=first),refs()
   assert len(refs())==2
  logger.remove(sink)
 assert owner.review_recovered_asset(media.root,asset).asset.state=='ready'
 assert media.resolve(asset)[1].read_bytes()==b'unchanged retained media'
 assert source.read_bytes()==b'unchanged retained media'
 assert not blocked_attempts(),blocked_attempts()
finally:db.close()
"""


@pytest.mark.parametrize(
    ("route", "mode"),
    [
        ("local", "success"),
        ("alias", "success"),
        ("library", "success"),
        ("local", "native_failure"),
        ("library", "native_failure"),
        ("local", "owner_failure"),
        ("library", "owner_failure"),
        ("local", "conversation_only"),
        ("local", "foreign_db"),
        ("library", "foreign_db"),
        ("local", "config_drift"),
        ("library", "config_drift"),
    ],
)
def test_direct_message_reference_lifecycle(tmp_path, route, mode):
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
            [sys.executable, "-c", _PROGRAM, route, mode],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )  # nosec B603 - fixed program and private fixture
    assert result.returncode == 0, log.read_text()[-12000:]
