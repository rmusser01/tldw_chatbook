"""Installed production routing, authoring and exact Console presentation ownership."""

import pytest

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 -- shared installed fixture
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.ProductionApp.test_backup_restore_composition import _ENTRY


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The installed child guards network before importing the actual app."""


_JOURNEY = (
    _ENTRY.split("async def main():", 1)[0]
    + r"""
from io import BytesIO
import time
from PIL import Image
from textual.widgets import Button,Collapsible
from tldw_chatbook.Constants import TAB_CHAT,TAB_PERSONAS
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import EditCharacterRequested
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import CharacterPerformanceRequested,PortraitPerformanceEditor
from tldw_chatbook.Widgets.Console.character_expression_avatar import CharacterExpressionAvatar
from tldw_chatbook.Widgets.Console.console_character_picker_modal import ConsoleCharacterChoice
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES
from tldw_chatbook.Audio.speech_presentation import SpeechPresentation,SpeechPresentationOwner,SpeechPresentationSnapshot

def png(color):
 with Image.new('RGB',(32,32),color) as image,BytesIO() as output:
  image.save(output,format='PNG');return output.getvalue()

async def main():
 app=TldwCli()
 async def offline_discovery(_config):return ()
 app.console_local_server_discovery=offline_discovery
 app.app_config['_first_run']=False
 app.app_config.setdefault('first_run',{})['setup_completed']=True
 # Only routing/artwork/ownership are qualified here; the provider/device is absent.
 db=app.chachanotes_db
 original=png('red')
 cid=db.add_character_card({'name':'Radio Journey','image':original})
 async with app.run_test(size=(180,72)) as pilot:
  async def ready(predicate):
   async with asyncio.timeout(30):
    while not predicate():await asyncio.sleep(.02)
  async def edit():
   app.post_message(NavigateToScreen('ccp'))
   await ready(lambda:isinstance(app.screen,PersonasScreen))
   screen=app.screen
   await pilot.pause()
   await ready(lambda:bool(screen.query('PersonasPreviewPane')))
   await screen._select_character(str(cid),'Radio Journey')
   screen.post_message(EditCharacterRequested(str(cid)))
   await ready(lambda:screen._edit_mode=='edit' and screen._character_editor_is_active())
   await pilot.pause()
   controller=screen._portrait_performance()
   assert await controller.handle(CharacterPerformanceRequested('prepare'))
   return screen,controller
  await ready(lambda:app._ui_ready and app._initial_screen_pushed)
  await pilot.pause()
  personas,controller=await edit()
  view=personas.query_one(PortraitPerformanceEditor)
  view.query_one(Collapsible).collapsed=False
  for index,role in enumerate(FRAME_ROLES):
   path=Path.home()/(role+'.png');path.write_bytes(png(('red','green','blue')[index%3]))
   assert await controller.handle(CharacterPerformanceRequested('import',role=role),path=str(path))
  assert not view.query_one('#portrait-performance-apply',Button).disabled
  assert await controller.handle(CharacterPerformanceRequested('apply'))
  saved=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
  assert saved.active and saved.revision==1
  app.post_message(NavigateToScreen('chat'))
  await ready(lambda:isinstance(app.screen,ChatScreen))
  chat=app.screen
  await chat._character._apply_console_character_choice_async(ConsoleCharacterChoice(cid,'Radio Journey','new'))
  header=chat.query_one('#console-rail-section-toggle-character',Button)
  if not chat.query_one('#console-rail-section-header-character').open:header.press()
  await ready(lambda:any(widget.current_image is not None for widget in chat.query(CharacterExpressionAvatar)))
  avatar=chat.query_one(CharacterExpressionAvatar)
  assert chat._active_character_avatar['performance'].revision==saved.revision
  rest=avatar.current_image.tobytes()
  store=chat._console_chat_store
  session=store.active_session_id
  epoch=store.active_session_epoch()
  speech=SpeechPresentation(SpeechPresentationOwner('journey',db,session,epoch,cid,None,None))
  current=chat._message.bind_speech_presentation(speech,lambda:store.active_session_id==session and store.active_session_epoch()==epoch)
  speech.publish(SpeechPresentationSnapshot(speech.owner,1,'playing',time.monotonic(),-12,'open',True))
  await ready(lambda:avatar.current_image.getpixel((avatar.current_image.width//2,avatar.current_image.height//2))[:3]==(0,0,255))
  assert chat.query_one(CharacterExpressionAvatar) is avatar and current()
  # The real timer paints; this test does not call _tick or replace the reader.
  store.create_session(title='Replacement context',runtime_backend='local')
  await chat._sync_native_console_chat_ui()
  assert not current() and chat._character._read_character_speech() is None
  personas,reopened=await edit()
  assert reopened.draft.expected_revision==saved.revision
  assert reopened.draft.source.source_sha256==saved.source.source_sha256
  assert reopened.draft.frames()==saved.frames
  assert await reopened.handle(CharacterPerformanceRequested('remove'))
  tombstone=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
  assert not tombstone.active and tombstone.revision>saved.revision
  assert await reopened.handle(CharacterPerformanceRequested('prepare'))
  for index,role in enumerate(FRAME_ROLES):
   path=Path.home()/(role+'.png')
   assert await reopened.handle(CharacterPerformanceRequested('import',role=role),path=str(path))
  assert await reopened.handle(CharacterPerformanceRequested('apply'))
  assert db.get_character_portrait_performance(character_id=cid,expression_key='neutral').revision>tombstone.revision
  assert db.get_character_card_by_id(cid)['image']==original
 assert not blocked_attempts(),blocked_attempts()
 print('retired and reopened')
asyncio.run(main())
"""
)


def test_installed_authoring_console_reopen_remove_and_reapply(
    tmp_path, native_package
):
    _run(
        tmp_path,
        "portraits",
        "journey",
        script=_JOURNEY,
        timeout=120,
        installed_package=native_package,
    )
