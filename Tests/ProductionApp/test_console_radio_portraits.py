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


from Tests.ProductionApp.portrait_qualification_helpers import TTS_CONFIG

_JOURNEY = (
    _ENTRY.split("async def main():", 1)[0].replace(
        "selector.chmod(0o600)",
        "selector.write_text(selector.read_text()+"
        + repr(TTS_CONFIG)
        + "); selector.chmod(0o600)",
    )
    + r"""
from io import BytesIO
import hashlib,json,time
from PIL import Image
from textual.widgets import Button,Checkbox,Collapsible,Input,Select
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import EditCharacterRequested
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import CharacterPerformanceRequested,PortraitPerformanceEditor
from tldw_chatbook.Widgets.Persona_Widgets.portrait_region_editor import PortraitRegionEditor
from tldw_chatbook.Widgets.Persona_Widgets.portrait_composition_review import PortraitCompositionReview
from tldw_chatbook.Widgets.Console.character_expression_avatar import CharacterExpressionAvatar
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES
from tldw_chatbook.Character_Chat.portrait_compositing import recover_components
from tldw_chatbook.Character_Chat.character_conversation_navigation import ResolvedLocalCharacterKey
from tldw_chatbook.UI.Console_Modules.wiring import _start_character_context_chat
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.character_expression_playback import PreparedPerformance
from Tests.ProductionApp.portrait_qualification_helpers import QualificationAudio,QualificationImages

def png(color):
 with Image.new('RGB',(32,32),color) as image,BytesIO() as output:
  image.save(output,format='PNG');return output.getvalue()

async def main():
 app=TldwCli()
 async def offline_discovery(_config):return ()
 app.console_local_server_discovery=offline_discovery
 app.app_config['_first_run']=False
 app.app_config.setdefault('first_run',{})['setup_completed']=True
 app.app_config.setdefault('console',{}).setdefault('onboarding',{})['first_send_completed']=True
 app.app_config.setdefault('appearance',{}).update(character_expression_mode='dynamic',animations_enabled=True,reduce_motion=False)
 db=app.chachanotes_db
 images=QualificationImages()
 cfg=images.install()
 audio=QualificationAudio();audio.install()
 if images.live:
  from importlib import resources
  original=resources.files('tldw_chatbook').joinpath('assets/characters/samira/expressions/neutral.webp').read_bytes()
  rectangles={'left_eye':(526,330,86,58),'right_eye':(635,344,70,58),'mouth':(560,440,110,85)}
 else:
  original=png('red')
  rectangles={'left_eye':(5,4,8,7),'right_eye':(20,4,8,7),'mouth':(9,18,14,10)}
 cid=db.add_character_card({'name':'Radio Journey','image':original})
 evidence={'real_image_adapter':images.live,'provider':cfg.default_backend,'source_sha256':hashlib.sha256(original).hexdigest(),'disclosures':[],'pair_views':[]}
 async with app.run_test(size=(180,72)) as pilot:
  async def ready(predicate):
   async with asyncio.timeout(30):
    while not predicate():await asyncio.sleep(.02)
  async def press(view,action):
   button=view.query_one('#portrait-performance-'+action,Button)
   button.focus();await pilot.press('enter');await pilot.pause()
  async def finish(controller):
   await ready(lambda:not controller._busy and controller._task is None)
  async def confirmed(view,controller,action):
   await press(view,action)
   await ready(lambda:isinstance(app.screen,ConfirmationDialog) or not controller._busy)
   assert isinstance(app.screen,ConfirmationDialog),str(view.query_one('#portrait-performance-status').render())
   evidence['disclosures'].append(app.screen.message)
   app.screen.query_one('#confirm-button',Button).focus();await pilot.press('enter')
   return controller._task
  async def edit():
   app.post_message(NavigateToScreen('ccp'))
   await ready(lambda:isinstance(app.screen,PersonasScreen))
   screen=app.screen
   await pilot.pause();await ready(lambda:bool(screen.query('PersonasPreviewPane')))
   await screen._select_character(str(cid),'Radio Journey')
   for worker in tuple(app.workers):
    if worker.group=='ccp-load-character':await worker.wait()
   await pilot.pause()
   await ready(lambda:str(screen.character_handler.current_character_id)==str(cid))
   screen.post_message(EditCharacterRequested(str(cid)))
   await ready(lambda:screen._edit_mode=='edit' and screen._character_editor_is_active())
   for worker in tuple(app.workers):
    if worker.group=='personas-visual-identity-load':await worker.wait()
   await pilot.pause()
   view=screen.query_one(PortraitPerformanceEditor)
   view.query_one(Collapsible).collapsed=False
   controller=screen._portrait_performance()
   await press(view,'prepare');await ready(lambda:controller.draft is not None);await finish(controller)
   return screen,controller,view
  await ready(lambda:app._ui_ready and app._initial_screen_pushed)
  personas,controller,view=await edit()
  await press(view,'regions');await ready(lambda:isinstance(app.screen,PortraitRegionEditor))
  modal=app.screen
  for role,values in rectangles.items():
   modal.query_one('#portrait-regions-role',Select).value=role;await pilot.pause()
   for key,value in zip(('x','y','width','height'),values):modal.query_one('#portrait-region-'+key,Input).value=str(value)
   await pilot.pause()
  assert modal.query_one('#portrait-regions-save',Button).disabled
  modal.query_one('#portrait-regions-neutral',Checkbox).value=True;await pilot.pause()
  modal.query_one('#portrait-regions-save',Button).focus();await pilot.press('enter')
  await ready(lambda:app.screen is personas);await finish(controller)
  settings=db.get_character_portrait_regions(character_id=cid,expression_key='neutral')
  assert settings.revision==1 and controller.draft.staged_roles==('rest',) and not controller.draft.can_apply
  selected_recipe=settings.recipe
  prepared_base=controller._reference.content
  await confirmed(view,controller,'generate');await finish(controller)
  assert len(images.calls)==3 and controller.draft.can_apply
  assert '3 image-edit requests' in evidence['disclosures'][-1]
  assert all(call.reference_image is images.calls[0].reference_image for call in images.calls)
  for component,affected in [('small',{'small','blink_small'}),('eyelids',{'blink_rest','blink_small','blink_open'})]:
   before=controller.draft.frames()
   view.query_one('#portrait-performance-component',Select).value=component;await pilot.pause()
   await confirmed(view,controller,'regenerate');await finish(controller)
   assert all(old.png==new.png for old,new in zip(before,controller.draft.frames()) if old.role not in affected)
  assert len(images.calls)==5
  assert recover_components(controller._reference.content,selected_recipe,controller.draft.frames())
  await press(view,'apply');await finish(controller)
  saved=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
  assert saved.active and saved.composition_recipe==selected_recipe
  if not images.live:
   active=saved;before=controller.draft.composition
   images.fail={len(images.calls)+1}
   await confirmed(view,controller,'generate');await finish(controller)
   assert controller.draft.composition is not before and not controller.draft.can_apply
   assert set(controller.draft.staged_roles)=={'rest','open','blink_rest','blink_open'}
   assert view.query_one('#portrait-performance-apply',Button).disabled and view.query_one('#portrait-performance-sample',Button).disabled
   assert db.get_character_portrait_performance(character_id=cid,expression_key='neutral')==active
   images.fail=set();view.query_one('#portrait-performance-component',Select).value='small';await pilot.pause()
   await confirmed(view,controller,'regenerate');await finish(controller)
   assert controller.draft.can_apply
   before=controller.draft.composition;images.fail={len(images.calls)}
   await confirmed(view,controller,'regenerate');await finish(controller)
   assert controller.draft.composition is before and controller.draft.can_apply
   images.fail=set()
   await press(view,'apply');await finish(controller)
   saved=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
   evidence['fresh_failure_preserved_active']=True
   evidence['failed_retry_preserved_candidate']=True
  before=controller.draft.composition;call_count=len(images.calls)
  await press(view,'review');await ready(lambda:isinstance(app.screen,PortraitCompositionReview))
  review=app.screen
  for region in ('overview','left_eye','right_eye','mouth'):
   review.query_one('#portrait-pairs-region',Select).value=region;await pilot.pause();await review.wait_for_render()
   raster=review._raster;assert raster and raster.width*raster.height<=1048576
   if region!='overview':
    with Image.open(BytesIO(raster.png)) as matrix:
     width,height=raster.width//3,raster.height//2
     for index,frame in enumerate(before.frames):
      with Image.open(BytesIO(frame.png)) as raw:
       actual=matrix.crop(((index%3)*width,(index//3)*height,(index%3+1)*width,(index//3+1)*height))
       expected=raw.crop(raster.crop_box)
       try:assert actual.tobytes()==expected.tobytes()
       finally:actual.close();expected.close()
   evidence['pair_views'].append({'region':region,'crop':raster.crop_box,'sha256':hashlib.sha256(raster.png).hexdigest()})
   if images.live:
    artifacts=Path.home()/'portrait-artifacts';artifacts.mkdir(exist_ok=True)
    (artifacts/('pair-'+region+'.png')).write_bytes(raster.png)
    app.save_screenshot('pair-'+region+'.svg',path=str(artifacts))
  review.query_one('#portrait-pairs-close',Button).focus();await pilot.press('enter');await finish(controller)
  assert controller.draft.composition is before and len(images.calls)==call_count
  # Real offline preview timer, driven by the shipped cycle selector.
  view.query_one('#portrait-performance-preview-mode',Select).value='cycle';await pilot.pause();await finish(controller)
  avatar=view.query_one('#portrait-performance-staged',CharacterExpressionAvatar)
  avatar.scroll_visible(animate=False,force=True)
  await pilot.pause();await ready(lambda:avatar.current_image is not None and avatar._visible())
  roles=set()
  async with asyncio.timeout(10):
   while roles!=set(FRAME_ROLES):
    if avatar._frame_index>=0:roles.add(FRAME_ROLES[avatar._frame_index])
    await asyncio.sleep(.02)
  evidence['offline_roles']=sorted(roles)
  # Real guarded TTS, PCM sink, device callbacks and amplitude observer.
  sample=await confirmed(view,controller,'sample')
  await ready(lambda:controller._speech_task is not None)
  avatar=view.query_one('#portrait-performance-staged',CharacterExpressionAvatar)
  avatar.scroll_visible(animate=False,force=True);await pilot.pause()
  await ready(lambda:avatar.current_image is not None and avatar._visible())
  audio.allow_synthesis()
  observed=False;sample_trace=[]
  while controller._busy:
   snap=controller._presentation.read() if controller._presentation else None
   if snap:
    sample_trace.append((snap.state,snap.mouth,snap.timing_valid,avatar._frame_index,avatar._visible()))
    if snap.state=='playing' and snap.timing_valid and snap.mouth!='rest' and avatar._frame_index>=0 and avatar._frame_index%3!=0:observed=True
   await asyncio.sleep(.02)
  assert observed and audio.synthesized,{'trace':sample_trace[-30:],'streams':len(audio.streams),'syntheses':len(audio.synthesized),'responses':audio.responses,'formats':audio.formats,'status':str(view.query_one('#portrait-performance-status').render())}
  evidence['guarded_sample_pcm_driven']=True
  app.post_message(NavigateToScreen('chat'));await ready(lambda:isinstance(app.screen,ChatScreen))
  chat=app.screen
  key=ResolvedLocalCharacterKey(db.get_local_authority_id(),cid)
  await _start_character_context_chat(chat,key,db,'Radio Journey',lambda:app.chachanotes_db is db)
  if not chat.query_one('#console-rail-section-header-character').open:chat.query_one('#console-rail-section-toggle-character',Button).press()
  await ready(lambda:any(widget.current_image is not None and widget._visible() for widget in chat.query(CharacterExpressionAvatar)))
  avatar=chat.query_one(CharacterExpressionAvatar)
  assert chat._active_character_avatar['performance'].revision==saved.revision
  store=chat._console_chat_store
  message=store.append_message(store.active_session_id,role=ConsoleMessageRole.ASSISTANT,content='Hello. Notice the pauses.')
  await chat._sync_native_console_chat_ui()
  outcomes=[]
  audio.hold_synthesis()
  assert await chat._message.request_console_message_speech(message.id,outcome_callback=outcomes.append)
  audio.allow_synthesis()
  observed=False;console_trace=[]
  async with asyncio.timeout(20):
   while not outcomes:
    channel=chat._message.current_speech_presentation()
    snap=channel.read() if channel else None
    visible=[item for item in chat.query(CharacterExpressionAvatar) if item.id=='console-character-avatar-image' and item.current_image is not None and item._visible()]
    current_avatar=visible[0] if visible else None
    if snap:
     console_trace.append((snap.state,snap.mouth,snap.timing_valid,current_avatar._frame_index if current_avatar else None,chat._character._read_character_speech() is not None,isinstance(current_avatar._prepared,PreparedPerformance) if current_avatar else False,chat._character._current_request()[8]))
     if snap.state=='playing' and snap.timing_valid and snap.mouth!='rest' and current_avatar is not None and isinstance(current_avatar._prepared,PreparedPerformance) and current_avatar._frame_index>=0 and current_avatar._frame_index%3!=0:
      assert chat._active_character_avatar['performance'].frames==saved.frames
      assert len(current_avatar._prepared.frames)==len(saved.frames)
      observed=True
    await asyncio.sleep(.02)
  assert observed and outcomes==[True],{'trace':console_trace[-40:],'outcomes':outcomes,'streams':len(audio.streams),'responses':audio.responses}
  evidence['console_pcm_driven']=True
  store.create_session(title='Replacement context',runtime_backend='local');await chat._sync_native_console_chat_ui()
  assert chat._character._read_character_speech() is None
  personas,reopened,view=await edit()
  assert reopened.draft.expected_revision==saved.revision and reopened.draft.frames()==saved.frames
  assert reopened._regions.recipe==selected_recipe and reopened._regions.revision==settings.revision
  assert reopened.draft.composition and reopened.draft.composition.recipe==selected_recipe
  evidence['reopened_exact_frames']=True
  await press(view,'remove');await finish(reopened)
  assert not db.get_character_portrait_performance(character_id=cid,expression_key='neutral').active
  assert db.get_character_portrait_regions(character_id=cid,expression_key='neutral').recipe==selected_recipe
  assert db.get_character_card_by_id(cid)['image']==original
  evidence['frames']=[frame.sha256 for frame in saved.frames]
  evidence['base_pixel_sha256']=selected_recipe.base_pixel_sha256
  evidence['region_revision']=settings.revision
  if images.live:
   from dataclasses import asdict
   artifacts=Path.home()/'portrait-artifacts';artifacts.mkdir(exist_ok=True)
   (artifacts/'recipe.json').write_text(json.dumps(asdict(selected_recipe),indent=2))
   (artifacts/'prepared-base.png').write_bytes(prepared_base)
   (artifacts/'source-artwork.webp').write_bytes(original)
   for frame in saved.frames:(artifacts/(frame.role+'.png')).write_bytes(frame.png)
  evidence['image_calls']=len(images.calls)
  evidence['image_results']=images.results
 await audio.settle()
 assert not blocked_attempts(),blocked_attempts()
 (Path.home()/'portrait-qualification.json').write_text(json.dumps(evidence,indent=2))
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
        timeout=180,
        installed_package=native_package,
    )
    import json
    import os
    from pathlib import Path

    evidence = (tmp_path / "home" / "portrait-qualification.json").read_text()
    result = json.loads(evidence)
    assert result["guarded_sample_pcm_driven"] and result["console_pcm_driven"]
    assert not result["real_image_adapter"] and result["image_calls"] == 10
    destination = os.environ.get("TLDW_PORTRAIT_EVIDENCE")
    if destination:
        Path(destination).write_text(evidence)
