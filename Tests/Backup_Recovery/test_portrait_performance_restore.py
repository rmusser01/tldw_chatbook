"""Actual full-profile F9 capture, isolated restore and fresh installed app open."""

from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 -- installed shared fixture
)
from Tests.ProductionApp.test_backup_restore_end_to_end import (
    _FLOW,
    _OPEN,
    _run_f9_journey,
)

_SEED = r"""
 from io import BytesIO
 from PIL import Image
 from dataclasses import asdict
 from tldw_chatbook.Character_Chat.portrait_performance import capture_performance_source
 from tldw_chatbook.Character_Chat.portrait_compositing import PixelRect,PortraitCompositionRecipe,prepared_base_pixel_sha256,compose_component,assemble_components
 with Image.new('RGB',(16,16),'red') as image,BytesIO() as output:
  image.save(output,format='PNG');portrait=output.getvalue()
 character=app.chachanotes_db.add_character_card({'name':'Restored radio','image':portrait})
 source,_=capture_performance_source(app.chachanotes_db,character_id=character,requested_state='idle')
 recipe=PortraitCompositionRecipe(1,16,16,prepared_base_pixel_sha256(portrait),PixelRect(0,0,4,4),PixelRect(10,0,4,4),PixelRect(4,9,7,6),True)
 regions_revision=app.chachanotes_db.save_character_portrait_regions(source,recipe,expected_revision=0)
 components={}
 for role,color in [('eyelids','green'),('small','blue'),('open','white')]:
  with Image.new('RGB',(16,16),color) as image,BytesIO() as output:
   image.save(output,format='PNG');components[role]=compose_component(portrait,recipe,role,output.getvalue())
 frames=assemble_components(portrait,recipe,components)
 revision=app.chachanotes_db.apply_character_portrait_performance(source,frames,expected_revision=0,composition_recipe=recipe,expected_regions_revision=regions_revision)
 portrait_evidence={'character':character,'revision':revision,'source':source.source_sha256,'frames':[frame.sha256 for frame in frames],'portrait':hashlib.sha256(portrait).hexdigest(),'authority':app.chachanotes_db.get_local_authority_id(),'recipe':asdict(recipe),'regions_revision':regions_revision}
"""
_CHECK = r"""
   import hashlib
   evidence=expected['portrait_performance']
   record=app.chachanotes_db.get_character_portrait_performance(character_id=evidence['character'],expression_key='neutral')
   assert record.active and record.revision==evidence['revision']
   assert record.source.source_sha256==evidence['source']
   assert [frame.sha256 for frame in record.frames]==evidence['frames']
   assert hashlib.sha256(app.chachanotes_db.get_character_card_by_id(evidence['character'])['image']).hexdigest()==evidence['portrait']
   assert app.chachanotes_db.get_local_authority_id()==evidence['authority']
   from dataclasses import asdict
   from io import BytesIO
   from PIL import Image
   from tldw_chatbook.Character_Chat.portrait_compositing import recover_components,compose_component,assemble_components
   settings=app.chachanotes_db.get_character_portrait_regions(character_id=evidence['character'],expression_key='neutral')
   assert settings.revision==evidence['regions_revision'] and asdict(settings.recipe)==evidence['recipe']
   assert record.composition_recipe==settings.recipe
   base=app.chachanotes_db.get_character_card_by_id(evidence['character'])['image']
   from dataclasses import replace
   from textual.widgets import Button
   from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
   from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
   from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import EditCharacterRequested
   from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import CharacterPerformanceRequested
   from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
   from Tests.ProductionApp.portrait_qualification_helpers import QualificationImages
   from tldw_chatbook.UI.Persona_Modules import portrait_regions
   images=QualificationImages();images.install()
   async def ready(predicate):
    async with asyncio.timeout(20):
     while not predicate():await asyncio.sleep(.02)
   app.post_message(NavigateToScreen('ccp'));await ready(lambda:isinstance(app.screen,PersonasScreen))
   personas=app.screen
   await pilot.pause();await ready(lambda:bool(personas.query('PersonasPreviewPane')))
   await personas._select_character(str(evidence['character']),'Restored radio')
   personas.post_message(EditCharacterRequested(str(evidence['character'])))
   await ready(lambda:personas._edit_mode=='edit' and personas._character_editor_is_active());await pilot.pause()
   controller=personas._portrait_performance()
   original_reference=portrait_regions.performance_reference
   def reencoded(raw,digest):
    reference=original_reference(raw,digest)
    with Image.open(BytesIO(reference.content)) as image,BytesIO() as output:
     image.save(output,format='PNG',compress_level=0)
     encoded=output.getvalue()
    assert encoded!=reference.content
    return replace(reference,content=encoded)
   portrait_regions.performance_reference=reencoded
   assert await controller.handle(CharacterPerformanceRequested('prepare'))
   assert controller._regions.revision==settings.revision and controller._regions.recipe==settings.recipe
   assert controller.draft.composition and controller.draft.frames()==record.frames
   before=controller.draft.frames()
   retry=personas.run_worker(controller.handle(CharacterPerformanceRequested('regenerate',component='small')))
   await ready(lambda:isinstance(app.screen,ConfirmationDialog))
   app.screen.query_one('#confirm-button',Button).focus();await pilot.press('enter')
   assert await retry.wait()
   assert len(images.calls)==1 and controller.draft.can_apply
   assert all(old.png==new.png for old,new in zip(before,controller.draft.frames()) if old.role not in {'small','blink_small'})
   assert app.chachanotes_db.get_character_portrait_performance(character_id=evidence['character'],expression_key='neutral')==record
   assert await controller.handle(CharacterPerformanceRequested('apply'))
   current=app.chachanotes_db.get_character_portrait_performance(character_id=evidence['character'],expression_key='neutral')
   assert current.revision==record.revision+1 and current.composition_recipe==settings.recipe
   assert recover_components(controller._reference.content,settings.recipe,current.frames)
   def changed_pixel(raw,digest):
    reference=reencoded(raw,digest)
    with Image.open(BytesIO(reference.content)) as image,BytesIO() as output:
     r,g,b,a=image.getpixel((0,0));image.putpixel((0,0),(r^1,g,b,a));image.save(output,format='PNG');encoded=output.getvalue()
    return replace(reference,content=encoded)
   portrait_regions.performance_reference=changed_pixel
   controller.discard();assert await controller.handle(CharacterPerformanceRequested('prepare'))
   assert controller._regions is None and controller.draft.composition is None
   assert not await controller.handle(CharacterPerformanceRequested('regenerate',component='small'))
   assert len(images.calls)==1
   portrait_regions.performance_reference=original_reference
   with Image.open(BytesIO(base)) as image,BytesIO() as output:
    image.save(output,format='PNG',compress_level=0);new_source=output.getvalue()
   card=app.chachanotes_db.get_character_card_by_id(evidence['character'])
   assert app.chachanotes_db.update_character_card(evidence['character'],{'image':new_source},card['version'])
   controller.discard();assert await controller.handle(CharacterPerformanceRequested('prepare'))
   assert controller._regions is None and not controller.draft.can_apply
   card=app.chachanotes_db.get_character_card_by_id(evidence['character'])
   assert app.chachanotes_db.update_character_card(evidence['character'],{'image':base},card['version'])
   controller.discard();assert await controller.handle(CharacterPerformanceRequested('prepare'))
   assert controller.draft.composition and controller._regions.revision==settings.revision
   assert app.chachanotes_db.get_character_card_by_id(evidence['character'])['image']==base
   (fixture/'portrait-restored-edit.json').write_text(json.dumps({'native_restore':True,'one_request_retry':len(images.calls),'equivalent_encoding_reused':True,'pixel_change_rejected':True,'source_change_rejected':True,'original_restored':True,'region_revision':settings.revision,'performance_revision':current.revision,'frames':[f.sha256 for f in current.frames]}))
   # The UUID is durable; exact DB-object scope still rejects a retired owner.
   from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
   from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
   from tldw_chatbook.Widgets.Console.console_character_picker_modal import ConsoleCharacterChoice
   from tldw_chatbook.Audio.speech_presentation import SpeechPresentation,SpeechPresentationOwner
   app.post_message(NavigateToScreen('chat'))
   async with asyncio.timeout(20):
    while not isinstance(app.screen,ChatScreen):await asyncio.sleep(.02)
   await pilot.pause()
   chat=app.screen
   from tldw_chatbook.Character_Chat.character_conversation_navigation import ResolvedLocalCharacterKey
   from tldw_chatbook.UI.Console_Modules.wiring import _start_character_context_chat
   key=ResolvedLocalCharacterKey(app.chachanotes_db.get_local_authority_id(),evidence['character'])
   await _start_character_context_chat(chat,key,app.chachanotes_db,'Restored radio',lambda:app.chachanotes_db is personas.app_instance.chachanotes_db)
   store=chat._console_chat_store
   retired=SpeechPresentation(SpeechPresentationOwner('retired',object(),store.active_session_id,store.active_session_epoch(),evidence['character'],None,None))
   chat._message.bind_speech_presentation(retired,lambda:True)
   retired.set_transport('playing')
   assert chat._character._read_character_speech() is None
"""


def test_native_full_profile_preserves_six_frames_on_fresh_open(
    tmp_path, native_package
):
    seed_at = (
        " note=app.chachanotes_db.add_note('UI captured note','Captured through F9')"
    )
    assert _FLOW.count(seed_at) == 1
    flow = _FLOW.replace(seed_at, _SEED + "\n" + seed_at)
    evidence_at = "  (fixture/'ui-installed.json').write_text(json.dumps(installed))"
    assert flow.count(evidence_at) == 1
    flow = flow.replace(
        evidence_at,
        "  installed['portrait_performance']=portrait_evidence\n" + evidence_at,
    )
    check_at = "   assert app._initial_screen_pushed and app._ui_ready"
    assert _OPEN.count(check_at) == 1
    child = _OPEN.replace(check_at, check_at + "\n" + _CHECK).replace(
        "async with app.run_test(size=(100,36)):",
        "async with app.run_test(size=(100,36)) as pilot:",
    )
    _run_f9_journey(
        tmp_path, False, False, False, native_package, flow=flow, open_child=child
    )
    import json
    import os
    from pathlib import Path

    report = (tmp_path / "portrait-restored-edit.json").read_text()
    assert json.loads(report)["one_request_retry"] == 1
    destination = os.environ.get("TLDW_PORTRAIT_RESTORE_EVIDENCE")
    if destination:
        Path(destination).write_text(report)
