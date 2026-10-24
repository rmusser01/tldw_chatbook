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
 from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES,capture_performance_source,normalize_performance_frames
 with Image.new('RGB',(16,16),'red') as image,BytesIO() as output:
  image.save(output,format='PNG');portrait=output.getvalue()
 character=app.chachanotes_db.add_character_card({'name':'Restored radio','image':portrait})
 source,_=capture_performance_source(app.chachanotes_db,character_id=character,requested_state='idle')
 frames=normalize_performance_frames({role:portrait for role in FRAME_ROLES})
 revision=app.chachanotes_db.apply_character_portrait_performance(source,frames,expected_revision=0)
 portrait_evidence={'character':character,'revision':revision,'source':source.source_sha256,'frames':[frame.sha256 for frame in frames],'portrait':hashlib.sha256(portrait).hexdigest(),'authority':app.chachanotes_db.get_local_authority_id()}
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
   await chat._character._apply_console_character_choice_async(ConsoleCharacterChoice(evidence['character'],'Restored radio','new'))
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
