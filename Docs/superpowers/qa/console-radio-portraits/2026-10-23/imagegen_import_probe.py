"""Opt-in generated-artwork journey; requires locally saved qualification assets."""

import tempfile
from pathlib import Path

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.ProductionApp.test_backup_restore_composition import _ENTRY

CHILD = (
    _ENTRY.split("async def main():", 1)[0]
    + r"""
import hashlib,json
from textual.widgets import Button,Collapsible
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import EditCharacterRequested
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import CharacterPerformanceRequested,PortraitPerformanceEditor
from tldw_chatbook.Widgets.Console.character_expression_avatar import CharacterExpressionAvatar
from tldw_chatbook.Widgets.Console.console_character_picker_modal import ConsoleCharacterChoice
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES

async def main():
 root=Path(__ASSET_ROOT__)
 output=root/'preview';output.mkdir(exist_ok=True)
 original=Path(__SOURCE__).read_bytes()
 manifest=json.loads((root/'manifest.json').read_text())
 for frame in manifest['frames']:
  assert hashlib.sha256((root/frame['file']).read_bytes()).hexdigest()==frame['sha256']
  assert hashlib.sha256((root/frame['prepared']['file']).read_bytes()).hexdigest()==frame['prepared']['sha256']
 app=TldwCli()
 async def offline_discovery(_config):return ()
 app.console_local_server_discovery=offline_discovery
 app.app_config['_first_run']=False
 app.app_config.setdefault('first_run',{})['setup_completed']=True
 # This private fixture represents an existing profile; no provider calls are made.
 app.app_config.setdefault('console',{}).setdefault('onboarding',{})['first_send_completed']=True
 app.no_color=False
 app.console.no_color=False
 db=app.chachanotes_db
 cid=db.add_character_card({'name':'Samira — artwork qualification','image':original})
 result={'synthetic_audio':False,'live_audio':False,'source_sha256':hashlib.sha256(original).hexdigest(),'roles':[]}
 async with app.run_test(size=(180,72)) as pilot:
  async def ready(predicate):
   try:
    async with asyncio.timeout(30):
     while not predicate():await asyncio.sleep(.02)
   except TimeoutError:
    app.save_screenshot('timeout.svg',path=str(output))
    print('AVATAR_DIAGNOSTIC',[(w._box,w._frame_index,w._preview_mode,w._visible(),type(w._prepared).__name__,w._animate,w._performance_disabled,str(w.tooltip),str(w.region)) for w in app.screen.query(CharacterExpressionAvatar)],flush=True)
    raise
  async def edit():
   app.post_message(NavigateToScreen('ccp'))
   await ready(lambda:isinstance(app.screen,PersonasScreen))
   screen=app.screen
   await pilot.pause()
   await ready(lambda:bool(screen.query('PersonasPreviewPane')))
   await screen._select_character(str(cid),'Samira — artwork qualification')
   screen.post_message(EditCharacterRequested(str(cid)))
   await ready(lambda:screen._edit_mode=='edit' and screen._character_editor_is_active())
   await pilot.pause()
   controller=screen._portrait_performance()
   assert await controller.handle(CharacterPerformanceRequested('prepare'))
   return screen,controller
  await ready(lambda:app._ui_ready and app._initial_screen_pushed)
  await pilot.pause()
  app._filters=[f for f in app._filters if type(f).__name__ not in {'Monochrome','NoColor'}]
  print('checkpoint: editor',flush=True)
  personas,controller=await edit()
  view=personas.query_one(PortraitPerformanceEditor)
  view.query_one(Collapsible).collapsed=False
  for role in FRAME_ROLES:
   assert await controller.handle(CharacterPerformanceRequested('import',role=role),path=str(root/'ready'/(role+'.png')))
  print('checkpoint: six imported',flush=True)
  before={f.role:f.sha256 for f in controller.draft.frames()}
  assert await controller.handle(CharacterPerformanceRequested('import',role='blink_small'),path=str(root/'ready'/'blink_small_v2.png'))
  after={f.role:f.sha256 for f in controller.draft.frames()}
  assert before['blink_small']!=after['blink_small']
  assert all(before[r]==after[r] for r in FRAME_ROLES if r!='blink_small')
  result['replacement_preserved_other_five']=True
  view.query_one('#portrait-performance-preview').scroll_visible(immediate=True)
  await ready(lambda:bool(view.query('#portrait-performance-staged')) and view.query_one('#portrait-performance-staged',CharacterExpressionAvatar).current_image is not None)
  await pilot.pause()
  app.save_screenshot('editor.svg',path=str(output))
  assert not view.query_one('#portrait-performance-apply',Button).disabled
  assert await controller.handle(CharacterPerformanceRequested('apply'))
  print('checkpoint: applied',flush=True)
  saved=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
  assert saved.active and saved.revision==1
  app.post_message(NavigateToScreen('chat'))
  await ready(lambda:isinstance(app.screen,ChatScreen))
  chat=app.screen
  await chat._character._apply_console_character_choice_async(ConsoleCharacterChoice(cid,'Samira — artwork qualification','new'))
  if not chat.query_one('#console-rail-section-header-character').open:
   chat.query_one('#console-rail-section-toggle-character',Button).press()
  await ready(lambda:any(w.current_image is not None and w._visible() for w in chat.query(CharacterExpressionAvatar)))
  await pilot.pause()
  print('checkpoint: console fitted',flush=True)
  for index,role in enumerate(FRAME_ROLES):
   print('checkpoint: pose '+role,flush=True)
   avatar=chat.query_one(CharacterExpressionAvatar)
   avatar.set_preview_mode(role)
   await ready(lambda:avatar._frame_index==index and avatar.current_image is not None and not avatar._painting)
   await pilot.pause(.1)
   assert chat.query_one(CharacterExpressionAvatar) is avatar
   avatar.current_image.save(output/(role+'-fitted.png'))
   app.save_screenshot(role+'.svg',path=str(output))
   result['roles'].append({'role':role,'box':list(avatar._box),'pixels':list(avatar.current_image.size),'sha256':hashlib.sha256(avatar.current_image.tobytes()).hexdigest()})
  assert len({r['sha256'] for r in result['roles']})==6
  avatar.set_preview_mode('idle')
  await pilot.pause(.4)
  assert avatar._timer is not None and not avatar._performance_disabled
  personas,reopened=await edit()
  assert reopened.draft.expected_revision==saved.revision
  assert reopened.draft.source==saved.source and reopened.draft.frames()==saved.frames
  assert await reopened.handle(CharacterPerformanceRequested('remove'))
  removed=db.get_character_portrait_performance(character_id=cid,expression_key='neutral')
  assert not removed.active and removed.revision>saved.revision
  assert db.get_character_card_by_id(cid)['image']==original
  result.update(reopened_matches=True,removed=True,original_unchanged=True,applied_revision=saved.revision,removed_revision=removed.revision)
 assert not blocked_attempts(),blocked_attempts()
 (output/'observations.json').write_text(json.dumps(result,indent=2)+'\n')
 print('retired and reopened')
asyncio.run(main())
"""
)


def main() -> None:
    """Run the actual app in the existing owned private-profile harness."""
    repo = Path(__file__).resolve().parents[5]
    assets = repo / "output/imagegen/console-radio-samira-2026-10-23"
    source = repo / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
    assert (assets / "ready/blink_small_v2.png").is_file()
    script = CHILD.replace("__ASSET_ROOT__", repr(str(assets))).replace(
        "__SOURCE__", repr(str(source))
    )
    (assets / "preview").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="radio-generated-art-") as folder:
        try:
            _run(Path(folder), "artwork", "journey", script=script, timeout=180)
        finally:
            log = Path(folder) / "restart-output.log"
            if log.is_file():
                (assets / "preview/probe.log").write_text(log.read_text())
    print(
        "PASS: generated-artwork mounted import/replacement/preview/apply/reopen/remove"
    )


if __name__ == "__main__":
    main()
