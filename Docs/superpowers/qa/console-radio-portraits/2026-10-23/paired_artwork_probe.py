"""Opt-in actual-app qualification of the authorized locally composited matrix."""

import runpy
import tempfile
from pathlib import Path

from Tests.Backup_Recovery.test_home_citation_retirement import _run

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Reuse the guarded artwork journey without changing its historical evidence."""
    repo = HERE.parents[4]
    assets = repo / "output/imagegen/console-radio-samira-2026-10-23/paired-v2"
    source = repo / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
    script = runpy.run_path(str(HERE / "imagegen_import_probe.py"))["CHILD"]
    changes = {
        " for frame in manifest['frames']:\n  assert hashlib.sha256((root/frame['file']).read_bytes()).hexdigest()==frame['sha256']\n  assert hashlib.sha256((root/frame['prepared']['file']).read_bytes()).hexdigest()==frame['prepared']['sha256']": " for role,frame in manifest['frames'].items():\n  assert hashlib.sha256((root/frame['file']).read_bytes()).hexdigest()==frame['sha256']\n assert set(manifest['frames'])==set(FRAME_ROLES)\n assert all(manifest['checks'].values())",
        "   screen.post_message(EditCharacterRequested(str(cid)))": "   await pilot.pause()\n   screen.post_message(EditCharacterRequested(str(cid)))",
        "path=str(root/'ready'/(role+'.png'))": "path=str(root.parent/'selected'/'blink_small.png' if role=='blink_small' else root/'selected'/(role+'.png'))",
        "path=str(root/'ready'/'blink_small_v2.png')": "path=str(root/'selected'/'blink_small.png')",
        "  result['replacement_preserved_other_five']=True": "  result['replacement_preserved_other_five']=True\n  from tldw_chatbook.Character_Chat.portrait_performance import normalize_performance_frames\n  expected=normalize_performance_frames({role:(root/'selected'/(role+'.png')).read_bytes() for role in FRAME_ROLES})\n  assert controller.draft.frames()==expected\n  result['final_frames_match_corrected_matrix']=True",
        "  await chat._character._apply_console_character_choice_async(ConsoleCharacterChoice(cid,'Samira — artwork qualification','new'))": "  from tldw_chatbook.Character_Chat.character_conversation_navigation import ResolvedLocalCharacterKey\n  from tldw_chatbook.UI.Console_Modules.wiring import _start_character_context_chat\n  key=ResolvedLocalCharacterKey(db.get_local_authority_id(),cid)\n  await _start_character_context_chat(chat,key,db,'Samira — artwork qualification',lambda:app.chachanotes_db is db)",
    }
    for before, after in changes.items():
        assert script.count(before) == 1, before
        script = script.replace(before, after)
    script = script.replace("__ASSET_ROOT__", repr(str(assets))).replace(
        "__SOURCE__", repr(str(source))
    )
    (assets / "preview").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="radio-paired-art-") as folder:
        try:
            _run(Path(folder), "paired", "journey", script=script, timeout=180)
        finally:
            log = Path(folder) / "restart-output.log"
            if log.is_file():
                (assets / "preview/probe.log").write_text(log.read_text())
    print("PASS: corrected matrix imported/replaced/previewed/applied/reopened/removed")


if __name__ == "__main__":
    main()
