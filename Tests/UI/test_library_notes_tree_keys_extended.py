"""F6 in the note body leaves the editor without selecting text (TASK-34000.28, N-32).

The real ``TldwCli``: the app's own non-priority F6 binding (ADR-031,
untouched), the ``TextualAppGuards`` (``DetachedFocusGuard`` first) and
Textual's ``TextArea`` are all in the path. LibraryScreen declares no ``f6``
of its own (dev's pin ``test_library_keeps_f6_owned_by_the_app_global_
binding``); instead the note body releases ``TextArea``'s own f6 ->
``select_line`` binding (``Widgets/text_area_pane_key.release_pane_key``,
fix round 1, I-1), so the key travels up to the app-global binding.

One real-app boot (about 7 s locally, 2026-10-24). It sits outside the PR
lane census because ``test_library_notes_tree_keys.py`` -- the lane file for
this task -- was at the 30 s budget with its three ``LibraryScreen`` boots;
the full ``Tests/UI`` run covers it.
"""

from __future__ import annotations

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _no_autosave,
    _open_note,
    _settings_without_splash,
)
from Tests.UI.test_library_notes_tree_keys import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
    _wait,
    pytestmark,  # noqa: F401 - asyncio + bootstrap_profile, as the lane file
)


async def test_f6_from_the_note_body_moves_focus_without_selecting_text(
    tmp_path, monkeypatch
) -> None:
    """AC#9: F6 in the body leaves the editor; nothing is selected or changed.

    The real ``TldwCli``: the app's own non-priority F6 binding (ADR-031,
    untouched) is what moves focus, through the app-wide guards. Base:
    Textual's ``TextArea`` f6 binding selects line 2 and focus stays on the
    body.
    """
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _p: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            # Reach the body with real Tabs, never a programmatic focus().
            for _ in range(12):
                if screen.focused is body:
                    break
                await pilot.press("tab")
                await pilot.pause()
            assert screen.focused is body, screen.focused
            await pilot.press("down")
            await pilot.pause()
            assert body.text == _BODY
            assert body.selection.start == body.selection.end
            work_pane = screen.query_one("#library-note-work-pane")

            await pilot.press("f6")
            await _wait(
                pilot,
                lambda: screen.focused is not body,
                lambda: f"F6 to leave the body (selection={body.selection!r})",
            )
            assert body.selection.start == body.selection.end, body.selection
            assert body.text == _BODY
            assert screen.focused is not None
            assert work_pane not in screen.focused.ancestors, screen.focused
            assert not screen._library_note_session.snapshot.dirty
            assert profile.note(profile.note_id)["content"] == _BODY

            await pilot.press("shift+f6")
            await _wait(
                pilot,
                lambda: screen.focused is not None
                and work_pane in screen.focused.ancestors,
                lambda: f"Shift+F6 to return to the note pane (focused={screen.focused!r})",
            )
            assert body.text == _BODY
            assert events == []
