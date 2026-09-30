"""The Conversations tray keeps a focused row focused across its own rebuild.

TASK-33621.12. Closing the row menu's Save .md… prompt re-synced the Console's
Conversations tray, whose recompose removed the focused row control; Textual
then moved focus to whatever preceded it -- the section toggle, or, live, the
Console header's Settings control, where the next Enter opened Settings. The
tray opts in to ``RecomposeCaptureGuard.RECOMPOSE_KEEPS_FOCUS`` so focus goes
back to the rebuilt row.

The first version of that opt-in pulled focus back to the row even when code
had moved focus away on purpose during the rebuild. Selecting a chat in the
rail focuses the composer while the tray is still rebuilding, so focus landed
on the row again and the next keystrokes went to it -- live, a typed "m"
opened the row's action menu (2026-09-30 review). Everything here drives the
real Console: the real tray, Textual's own focus reset, the real rail.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

_ROW_CONTROL_ID = "console-conversation-actions-0"


async def _wait_until(pilot, predicate, *, timeout: float = 5.0) -> bool:
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


def _describe(widget) -> str:
    return "None" if widget is None else f"{type(widget).__name__}#{widget.id}"


def test_the_conversations_tray_opts_in_to_keeping_focus() -> None:
    """The opt-in is what rescues focus when the reset lands outside the rail."""
    assert ConsoleWorkspaceContextTray.RECOMPOSE_KEEPS_FOCUS is True


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_lands", ["in-rail", "outside-rail"])
@private_profile_test
async def test_a_tray_rebuild_returns_focus_to_the_rebuilt_row(
    request, reset_lands
) -> None:
    """A rebuild that removes the focused row control puts focus on its twin.

    ``in-rail``: Textual's reset lands on the Conversations section toggle,
    where the rail's own focus recovery could also rescue it.
    ``outside-rail``: the rail's controls in front of the tray are taken out
    of the focus chain, so the reset lands on a Console header control. The
    rail's recovery gives up once focus is outside the rail, so only the
    tray's opt-in can bring focus back -- the live header-Settings case.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await pilot.pause(0.5)
        tray = screen.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        rail = screen.query_one("#console-left-rail")
        if reset_lands == "outside-rail":
            rail.can_focus = False
            for control in rail.query(Button):
                if tray not in control.ancestors:
                    control.can_focus = False
        row = screen.query_one(f"#{_ROW_CONTROL_ID}", Button)
        row.focus()
        await pilot.pause()
        assert pilot.app.focused is row

        tray.refresh(recompose=True)
        assert await _wait_until(
            pilot, lambda: not row.is_attached and not tray.recompose_in_flight
        ), "the tray never rebuilt the row"
        await pilot.pause(0.3)

        focused = pilot.app.focused
        assert focused is not None and focused.id == _ROW_CONTROL_ID, (
            f"focus was stranded on {_describe(focused)}"
        )
        assert focused is not row and focused.is_attached
        assert tray in focused.ancestors


@pytest.mark.asyncio
@private_profile_test
async def test_a_focus_move_queued_during_a_tray_rebuild_wins(request) -> None:
    """The deterministic form of the row-selection journey below.

    The row is focused and the tray starts rebuilding; while it is torn down
    the Console asks for the composer exactly as a row selection does. That
    request is queued (``Widget.focus`` defers through ``call_later``), so it
    may land before or after the rebuild finishes -- either way it is the
    newer, deliberate move and must win over the tray's restore.
    """
    import asyncio

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await pilot.pause(0.5)
        tray = screen.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        composer = screen.query_one("#console-native-composer")
        row = screen.query_one(f"#{_ROW_CONTROL_ID}", Button)
        row.focus()
        await pilot.pause()
        assert pilot.app.focused is row

        tray.refresh(recompose=True)
        # Plain sleeps: `pilot.pause()` waits for the screen to go idle,
        # which is only after the rebuild has finished.
        for _ in range(400):
            if tray.recompose_in_flight:
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("the tray never started rebuilding")
        screen._focus_console_composer_if_needed(force=True)
        assert await _wait_until(
            pilot, lambda: not row.is_attached and not tray.recompose_in_flight
        ), "the tray never finished rebuilding"
        await pilot.pause(0.5)

        focused = pilot.app.focused
        assert focused is not None and (
            focused is composer or composer in focused.ancestors
        ), f"the rebuild took focus back to {_describe(focused)}"


async def _open_second_tab(screen, store, pilot) -> None:
    previous = store.active_session_id
    screen.query_one("#console-new-chat-tab", Button).press()
    assert await _wait_until(
        pilot, lambda: store.active_session_id not in (None, previous)
    ), "no second tab opened"
    await pilot.pause(0.8)


def _rail_row_for(screen, session_id: str) -> Button:
    for button in screen.query(Button):
        if str(button.id or "").startswith("console-workspace-conversation-") and (
            getattr(button, "conversation_id", None) == f"native:{session_id}"
        ):
            return button
    raise AssertionError(f"no rail row for session {session_id}")


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["click", "enter"])
@private_profile_test
async def test_choosing_a_rail_chat_leaves_typing_in_the_composer(request, how) -> None:
    """Selecting a chat focuses the composer mid-rebuild; the tray keeps out.

    Before the fix, a click on the row left focus on the rebuilt row in 3 of
    3 runs, and the next key went to the row instead of the draft ("m" opens
    the row's action menu). The key is pressed as real input.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        store = screen._ensure_console_chat_store()
        first_id = store.ensure_session().id
        await _open_second_tab(screen, store, pilot)
        tray = screen.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        row = _rail_row_for(screen, first_id)
        if how == "enter":
            row.focus()
            await pilot.pause()
            assert pilot.app.focused is row
            await pilot.press("enter")
        else:
            row.scroll_visible(animate=False)
            await pilot.pause(0.2)
            assert await pilot.click(f"#{row.id}")
        assert await _wait_until(pilot, lambda: store.active_session_id == first_id), (
            "choosing the row did not switch to its chat"
        )
        # The selection's tray rebuild can start seconds later (traced: about
        # 2 s after the click, with a second rebuild right behind it), and the
        # composer's focus arrives in the middle of it. Wait for the rebuild
        # that replaces the chosen row, then for every follow-up to settle.
        assert await _wait_until(
            pilot,
            lambda: not row.is_attached and not tray.recompose_in_flight,
            timeout=10.0,
        ), "the selection never rebuilt the Conversations tray"
        await pilot.pause(1.5)

        composer = screen.query_one("#console-native-composer")
        focused = pilot.app.focused
        assert focused is not None and (
            focused is composer or composer in focused.ancestors
        ), f"focus after choosing the chat: {_describe(focused)}"

        await pilot.press("m")
        await pilot.pause(0.3)
        assert composer.draft_text().endswith("m"), (
            f"the key went elsewhere; draft={composer.draft_text()!r}, "
            f"focus={_describe(pilot.app.focused)}"
        )


def _chats_state(order: tuple[str, ...]):
    """A context snapshot whose Chats list shows ``order``, newest first."""
    from dataclasses import replace

    from Tests.UI.test_console_rail_reconciliation import _workspace_state
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
        build_console_conversation_browser_state,
    )

    rows = tuple(
        ConsoleConversationBrowserInputRow(
            row_key=key,
            conversation_id=key,
            native_session_id=None,
            title=f"Chat {key}",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
            # Newest first: the first key in `order` gets the latest stamp.
            updated_sort=f"2026-09-{30 - position:02d}T00:00:00",
        )
        for position, key in enumerate(order)
    )
    return replace(
        _workspace_state(),
        conversation_browser=build_console_conversation_browser_state(
            rows=rows,
            active_workspace_id=None,
            group_collapse_preferences={"section:chats": False},
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("restorer", ["tray", "rail"])
@pytest.mark.parametrize(
    "control_prefix",
    ["console-workspace-conversation-", "console-conversation-actions-"],
)
async def test_a_rebuild_that_reorders_chats_keeps_focus_on_the_same_chat(
    control_prefix, restorer, monkeypatch
) -> None:
    """Qodo #2932: row ids are positional, so an id alone names another chat.

    A tray rebuild that reorders the list -- a star, or another chat getting
    newer -- used to put focus on whichever chat now held the focused
    control's index, so the next Enter or ``m`` acted on the wrong chat.
    Focus must follow the chat, onto the same kind of control. Drives the
    real rail and the real tray. ``tray``: the tray's own restore acts first.
    ``rail``: with the tray's opt-in off, only the rail's focus recovery can
    put focus back -- it resolves the same identity and must agree.
    """
    from Tests.UI.test_console_rail_reconciliation import _RailHarness, _settle
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    if restorer == "rail":
        monkeypatch.setattr(
            ConsoleWorkspaceContextTray, "RECOMPOSE_KEEPS_FOCUS", False
        )
    app = _RailHarness(workspace_state=_chats_state(("c0", "c1", "c2")))
    async with app.run_test(size=(60, 40)) as pilot:
        await _settle(pilot)
        rail = app.query_one(ConsoleLeftRail)
        tray = app.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        control = tray.query_one(f"#{control_prefix}1", Button)
        assert control.row_key == "c1"
        control.focus()
        await pilot.pause()
        assert app.focused is control

        # c2 becomes the newest chat: c1 moves from index 1 to index 2, and
        # index 1 now belongs to c0.
        rail.sync_workspace_context(_chats_state(("c2", "c0", "c1")))
        assert await _wait_until(
            pilot, lambda: not control.is_attached and not tray.recompose_in_flight
        ), "the tray never rebuilt the rows"
        await pilot.pause(0.3)
        await _settle(pilot)

        assert tray.query_one(f"#{control_prefix}2", Button).row_key == "c1"
        focused = app.focused
        assert focused is not None and focused.is_attached, _describe(focused)
        assert str(focused.id or "").startswith(control_prefix), (
            f"focus moved to another kind of control: {_describe(focused)}"
        )
        assert getattr(focused, "row_key", None) == "c1", (
            f"focus followed the position, not the chat: {_describe(focused)} "
            f"is chat {getattr(focused, 'row_key', None)!r}"
        )
