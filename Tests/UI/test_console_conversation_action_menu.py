"""The conversation row action menu, driven through the real Console.

TASK-23200. The rail's conversation rows carried a star button that shipped
disabled on a fresh install, stretched to the full height of a multi-line row,
and was explained by "Local stars unavailable" printed beside it. This suite
pins the replacement: a one-row asterisk that opens an anchored, keyboard
operable menu.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)


def _opener(screen) -> Button:
    return screen.query_one("#console-conversation-actions-0", Button)


@pytest.mark.asyncio
@private_profile_test
async def test_row_carries_one_right_conversation_icon(request) -> None:
    """The control must not reserve the row's whole height any more."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        opener = _opener(screen)

        assert str(opener.label).strip() == "💬"
        assert opener.disabled is False
        assert opener.region.height == 1, (
            "the action opener is still reserving full row height"
        )
        assert not screen.query(".console-conversation-star"), (
            "the retired star control is still being composed"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_local_stars_unavailable_jargon_is_gone(request) -> None:
    """The developer-facing line must not appear in the rail at all."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        rail = screen.query_one("#console-left-rail")
        text = " ".join(
            str(getattr(widget, "renderable", ""))
            for widget in rail.query("*")
            if widget.display
        )
        assert "Local stars unavailable" not in text
        assert not screen.query("#console-conversation-browser-marks-unavailable")


@pytest.mark.asyncio
@private_profile_test
async def test_asterisk_opens_the_menu_with_the_expected_entries(request) -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        labels = [str(button.label).strip() for button in menu.query(Button)]
        assert labels == [
            "Favourite",
            "Mark as unread",
            "Change status ▸",
            "Archive",
            "Rename…",
            "Copy as ▸",
            "Icon and colour…",
            "More ▸",
        ]


@pytest.mark.asyncio
@private_profile_test
async def test_every_disabled_entry_states_its_precondition(request) -> None:
    """A greyed control with no explanation is the defect being removed."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        for button in menu.query(Button):
            if button.disabled:
                assert button.tooltip, f"{button.id} is disabled with no stated reason"


@pytest.mark.asyncio
@private_profile_test
async def test_more_opens_delete_and_back_returns(request) -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        more = next(
            button
            for button in menu.query(Button)
            if getattr(button, "console_action_id", "") == "page:more"
        )
        more.press()
        await pilot.pause(0.5)
        assert menu.page == "more"
        assert [
            getattr(button, "console_action_id", "") for button in menu.query(Button)
        ] == ["page:root", "delete"]

        back = next(iter(menu.query(Button)))
        back.press()
        await pilot.pause(0.5)
        assert menu.page == "root"


@pytest.mark.asyncio
@private_profile_test
async def test_escape_steps_out_of_a_submenu_before_closing(request) -> None:
    """Escape in a submenu returns to the root rather than dropping the row."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        next(
            button
            for button in menu.query(Button)
            if getattr(button, "console_action_id", "") == "page:more"
        ).press()
        await pilot.pause(0.5)
        assert menu.page == "more"

        await pilot.press("escape")
        await pilot.pause(0.5)
        assert menu.page == "root", "escape closed the menu instead of stepping back"
        assert screen.query(ConsoleConversationActionMenu)

        await pilot.press("escape")
        await pilot.pause(0.5)
        assert not screen.query(ConsoleConversationActionMenu), (
            "escape at the root did not close the menu"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_menu_focuses_its_first_actionable_entry_on_open(request) -> None:
    """Keyboard users must land on something they can actually choose."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.4)

        menu = screen.query_one(ConsoleConversationActionMenu)
        focused = pilot.app.focused
        assert focused is not None
        assert focused in list(menu.query(Button))
        assert not focused.disabled


@pytest.mark.asyncio
@private_profile_test
async def test_click_outside_closes_the_menu_without_dispatching(
    request, monkeypatch
) -> None:
    """ADR-068 dismiss contract: a click elsewhere folds the menu, no actions.

    Clicking the composer is the canonical stranding path: Textual moves
    focus to the clicked widget before the press bubbles to the screen, so
    the dismissal must also leave focus exactly where the click put it.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        assert screen.query(ConsoleConversationActionMenu)

        dispatched: list[object] = []
        monkeypatch.setattr(
            screen,
            "on_conversation_action_chosen",
            lambda event: dispatched.append(event),
        )

        assert await pilot.click("#console-native-composer")
        await pilot.pause(0.3)

        assert not screen.query(ConsoleConversationActionMenu), (
            "a click outside the menu left it open"
        )
        assert dispatched == [], "an outside click dispatched a menu action"
        assert pilot.app.focused is not None
        assert pilot.app.focused is not _opener(screen), (
            "outside-click dismissal pulled focus back to the opener"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_click_on_menu_chrome_keeps_the_menu_open(request) -> None:
    """A click on the menu's border must not fold it mid-inspection.

    Targets the top border row (offset y=0) -- menu chrome, not a button --
    through the same screen-level mouse path a real terminal press takes.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        await pilot.click(ConsoleConversationActionMenu, offset=(2, 0))
        await pilot.pause(0.3)

        assert screen.query_one(ConsoleConversationActionMenu), (
            "a click on the menu itself dismissed it"
        )
        assert menu.page == "root"


@pytest.mark.asyncio
@private_profile_test
async def test_escape_with_focus_outside_the_menu_closes_it(request) -> None:
    """Escape must reach a stranded menu even after focus moved elsewhere.

    Focus is moved to the composer without a mouse press (the screen seam
    directly), which is the state a user reaches via keyboard pane cycling
    once click-outside dismissal exists.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        assert screen.query(ConsoleConversationActionMenu)

        composer = screen.query_one("#console-native-composer")
        screen.set_focus(composer)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause(0.3)

        assert not screen.query(ConsoleConversationActionMenu), (
            "escape from outside the menu left it stranded"
        )
        assert pilot.app.focused is composer, (
            "escape-from-elsewhere moved focus instead of only closing the menu"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_pressing_the_asterisk_again_replaces_rather_than_stacks(request) -> None:
    """The opener's press path still ends with exactly one menu mounted."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        await pilot.click("#console-conversation-actions-0")
        await pilot.pause(0.3)

        mounted = screen.query(ConsoleConversationActionMenu)
        assert len(mounted) == 1, f"expected one replaced menu, found {len(mounted)}"


@pytest.mark.unit
def test_menu_width_constant_and_stylesheet_cannot_drift() -> None:
    """The two encodings of the menu's width must agree.

    Qodo review, PR #2233: anchoring clamps against `MENU_WIDTH` while
    rendering uses the CSS `width`, so if one changes alone the menu is
    positioned for a size it is not drawn at.

    Qodo's suggested fix -- interpolate the constant into the stylesheet --
    is not available here: `css/build_css.py` lifts `BUNDLED_CSS` into the
    built stylesheet statically and rejects anything that is not a plain
    string literal, so an f-string breaks the CSS bundle build outright
    (observed: "BUNDLED_CSS is not a plain string literal"). Pinning them
    together in a test gives the same protection within that constraint.
    """
    import re

    from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
        ConsoleConversationActionMenu,
    )

    declared = re.search(
        r"ConsoleConversationActionMenu\s*\{[^}]*?\bwidth:\s*(\d+)\s*;",
        ConsoleConversationActionMenu.BUNDLED_CSS,
        re.S,
    )
    assert declared, "the menu stylesheet no longer declares an explicit width"
    assert int(declared.group(1)) == ConsoleConversationActionMenu.MENU_WIDTH, (
        f"stylesheet width {declared.group(1)} != MENU_WIDTH "
        f"{ConsoleConversationActionMenu.MENU_WIDTH}; anchoring and rendering "
        "have drifted apart"
    )


# ---- Copy as markdown (TASK-25886) ---------------------------------------


def _copy_target(**overrides):
    from tldw_chatbook.Chat.console_conversation_actions import (
        ConversationMenuTarget,
    )

    base = {
        "conversation_id": "conv-copy",
        "title": "Copyable chat",
        "has_messages": True,
    }
    base.update(overrides)
    return ConversationMenuTarget(**base)


@pytest.mark.asyncio
@private_profile_test
async def test_root_menu_offers_copy_as_with_disclosure_glyph(request) -> None:
    """The Copy as opener carries the ▸ like the other page openers."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        labels = [
            str(button.label).strip()
            for button in screen.query_one(ConsoleConversationActionMenu).query(Button)
        ]
        assert labels[5] == "Copy as ▸"


@pytest.mark.asyncio
@private_profile_test
async def test_copy_page_offers_clean_full_and_save(request) -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)
        next(
            b
            for b in menu.query(Button)
            if getattr(b, "console_action_id", "") == "page:copy"
        ).press()
        await pilot.pause(0.5)
        assert menu.page == "copy"
        actions = [getattr(b, "console_action_id", "") for b in menu.query(Button)]
        assert actions == [
            "page:root",
            "copy-markdown:clean",
            "copy-markdown:full",
            "save-markdown",
        ]


@pytest.mark.asyncio
@private_profile_test
async def test_copy_clean_routes_to_clipboard_with_markdown(
    request, monkeypatch, tmp_path
) -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        copied: list[str] = []
        monkeypatch.setattr(
            screen.app_instance,
            "copy_to_clipboard",
            lambda text: copied.append(text),
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        screen.app_instance.chachanotes_db = CharactersRAGDB(
            str(tmp_path / "copy.db"), "copy-test"
        )
        db = screen.app_instance.chachanotes_db
        db = screen.app_instance.chachanotes_db
        conv_id = db.add_conversation({"title": "Copyable chat"})
        db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "first question",
            }
        )
        db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "the answer",
            }
        )
        monkeypatch.setattr(
            screen,
            "_console_conversation_state",
            lambda cid: "in-progress",
        )

        from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
            ConversationActionChosen,
        )

        screen.on_conversation_action_chosen(
            ConversationActionChosen(
                "copy-markdown:clean", _copy_target(conversation_id=conv_id)
            )
        )
        await pilot.pause(1.0)

        assert len(copied) == 1
        markdown = copied[0]
        assert markdown.startswith("# ")
        assert "## User" in markdown and "first question" in markdown
        assert "## Assistant" in markdown and "the answer" in markdown


@pytest.mark.asyncio
async def test_copy_empty_chat_is_gated_and_copies_nothing(monkeypatch) -> None:
    from tldw_chatbook.Chat.console_conversation_actions import (
        build_conversation_menu,
    )

    items = {
        item.action_id: item
        for item in build_conversation_menu(
            _copy_target(has_messages=False), page="copy"
        )
    }
    assert items["copy-markdown:clean"].enabled is False
    assert items["copy-markdown:clean"].disabled_reason == (
        "This chat has no messages yet."
    )


# ---- Save .md… through the real row menu (TASK-33621.12) -----------------
#
# G3-02 (2026-09-29 Console UX review): the save handler called
# `push_screen` on the ChatScreen -- a Screen, which has none -- so the
# worker's AttributeError ended the whole app. The test that used to stand
# here replaced `_save_console_conversation_markdown` with a fake, so the
# broken call never ran under test. Everything below drives the real menu,
# the real push and the real modal; nothing on that path is monkeypatched.

_OPENER_ID = "console-conversation-actions-0"


def _menu_button(menu, action_id: str) -> Button:
    return next(
        button
        for button in menu.query(Button)
        if getattr(button, "console_action_id", "") == action_id
    )


async def _wait_until(pilot, predicate, *, timeout: float = 5.0) -> bool:
    """Pump the app until ``predicate()`` holds or the app has died."""
    steps = int(timeout / 0.05)
    for _ in range(steps):
        if predicate():
            return True
        if pilot.app._exception is not None or not pilot.app.is_running:
            return False
        await pilot.pause(0.05)
    return predicate()


def _assert_app_alive(pilot) -> None:
    assert pilot.app._exception is None, f"the app died: {pilot.app._exception!r}"
    assert pilot.app.is_running


def _notifications(pilot, severity: str) -> list[str]:
    return [
        str(note.message)
        for note in pilot.app._notifications
        if note.severity == severity
    ]


async def _seed_row_zero_messages(pilot) -> None:
    """Give the open "Chat 1" row real messages in the live chat store."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    store = pilot.app.screen._ensure_console_chat_store()
    session_id = store.active_session_id
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="saved question"
    )
    store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="saved answer"
    )
    await pilot.pause(0.2)


def _ready_save_prompt(pilot):
    """The save prompt once it is mounted with its path field focused.

    ``app.screen`` switches to a pushed screen before that screen has
    composed, so "the top screen is the prompt" alone races its mount.
    """
    from textual.widgets import Input

    from tldw_chatbook.Widgets.Console.console_save_markdown_modal import (
        ConsoleSaveMarkdownModal,
    )

    screen = pilot.app.screen
    if not isinstance(screen, ConsoleSaveMarkdownModal) or not screen.is_mounted:
        return None
    fields = screen.query("#console-save-markdown-input").results(Input)
    field = next(fields, None)
    return screen if field is not None and field.has_focus else None


async def _open_save_prompt_from_row_menu(pilot):
    """Row 0 ▸ Copy as ▸ Save .md…, by keyboard, exactly as a user would.

    Returns:
        The mounted save prompt. Fails -- naming the app's own exception --
        when the prompt never opens, which is what the pre-fix crash does.
    """
    screen = pilot.app.screen
    _opener(screen).focus()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.3)
    menu = screen.query_one(ConsoleConversationActionMenu)
    _menu_button(menu, "page:copy").focus()
    await pilot.press("enter")
    await pilot.pause(0.3)
    save = _menu_button(menu, "save-markdown")
    assert not save.disabled, "the seeded row should offer Save .md…"
    save.focus()
    await pilot.press("enter")
    opened = await _wait_until(pilot, lambda: _ready_save_prompt(pilot) is not None)
    assert opened, (
        "Save .md… never opened its path prompt; "
        f"app exception: {pilot.app._exception!r}"
    )
    return _ready_save_prompt(pilot)


async def _submit_save_path(pilot, modal, path) -> None:
    from textual.widgets import Input

    field = modal.query_one("#console-save-markdown-input", Input)
    assert field.has_focus, "the prompt should open with its path field focused"
    field.value = str(path)
    await pilot.press("enter")


def _focused_id(pilot) -> str | None:
    focused = pilot.app.focused
    return focused.id if focused is not None else None


@pytest.mark.asyncio
@private_profile_test
async def test_row_menu_save_md_writes_the_file_and_keeps_the_app_running(
    request, tmp_path
) -> None:
    """AC#1/#4: the real row-menu Save .md… writes the file; the app lives."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        chat_screen = pilot.app.screen
        await _seed_row_zero_messages(pilot)
        modal = await _open_save_prompt_from_row_menu(pilot)
        target = tmp_path / "exports" / "saved-chat.md"

        await _submit_save_path(pilot, modal, target)

        assert await _wait_until(pilot, target.exists), (
            f"no file was written; app exception: {pilot.app._exception!r}"
        )
        written = target.read_text(encoding="utf-8")
        assert written.startswith("# ")
        assert "saved question" in written and "saved answer" in written
        _assert_app_alive(pilot)
        assert pilot.app.screen is chat_screen
        # The toast names the folder too, so a file saved under a bare name
        # (relative to where the app was started) can be found.
        saved_message = f"Saved saved-chat.md to {target.parent}."
        assert await _wait_until(
            pilot, lambda: saved_message in _notifications(pilot, "information")
        ), _notifications(pilot, "information")
        # Save closes through the same dismiss-once path as Cancel, so focus
        # goes back to the row control the user opened the menu from.
        assert await _wait_until(pilot, lambda: _focused_id(pilot) == _OPENER_ID), (
            f"focus landed on {pilot.app.focused!r}"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_save_md_for_a_persisted_conversation_writes_its_database_rows(
    request, tmp_path
) -> None:
    """A saved (not open) chat exports its database rows through the prompt."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
        ConversationActionChosen,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        screen.app_instance.chachanotes_db = CharactersRAGDB(
            str(tmp_path / "copy.db"), "copy-test"
        )
        db = screen.app_instance.chachanotes_db
        conv_id = db.add_conversation({"title": "Copyable chat"})
        db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "persisted question",
            }
        )
        screen.post_message(
            ConversationActionChosen(
                "save-markdown", _copy_target(conversation_id=conv_id)
            )
        )
        assert await _wait_until(
            pilot, lambda: _ready_save_prompt(pilot) is not None
        ), f"no prompt; app exception: {pilot.app._exception!r}"
        modal = _ready_save_prompt(pilot)
        # The default is a slug of the title under ~/Downloads.
        default = modal.query_one("#console-save-markdown-input").value
        assert default.endswith("copyable-chat.md"), default
        target = tmp_path / "exported.md"

        await _submit_save_path(pilot, modal, target)

        assert await _wait_until(pilot, target.exists)
        written = target.read_text(encoding="utf-8")
        assert written.startswith("# ")
        assert "persisted question" in written
        _assert_app_alive(pilot)


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["escape", "cancel", "backdrop"])
@private_profile_test
async def test_save_prompt_closes_without_writing_and_restores_focus(
    request, tmp_path, how
) -> None:
    """AC#2: Esc and Cancel both close the prompt, write nothing, and return
    focus to the row control that opened the menu. A click on the dimmed
    backdrop is the same cancel (the ADR-031 modal contract)."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        chat_screen = pilot.app.screen
        await _seed_row_zero_messages(pilot)
        modal = await _open_save_prompt_from_row_menu(pilot)
        target = tmp_path / "never-written.md"
        from textual.widgets import Input

        modal.query_one("#console-save-markdown-input", Input).value = str(target)
        await pilot.pause()

        if how == "escape":
            await pilot.press("escape")
        elif how == "cancel":
            cancel = modal.query_one("#console-save-markdown-cancel", Button)
            cancel.focus()
            await pilot.press("enter")
        else:
            box = modal.query_one("#console-save-markdown-box")
            assert not box.region.contains(1, 1), "the prompt covers the corner"
            assert await pilot.click(offset=(1, 1))

        assert await _wait_until(pilot, lambda: pilot.app.screen is chat_screen), (
            f"{how} left the save prompt open"
        )
        assert modal not in pilot.app.screen_stack
        await pilot.pause(0.5)
        assert not target.exists(), f"{how} still wrote the file"
        _assert_app_alive(pilot)
        assert await _wait_until(pilot, lambda: _focused_id(pilot) == _OPENER_ID), (
            f"focus after {how} landed on {pilot.app.focused!r}"
        )


async def _make_newest(pilot, chat_screen, session_id: str, opener_id: str) -> None:
    """Give ``session_id`` a new message, so it rises to the top of the rail.

    Waits until the rail has re-synced and ``opener_id`` -- its old slot --
    belongs to another chat.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    store = chat_screen._ensure_console_chat_store()
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="a newer question"
    )
    chat_screen._sync_console_workspace_context()
    tray = chat_screen.query_one("#console-workspace-context")

    def _slot_moved_on() -> bool:
        slot = next(iter(chat_screen.query(f"#{opener_id}")), None)
        return (
            slot is not None
            and not tray.recompose_in_flight
            and getattr(slot, "row_key", None) not in (None, f"native:{session_id}")
        )

    assert await _wait_until(pilot, _slot_moved_on), (
        "the chat never moved: its old slot still names it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reorder_while", ["menu-open", "prompt-open"])
@private_profile_test
async def test_a_reorder_during_save_md_returns_focus_to_the_same_chat(
    request, reorder_while
) -> None:
    """Checkpoint review of Qodo #2932: the opener is its chat, not its slot.

    Row ids are positional. The menu captured its opener's id when it
    opened, and the Save .md prompt captured the same id under it, so a tray
    rebuild that reordered the chats while either was open sent focus to the
    chat that now held that slot -- the next Enter or ``m`` acted on the
    wrong chat. Two open chats; the menu is opened from the older one's
    row, that chat then gets a message (so it moves to the top), and Esc
    closes the prompt.
    """
    from Tests.UI.test_console_tray_rebuild_focus import _open_second_tab

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        chat_screen = pilot.app.screen
        await _seed_row_zero_messages(pilot)
        store = chat_screen._ensure_console_chat_store()
        first_id = store.active_session_id
        first_key = f"native:{first_id}"
        await _open_second_tab(chat_screen, store, pilot)
        (opener,) = [
            button
            for button in chat_screen.query(Button)
            if str(button.id or "").startswith("console-conversation-actions-")
            and getattr(button, "row_key", None) == first_key
        ]
        opener_id = str(opener.id)

        opener.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.3)
        menu = chat_screen.query_one(ConsoleConversationActionMenu)
        _menu_button(menu, "page:copy").focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        if reorder_while == "menu-open":
            await _make_newest(pilot, chat_screen, first_id, opener_id)
        _menu_button(menu, "save-markdown").focus()
        await pilot.press("enter")
        assert await _wait_until(
            pilot, lambda: _ready_save_prompt(pilot) is not None
        ), f"no prompt; app exception: {pilot.app._exception!r}"
        if reorder_while == "prompt-open":
            await _make_newest(pilot, chat_screen, first_id, opener_id)

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: pilot.app.screen is chat_screen)
        await pilot.pause(0.5)
        _assert_app_alive(pilot)

        focused = pilot.app.focused
        assert focused is not None and str(focused.id or "").startswith(
            "console-conversation-actions-"
        ), f"focus landed on {focused!r}"
        assert getattr(focused, "row_key", None) == first_key, (
            f"focus followed the opener's old slot to chat "
            f"{getattr(focused, 'row_key', None)!r}, not {first_key!r}"
        )


def _parent_is_a_file(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("occupied", encoding="utf-8")
    return blocker / "chat.md", lambda: None


def _read_only_folder(tmp_path):
    import os
    import stat

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory write permission")
    folder = tmp_path / "read-only"
    folder.mkdir()
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return folder / "chat.md", lambda: folder.chmod(stat.S_IRWXU)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unwritable",
    [_parent_is_a_file, _read_only_folder],
    ids=["file-parent", "read-only"],
)
@private_profile_test
async def test_unwritable_save_path_shows_an_error_and_keeps_the_app_running(
    request, tmp_path, unwritable
) -> None:
    """AC#3: a save that cannot complete says why and never ends the app."""
    target, restore = unwritable(tmp_path)
    try:
        async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
            chat_screen = pilot.app.screen
            await _seed_row_zero_messages(pilot)
            modal = await _open_save_prompt_from_row_menu(pilot)

            await _submit_save_path(pilot, modal, target)

            def _save_errors() -> list[str]:
                return [
                    text
                    for text in _notifications(pilot, "error")
                    if text.startswith("Could not save chat.md")
                ]

            assert await _wait_until(pilot, lambda: bool(_save_errors())), (
                "an unwritable path produced no visible error: "
                f"{_notifications(pilot, 'error')}"
            )
            (message,) = _save_errors()
            assert str(target.parent) in message, message
            # The path is shown verbatim, never parsed as markup.
            assert all(
                note.markup is False
                for note in pilot.app._notifications
                if str(note.message) == message
            )
            assert not target.exists()
            _assert_app_alive(pilot)
            assert pilot.app.screen is chat_screen
    finally:
        restore()


@pytest.mark.asyncio
@private_profile_test
async def test_blank_save_path_keeps_the_prompt_open_and_says_why(
    request, tmp_path
) -> None:
    """Save with an empty path is refused visibly, not silently ignored."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await _seed_row_zero_messages(pilot)
        modal = await _open_save_prompt_from_row_menu(pilot)

        await _submit_save_path(pilot, modal, "   ")
        await pilot.pause(0.3)

        assert pilot.app.screen is modal, "a blank path closed the prompt"
        assert "Enter a file path to save to." in _notifications(pilot, "warning")
        _assert_app_alive(pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_copy_follows_the_active_branch_not_every_sibling(
    request, monkeypatch, tmp_path
) -> None:
    """Regenerated branches must not bleed into the export (PR #2262)."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
        ConversationActionChosen,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        screen.app_instance.chachanotes_db = CharactersRAGDB(
            str(tmp_path / "branch.db"), "branch-test"
        )
        db = screen.app_instance.chachanotes_db
        conv_id = db.add_conversation({"title": "Branched chat"})
        root = db.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "question"}
        )
        db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "first attempt",
                "parent_message_id": root,
            }
        )
        db.update_conversation(
            conv_id, {"active_leaf_message_id": root}, expected_version=1
        )
        # Direct leaf update (update_conversation whitelists fields): the
        # leaf points at root, so neither assistant branch exports.
        import sqlite3 as _sqlite

        conn = _sqlite.connect(str(tmp_path / "branch.db"))
        second = conn.execute(
            "SELECT id FROM messages WHERE content='first attempt'"
        ).fetchone()[0]
        conn.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (second, conv_id),
        )
        conn.commit()
        conn.close()

        copied: list[str] = []
        monkeypatch.setattr(
            screen.app_instance, "copy_to_clipboard", lambda t: copied.append(t)
        )
        screen.on_conversation_action_chosen(
            ConversationActionChosen(
                "copy-markdown:full",
                _copy_target(conversation_id=conv_id, title="Branched chat"),
            )
        )
        await pilot.pause(1.0)

        assert len(copied) == 1
        assert "question" in copied[0]
        assert "first attempt" in copied[0]
        assert copied[0].count("## Assistant") == 1


@pytest.mark.asyncio
@private_profile_test
async def test_transcript_click_folds_the_menu(request, monkeypatch) -> None:
    """ADR-068 completion: transcript presses own the biggest screen area.

    The screen-level outside-click dismissal returns early for transcript
    targets (the transcript owns its in-area interaction), and the
    transcript's own cleanup only knew its selection UI -- so a click on
    the transcript left a row action menu floating. The transcript's
    pointer press now folds the row menus itself.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        assert screen.query(ConsoleConversationActionMenu)

        assert await pilot.click("#console-native-transcript")
        await pilot.pause(0.3)
        assert not screen.query(ConsoleConversationActionMenu), (
            "a transcript press left the row action menu mounted"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_fold_helper_dismisses_both_registries_without_focus_restore(
    request,
) -> None:
    """Unit seam for _fold_row_action_menus_for_pointer (PR #2593 review).

    Mounts one menu of EACH registry directly (no UI open path), parks
    focus somewhere deliberate, calls the transcript helper, and asserts
    every registered menu was dismissed with restore_focus=False -- i.e.
    both gone and focus untouched (no yank back to the rail opener).
    """
    from tldw_chatbook.Chat.console_conversation_actions import (
        ConversationMenuTarget,
    )
    from tldw_chatbook.Chat.console_workspace_actions import WorkspaceMenuTarget
    from tldw_chatbook.Widgets.Console.console_workspace_action_menu import (
        ConsoleWorkspaceActionMenu,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript")
        await screen.mount(
            ConsoleConversationActionMenu(
                target=ConversationMenuTarget(conversation_id="c1"),
                opener_id="console-conversation-actions-0",
                screen_x=4,
                screen_y=6,
            ),
            ConsoleWorkspaceActionMenu(
                target=WorkspaceMenuTarget(workspace_id="w1"),
                opener_id="console-workspace-tree",
                screen_x=4,
                screen_y=14,
            ),
        )
        await pilot.pause(0.3)
        composer = screen.query_one("#console-native-composer")
        screen.set_focus(composer)
        await pilot.pause()

        transcript._fold_row_action_menus_for_pointer()
        await pilot.pause(0.3)

        assert not screen.query(ConsoleConversationActionMenu)
        assert not screen.query(ConsoleWorkspaceActionMenu)
        assert pilot.app.focused is composer, (
            "fold restored opener focus instead of honouring the press"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_ascii_attention_indicators_fit_painted_action_cells(request) -> None:
    from rich.text import Text
    from textual.geometry import Region

    from tldw_chatbook.Workspaces.conversation_attention import ATTENTION_PRESENTATIONS

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        control = _opener(pilot.app.screen)
        control.add_class("conversation-actions-ascii")
        for _unicode, ascii_icon, _status in ATTENTION_PRESENTATIONS.values():
            control.label = Text(ascii_icon)
            await pilot.pause()
            painted = "".join(
                strip.text
                for strip in control.render_lines(
                    Region(0, 0, control.size.width, control.size.height)
                )
            )
            assert ascii_icon in painted, (ascii_icon, painted)
