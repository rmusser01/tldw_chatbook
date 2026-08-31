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

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)


def _opener(screen) -> Button:
    return screen.query_one("#console-conversation-actions-0", Button)


@pytest.mark.asyncio
async def test_row_carries_a_one_row_asterisk_not_a_full_height_star() -> None:
    """The control must not reserve the row's whole height any more."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        opener = _opener(screen)

        assert str(opener.label).strip() == "*"
        assert opener.disabled is False
        assert opener.region.height == 1, (
            "the action opener is still reserving full row height"
        )
        assert not screen.query(".console-conversation-star"), (
            "the retired star control is still being composed"
        )


@pytest.mark.asyncio
async def test_local_stars_unavailable_jargon_is_gone() -> None:
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
async def test_asterisk_opens_the_menu_with_the_expected_entries() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        labels = [str(button.label).strip() for button in menu.query(Button)]
        assert labels == [
            "Favourite",
            "Change status ▸",
            "Archive",
            "Rename…",
            "Copy as ▸",
            "More ▸",
        ]


@pytest.mark.asyncio
async def test_every_disabled_entry_states_its_precondition() -> None:
    """A greyed control with no explanation is the defect being removed."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        for button in menu.query(Button):
            if button.disabled:
                assert button.tooltip, (
                    f"{button.id} is disabled with no stated reason"
                )


@pytest.mark.asyncio
async def test_more_opens_delete_and_back_returns() -> None:
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
async def test_escape_steps_out_of_a_submenu_before_closing() -> None:
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
async def test_menu_focuses_its_first_actionable_entry_on_open() -> None:
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
async def test_click_outside_closes_the_menu_without_dispatching(monkeypatch) -> None:
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
async def test_click_on_menu_chrome_keeps_the_menu_open() -> None:
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
async def test_escape_with_focus_outside_the_menu_closes_it() -> None:
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
async def test_pressing_the_asterisk_again_replaces_rather_than_stacks() -> None:
    """The opener's press path still ends with exactly one menu mounted."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        await pilot.click("#console-conversation-actions-0")
        await pilot.pause(0.3)

        mounted = screen.query(ConsoleConversationActionMenu)
        assert len(mounted) == 1, (
            f"expected one replaced menu, found {len(mounted)}"
        )


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


# ---- Copy as markdown (TASK-25836) ---------------------------------------


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
async def test_root_menu_offers_copy_as_with_disclosure_glyph() -> None:
    """The Copy as opener carries the ▸ like the other page openers."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        labels = [
            str(button.label).strip()
            for button in screen.query_one(ConsoleConversationActionMenu).query(Button)
        ]
        assert labels[4] == "Copy as ▸"


@pytest.mark.asyncio
async def test_copy_page_offers_clean_full_and_save() -> None:
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
        actions = [
            getattr(b, "console_action_id", "") for b in menu.query(Button)
        ]
        assert actions == [
            "page:root",
            "copy-markdown:clean",
            "copy-markdown:full",
            "save-markdown",
        ]


@pytest.mark.asyncio
async def test_copy_clean_routes_to_clipboard_with_markdown(
    monkeypatch, tmp_path
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
            ConversationActionChosen("copy-markdown:clean", _copy_target(conversation_id=conv_id))
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
        for item in build_conversation_menu(_copy_target(has_messages=False), page="copy")
    }
    assert items["copy-markdown:clean"].enabled is False
    assert items["copy-markdown:clean"].disabled_reason == (
        "This chat has no messages yet."
    )


@pytest.mark.asyncio
async def test_save_writes_validated_markdown_file(monkeypatch, tmp_path) -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
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
                "content": "persisted question",
            }
        )
        monkeypatch.setattr(
            screen,
            "_console_conversation_state",
            lambda cid: "in-progress",
        )
        target = tmp_path / "exported.md"

        from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
            ConversationActionChosen,
        )

        async def _fake_save(t):
            markdown = screen._render_console_conversation_markdown(t, "clean")
            assert markdown is not None
            await screen._write_console_markdown_file(str(target), markdown)

        monkeypatch.setattr(screen, "_save_console_conversation_markdown", _fake_save)
        screen.on_conversation_action_chosen(
            ConversationActionChosen("save-markdown", _copy_target(conversation_id=conv_id))
        )
        await pilot.pause(1.0)

        written = target.read_text(encoding="utf-8")
        assert "persisted question" in written
        assert written.startswith("# ")


@pytest.mark.asyncio
async def test_copy_follows_the_active_branch_not_every_sibling(
    monkeypatch, tmp_path
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
