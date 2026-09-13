"""Conversations "Open in Console" recovery (critique #8 row 7, task-32056).

The action sat at the bottom of the reader under a 30-message transcript and
refused with a toast naming a workspace the user had no way to link into.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Widgets.Library import LibraryConversationReader


def _loaded_reader_state() -> ConversationReaderState:
    return ConversationReaderState(
        selected_id="chat-a",
        selected_version=4,
        loaded_id="chat-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=(
            ConversationMessageView(
                "message-a", "user", "2026-08-23T12:01:00Z", "revision-a", 5, "hello"
            ),
        ),
        message_total=1,
        complete=True,
    )


@pytest.mark.asyncio
async def test_open_console_sits_in_the_header_beside_read_and_info(
    widget_pilot,
) -> None:
    """AC#1: the action is header chrome, not a footer under the transcript."""
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={"title": "Alpha planning"},
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        order = [
            child.id
            for child in reader.walk_children()
            if child.id
            in {
                "library-conversation-reader-read",
                "library-conversation-reader-info",
                "library-conversation-open-console",
                "library-conversation-reader-status",
                "library-conversation-reader-messages",
            }
        ]
        assert order.index("library-conversation-open-console") < order.index(
            "library-conversation-reader-status"
        )
        assert order.index("library-conversation-open-console") < order.index(
            "library-conversation-reader-messages"
        )


@pytest.mark.asyncio
async def test_workspace_refusal_is_inline_with_a_link_action(
    widget_pilot,
) -> None:
    """AC#2/#3: the reason and the remedy render on the action itself."""
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={
            "title": "Alpha planning",
            "_workspace_block": "not in this workspace",
            "_workspace_block_linkable": True,
        },
        id="library-conversation-reader",
    ) as pilot:
        open_console = pilot.app.query_one(
            "#library-conversation-use-source", Button
        )
        blocked = pilot.app.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        link = pilot.app.query_one("#library-conversation-link-workspace", Button)

        # The reason wraps in a Static, not the Button label: a Button label
        # is single-line and truncates (fix round 1). It is the refusal
        # SENTENCE, not a second copy of the action name (task-32101).
        assert str(blocked.renderable) == (
            "This conversation is not in this workspace. Pressing this "
            "adds it to the active workspace first, and you can undo that."
        )
        assert blocked.display is True
        # task-32107 (user decision, critique #10): a block a LINK can
        # resolve no longer disables the hand-off -- the press links and
        # proceeds in one undoable step, and the sentence above says so.
        # "Link to workspace" stays for membership without a hand-off.
        assert open_console.disabled is False
        assert link.display is True
        assert link.disabled is False
        assert str(link.label) == "Link to workspace"


@pytest.mark.asyncio
async def test_linking_clears_the_inline_refusal(widget_pilot) -> None:
    """After the link lands, the same reader enables the hand-off."""
    state = _loaded_reader_state()
    async with await widget_pilot(
        LibraryConversationReader,
        state=state,
        loaded_metadata={
            "title": "Alpha planning",
            "_workspace_block": "not in this workspace",
            "_workspace_block_linkable": True,
        },
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        reader.sync_state(state, loaded_metadata={"title": "Alpha planning"})
        await pilot.pause()

        open_console = pilot.app.query_one(
            "#library-conversation-use-source", Button
        )
        blocked = pilot.app.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        link = pilot.app.query_one("#library-conversation-link-workspace", Button)
        assert str(open_console.label) == "Use as source"
        assert open_console.disabled is False
        assert blocked.display is False
        assert link.display is False


def test_link_to_workspace_makes_the_conversation_eligible() -> None:
    """AC#2: the remedy writes the membership the eligibility rule wants."""
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Workspaces.display_state import (
        build_library_workspace_depth_state,
        library_item_context_handoff,
    )

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    registry.set_active_workspace("workspace-a")
    records = {
        "conversations": [{"id": "chat-a", "title": "Alpha planning"}],
        "notes": [],
        "media": [],
    }

    before = build_library_workspace_depth_state(
        registry_service=registry, source_records=records
    )
    eligible, reason = library_item_context_handoff(
        before, item_type="conversation", item_id="chat-a"
    )
    assert eligible is False
    assert reason

    registry.link_membership(
        "workspace-a",
        item_type="conversation",
        item_id="chat-a",
        title="Alpha planning",
    )

    after = build_library_workspace_depth_state(
        registry_service=registry, source_records=records
    )
    assert library_item_context_handoff(
        after, item_type="conversation", item_id="chat-a"
    ) == (True, "")


def test_conversation_open_console_has_a_keyboard_route() -> None:
    """AC#1: 'c' reaches the hand-off, mirroring Media's own binding."""
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LIBRARY_ROW_BROWSE_MEDIA,
        LibraryScreen,
    )

    actions = {
        binding.action
        for binding in LibraryScreen.BINDINGS
        if getattr(binding, "key", None) == "c"
    }
    assert "library_conversation_open_console" in actions
    assert hasattr(LibraryScreen, "action_library_conversation_open_console")

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    registry.set_active_workspace("workspace-a")
    screen = LibraryScreen(app)
    screen.restore_state(
        {"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS}
    )
    screen._conversations_state.reader_state = _loaded_reader_state()
    screen._local_source_records["conversations"] = [
        {"id": "chat-a", "title": "Alpha planning"}
    ]

    # Resume reopens the original identity; source membership gates only
    # the separate Use as source action.
    assert screen.check_action("library_conversation_open_console", ()) is True

    registry.link_membership(
        "workspace-a",
        item_type="conversation",
        item_id="chat-a",
        title="Alpha planning",
    )
    screen._invalidate_library_workspace_depth_state()
    assert screen.check_action("library_conversation_open_console", ()) is True

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    assert screen.check_action("library_conversation_open_console", ()) is False


@pytest.mark.asyncio
async def test_mounted_reader_offers_the_link_then_enables_the_handoff() -> None:
    """End to end: the seeded (unlinked) conversation, then the remedy.

    The refusal names the block on the action itself and the adjacent
    "Link to workspace" resolves it in place -- previously the press only
    raised a toast naming a workspace with nothing on screen to link into.
    """
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _build_test_app,
        _seed_conversations,
        _wait_for_library_shell,
        _wait_for_selector,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LibraryScreen,
    )

    app = _build_test_app()
    _seed_conversations(
        app,
        [
            {
                "id": "chat-a",
                "title": "Alpha planning",
                "version": 4,
                "message_count": 1,
                "last_modified": "2026-08-23T12:00:00Z",
            }
        ],
    )
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    registry.set_active_workspace("workspace-a")

    screen = LibraryScreen(app)
    screen.restore_state(
        {"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS}
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-conversation-reader")
        for _ in range(20):
            if screen._conversations_state.reader_state.loaded_id:
                break
            await pilot.pause(0.01)

        open_console = screen.query_one(
            "#library-conversation-use-source", Button
        )
        blocked = screen.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        link = screen.query_one("#library-conversation-link-workspace", Button)
        assert str(blocked.renderable) == (
            "This conversation is not in this workspace. Pressing this "
            "adds it to the active workspace first, and you can undo that."
        )
        # task-32107: pressable, and unmarked -- the "○" marker means
        # blocked, so it follows the button's real disabled state.
        assert str(open_console.label) == "Use as source"
        assert open_console.disabled is False
        assert link.display is True
        # Resume is independent of source workspace eligibility.
        assert screen.check_action("library_conversation_open_console", ()) is True

        link.press()
        await pilot.pause()
        await pilot.pause()

        assert registry.get_item_memberships(
            item_type="conversation", item_id="chat-a"
        )
        open_console = screen.query_one(
            "#library-conversation-use-source", Button
        )
        assert str(open_console.label) == "Use as source"
        assert open_console.disabled is False
        assert (
            screen.query_one(
                "#library-conversation-open-console-blocked", Static
            ).display
            is False
        )
        assert (
            screen.query_one(
                "#library-conversation-link-workspace", Button
            ).display
            is False
        )
        assert screen.check_action("library_conversation_open_console", ()) is True


@pytest.mark.asyncio
async def test_link_is_withheld_while_the_selection_outruns_the_transcript(
    widget_pilot,
) -> None:
    """Review round 2: the remedy writes ``loaded_id``, so it obeys the load fence.

    Selecting another conversation leaves the previous transcript on screen
    until the new one loads; offering "Link to workspace" there would link
    the conversation the user just navigated away from.
    """
    from dataclasses import replace

    state = replace(
        _loaded_reader_state(),
        selected_id="chat-b",
        selected_version=7,
        loading=True,
    )
    async with await widget_pilot(
        LibraryConversationReader,
        state=state,
        loaded_metadata={
            "title": "Alpha planning",
            "_workspace_block": "not in this workspace",
            "_workspace_block_linkable": True,
        },
        id="library-conversation-reader",
    ) as pilot:
        link = pilot.app.query_one("#library-conversation-link-workspace", Button)
        assert link.display is False


@pytest.mark.asyncio
async def test_non_linkable_block_keeps_its_own_recovery_copy(
    widget_pilot,
) -> None:
    """Review round 2: never name a control the block does not offer.

    A block linking cannot resolve (no active workspace) hides the link, so
    the disabled hand-off must repeat the eligibility rule's own remedy
    instead of pointing at a button that is not on screen.
    """
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={
            "title": "Alpha planning",
            "_workspace_block": "blocked for this workspace",
            "_workspace_block_linkable": False,
            "_workspace_block_detail": (
                "Select an active workspace before using this item in Console."
            ),
        },
        id="library-conversation-reader",
    ) as pilot:
        open_console = pilot.app.query_one(
            "#library-conversation-use-source", Button
        )
        link = pilot.app.query_one("#library-conversation-link-workspace", Button)
        assert link.display is False
        assert open_console.disabled is True
        tooltip = str(open_console.tooltip)
        assert "Link to workspace" not in tooltip
        assert "Select an active workspace" in tooltip


def test_link_remedy_refuses_a_stale_retained_transcript() -> None:
    """Review round 2: the persisting seam re-checks the fence it renders."""
    from dataclasses import replace

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LibraryScreen,
    )

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    registry.set_active_workspace("workspace-a")
    screen = LibraryScreen(app)
    screen.restore_state(
        {"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS}
    )
    screen._local_source_records["conversations"] = [
        {"id": "chat-a", "title": "Alpha planning"}
    ]
    screen._conversations_state.reader_state = replace(
        _loaded_reader_state(), selected_id="chat-b", selected_version=7, loading=True
    )

    screen._link_selected_conversation_to_workspace()

    assert not registry.get_item_memberships(
        item_type="conversation", item_id="chat-a"
    )


@pytest.mark.asyncio
async def test_blocked_state_paints_the_action_name_once(widget_pilot) -> None:
    """task-32101 AC#4: one control, one action name, one sentence.

    The blocked state used to render the disabled Button ("Open in Console")
    above a Static repeating "○ Open in Console · not in this workspace" --
    two paints of the same action name, read on screen as two controls. The
    marker now lives on the button label and the line beneath it carries the
    reason SENTENCE only.
    """
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_reader_state(),
        loaded_metadata={
            "title": "Alpha planning",
            "_workspace_block": "not in this workspace",
            "_workspace_block_linkable": True,
        },
        id="library-conversation-reader",
    ) as pilot:
        open_console = pilot.app.query_one(
            "#library-conversation-use-source", Button
        )
        blocked = pilot.app.query_one(
            "#library-conversation-open-console-blocked", Static
        )
        # task-32107: the action is pressable now, so it carries no "○";
        # the one-name/one-sentence rule this test exists for is unchanged.
        assert str(open_console.label) == "Use as source"
        assert open_console.disabled is False
        assert blocked.display is True
        assert "Use as source" not in str(blocked.renderable)
        # ...and it is the very sentence the tooltip gives.
        assert str(blocked.renderable) == str(open_console.tooltip)
        assert "not in this workspace" in str(blocked.renderable)


def test_blocked_c_key_says_what_the_resume_control_says() -> None:
    """Keep dev's explanatory key while fencing the original loaded identity."""
    from dataclasses import replace
    from unittest.mock import Mock

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LIBRARY_ROW_BROWSE_MEDIA,
        LibraryScreen,
    )

    app = _build_test_app()
    app.notify = Mock()
    app.resume_console_conversation = Mock()
    app.open_chat_with_handoff = Mock()
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    screen._conversations_state.reader_state = replace(
        _loaded_reader_state(), loading=True, complete=False
    )
    assert screen.check_action("library_conversation_open_console", ()) is True
    assert ("c", "resume conversation") not in screen._library_route_shortcuts_for_current_state()
    screen.action_library_conversation_open_console()
    app.resume_console_conversation.assert_not_called()
    app.open_chat_with_handoff.assert_not_called()
    app.notify.assert_called_once_with(
        "Wait for the selected conversation to finish loading.", severity="warning"
    )
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    assert screen.check_action("library_conversation_open_console", ()) is False


def test_ready_resume_key_ignores_source_workspace_membership() -> None:
    """Original-ID resume remains available when the source handoff is refused."""
    from unittest.mock import Mock

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LibraryScreen,
    )

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    registry.set_active_workspace("workspace-a")
    app.resume_console_conversation = Mock()
    app.open_chat_with_handoff = Mock()
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    screen._conversations_state.reader_state = _loaded_reader_state()
    screen._conversations_state.freshness = "fresh"
    screen._local_source_records["conversations"] = [{"id": "chat-a", "title": "Alpha planning"}]
    assert screen._library_conversation_workspace_block()[0]
    assert ("c", "resume conversation") in screen._library_route_shortcuts_for_current_state()
    screen.action_library_conversation_open_console()
    app.resume_console_conversation.assert_called_once_with("chat-a")
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatch", ["button", "key"])
@pytest.mark.parametrize("exists_now", [True, False])
async def test_stale_list_resume_dispatches_original_and_checks_current_storage(
    dispatch, exists_now, monkeypatch
) -> None:
    """List freshness cannot silence Resume; fresh storage still decides recovery."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Console_Modules.archive import request_conversation_resume
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_ROW_BROWSE_CONVERSATIONS,
        LibraryScreen,
    )

    app = _build_test_app()
    app.notify = Mock()
    app.post_message = Mock()
    app.resume_console_conversation = Mock()
    app.open_chat_with_handoff = Mock()
    registry = app.workspace_registry_service
    registry.create_workspace(
        workspace_id="current-workspace", name="Current workspace"
    )
    read_workspace = Mock(wraps=registry.get_workspace)
    monkeypatch.setattr(registry, "get_workspace", read_workspace)
    current_metadata = {
        "id": "chat-a",
        "version": 9,
        "archived": False,
        "workspace_id": "current-workspace",
    }
    read_metadata = Mock(return_value=current_metadata if exists_now else None)
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_metadata=read_metadata
    )
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    screen._conversations_state.reader_state = _loaded_reader_state()
    screen._conversations_state.freshness = "stale"
    assert screen._library_conversation_handoff_ready()
    assert (
        "c",
        "resume conversation",
    ) in screen._library_route_shortcuts_for_current_state()

    if dispatch == "button":
        screen.open_selected_conversation_in_console(
            Button.Pressed(Button(id="library-conversation-open-console"))
        )
    else:
        screen.action_library_conversation_open_console()
    app.resume_console_conversation.assert_called_once_with("chat-a")

    screen.use_selected_conversation_as_source(
        Button.Pressed(Button(id="library-conversation-use-source"))
    )
    app.open_chat_with_handoff.assert_not_called()
    # Follow the same typed request invoked by the application dispatcher.
    await request_conversation_resume(
        app, app.resume_console_conversation.call_args.args[0]
    )
    assert read_metadata.call_count == (2 if exists_now else 1)
    assert all(call.args == ("chat-a",) for call in read_metadata.call_args_list)
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
    if exists_now:
        read_workspace.assert_called_once_with("current-workspace")
        assert claim is not None and claim.value.conversation_id == "chat-a"
        app.post_message.assert_called_once()
    else:
        read_workspace.assert_not_called()
        assert claim is None
        app.post_message.assert_not_called()
        app.notify.assert_called_once_with(
            "This conversation is no longer available.", severity="warning"
        )
