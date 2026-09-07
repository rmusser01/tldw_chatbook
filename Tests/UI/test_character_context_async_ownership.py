"""Character navigation preserves async generation, profile and focus ownership."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.test_console_character_context import (
    _CharacterApp,
    _controller,
    _groups,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationRow,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextState,
)
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)


@pytest.mark.asyncio
async def test_removed_browser_rejects_late_presentation():
    controller = _controller()
    state = ConsoleCharacterContextState(groups=_groups())
    app = _CharacterApp(controller, state)
    async with app.run_test() as pilot:
        widget = app.screen.query_one(ConsoleCharacterContext)
        await widget.remove()
        widget.sync_state(replace(state, data_revision=999))
        await pilot.pause()
        assert widget.state is state


@pytest.mark.asyncio
@pytest.mark.parametrize("churn", ["before", "during", "none"])
async def test_start_keeps_exact_profile_through_picker_fetch(tmp_path, churn):
    from Tests.UI.test_console_character_controller import (
        _controller as picker_controller,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.wiring import _start_character_context_chat

    first = CharactersRAGDB(tmp_path / "first.sqlite", client_id="first")
    second = CharactersRAGDB(tmp_path / "second.sqlite", client_id="second")
    release = threading.Event()
    try:
        first_id = first.add_character_card({"name": "Original"})
        second_id = second.add_character_card({"name": "Different"})
        assert first_id == second_id
        key = ResolvedLocalCharacterKey(first.get_local_authority_id(), first_id)
        group = CharacterConversationGroup(key, "Original", (), 0, True)
        active = [first]
        store = ConsoleChatStore()
        picker = picker_controller(
            character_db_accessor=lambda: active[0],
            ensure_chat_store=lambda: store,
            default_session_settings=lambda: ConsoleSessionSettings(
                provider="synthetic"
            ),
        )
        screen = SimpleNamespace(_character=picker)
        controller = _controller(
            database_accessor=lambda: active[0],
            current_character_accessor=lambda: (first_id, "Original"),
            start_console=lambda *args: _start_character_context_chat(screen, *args),
        )
        controller.state = replace(
            controller.state, scope_fingerprint=await controller._fingerprint()
        )
        entered = threading.Event()
        fetch = first.get_character_card_by_id

        def delayed(character_id):
            entered.set()
            assert release.wait(2)
            return fetch(character_id)

        if churn == "before":
            active[0] = second
        if churn == "during":
            first.get_character_card_by_id = delayed
        pending = asyncio.create_task(controller.start_current(group))
        if churn == "during":
            assert await asyncio.to_thread(entered.wait, 2)
            active[0] = second
            release.set()
        await pending
        if churn != "none":
            assert store.active_session_id is None
        else:
            session = store.switch_session(store.active_session_id)
            assert session.character_name == "Original"
            assert session.assistant_authority_id == key.data_authority_id
    finally:
        release.set()
        first.close()
        second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "refresh",
        "refresh_error",
        "search",
        "search_error",
        "refresh_missing",
        "search_missing",
        "details",
        "repair",
        "inspection",
        "browse",
    ],
)
async def test_superseded_operation_cannot_publish_after_final_scope_await(operation):
    dispatched = []
    controller = _controller(
        navigate_inspection=dispatched.append,
        navigate_repair=dispatched.append,
        navigate_unavailable_browse=dispatched.append,
    )
    key = UnresolvedConversationKey("authority", "lost")
    row = CharacterConversationRow.unavailable(
        key,
        reason=UnavailableCharacterReason.DELETED_CARD,
        character_label="Lost",
        title="Lost",
        last_modified="2026-09-01",
        created_at="2026-09-01",
    )
    group = CharacterConversationGroup(key, "Unavailable", (row,), 1, False)
    original_search = controller._search_sync
    if operation.endswith("missing"):
        controller._database_accessor = lambda: None
    if operation.endswith("error"):

        def fail(*args):
            raise RuntimeError("synthetic read failure")

        if operation.startswith("search"):
            controller._search_sync = fail
        else:
            controller._load_recent_sync = fail
    entered, release = asyncio.Event(), asyncio.Event()
    validate = controller._scope_is_current
    old = None

    async def delayed(snapshot):
        if asyncio.current_task() is old and not entered.is_set():
            entered.set()
            await release.wait()
        return await validate(snapshot)

    controller._scope_is_current = delayed
    if operation.startswith("refresh"):
        work = controller.refresh()
    elif operation.startswith("search"):
        work = controller.search("old")
    elif operation == "details":
        work = controller.refresh_unavailable_details((group,))
    elif operation == "repair":
        work = controller.repair_unavailable(key, row_key=row.row_key)
    elif operation == "inspection":
        work = controller.open_unavailable(key, row_key=row.row_key)
    else:
        work = controller.view_group(group)
    old = asyncio.create_task(work)
    await asyncio.wait_for(entered.wait(), 2)
    controller._search_sync = original_search
    await controller.search("new")
    newer = controller.state
    release.set()
    await old
    assert controller.state is newer
    assert not dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize("leave_character", [True, False])
async def test_recompose_preserves_only_owned_focus_intent(leave_character):
    controller = _controller()
    state = ConsoleCharacterContextState(
        groups=_groups(), expanded_key=_groups()[0].key
    )
    app = _CharacterApp(controller, state)
    async with app.run_test(size=(80, 35)) as pilot:
        widget = app.screen.query_one(ConsoleCharacterContext)
        external = Button("Composer", id="outside-character")
        await app.screen.mount(external)
        await pilot.pause()
        row = widget.query_one(".console-character-row", Button)
        row.focus()
        await pilot.pause()
        entered, release = asyncio.Event(), asyncio.Event()
        original = widget.recompose

        async def delayed():
            entered.set()
            await release.wait()
            await original()

        widget.recompose = delayed
        widget.sync_state(replace(widget.state, data_revision=99))
        await asyncio.wait_for(entered.wait(), 2)
        if leave_character:
            external.focus()
        else:
            widget.sync_state(replace(widget.state, data_revision=100))
        await pilot.pause()
        release.set()
        await pilot.pause()
        if leave_character:
            assert app.focused is external
        else:
            assert app.focused.id == row.id
            assert app.focused is not row
