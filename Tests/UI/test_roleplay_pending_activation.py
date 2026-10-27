"""The published four-key find survives a pending actual character read."""

import asyncio
import threading

import pytest
from textual.widgets import Input, Static

from Tests.UI.roleplay_frame_harness import (
    ISOLDE,
    highlighted_item_id,
    open_styled_roleplay,
    open_volume_db,
    roleplay_full_app,
    settle,
    volume_db_path,
    wait_until,
)
from Tests.UI.test_roleplay_journeys import _start_on_another_card
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaEntitySelected,
)
from tldw_chatbook.UI.Screens import personas_screen as module
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("publication", ["ordinary", "held", "repeated", "late"])
async def test_four_key_find_opens_once_after_matching_publication(
    tmp_path_factory, tmp_path, monkeypatch, publication
):
    """Actual keys open Isolde without a third Enter; held rows never open an old card."""
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 348))
    release, reading = threading.Event(), threading.Event()
    page = module.get_character_page_for_ui
    if publication == "ordinary":
        release.set()

    def held_page(actual_db, **kwargs):
        if "isolde" in str(kwargs.get("search_term") or "").lower():
            reading.set()
            assert release.wait(30), "held Isolde read was not released"
        return page(actual_db, **kwargs)

    try:
        async with roleplay_full_app(size=(120, 36), character_db=db) as pilot:
            screen = pilot.app.screen
            other = await _start_on_another_card(pilot, db)
            isolde = str(db.get_character_card_by_name(ISOLDE)["id"])
            monkeypatch.setattr(module, "get_character_page_for_ui", held_page)
            library = screen.query_one("#personas-library-pane")
            post = library.post_message
            selections = []

            def observe_selection(message):
                if isinstance(message, PersonaEntitySelected):
                    selections.append((message.entity_kind, message.entity_id))
                return post(message)

            monkeypatch.setattr(library, "post_message", observe_selection)
            try:
                if publication == "late":
                    await pilot.press("ctrl+f", *"isolde", "enter")
                else:
                    await pilot.press("ctrl+f", *"isolde", "enter", "enter")
                await wait_until(pilot, reading.is_set, what="the Isolde read")
                if publication != "ordinary":
                    assert screen.state.selected_entity_id == other
                    assert screen._window_state.landing_owed()
                    if publication == "repeated":
                        await pilot.press("enter", "enter", "enter")
                        assert screen.state.selected_entity_id == other
            finally:
                release.set()
            if publication == "late":
                await wait_until(
                    pilot,
                    lambda: highlighted_item_id(screen) == f"character:{isolde}",
                    what="the published first match before a late Enter",
                )
                assert screen.state.selected_entity_id == other
                await pilot.press("enter")
            await wait_until(
                pilot,
                lambda: screen.state.selected_entity_id == isolde,
                what="Isolde's card after the four-key burst",
            )
            await settle(pilot)
            assert highlighted_item_id(screen) == f"character:{isolde}"
            assert selections == [("character", isolde)]
            assert str(
                screen.query_one("#personas-character-card-name", Static).render()
            ) == (f"Name: {ISOLDE}")
    finally:
        release.set()
        db.close_connection()


@pytest.mark.parametrize("newer_intent", ["cursor", "query", "reload"])
async def test_newer_intent_retires_activation_before_selection_dispatch(
    tmp_path_factory, tmp_path, monkeypatch, newer_intent
):
    """A delayed row-selection message cannot outrun an accepted newer user intent."""
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 348))
    release, reading = threading.Event(), threading.Event()
    page = module.get_character_page_for_ui

    def held_page(actual_db, **kwargs):
        if "isolde" in str(kwargs.get("search_term") or "").lower():
            reading.set()
            assert release.wait(30), "held Isolde read was not released"
        return page(actual_db, **kwargs)

    try:
        async with roleplay_full_app(size=(120, 36), character_db=db) as pilot:
            screen = pilot.app.screen
            other = await _start_on_another_card(pilot, db)
            library = screen.query_one("#personas-library-pane")
            post = library.post_message
            queued = []

            def hold_selection(message):
                if isinstance(message, PersonaEntitySelected):
                    queued.append(message)
                    return True
                return post(message)

            monkeypatch.setattr(module, "get_character_page_for_ui", held_page)
            monkeypatch.setattr(library, "post_message", hold_selection)
            try:
                await pilot.press("ctrl+f", *"isolde", "enter", "enter")
                await wait_until(pilot, reading.is_set, what="the held Isolde read")
                assert screen.state.selected_entity_id == other
                release.set()
                await wait_until(
                    pilot, lambda: bool(queued), what="the deferred selection"
                )
                assert len(queued) == 1
                if newer_intent == "cursor":
                    await pilot.press("down")
                elif newer_intent == "reload":
                    await screen._reload_character_page()
                else:
                    screen.query_one(
                        "#personas-library-search", Input
                    ).value = "Character 03"
                    await wait_until(
                        pilot,
                        lambda: screen.state.search_query == "Character 03",
                        what="the accepted newer query",
                    )
                post(queued.pop())
                await settle(pilot)
                assert screen.state.selected_entity_id == other
            finally:
                release.set()
    finally:
        release.set()
        db.close_connection()


async def test_window_state_reexports_preserve_the_existing_import_identity():
    """Moving the pure owner must not split old and new consumers' class identity."""
    from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as old
    from tldw_chatbook.UI.Persona_Modules import roleplay_list_window_state as new

    for name in (
        "ListFence",
        "ListAnchor",
        "ListWindowRequest",
        "WindowIntent",
        "ListWindowState",
        "same_list",
        "new_option_ids",
        "window_wants_more",
    ):
        assert getattr(old, name) is getattr(new, name)


async def test_superseded_dictionary_search_waits_for_authoritative_publication(
    tmp_path_factory, tmp_path, mock_app_instance, monkeypatch
):
    """A same-query refresh cannot turn search-return into stale-row activation."""
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 348))
    search_reading, search_release = asyncio.Event(), asyncio.Event()
    refresh_reading, refresh_release = asyncio.Event(), asyncio.Event()
    local = LocalChatDictionaryService(db)
    local.create_dictionary({"name": "Old first"})
    local.create_dictionary({"name": "Old second"})
    service = ChatDictionaryScopeService(local_service=local, server_service=None)
    actual_list = service.list_dictionaries
    reads = 0

    async def list_dictionaries(**kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            search_reading.set()
            await search_release.wait()
        elif reads == 3:
            refresh_reading.set()
            await refresh_release.wait()
        return await actual_list(**kwargs)

    monkeypatch.setattr(service, "list_dictionaries", list_dictionaries)
    mock_app_instance.chat_dictionary_scope_service = service
    refresh = None
    try:
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=(120, 36), character_db=db
        ) as pilot:
            screen = pilot.app.screen
            assert await pilot.click("#personas-mode-dictionaries")
            await wait_until(pilot, lambda: reads == 1, what="initial dictionary rows")
            await settle(pilot)
            target = local.create_dictionary({"name": "Target fresh"})
            target_id = str(target["id"])
            library = screen.query_one("#personas-library-pane")
            post = library.post_message
            selections = []

            def observe_selection(message):
                if isinstance(message, PersonaEntitySelected):
                    selections.append(message.entity_id)
                return post(message)

            monkeypatch.setattr(library, "post_message", observe_selection)
            monkeypatch.setattr(module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 60.0)
            try:
                await pilot.press("ctrl+f", *"Target", "enter", "enter")
                await wait_until(pilot, search_reading.is_set, what="search S held")
                refresh = asyncio.create_task(
                    screen._render_dictionary_rows(query="Target")
                )
                await wait_until(pilot, refresh_reading.is_set, what="refresh R held")
                search_release.set()
                await wait_until(
                    pilot,
                    lambda: not any(
                        worker.group == "personas-library-search"
                        and not worker.is_finished
                        for worker in screen.workers
                    ),
                    what="superseded S returned",
                )
                await pilot.pause()
                assert selections == [], "search return activated an old dictionary row"
                assert screen._window_state.landing_owed()
                refresh_release.set()
                await refresh
                await wait_until(
                    pilot,
                    lambda: screen.state.selected_entity_kind == "dictionary"
                    and screen.state.selected_entity_id == target_id,
                    what="the authoritative first matching dictionary opened",
                )
                assert selections == [target_id]
            finally:
                search_release.set()
                refresh_release.set()
                if refresh is not None:
                    await refresh
    finally:
        db.close_connection()
