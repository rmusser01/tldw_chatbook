"""TASK-33910.35: only published Lore search rows satisfy Enter's landing."""

import asyncio
import threading

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.roleplay_frame_harness import select_roleplay_item, visible_item_ids
from Tests.UI.test_personas_lore_save_responsiveness import lore_app, submit, until
from tldw_chatbook.UI.Screens import personas_screen as module
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaEntitySelected,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]


@pytest.mark.parametrize("case", ["filtered-away", "included-not-first"])
@private_profile_test
async def test_save_refresh_keeps_enter_fenced_until_matching_rows_land(
    request, mock_app_instance, tmp_path, monkeypatch, case
):
    """A superseded read must not make Enter open the previous results."""
    search_entered, search_release = threading.Event(), threading.Event()
    refresh_entered, refresh_release = threading.Event(), threading.Event()
    async with lore_app(mock_app_instance, tmp_path, monkeypatch) as context:
        pilot, screen, detail, manager, first, second, hold, _ = context
        library = screen.query_one(PersonasLibraryPane)
        saved_id, target_id = first, second
        query = "Second"
        if case == "included-not-first":
            await select_roleplay_item(pilot, library, f"lore:{second}")
            await until(lambda: screen.state.selected_entity_id == str(second))
            saved_id, target_id = second, first
            query = "Lore"
        rows_before = list(visible_item_ids(library))
        original_read = screen._list_world_books_with_counts
        read_lock = threading.Lock()
        reads = []
        selected = []

        def held_read(actual_manager):
            with read_lock:
                index = len(reads)
                reads.append(index)
            if index == 0:
                search_entered.set()
                assert search_release.wait(90), "search read not released"
            elif index == 1:
                refresh_entered.set()
                assert refresh_release.wait(90), "save refresh read not released"
            return original_read(actual_manager)

        original_post = library.post_message

        def observe_post(message):
            if isinstance(message, PersonaEntitySelected):
                selected.append((message.entity_kind, message.entity_id))
            return original_post(message)

        monkeypatch.setattr(library, "post_message", observe_post)
        monkeypatch.setattr(module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 60.0)
        try:
            await submit(pilot, detail)
            await until(hold.entered.is_set)
            monkeypatch.setattr(screen, "_list_world_books_with_counts", held_read)
            search = library.query_one("#personas-library-search", Input)
            search.value = query
            search.focus()
            await until(
                lambda: screen.state.search_query == query
                and screen._search_debounce_timer is not None
            )
            await pilot.press("enter")
            await until(search_entered.is_set)
            assert screen._window_state.landing_owed()
            assert list(visible_item_ids(library)) == rows_before

            hold.release.set()
            await until(refresh_entered.is_set)
            assert manager.get_world_book(saved_id)["name"] == "Submitted Lore"
            search_release.set()
            await until(
                lambda: not any(
                    worker.group == "personas-library-search" and not worker.is_finished
                    for worker in pilot.app.workers
                )
            )
            assert not refresh_release.is_set()
            assert list(visible_item_ids(library)) == rows_before
            assert pilot.app.focused is library.item_list
            await pilot.press("enter")
            queued = asyncio.Event()
            screen.call_later(queued.set)
            await asyncio.wait_for(queued.wait(), 5)
            await pilot.pause()
            assert selected == [], (
                "Enter selected old rows before the matching save refresh landed: "
                f"{selected}"
            )
            assert screen._window_state.landing_owed()
            assert screen.state.selected_entity_id == str(saved_id)

            refresh_release.set()
            await until(
                lambda: f"lore:{target_id}" in visible_item_ids(library)
                and not screen._window_state.landing_owed()
            )
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            expected_rows = (
                [f"lore:{second}"]
                if case == "filtered-away"
                else [f"lore:{first}", f"lore:{second}"]
            )
            assert list(visible_item_ids(library)) == expected_rows
            assert library.item_list.highlighted_id() == f"lore:{target_id}"
            await until(lambda: screen.state.selected_entity_id == str(target_id))
            assert selected == [("lore", str(target_id))]
            saved = manager.get_world_book(saved_id)
            assert saved["name"] == "Submitted Lore" and saved["version"] == 2
            assert manager.get_world_book(target_id)["name"] == (
                "Second Lore" if case == "filtered-away" else "First Lore"
            )
            assert len(hold.calls) == 1
        finally:
            # Every intended RED must drain the real native reads and write too.
            search_release.set()
            refresh_release.set()
            hold.release.set()
