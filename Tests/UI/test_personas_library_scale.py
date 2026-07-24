"""P3a Task 4: PersonasScreen library-scale wiring — mounted integration
against a REAL CharactersRAGDB.

Proves the character library is now page-at-a-time (Task 1 DB seam + Task 3
pane controls wired through the screen):

- first page shows PERSONAS_LIBRARY_PAGE_SIZE rows + a "1-50 of N" page bar,
- next page is disjoint from the first,
- a tag filter narrows the count and resets the page offset,
- a sort change resets the page offset,
- importing a card whose name conflicts with an OFF-PAGE character reports
  "already existed" (drives the count-based pre-import existence check that
  replaced the page-cache membership snapshot).

Harness: mirrors ``test_personas_character_world_books_screen.py`` — a real
``CharactersRAGDB`` seeded via ``add_character_card`` and fed to the delegating
``PersonasTestApp`` from ``test_personas_dictionaries.py``. The screen's paged
loader (and its import/save paths) resolve the character DB through
``ccp_character_handler._default_character_db()`` →
``config.get_chachanotes_db_lazy()``, so the harness points that resolver at the
seeded DB (and also sets ``chachanotes_db`` on the mock app for good measure).
"""

import json
from contextlib import asynccontextmanager

import pytest

import tldw_chatbook.config as config_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.personas_screen import PERSONAS_LIBRARY_PAGE_SIZE
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp

pytestmark = pytest.mark.asyncio


@pytest.fixture
def scaled_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "p3a_scale.db", "test-client")
    yield db
    db.close_connection()


def _seed(db, count, *, tags_for=None):
    """Seed ``count`` characters named c000.. with optional per-index tags."""
    for i in range(count):
        db.add_character_card(
            {
                "name": f"c{i:03d}",
                "description": "",
                "tags": list(tags_for(i)) if tags_for else [],
            }
        )


def _pane_visible_rows(pane: PersonasLibraryPane) -> int:
    """Number of selectable library rows currently rendered."""
    return len(pane._row_lookup)


@asynccontextmanager
async def _personas(mock_app_instance, db, monkeypatch):
    """Mount PersonasScreen over a real CharactersRAGDB and settle mount work."""
    # The screen's character DB resolver (import/export/save + the paged loader)
    # goes through the lazy global getter; route it at the seeded DB.
    monkeypatch.setattr(config_module, "get_chachanotes_db_lazy", lambda: db)
    mock_app_instance.chachanotes_db = db
    mock_app_instance.chat_dictionary_scope_service = None
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        yield pilot, pilot.app.screen


async def test_first_page_and_next(mock_app_instance, scaled_db, monkeypatch):
    baseline = scaled_db.count_character_cards()
    _seed(scaled_db, 130)
    total = baseline + 130
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (
        pilot,
        screen,
    ):
        pane = screen.query_one(PersonasLibraryPane)
        assert screen.state.page_offset == 0
        assert screen._character_total == total
        # First page: exactly PAGE_SIZE rows + a "1-50 of N" page bar.
        assert _pane_visible_rows(pane) == PERSONAS_LIBRARY_PAGE_SIZE
        info = screen.query_one("#personas-library-page-info").renderable
        assert f"1-{PERSONAS_LIBRARY_PAGE_SIZE} of {total}" in str(info)
        assert screen.query_one("#personas-library-pagebar").display is True

        first_ids = set(pane._row_lookup and _row_ids(pane))
        await screen._on_page_changed_delta(1)
        await pilot.pause()
        assert screen.state.page_offset == PERSONAS_LIBRARY_PAGE_SIZE
        second_ids = set(_row_ids(pane))
        # Next page is a disjoint window.
        assert first_ids and second_ids and first_ids.isdisjoint(second_ids)


def _row_ids(pane: PersonasLibraryPane) -> list[str]:
    return [row.item_id for row in pane._row_lookup.values()]


async def test_tag_filter_narrows_count_and_resets_offset(
    mock_app_instance, scaled_db, monkeypatch
):
    _seed(
        scaled_db,
        10,
        tags_for=lambda i: ["hero"] if i < 3 else ["villain"],
    )
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (
        pilot,
        screen,
    ):
        screen.state.page_offset = PERSONAS_LIBRARY_PAGE_SIZE  # pretend we paged
        await screen._apply_tag_filter("hero")
        await pilot.pause()
        assert screen._character_total == 3
        assert screen.state.page_offset == 0
        assert screen.state.tag_filter == "hero"
        # Clearing the filter (None) restores the full count.
        await screen._apply_tag_filter(None)
        await pilot.pause()
        assert screen.state.tag_filter is None
        assert screen._character_total == scaled_db.count_character_cards()


async def test_sort_change_resets_offset(mock_app_instance, scaled_db, monkeypatch):
    _seed(scaled_db, 130)
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (
        pilot,
        screen,
    ):
        assert screen.state.sort_key == "name_asc"
        screen.state.page_offset = PERSONAS_LIBRARY_PAGE_SIZE
        await screen._cycle_sort()
        await pilot.pause()
        assert screen.state.page_offset == 0
        # Cycle advanced off the default sort.
        assert screen.state.sort_key != "name_asc"


async def test_character_page_reads_run_off_the_event_loop(
    mock_app_instance, scaled_db, monkeypatch
):
    """The count + page DB reads must be dispatched off the UI event loop."""
    import asyncio

    import tldw_chatbook.UI.Screens.personas_screen as ps

    _seed(scaled_db, 60)
    on_loop: list[bool] = []
    real_page = ps.get_character_page_for_ui
    real_count = ps.count_character_page

    def _spy_page(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real_page(*args, **kwargs)

    def _spy_count(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real_count(*args, **kwargs)

    monkeypatch.setattr(ps, "get_character_page_for_ui", _spy_page)
    monkeypatch.setattr(ps, "count_character_page", _spy_count)
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (pilot, screen):
        await screen._reload_character_page()
        await pilot.pause()
        assert on_loop, "the page/count DB reads must have run"
        assert all(seen is False for seen in on_loop)


async def test_character_search_defaults_to_relevance_sort(
    mock_app_instance, scaled_db, monkeypatch
):
    from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
        PersonaSearchChanged,
    )

    _seed(scaled_db, 5)
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (pilot, screen):
        assert screen.state.sort_key == "name_asc"
        screen._handle_search_changed(PersonaSearchChanged(query="c00"))
        assert screen.state.sort_key == "relevance"
        assert screen.state.page_offset == 0
        # Clearing the search reverts relevance (search-only) to name.
        screen._handle_search_changed(PersonaSearchChanged(query=""))
        assert screen.state.sort_key == "name_asc"
        await pilot.pause()


async def test_sort_cycle_excludes_relevance_outside_characters_mode(
    mock_app_instance, scaled_db, monkeypatch
):
    """task-463 #4: personas page in-memory (no FTS), so a searching
    non-characters mode must not offer a "Relevance" option that would be
    silently remapped to name_asc."""
    _seed(scaled_db, 5)
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (pilot, screen):
        screen.state.search_query = "abc"
        assert screen.state.active_mode == "characters"
        assert any(key == "relevance" for key, _ in screen._character_sort_cycle())
        screen.state.active_mode = "personas"
        assert all(key != "relevance" for key, _ in screen._character_sort_cycle())
        await pilot.pause()


async def test_page_nav_reuses_count_cache(mock_app_instance, scaled_db, monkeypatch):
    """Sort/page navigation must not recompute the count for an unchanged filter."""
    _seed(scaled_db, 130)
    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (
        pilot,
        screen,
    ):
        cache_key = screen._count_cache_key
        assert cache_key == (None, None)
        await screen._on_page_changed_delta(1)
        await pilot.pause()
        # Page nav within the same (search, tag) keeps the cached key.
        assert screen._count_cache_key == cache_key


async def test_import_offpage_name_conflict_message(
    mock_app_instance, scaled_db, tmp_path, monkeypatch
):
    """A name conflict with an OFF-PAGE character reports 'already existed',
    not 'imported new' — the page-cache snapshot would have missed it."""
    # Seed enough alphabetically-earlier characters that the conflict target
    # ("Zzz Offpage") lands beyond the first page.
    _seed(scaled_db, 60)
    conflict_name = "Zzz Offpage"
    scaled_db.add_character_card({"name": conflict_name, "description": "orig"})

    # A complete V2 card (all required fields) so the real importer parses it and
    # only the duplicate NAME triggers the conflict path.
    card_path = tmp_path / "conflict.json"
    card_path.write_text(
        json.dumps(
            {
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    "name": conflict_name,
                    "description": "reimport",
                    "personality": "",
                    "scenario": "",
                    "first_mes": "Hello.",
                    "mes_example": "",
                },
            }
        ),
        encoding="utf-8",
    )

    async with _personas(mock_app_instance, scaled_db, monkeypatch) as (pilot, screen):
        # The conflict character is not on page 0 (name_asc), proving the fix
        # cannot lean on the page cache.
        pane = screen.query_one(PersonasLibraryPane)
        assert conflict_name not in {
            row.name for row in pane._row_lookup.values()
        }

        notes: list[tuple[str, str]] = []
        monkeypatch.setattr(
            screen, "_notify", lambda msg, sev="warning": notes.append((msg, sev))
        )

        selected: list[str] = []

        async def _fake_select(entity_id, entity_name):
            selected.append(str(entity_id))

        monkeypatch.setattr(screen, "_select_character", _fake_select)

        before = scaled_db.count_character_cards()
        await screen._import_character_from_path(str(card_path))
        await pilot.pause()

        # No new row was created (conflict returned the existing id).
        assert scaled_db.count_character_cards() == before
        assert notes, "import must surface a notification"
        message = notes[-1][0].lower()
        assert "already existed" in message
        assert "imported" not in message or "already" in message
