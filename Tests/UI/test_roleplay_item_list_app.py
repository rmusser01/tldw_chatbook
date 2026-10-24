"""Roleplay frame B2a: the items list mounted on a real ChaChaNotes (spec 1.5.6).

Database windows, load-through, the fence and reload-from-anchor on the
styled mock tier and the DB-seeded full tier (348 and 5,000 characters), the
CI counts per settled keystroke (AC#8, AC#10), the marks bar (AC#5), the
persona exhaustion paging (AC#6) and the visit-lifecycle hooks (AC#11).
Mounts ``PersonasScreen``: bootstrap profile, admission-sensitive step.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.worker import WorkerState

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.Persona_Modules.roleplay_frame_state as frame_state
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from Tests.UI.roleplay_frame_harness import (
    VOLUME_SIZES,
    StyledRoleplayMockApp,
    highlighted_item_id,
    item_at,
    open_styled_roleplay,
    open_volume_db,
    painted_rows,
    roleplay_full_app,
    select_roleplay_item,
    settle,
    styled_tiers,
    visible_item_ids,
    volume_db_path,
    wait_until,
)
from tldw_chatbook.Character_Chat import Character_Chat_Lib
from tldw_chatbook.Character_Chat.character_events import CharacterCardChanged
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
    RoleplayItemList,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

SIZE = (120, 36)


@pytest.fixture
def volume_348(tmp_path_factory, tmp_path):
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 348))
    yield db
    db.close_connection()


def _db_ids(db) -> list[str]:
    """Every visible character's option id, in the list's name order."""
    rows = db.list_character_cards_page(limit=100_000, offset=0)
    return [f"character:{row['id']}" for row in rows]


async def _to_the_end(pilot, screen) -> None:
    item_list = screen.query_one(RoleplayItemList)
    item_list.focus()
    await pilot.press("end")
    await wait_until(
        pilot,
        lambda: not item_list.more_available,
        what="every window loaded",
    )
    await settle(pilot)


# ---- AC#7: the window rules on a real database ----


async def test_a_selection_past_row_200_restores(mock_app_instance, volume_348):
    target = volume_348.list_character_cards_page(limit=1, offset=300)[0]
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        assert len(visible_item_ids(screen)) == 200
        screen._pending_restore = {
            "kind": "character",
            "id": str(target["id"]),
            "name": target["name"],
            "preview": None,
        }
        await screen._apply_pending_restore()
        option_id = f"character:{target['id']}"
        await wait_until(
            pilot,
            lambda: highlighted_item_id(screen) == option_id,
            what="the restored row past row 200",
        )
        assert screen.query_one(RoleplayItemList).active_id == option_id
        assert len(visible_item_ids(screen)) > 300


async def test_an_insert_and_a_delete_ahead_then_the_end_keep_every_id_once(
    mock_app_instance, volume_348
):
    """RP-080's class: writes ahead of the loaded end reload from the anchor."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        cursor = visible_item_ids(screen)[150]
        await select_roleplay_item(pilot, screen, cursor, load=False)
        volume_348.add_character_card({"name": "Aaron Early", "description": "x"})
        await screen.refresh_character_library_list(None)
        await settle(pilot)
        assert highlighted_item_id(screen) == cursor
        doomed = volume_348.list_character_cards_page(limit=1, offset=5)[0]
        volume_348.soft_delete_character_card(doomed["id"], doomed["version"])
        await screen.refresh_character_library_list(None)
        await settle(pilot)
        assert highlighted_item_id(screen) == cursor
        await _to_the_end(pilot, screen)
        ids = visible_item_ids(screen)
        assert len(ids) == len(set(ids))
        assert list(ids) == _db_ids(volume_348)
        assert f"character:{doomed['id']}" not in ids


async def test_deleting_the_cursor_row_past_row_200_leaves_the_cursor_on_its_neighbour(
    mock_app_instance, volume_348
):
    """Spec 1.5.6 "Writes" and load-through: a delete reloads from the anchor,
    every loaded row stays loaded, and the cursor takes the deleted row's
    neighbour at the same index (never row 0)."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _to_the_end(pilot, screen)
        before = visible_item_ids(screen)
        doomed, neighbour = before[250], before[251]
        await select_roleplay_item(pilot, screen, doomed, load=False)
        record = volume_348.get_character_card_by_id(int(doomed.split(":")[1]))
        volume_348.soft_delete_character_card(record["id"], record["version"])
        await screen.refresh_character_library_list(None)
        await settle(pilot)
        after = visible_item_ids(screen)
        assert len(after) == len(before) - 1 and doomed not in after
        assert highlighted_item_id(screen) == neighbour
        assert after.index(neighbour) == 250


async def test_a_late_window_from_a_superseded_query_is_dropped(
    mock_app_instance, volume_348, monkeypatch
):
    gate = threading.Event()
    blocked = threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("search_term") and "0001" in kwargs["search_term"]:
            blocked.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        screen.state.search_query = "Character 0001"
        older = asyncio.create_task(screen._reload_character_page())
        await wait_until(pilot, blocked.is_set, what="the older window in flight")
        screen.state.search_query = "Character 0002"
        await screen._reload_character_page()
        gate.set()
        await older
        await settle(pilot)
        item_list = screen.query_one(RoleplayItemList)
        names = [item_list.row(oid).name for oid in visible_item_ids(screen)]
        assert names == ["Character 0002"]


async def test_scrolling_to_the_end_then_searching_and_clearing_skips_no_name(
    mock_app_instance, volume_348
):
    """AC#4 (RP-080: on dev about 41 names were skipped across page edges)."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _to_the_end(pilot, screen)
        search = screen.query_one("#personas-library-search", Input)
        search.value = "Character 03"
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 48,
            what="the search results",
        )
        search.value = ""
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 200,
            what="the cleared list's first window",
        )
        await _to_the_end(pilot, screen)
        ids = visible_item_ids(screen)
        assert list(ids) == _db_ids(volume_348)
        assert len(ids) == 348 + 1


# ---- Personas paged until exhausted (RP-093; Review Focus 2) ----


def _persona_service(records: list[dict], *, ignore_offset: bool = False):
    calls: list[tuple[int, int]] = []

    async def list_persona_profiles(
        mode="local", active_only=False, include_deleted=False, limit=100, offset=0
    ):
        calls.append((limit, offset))
        start = 0 if ignore_offset else offset
        return {"items": [dict(r) for r in records[start : start + limit]]}

    service = SimpleNamespace(
        list_persona_profiles=list_persona_profiles,
        get_persona_profile=AsyncMock(return_value=dict(records[0])),
    )
    return service, calls


def _personas(count: int) -> list[dict]:
    """Newest first, as the local store lists them; Default Narrator is oldest."""
    records = [
        {
            "id": f"p-{n:03d}",
            "name": f"Persona {n:03d}",
            "system_prompt": f"You are persona {n}.",
            "created_at": f"2026-10-01T00:{n % 60:02d}:00",
        }
        for n in range(count - 1, 0, -1)
    ]
    records.append(
        {
            "id": "p-narrator",
            "name": "Default Narrator",
            "system_prompt": "Narrate neutrally.",
            "created_at": "2026-01-01T00:00:00",
        }
    )
    return records


async def test_persona_paging_reads_until_a_short_page(mock_app_instance, volume_348):
    service, calls = _persona_service(_personas(114))
    mock_app_instance.character_persona_scope_service = service
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-personas")
        await settle(pilot)
        ids = visible_item_ids(screen)
        assert len(ids) == 114
        assert "persona:p-narrator" in ids
        assert calls == [(100, 0), (100, 100)]
        # AC#6 (spec 1.5.1): the pane's title counts every persona.
        pane = screen.query_one("#personas-library-pane")
        assert str(pane.border_title) == "Personas 114"


async def test_persona_paging_stops_when_a_page_adds_no_new_id(
    mock_app_instance, volume_348
):
    service, calls = _persona_service(_personas(150), ignore_offset=True)
    mock_app_instance.character_persona_scope_service = service
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-personas")
        await settle(pilot)
        assert len(visible_item_ids(screen)) == 100
        assert calls == [(100, 0), (100, 100)]


# ---- Titles and the 4-key find (spec 1.5.1, 1.6) ----


async def test_the_title_counts_the_kind_and_narrows_with_a_search(
    mock_app_instance, volume_348
):
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        pane = screen.query_one("#personas-library-pane")
        assert str(pane.border_title) == "Characters 349"
        screen.query_one("#personas-library-search", Input).value = "Character 03"
        await wait_until(
            pilot,
            lambda: str(pane.border_title) == "Characters 48 of 349",
            what="the narrowed title",
        )
        first = visible_item_ids(screen)[0]
        await select_roleplay_item(pilot, screen, first)
        await settle(pilot)
        assert str(pane.border_subtitle) == "#1 of 48"


async def test_enter_in_the_search_box_lands_on_the_first_match_before_the_debounce(
    mock_app_instance, volume_348
):
    """On dev, Enter within 0.2 s of the last keystroke used the previous
    result; B2a renders the pending search first (spec 1.6's 4-key find)."""
    target = volume_348.get_character_card_by_name("Character 0042")
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await pilot.press("ctrl+f")
        await pilot.press(*"Character 0042")
        await pilot.press("enter")
        await wait_until(
            pilot,
            lambda: highlighted_item_id(screen) == f"character:{target['id']}",
            what="the first match under the cursor",
        )
        assert screen.focused is screen.query_one(RoleplayItemList)


async def test_enter_after_the_debounce_fired_waits_for_its_window_and_lands_on_the_first_match(
    mock_app_instance, volume_348, monkeypatch
):
    """Enter after the 0.2 s debounce fired but before its window read returned:
    the screen waits for that render, so the cursor lands on the new first
    match. Never on dev (where Enter used the previous result); B2a's window
    path had it until this test: the cursor stayed on rows the render then
    replaced, and ``set_options`` cleared it, so a second Enter opened nothing.

    The read returns 1 s after Enter, inside the 2 s Enter waits at most
    (B2a's final review, I2). ``pilot.press`` returns only once the screen
    has handled Enter, so the read cannot be released after it: until then
    this test held the read for its whole 10 s timeout."""
    target = volume_348.get_character_card_by_name("Character 0042")
    page = personas_screen_module.get_character_page_for_ui
    release = threading.Event()
    reading = threading.Event()

    def held(db, **kwargs):
        if "0042" in str(kwargs.get("search_term") or ""):
            reading.set()
            release.wait(10)
        return page(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", held)
        await pilot.press("ctrl+f", *"Character 0042")
        await wait_until(pilot, reading.is_set, what="the debounced search's read")
        assert screen._search_debounce_timer is None
        returns = threading.Timer(1.0, release.set)
        returns.start()
        try:
            await pilot.press("enter")
        finally:
            returns.cancel()
            release.set()
        await wait_until(
            pilot,
            lambda: highlighted_item_id(screen) == f"character:{target['id']}",
            what="the first match under the cursor",
        )
        assert screen.focused is screen.query_one(RoleplayItemList)


@pytest.mark.parametrize("debounce", ["pending", "fired"])
async def test_enter_stops_waiting_for_a_slow_render_at_the_cap_and_keys_flow_meanwhile(
    mock_app_instance, volume_348, monkeypatch, debounce
):
    """B2a's final review (I2): Enter waits for its search's render inside the
    screen's message handler, and that holds every non-priority key (on a
    server source, for the server's round trip). New with B2a; bounded here:
    after ``_SEARCH_SUBMIT_WAIT_SECONDS`` Enter does what dev did (the cursor
    goes to the first row shown now), the screen takes keys again while the
    read is still held, and the slow render still lands its rows.

    ``pilot.press`` returns once the screen has handled the key, so its
    duration is how long Enter held the screen."""
    target = volume_348.get_character_card_by_name("Character 0042")
    cap = personas_screen_module._SEARCH_SUBMIT_WAIT_SECONDS
    page = personas_screen_module.get_character_page_for_ui
    release = threading.Event()
    reading = threading.Event()
    escapes: list[bool] = []

    def held(db, **kwargs):
        if "0042" in str(kwargs.get("search_term") or ""):
            reading.set()
            release.wait(15)
        return page(db, **kwargs)

    escape = PersonasScreen.action_personas_escape

    def record_escape(self) -> None:
        escapes.append(release.is_set())
        escape(self)

    monkeypatch.setattr(PersonasScreen, "action_personas_escape", record_escape)
    if debounce == "pending":
        # The search is still waiting on its timer when Enter arrives.
        monkeypatch.setattr(
            personas_screen_module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 60.0
        )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", held)
        try:
            await pilot.press("ctrl+f", *"Character 0042")
            if debounce == "fired":
                await wait_until(pilot, reading.is_set, what="the debounced read")
            else:
                assert screen._search_debounce_timer is not None
            started = time.monotonic()
            await pilot.press("enter")
            held_for = time.monotonic() - started
            assert reading.is_set() and not release.is_set()
            assert cap - 0.25 <= held_for < cap + 3.0
            assert screen.focused is item_list
            await pilot.press("escape")
            assert escapes == [False]
        finally:
            release.set()
        await wait_until(
            pilot,
            lambda: visible_item_ids(screen) == (f"character:{target['id']}",),
            what="the slow search's rows",
        )


# ---- Marks by id, the marks bar and the one delete modal (AC#5, R32; Task 5) ----


def _record_notices(monkeypatch, screen) -> list[str]:
    """Every message the screen notifies, in order (toasts are ``_notify``)."""
    seen: list[str] = []
    original = screen._notify

    def record(message, severity="warning", **kwargs):
        seen.append(message)
        return original(message, severity, **kwargs)

    monkeypatch.setattr(screen, "_notify", record)
    return seen


def _option_id(db, name: str) -> str:
    return f"character:{db.get_character_card_by_name(name)['id']}"


async def _mark_rows(pilot, screen, option_ids) -> None:
    screen.query_one(RoleplayItemList).focus()
    for option_id in option_ids:
        await select_roleplay_item(pilot, screen, option_id, load=False)
        await pilot.press("m")
    await settle(pilot)


def _bar(screen) -> str:
    """The marks bar's row as painted ("" while it is hidden)."""
    bar = screen.query_one("#personas-library-marks")
    if not bar.display:
        return ""
    region = bar.content_region
    return painted_rows(screen)[region.y][region.x : region.right].rstrip()


def _dialog_text(dialog: ConfirmationDialog) -> tuple[str, str, str]:
    title = str(dialog.query_one(".dialog-title").render())
    message = str(dialog.query_one(".dialog-message").render())
    confirm = str(dialog.query_one("#confirm-button").label)
    return title, message, confirm


async def test_marks_survive_search_and_sort_and_count_the_hidden_ones(
    mock_app_instance, volume_348
):
    """AC#5: on dev a search or a page change dropped every mark (RP-039)."""
    marked = [_option_id(volume_348, f"Character {n:04d}") for n in (1, 2, 3)]
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=(220, 55), character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, marked)
        assert _bar(screen).startswith("✓ 3 marked")
        screen.query_one("#personas-library-search", Input).value = "Character 0003"
        await wait_until(
            pilot,
            lambda: visible_item_ids(screen) == (marked[2],),
            what="the narrowed list",
        )
        await settle(pilot)
        assert _bar(screen).startswith("✓ 3 marked (2 hidden)")
        screen.query_one("#personas-library-search", Input).value = ""
        await wait_until(
            pilot, lambda: len(visible_item_ids(screen)) == 200, what="the full list"
        )
        screen.query_one(RoleplayItemList).focus()
        await pilot.press("s")
        await settle(pilot)
        assert screen.state.sort_key != "name_asc"
        assert _bar(screen).startswith("✓ 3 marked")
        assert screen._marked_rows and len(screen._marked_rows) == 3


async def test_bulk_delete_names_five_and_the_rest_and_deletes_only_the_marked(
    mock_app_instance, volume_348
):
    """R32: one modal names up to 5 items plus "and N more"; Cancel keeps all."""
    names = [f"Character {n:04d}" for n in range(1, 8)]
    marked = [_option_id(volume_348, name) for name in names]
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, marked)
        assert await pilot.click("#personas-library-marks-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal",
        )
        title, message, confirm = _dialog_text(pilot.app.screen)
        assert title == "Delete 7 marked"
        assert confirm == "Delete 7 marked"
        assert message.splitlines() == [
            "Delete these 7 characters? This cannot be undone here.",
            "",
            *[f"  {name}" for name in names[:5]],
            "  and 2 more",
        ]
        await pilot.press("escape")
        await settle(pilot)
        assert volume_348.count_character_cards() == 348 + 1
        assert len(screen._marked_rows) == 7
        # Rung 0: Escape is the modal's safe negative; focus goes back to its opener.
        assert pilot.app.focused is screen.query_one("#personas-library-marks-delete")
        assert await pilot.click("#personas-library-marks-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal again",
        )
        assert await pilot.click("#confirm-button")
        await wait_until(
            pilot,
            lambda: volume_348.count_character_cards() == 348 + 1 - 7,
            what="the seven deletes",
        )
        await settle(pilot)
        assert screen._marked_rows == ()
        assert _bar(screen) == ""
        assert not set(marked) & set(visible_item_ids(screen))
        # R-120 (I1): the bar hid under its own Delete button; focus stays on
        # the list (spec 4.11: "Delete, other rows only" -> stays on the list).
        assert screen.focused is screen.query_one(RoleplayItemList)


async def test_the_inspector_delete_acts_on_the_open_item_only(
    mock_app_instance, volume_348
):
    """On dev an active mark set silently retargeted the Inspector's Delete
    at the marked rows (RP-039); bulk Delete now lives in the marks bar."""
    opened = _option_id(volume_348, "Character 0010")
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=(160, 45), character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await select_roleplay_item(pilot, screen, opened)
        await settle(pilot)
        await _mark_rows(
            pilot,
            screen,
            [_option_id(volume_348, f"Character {n:04d}") for n in (1, 2)],
        )
        assert await pilot.click("#personas-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal",
        )
        assert _dialog_text(pilot.app.screen) == (
            "Delete",
            "Delete Character 0010? This cannot be undone here.",
            "Delete",
        )
        await pilot.press("escape")
        await settle(pilot)
        assert len(screen._marked_rows) == 2
        assert item_at(screen, opened).active


async def test_space_turns_a_lore_book_off_with_the_undo_toast_and_on_again(
    mock_app_instance, volume_348, monkeypatch
):
    """Spec 4.6 / R33: on dev Space toggled only dictionary rows, untoasted."""
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager

    manager = WorldBookManager(volume_348)
    book = f"lore:{manager.create_world_book('Codex Arcana')}"
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        assert await pilot.click("#personas-mode-lore")
        await wait_until(
            pilot, lambda: book in visible_item_ids(screen), what="the lore list"
        )
        screen.query_one(RoleplayItemList).focus()
        await select_roleplay_item(pilot, screen, book, load=False)
        await pilot.press("space")
        await wait_until(
            pilot, lambda: item_at(screen, book).state == "off", what="the off word"
        )
        assert not manager.get_world_book(int(book.split(":")[1]))["enabled"]
        assert notices == ["Codex Arcana turned off · space to undo"]
        await pilot.press("space")
        await wait_until(
            pilot, lambda: item_at(screen, book).state == "on", what="the on word"
        )
        assert highlighted_item_id(screen) == book
        assert notices == ["Codex Arcana turned off · space to undo"]


async def test_the_footer_advertises_space_where_it_toggles(
    mock_app_instance, volume_348
):
    """Spec 4.6, survey R-040: Space toggles dictionaries and lore books.

    On dev the footer advertised ``space toggle`` in dictionaries mode only,
    the one mode where Space acted. Lore books toggle too since this slice, so
    the hint follows them; characters never advertise it.
    """
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        footer = screen.query_one(AppFooterStatus)
        assert "space" not in footer.shortcut_text
        assert await pilot.click("#personas-mode-lore")
        await settle(pilot)
        assert "space toggle" in footer.shortcut_text
        assert await pilot.click("#personas-mode-characters")
        await settle(pilot)
        assert "space" not in footer.shortcut_text


def _dictionary_service(*books: tuple[str, object]) -> SimpleNamespace:
    """A dictionaries scope service listing ``(name, entry_count)`` books (all on)."""
    records = [
        {"id": number, "name": name, "entry_count": count, "enabled": True}
        for number, (name, count) in enumerate(books, start=1)
    ]
    return SimpleNamespace(
        list_dictionaries=AsyncMock(return_value={"dictionaries": records})
    )


@pytest.mark.parametrize("mode", ["lore", "dictionaries"])
async def test_end_reaches_the_last_book_after_a_partial_character_list(
    mock_app_instance, volume_348, mode
):
    """B2a's final review (I1): with 348 characters the first window leaves
    rows past its end, and End in Lore books or Chat dictionaries then asked
    for a character window the screen drops there, so the cursor never moved.
    New with B2a (dev's list had no End); fixed here. Each book render also
    says itself that no rows lie past its end."""
    names = [f"Book {letter}" for letter in "ABCDE"]
    if mode == "lore":
        from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager

        manager = WorldBookManager(volume_348)
        for name in names:
            manager.create_world_book(name)
    else:
        mock_app_instance.chat_dictionary_scope_service = _dictionary_service(
            *((name, 2) for name in names)
        )
    kind = frame_state.KIND_FOR_MODE[mode]
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        assert item_list.more_available
        assert await pilot.click(f"#personas-mode-{mode}")
        await wait_until(
            pilot,
            lambda: [item_list.row(i).name for i in visible_item_ids(screen)] == names,
            what=f"the {mode} list",
        )
        assert all(i.startswith(f"{kind}:") for i in visible_item_ids(screen))
        item_list.focus()
        await select_roleplay_item(pilot, screen, index=0, load=False)
        await pilot.press("end")
        await settle(pilot)
        assert highlighted_item_id(screen) == visible_item_ids(screen)[-1]
        item_list.set_more_available(True)
        await screen._render_search_query(query="", mode=mode)
        assert not item_list.more_available


async def test_a_dictionary_whose_count_is_not_a_number_still_lists(
    mock_app_instance, volume_348
):
    """B2a's final review (M10): one non-numeric ``entry_count`` failed the
    whole dictionaries render. New with B2a (dev only formatted the value);
    fixed here: that row lists without a count, and every other row lists."""
    mock_app_instance.chat_dictionary_scope_service = _dictionary_service(
        ("Odd Rules", "n/a"), ("Plain Rules", 3)
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-dictionaries")
        await settle(pilot)
        item_list = screen.query_one(RoleplayItemList)
        assert visible_item_ids(screen) == ("dictionary:1", "dictionary:2")
        assert item_list.row("dictionary:1").meta is None
        assert item_list.row("dictionary:2").meta == "3 rules"


# ---- CI counts per settled keystroke, 348 and 5,000 (AC#8, AC#10; Task 8) ----
#
# Spec 5.2 B2a (a): deterministic counts, never a wall-clock threshold taken
# through pilot.press() + pause() (lessons-testing-evidence). The wall clock
# is the live harness's job (Task 13).


class _Work:
    """What one settled keystroke, resize or End did, counted at its seams."""

    def __init__(self) -> None:
        self.armed = False
        self.reset()

    def reset(self) -> None:
        self.windows: list[tuple[int, int]] = []
        self.set_options = 0
        self.add_options = 0
        self.prompts = 0
        self.mounts = 0
        self.recomposes = 0


@pytest.fixture
def work(monkeypatch) -> _Work:
    tally = _Work()
    page = personas_screen_module.get_character_page_for_ui

    def window(db, **kwargs):
        if tally.armed:
            tally.windows.append((kwargs["limit"], kwargs["offset"]))
        return page(db, **kwargs)

    set_options = RoleplayItemList.set_options
    add_options = RoleplayItemList.add_options
    build = frame_state.build_item_prompt
    register = App._register
    recompose = Widget.recompose

    def counted_set_options(self, options):
        tally.set_options += tally.armed
        self._counting_set_options = True  # its own add_options is not an append
        try:
            return set_options(self, options)
        finally:
            self._counting_set_options = False

    def counted_add_options(self, options):
        if not getattr(self, "_counting_set_options", False):
            tally.add_options += tally.armed
        return add_options(self, options)

    def counted_build(**kwargs):
        tally.prompts += tally.armed
        return build(**kwargs)

    def counted_register(self, parent, *widgets, **kwargs):
        if tally.armed:
            tally.mounts += len(widgets)
        return register(self, parent, *widgets, **kwargs)

    async def counted_recompose(self, *args, **kwargs):
        tally.recomposes += tally.armed
        return await recompose(self, *args, **kwargs)

    monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", window)
    monkeypatch.setattr(RoleplayItemList, "set_options", counted_set_options)
    monkeypatch.setattr(RoleplayItemList, "add_options", counted_add_options)
    monkeypatch.setattr(frame_state, "build_item_prompt", counted_build)
    monkeypatch.setattr(App, "_register", counted_register)
    monkeypatch.setattr(Widget, "recompose", counted_recompose)
    return tally


@pytest.fixture(params=VOLUME_SIZES, ids=lambda count: f"{count}-characters")
def volume_db(request, tmp_path_factory, tmp_path):
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, request.param))
    yield db
    db.close_connection()


class _RecomposeProbe(Widget):
    """A widget whose reactive recomposes it: Textual's own recompose path."""

    value = reactive(0, recompose=True)

    def compose(self) -> ComposeResult:
        yield Static(f"value {self.value}")


async def test_the_mount_and_recompose_counters_count_a_known_mount_and_recompose(
    mock_app_instance, volume_348, work
):
    """Positive control for the "0 mounts, 0 recomposes" counts below (Task 8
    review; Global 38): the same seams count a known mount and a known
    recompose, so a zero there is a measurement, never a seam that is dead."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        work.armed = True
        await screen.mount(Static("positive control"))
        await settle(pilot)
        work.armed = False
        assert work.mounts >= 1 and work.recomposes == 0
        probe = _RecomposeProbe()
        await screen.mount(probe)
        await settle(pilot)
        work.reset()
        work.armed = True
        probe.value = 1
        await settle(pilot)
        work.armed = False
        assert work.recomposes == 1
        assert work.mounts >= 1  # the recompose mounts the probe's new child
        assert str(probe.query_one(Static).render()) == "value 1"


async def test_a_search_keystroke_reads_one_window_and_mounts_nothing(volume_db, work):
    """AC#8: one window query (<=200 rows), one set_options, prompts built ==
    the window's rows, 0 mounts, 0 recomposes. On dev every keystroke cleared
    and remounted the page's rows (PERF-22's per-row widgets)."""
    async with roleplay_full_app(size=SIZE, character_db=volume_db) as pilot:
        screen = pilot.app.screen
        screen.query_one("#personas-library-search", Input).focus()
        await settle(pilot)
        work.armed = True
        await pilot.press("C")
        await wait_until(pilot, lambda: bool(work.windows), what="the window query")
        await settle(pilot)
        work.armed = False
        rows = len(visible_item_ids(screen))
        assert screen.state.search_query == "C"
        assert work.windows == [(200, 0)]
        assert (work.set_options, work.add_options) == (1, 0)
        assert work.prompts == rows == 200
        assert (work.mounts, work.recomposes) == (0, 0)


async def test_a_cursor_keystroke_reads_nothing_and_builds_nothing(volume_db, work):
    async with roleplay_full_app(size=SIZE, character_db=volume_db) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        item_list.focus()
        await settle(pilot)
        # The first paint read one window and rows remain: a near-end check
        # that ignored the cursor would have read the rest of 348 already.
        assert len(visible_item_ids(screen)) == 200 and item_list.more_available
        work.armed = True
        await pilot.press("down")
        await settle(pilot)
        work.armed = False
        assert work.windows == []
        assert (work.set_options, work.add_options, work.prompts) == (0, 0, 0)
        assert (work.mounts, work.recomposes) == (0, 0)


async def test_a_slow_window_read_never_stalls_the_event_loop(volume_db, monkeypatch):
    """AC#8's stall check, in the discriminating heartbeat form
    (test_personas_deferred_center_views.py): the read is made deliberately
    slow, so the heartbeat fails if it ever runs on the event loop."""
    page = personas_screen_module.get_character_page_for_ui

    def slow(db, **kwargs):
        time.sleep(0.6)
        return page(db, **kwargs)

    async with roleplay_full_app(size=SIZE, character_db=volume_db) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", slow)
        screen.query_one("#personas-library-search", Input).focus()
        await settle(pilot)
        gaps: list[float] = []
        running = True

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            previous = loop.time()
            while running:
                await asyncio.sleep(0.005)
                now = loop.time()
                gaps.append(now - previous)
                previous = now

        beat = asyncio.create_task(heartbeat())
        await pilot.press("C")
        await wait_until(
            pilot,
            lambda: (
                " of " in str(screen.query_one("#personas-library-pane").border_title)
                and not _window_workers(pilot.app)
            ),
            what="the slow window",
        )
        await asyncio.sleep(0.7)
        running = False
        await beat
        assert gaps
        # A read on the loop would stall it for >= 0.6 s; 0.45 leaves room for
        # a loaded CI runner without letting that through.
        assert max(gaps) < 0.45, f"event loop stalled for {max(gaps):.3f}s"


async def test_with_5000_rows_loaded_a_resize_and_end_do_no_data_work(
    tmp_path_factory, tmp_path, work
):
    """AC#10 (RC-6): scrolled to the end with every window resident, a
    120 <-> 160 resize rebuilds only the prompts the new width changes, each
    once, with at most one set_options; End then does nothing at all."""
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 5000))
    for n in range(3):
        db.add_character_card({"name": f"Zz {'Long ' * 9}Name {n}", "description": "x"})
    try:
        async with roleplay_full_app(size=SIZE, character_db=db) as pilot:
            screen = pilot.app.screen
            item_list = screen.query_one(RoleplayItemList)
            item_list.focus()
            work.armed = True
            await pilot.press("end")
            await wait_until(
                pilot, lambda: not item_list.more_available, what="every window"
            )
            await settle(pilot)
            work.armed = False
            total = 5000 + 1 + 3
            assert len(visible_item_ids(screen)) == total
            assert work.windows == [(200, offset) for offset in range(200, total, 200)]
            assert (work.set_options, work.add_options) == (0, 1)
            assert work.prompts == total - 200
            assert (work.mounts, work.recomposes) == (0, 0)
            for size in ((160, 45), (120, 36)):
                before = {
                    oid: item_list.prompt_for(oid) for oid in item_list.loaded_ids()
                }
                work.reset()
                work.armed = True
                await pilot.resize_terminal(*size)
                await settle(pilot)
                work.armed = False
                width = item_list.scrollable_content_region.width
                changed = sum(
                    prompt.needs_rebuild(width=width, density=item_list.density)
                    for prompt in before.values()
                )
                assert 3 <= changed < total
                assert work.prompts == changed
                assert work.set_options == 1 and work.add_options == 0
                assert work.windows == []
                assert (work.mounts, work.recomposes) == (0, 0)
            work.reset()
            work.armed = True
            await pilot.press("end")
            await settle(pilot)
            work.armed = False
            assert highlighted_item_id(screen) == visible_item_ids(screen)[-1]
            assert work.windows == []
            assert (work.set_options, work.add_options, work.prompts) == (0, 0, 0)
            assert (work.mounts, work.recomposes) == (0, 0)
    finally:
        db.close_connection()


# ---- The visit-lifecycle hooks (spec 4.11, AC#11; Task 7) ----


class _ReusedRoleplayApp(StyledRoleplayMockApp):
    """Roleplay as an installed screen, so a switch suspends it (PERF-22's path)."""

    def on_mount(self, event: events.Mount) -> None:
        event.prevent_default()  # not the parent's push of a fresh screen
        self.install_screen(PersonasScreen(self), name="roleplay")
        self.install_screen(Screen(), name="elsewhere")
        self.push_screen("roleplay")


def _window_workers(app) -> list:
    return [
        worker
        for worker in app.workers
        if worker.group == "roleplay-list-window"
        and worker.state in (WorkerState.PENDING, WorkerState.RUNNING)
    ]


async def test_a_modal_over_roleplay_keeps_the_marks(mock_app_instance, volume_348):
    """A pushed modal suspends Roleplay, but it is no leave (spec R16)."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[:2])
        await pilot.app.push_screen(ConfirmationDialog(title="t", message="m"))
        await settle(pilot)
        pilot.app.pop_screen()
        await settle(pilot)
        assert len(screen._marked_rows) == 2
        assert _bar(screen)


async def test_leaving_by_suspend_clears_the_marks_and_stops_the_window_worker(
    mock_app_instance, volume_348, monkeypatch
):
    gate = threading.Event()
    reading = threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset"):
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[:3])
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        screen.query_one(RoleplayItemList).focus()
        await pilot.press("end")
        await wait_until(pilot, reading.is_set, what="the window read")
        assert _window_workers(app)
        try:
            await app.switch_screen("elsewhere")
            await settle(pilot)
            assert screen not in app.screen_stack and screen.is_attached
            assert _window_workers(app) == []
            assert screen._marked_rows == ()
            assert not screen.query_one(RoleplayItemList).marked_rows("character")
        finally:
            gate.set()
        await app.switch_screen("roleplay")
        await settle(pilot)
        assert _bar(screen) == ""
        assert _window_workers(app) == []  # the resume starts none either
        ids = visible_item_ids(screen)
        assert len(ids) == 200 and len(ids) == len(set(ids))  # the late window dropped


@pytest.mark.parametrize("trigger", ["end", "near_end"])
async def test_a_list_loaded_before_a_leave_still_loads_through_after_the_return(
    mock_app_instance, volume_348, trigger
):
    """The leave bumps the page generation; the loaded rows stay the list's rows."""
    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        assert len(visible_item_ids(screen)) == 200
        await app.switch_screen("elsewhere")
        await settle(pilot)
        await app.switch_screen("roleplay")
        await settle(pilot)
        item_list = screen.query_one(RoleplayItemList)
        item_list.focus()
        if trigger == "end":
            await pilot.press("end")
        else:
            item_list.highlighted = 195
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) > 200,
            what="the next window after the return",
        )
        if trigger == "end":
            await wait_until(
                pilot,
                lambda: len(visible_item_ids(screen)) == 349,
                what="the whole list",
            )
        await settle(pilot)
        ids = list(visible_item_ids(screen))
        assert len(ids) == len(set(ids))


async def test_a_near_end_read_cancelled_by_a_leave_does_not_stall_the_list(
    mock_app_instance, volume_348, monkeypatch
):
    """The leave cancels an append in flight; the request latch must not stay set."""
    gate = threading.Event()
    reading = threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset") and not gate.is_set():
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        item_list.focus()
        item_list.highlighted = 195
        await wait_until(pilot, reading.is_set, what="the near-end read")
        try:
            await app.switch_screen("elsewhere")
            await settle(pilot)
        finally:
            gate.set()
        await app.switch_screen("roleplay")
        await settle(pilot)
        item_list.focus()
        item_list.highlighted = 190
        await settle(pilot)
        item_list.highlighted = 198
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) > 200,
            what="a near-end window after the return",
        )


async def test_a_replace_cancelled_by_a_leave_reloads_on_the_return(
    mock_app_instance, volume_348, monkeypatch
):
    gate = threading.Event()
    blocked = threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("search_term") and kwargs.get("offset") == 0:
            blocked.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        screen.state.search_query = "Character 03"
        reload = asyncio.create_task(screen._reload_character_page())
        await wait_until(pilot, blocked.is_set, what="the search's first window")
        try:
            await app.switch_screen("elsewhere")
            await settle(pilot)
        finally:
            gate.set()
        await asyncio.gather(reload, return_exceptions=True)
        await app.switch_screen("roleplay")
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 48,
            what="the search's results after the return",
        )
        assert [f"character:{row['id']}" for row in screen._characters] == list(
            visible_item_ids(screen)
        )


async def test_a_modal_over_roleplay_starts_no_reload(mock_app_instance, volume_348):
    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        screen = pilot.app.screen
        reloads: list[int] = []
        real = screen._reload_character_page

        async def counted(**kwargs):
            reloads.append(1)
            await real(**kwargs)

        screen._reload_character_page = counted
        await pilot.app.push_screen(ConfirmationDialog(title="t", message="m"))
        await settle(pilot)
        pilot.app.pop_screen()
        await settle(pilot)
        assert reloads == []
        assert len(visible_item_ids(screen)) == 200


def _keys_naming_marks(state) -> list[str]:
    """Every key, at any depth of a ``save_state`` snapshot, that names marks."""
    if isinstance(state, dict):
        found = [key for key in state if "mark" in str(key).lower()]
        for value in state.values():
            found += _keys_naming_marks(value)
        return found
    if isinstance(state, (list, tuple)):
        return [key for value in state for key in _keys_naming_marks(value)]
    return []


async def test_leaving_and_returning_clears_the_marks(volume_348, monkeypatch):
    """The real routes: Roleplay unmounts on leave today, and the hook runs then."""
    ended: list[PersonasScreen] = []
    end_visit = PersonasScreen._end_visit

    def record(self: PersonasScreen) -> None:
        ended.append(self)
        end_visit(self)

    monkeypatch.setattr(PersonasScreen, "_end_visit", record)
    async with roleplay_full_app(size=SIZE, character_db=volume_348) as pilot:
        app = pilot.app
        first = app.screen
        await _mark_rows(pilot, first, visible_item_ids(first)[:2])
        assert len(first._marked_rows) == 2
        assert not _keys_naming_marks(first.save_state())  # R16: never saved
        await pilot.press("ctrl+1")
        await wait_until(
            pilot,
            lambda: not isinstance(app.screen, PersonasScreen),
            what="Home",
        )
        await settle(pilot)
        assert first in ended and first._marked_rows == ()
        await pilot.press("ctrl+4")
        await wait_until(
            pilot,
            lambda: (
                isinstance(app.screen, PersonasScreen)
                and len(visible_item_ids(app.screen)) == 200
            ),
            what="Roleplay again",
        )
        await settle(pilot)
        assert app.screen._marked_rows == ()
        assert _bar(app.screen) == ""
        assert _window_workers(app) == []


async def test_quitting_from_roleplay_runs_the_leave_hook_on_unmount(
    mock_app_instance, volume_348, monkeypatch
):
    """The unmount call is the hook's net for a removal that sends no suspend.

    Textual posts ``ScreenSuspend`` to every screen it replaces before
    removing it, so a leave reaches the hook through ``on_screen_suspend``
    first. Only the app's shutdown (``App._close_all``) prunes a screen with
    no suspend: quitting with marks and a window read in flight still ends
    the visit, and the window is fenced.
    """
    gate = threading.Event()
    reading = threading.Event()
    original = personas_screen_module.get_character_page_for_ui
    suspended: list[PersonasScreen] = []
    on_suspend = PersonasScreen.on_screen_suspend

    def gated(db, **kwargs):
        if kwargs.get("offset"):
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    def record_suspend(self: PersonasScreen) -> None:
        suspended.append(self)
        on_suspend(self)

    monkeypatch.setattr(PersonasScreen, "on_screen_suspend", record_suspend)
    try:
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=volume_348
        ) as pilot:
            app = pilot.app
            screen = app.screen
            item_list = screen.query_one(RoleplayItemList)
            await _mark_rows(pilot, screen, visible_item_ids(screen)[:3])
            monkeypatch.setattr(
                personas_screen_module, "get_character_page_for_ui", gated
            )
            item_list.focus()
            await pilot.press("end")
            await wait_until(pilot, reading.is_set, what="the window read")
            assert _window_workers(app)
            generation = screen._window_state.generation
            assert len(screen._marked_rows) == 3
    finally:
        gate.set()
    assert suspended == []
    # The pane and its list are pruned before the screen's own unmount, so
    # the screen's mark set is what the hook clears here.
    assert screen._marked_rows == ()
    assert screen._window_state.generation == generation + 1
    assert _window_workers(app) == []


async def test_escape_clears_the_marks(mock_app_instance, volume_348):
    """The bar advertises "Clear esc"; Escape on the list clears this kind's marks."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[:2])
        assert _bar(screen)
        await pilot.press("escape")
        await settle(pilot)
        assert _bar(screen) == ""
        assert screen._marked_rows == ()
        assert screen.focused is screen.query_one(RoleplayItemList)


@styled_tiers
@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=["120x36", "80x24"])
async def test_the_marks_bar_paints_every_action_it_shows_whole(
    styled_tier, size, mock_app_instance, volume_348
):
    """Spec 1.5.5: the bar degrades by width, never by clipping.

    The bar's content is two cells narrower than the pane's (its padding), so
    a ladder fed the pane's width cut Clear to "Clea" at 120x36 and painted no
    Delete or Clear at 80x24. Every action shown lies inside the bar with its
    whole label painted, and Delete, which stays down to the count alone,
    opens the one delete modal at both sizes.
    """
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=size, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[1:3])
        bar = screen.query_one("#personas-library-marks")
        inside = bar.content_region
        row = painted_rows(screen)[inside.y]
        shown = [
            button
            for button in screen.query(".personas-library-marks-action")
            if button.display
        ]
        assert _bar(screen).startswith("✓ 2")
        for button in shown:
            label = str(button.label)
            region = button.region
            assert inside.x <= region.x and region.right <= inside.right, (
                label,
                region,
                inside,
            )
            assert row[region.x : region.right] == label
        assert "personas-library-marks-delete" in [button.id for button in shown]
        assert await pilot.click("#personas-library-marks-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal",
        )
        assert _dialog_text(pilot.app.screen)[0] == "Delete 2 marked"
        await pilot.press("escape")
        await settle(pilot)


@styled_tiers
@pytest.mark.parametrize(
    ("marks", "ascii_glyphs"),
    [(12, False), (2, True)],
    ids=["12-marks", "ascii-2-marks"],
)
async def test_delete_stays_whole_and_clickable_when_the_count_cannot_fit(
    styled_tier, marks, ascii_glyphs, mock_app_instance, volume_348
):
    """At 80x24 the bar has 10 content cells; the count yields before Delete.

    ``✓ 12 Delete`` needs 11 cells and ``[x] 2 Delete`` needs 12, so the
    count moves into Delete's tooltip and Delete stays, whole and clickable.
    The Task 5 review found the count alone there, with no bulk action at
    the standard terminal size.
    """
    async with open_styled_roleplay(
        styled_tier,
        mock_app_instance,
        size=(80, 24),
        ascii_glyphs=ascii_glyphs,
        character_db=volume_348,
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[1 : 1 + marks])
        inside = screen.query_one("#personas-library-marks").content_region
        delete = screen.query_one("#personas-library-marks-delete")
        assert delete.display
        region = delete.region
        assert inside.x <= region.x and region.right <= inside.right, (region, inside)
        assert painted_rows(screen)[inside.y][region.x : region.right] == "Delete"
        assert _bar(screen) == "Delete"
        assert str(delete.tooltip) == f"{marks} marked"
        assert await pilot.click("#personas-library-marks-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal",
        )
        assert _dialog_text(pilot.app.screen)[0] == f"Delete {marks} marked"
        await pilot.press("escape")
        await settle(pilot)


@styled_tiers
@pytest.mark.parametrize("size", [(100, 24), (120, 24)], ids=["100x24", "120x24"])
async def test_twelve_marks_at_24_rows_paint_the_count_and_a_whole_delete(
    styled_tier, size, mock_app_instance, volume_348
):
    """At 100-120 columns and 24 rows the pane has two rows under its
    controls. The list's three-row minimum pushed the marks bar below the
    pane, where nothing of it painted (Task 5's re-review probe read it as
    ""; new with B2a, whose bar this is). The list now takes the rows left,
    so the bar paints the ladder's rung for its own width, count first, with
    Delete whole inside it."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=size, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[1:13])
        assert len(screen._marked_rows) == 12
        pane = screen.query_one("#personas-library-pane").content_region
        bar = screen.query_one("#personas-library-marks")
        item_list = screen.query_one(RoleplayItemList)
        assert pane.contains_region(bar.region), (bar.region, pane)
        assert pane.contains_region(item_list.region), (item_list.region, pane)
        inside = bar.content_region
        view = frame_state.marks_bar_view(12, 0, inside.width, can_export=True)
        assert view.count_text.startswith("✓ 12"), (inside, view)
        painted = view.count_text + "".join(
            view.separator + label for label in view.labels
        )
        assert _bar(screen) == painted, (inside, view)
        delete = screen.query_one("#personas-library-marks-delete")
        region = delete.region
        assert inside.x <= region.x and region.right <= inside.right, (region, inside)
        row = painted_rows(screen)[inside.y]
        assert row[region.x : region.right] == str(delete.label)
        assert await pilot.click("#personas-library-marks-delete")
        await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, ConfirmationDialog)
                and bool(pilot.app.screen.query("#confirm-button"))
            ),
            what="the delete modal",
        )
        assert _dialog_text(pilot.app.screen)[0] == "Delete 12 marked"
        await pilot.press("escape")
        await settle(pilot)


@pytest.mark.parametrize(
    "focus_id",
    ["personas-delete", "personas-conversations-search"],
    ids=["inspector-button", "inspector-input"],
)
async def test_escape_away_from_the_list_keeps_the_marks(
    focus_id, mock_app_instance, volume_348
):
    """Spec Q-2: Escape clears marks only on the items list or its marks bar.

    Escape in the Inspector, on a button or in an input other than the
    library's search box, never clears them (the Task 5 review found it
    clearing every mark from any focus).
    """
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=(160, 45), character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await select_roleplay_item(
            pilot, screen, _option_id(volume_348, "Character 0010")
        )
        await settle(pilot)
        await _mark_rows(
            pilot,
            screen,
            [_option_id(volume_348, f"Character {n:04d}") for n in (1, 2)],
        )
        target = screen.query_one(f"#{focus_id}")
        target.focus()
        await settle(pilot)
        assert screen.focused is target
        await pilot.press("escape")
        await settle(pilot)
        assert len(screen._marked_rows) == 2
        assert _bar(screen).startswith("✓ 2 marked")


async def test_escape_on_the_marks_bar_clears_the_marks_and_focus_lands_on_the_list(
    mock_app_instance, volume_348
):
    """The marks bar is the list's own context: Escape there clears too (I1)."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        await _mark_rows(pilot, screen, visible_item_ids(screen)[:2])
        delete = screen.query_one("#personas-library-marks-delete")
        delete.focus()
        await settle(pilot)
        assert screen.focused is delete
        await pilot.press("escape")
        await settle(pilot)
        assert _bar(screen) == ""
        assert screen._marked_rows == ()
        assert screen.focused is screen.query_one(RoleplayItemList)


# ---- An append extends only the list its rows came from (spec 1.5.6 "Window fence") ----


async def _pause_without_settling(pilot) -> None:
    """Let messages run while a held window blocks (``settle`` would wait on it)."""
    for _ in range(20):
        await pilot.pause(0.02)


@pytest.mark.parametrize("trigger", ["near_end", "end"])
@pytest.mark.parametrize("change", ["sort", "search"])
async def test_a_window_asked_for_during_a_reload_never_lands_on_the_old_rows(
    mock_app_instance, volume_348, monkeypatch, trigger, change
):
    """A near-end move or End while a sort or search reload reads its first
    window. Starting an append then cancelled the reload (one exclusive group)
    and laid the new order's rows onto the old ones (B2a review, fix round 1;
    never on dev, whose pager had no appends). End is asked again once the
    reload lands."""
    gate = threading.Event()
    blocked = threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        reloading = (
            kwargs.get("order_by") == "modified_desc"
            if change == "sort"
            else bool(kwargs.get("search_term"))
        )
        if reloading and kwargs.get("offset") == 0:
            blocked.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        old = visible_item_ids(screen)
        assert len(old) == 200 and item_list.more_available
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        if change == "sort":
            screen.state.sort_key = "modified_desc"
        else:
            screen.state.search_query = "Character 03"
        reload = asyncio.create_task(screen._reload_character_page())
        await wait_until(pilot, blocked.is_set, what="the reload's first window")
        if trigger == "near_end":
            item_list.highlighted = len(old) - 5
        else:
            item_list.focus()
            await pilot.press("end")
        await _pause_without_settling(pilot)
        gate.set()
        await reload
        expected = [
            f"character:{row['id']}"
            for row in original(
                volume_348,
                limit=100_000,
                offset=0,
                order_by=screen.state.sort_key,
                search_term=(
                    personas_screen_module.quote_fts5_prefix("Character 03")
                    if change == "search"
                    else None
                ),
                tag=None,
            )
        ]
        if trigger == "end":
            await wait_until(
                pilot,
                lambda: (
                    list(visible_item_ids(screen)) == expected
                    and highlighted_item_id(screen) == expected[-1]
                ),
                what="End, asked again once the reload landed",
            )
        await settle(pilot)
        ids = list(visible_item_ids(screen))
        assert ids == expected[: len(ids)]
        assert [f"character:{row['id']}" for row in screen._characters] == ids
        if change == "search":
            assert ids == expected


async def test_a_near_end_move_while_a_search_debounces_appends_nothing(
    mock_app_instance, volume_348, monkeypatch
):
    """The query changes at the keystroke; its reload runs after the debounce.
    A window asked for in between would read the new query at the old list's
    offset and append it to the old rows."""
    monkeypatch.setattr(
        personas_screen_module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 30.0
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        old = visible_item_ids(screen)
        screen.query_one("#personas-library-search", Input).value = "Character"
        await wait_until(
            pilot,
            lambda: screen.state.search_query == "Character",
            what="the keystroke",
        )
        item_list.highlighted = len(old) - 5
        await _pause_without_settling(pilot)
        assert visible_item_ids(screen) == old
        assert len(screen._characters) == len(old)


async def test_a_server_list_without_a_total_counts_its_windows(mock_app_instance):
    """The tldw_api client's ``list_characters`` returns a bare list with no
    total: the title reads ``100+`` until a short window arrives, then the
    loaded count, and the bottom border's M follows (spec 1.5.1, 1.5.6
    "Server"; B2a review, fix round 1)."""
    records = [{"id": n, "name": f"Remote {n:03d}"} for n in range(1, 151)]

    async def list_characters(*, mode, limit, offset):
        return [dict(record) for record in records[offset : offset + limit]]

    mock_app_instance.runtime_backend = "server"
    mock_app_instance.active_server_id = "server-a"
    mock_app_instance.character_persona_scope_service = SimpleNamespace(
        list_characters=list_characters,
        search_characters=AsyncMock(return_value=[]),
        get_character=AsyncMock(return_value=dict(records[0])),
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=SIZE) as pilot:
        screen = pilot.app.screen
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 100,
            what="the first server window",
        )
        assert screen.state.runtime_source == "server"
        pane = screen.query_one("#personas-library-pane")
        item_list = screen.query_one(RoleplayItemList)
        assert str(pane.border_title) == "Characters 100+"
        item_list.focus()
        await pilot.press("end")
        await wait_until(
            pilot,
            lambda: (
                len(visible_item_ids(screen)) == 150 and not item_list.more_available
            ),
            what="every server window",
        )
        await settle(pilot)
        assert str(pane.border_title) == "Characters 150"
        item_list.set_active(visible_item_ids(screen)[-1])
        await settle(pilot)
        assert str(pane.border_subtitle) == "#150 of 150"


# ---- The window state's one owner and its edge cases (TASK-33910.34) ----
#
# Each case below is new with B2a (dev's paged list had no windows) and is
# fixed by TASK-33910.34, which gives the list's window state one owner
# (``frame_state.ListWindowState``).


def _listed(db, *, order_by: str = "name_asc", query: str | None = None) -> list[str]:
    """Every option id one sort and search lists, in the list's order."""
    search = personas_screen_module.quote_fts5_prefix(query) if query else None
    rows = Character_Chat_Lib.get_character_page_for_ui(
        db, limit=100_000, offset=0, order_by=order_by, search_term=search, tag=None
    )
    return [f"character:{row['id']}" for row in rows]


def _hold_the_sorts_first_window(
    monkeypatch,
) -> tuple[threading.Event, threading.Event]:
    """Hold the ``modified_desc`` reload's first read until the gate opens."""
    gate, blocked = threading.Event(), threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("order_by") == "modified_desc" and kwargs.get("offset") == 0:
            blocked.set()
            gate.wait(10)
        return original(db, **kwargs)

    monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
    return gate, blocked


async def _open_past_row_200(screen, db, offset: int = 300) -> str:
    """Open the character at name row ``offset`` as a restore does: a load-through."""
    target = db.list_character_cards_page(limit=1, offset=offset)[0]
    screen._pending_restore = {
        "kind": "character",
        "id": str(target["id"]),
        "name": target["name"],
        "preview": None,
    }
    await screen._apply_pending_restore()
    return f"character:{target['id']}"


@pytest.mark.parametrize("newer", [None, "home"])
async def test_end_asked_for_during_a_reload_replays_only_if_no_cursor_move_came_after(
    mock_app_instance, volume_348, monkeypatch, newer
):
    """AC#2: End while a sort reloads is asked again once the sorted list
    lands, unless the user moved the cursor after it (Home here). B2a
    replayed it regardless: End then Home still loaded every row and left
    the cursor on the last one."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        gate, blocked = _hold_the_sorts_first_window(monkeypatch)
        screen.state.sort_key = "modified_desc"
        reload = asyncio.create_task(screen._reload_character_page())
        try:
            await wait_until(pilot, blocked.is_set, what="the sort's first window")
            item_list.focus()
            await pilot.press("end")
            if newer == "home":
                await pilot.press("home")
            await _pause_without_settling(pilot)
        finally:
            gate.set()
        await reload
        expected = _listed(volume_348, order_by="modified_desc")
        if newer is None:
            await wait_until(
                pilot,
                lambda: (
                    list(visible_item_ids(screen)) == expected
                    and highlighted_item_id(screen) == expected[-1]
                ),
                what="End, asked again on the sorted list",
            )
            return
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert list(visible_item_ids(screen)) == expected[:200]
        assert highlighted_item_id(screen) != expected[-1]


@pytest.mark.parametrize("newer", [None, "selection"])
async def test_a_load_through_asked_for_during_a_reload_replays_only_if_no_selection_came_after(
    mock_app_instance, volume_348, monkeypatch, newer
):
    """AC#2: a load-through (a restore of a row past the loaded end) while a
    sort reloads is asked again once the sorted list lands, unless the user
    opened another row after it. B2a replayed it regardless, so the row the
    user opened lost the loaded marker and the cursor to the restored one."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        gate, blocked = _hold_the_sorts_first_window(monkeypatch)
        screen.state.sort_key = "modified_desc"
        reload = asyncio.create_task(screen._reload_character_page())
        try:
            await wait_until(pilot, blocked.is_set, what="the sort's first window")
            target = await _open_past_row_200(screen, volume_348)
            assert target not in visible_item_ids(screen)
            if newer == "selection":
                chosen = visible_item_ids(screen)[10]
                await select_roleplay_item(pilot, screen, chosen)
                await wait_until(
                    pilot,
                    lambda: item_list.active_id == chosen,
                    what="the newer selection",
                )
        finally:
            gate.set()
        await reload
        if newer is None:
            await wait_until(
                pilot,
                lambda: (
                    item_list.active_id == target
                    and highlighted_item_id(screen) == target
                ),
                what="the load-through, asked again on the sorted list",
            )
            return
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert item_list.active_id == chosen
        assert highlighted_item_id(screen) != target


async def test_end_pressed_during_the_search_debounce_lands_on_the_new_lists_last_row(
    mock_app_instance, volume_348, monkeypatch
):
    """AC#3: End while a search waits out its debounce. B2a deferred End
    under the fence taken before the search's reload bumped the generation,
    so the landed list never matched it and End was dropped. The debounce
    is long here and fired by hand (its own callback), so End is certain to
    land inside it."""
    monkeypatch.setattr(
        personas_screen_module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 60.0
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        screen.query_one("#personas-library-search", Input).value = "Seeded"
        await wait_until(
            pilot,
            lambda: screen.state.search_query == "Seeded",
            what="the keystroke",
        )
        item_list.focus()
        await pilot.press("end")
        await _pause_without_settling(pilot)
        assert screen._search_debounce_timer is not None
        assert len(visible_item_ids(screen)) == 200
        screen._cancel_search_debounce()
        screen._start_debounced_search_render(query="Seeded", mode="characters")
        expected = _listed(volume_348, order_by=screen.state.sort_key, query="Seeded")
        assert len(expected) > 200
        await wait_until(
            pilot,
            lambda: (
                list(visible_item_ids(screen)) == expected
                and highlighted_item_id(screen) == expected[-1]
            ),
            what="End on the searched list's last row",
        )


async def test_enter_after_end_during_the_search_debounce_lands_on_the_first_match(
    mock_app_instance, volume_348, monkeypatch
):
    """End during the debounce is owed to the searched list (AC#3), and Enter
    in the search box after it is a newer cursor move (AC#2): the searched
    list lands with the cursor on its first match and End is not asked
    again. Dev dropped End here anyway (AC#3's defect), so this is green on
    dev; it pins that AC#3's fix leaves Enter in charge.

    Enter waits at most ``_SEARCH_SUBMIT_WAIT_SECONDS`` (2 s) for its render
    on dev and here (TASK-33910.35 removes the wait). Under xdist load the
    render can outlast that cap, and Enter then puts the cursor on the old
    rows, as B2a's own Enter test can show. The cap is raised to 30 s here,
    so Enter always waits for the render it pins, and every check waits on
    its condition, not on a fixed time."""
    monkeypatch.setattr(
        personas_screen_module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 60.0
    )
    monkeypatch.setattr(personas_screen_module, "_SEARCH_SUBMIT_WAIT_SECONDS", 30.0)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        search = screen.query_one("#personas-library-search", Input)
        search.value = "Seeded"
        await wait_until(
            pilot,
            lambda: screen.state.search_query == "Seeded",
            what="the keystroke",
        )
        item_list.focus()
        await pilot.press("end")
        await _pause_without_settling(pilot)
        assert screen._search_debounce_timer is not None
        search.focus()
        await pilot.press("enter")
        expected = _listed(volume_348, order_by=screen.state.sort_key, query="Seeded")
        await wait_until(
            pilot,
            lambda: (
                list(visible_item_ids(screen)) == expected[:200]
                and highlighted_item_id(screen) == expected[0]
            ),
            what="the searched list with the cursor on its first match",
        )
        await settle(pilot)
        await _pause_without_settling(pilot)
        await settle(pilot)
        assert list(visible_item_ids(screen)) == expected[:200]
        assert highlighted_item_id(screen) == expected[0]
        assert screen.focused is item_list


async def test_an_end_still_reading_yields_to_a_newer_cursor_move(
    mock_app_instance, volume_348, monkeypatch
):
    """End reads every window, then moves the cursor to the last row; Home
    pressed while it reads wins, and the rows still load. B2a moved the
    cursor to the last row regardless (the same rule as AC#2's, for a read
    no reload deferred)."""
    gate, reading = threading.Event(), threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset") and not gate.is_set():
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        item_list.focus()
        await select_roleplay_item(pilot, screen, index=5, load=False)
        try:
            await pilot.press("end")
            await wait_until(pilot, reading.is_set, what="End's read")
            await pilot.press("home")
        finally:
            gate.set()
        expected = _db_ids(volume_348)
        await wait_until(
            pilot,
            lambda: list(visible_item_ids(screen)) == expected,
            what="every row, loaded by End",
        )
        await settle(pilot)
        assert highlighted_item_id(screen) == expected[0]


@pytest.mark.parametrize("move", ["end", "load_through"])
async def test_a_move_whose_append_a_write_reload_cancels_completes_on_the_reloaded_list(
    mock_app_instance, volume_348, monkeypatch, move
):
    """AC#4: a write (a card saved elsewhere) reloads the list from its anchor
    while End's or a load-through's append is still reading, and the reload
    cancels that append (one exclusive worker group). B2a lost the move:
    the list stayed at its first 200 rows."""
    gate, reading = threading.Event(), threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset") and not gate.is_set():
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        try:
            if move == "end":
                item_list.focus()
                await pilot.press("end")
            else:
                target = await _open_past_row_200(screen, volume_348)
            await wait_until(pilot, reading.is_set, what="the append's read")
            await screen.refresh_character_library_list(None)
        finally:
            gate.set()
        expected = _db_ids(volume_348)
        if move == "end":
            await wait_until(
                pilot,
                lambda: (
                    list(visible_item_ids(screen)) == expected
                    and highlighted_item_id(screen) == expected[-1]
                ),
                what="End, completed on the reloaded list",
            )
        else:
            await wait_until(
                pilot,
                lambda: (
                    item_list.active_id == target
                    and highlighted_item_id(screen) == target
                ),
                what="the load-through, completed on the reloaded list",
            )
        ids = list(visible_item_ids(screen))
        assert ids == expected[: len(ids)]


async def test_a_load_through_a_newer_list_cancelled_never_moves_the_cursor_later(
    mock_app_instance, volume_348, monkeypatch
):
    """AC#5: a search starts a newer list while a load-through is still
    reading. B2a kept the row waiting, so when the user later moved far
    enough down the searched list for a window to bring that row in, the
    cursor jumped to it (B2a's final review fixed the kind-switch case)."""
    gate, reading = threading.Event(), threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset") and not kwargs.get("search_term") and not gate.is_set():
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        try:
            target = await _open_past_row_200(screen, volume_348)
            await wait_until(pilot, reading.is_set, what="the load-through's read")
            screen.state.search_query = "Seeded"
            await screen._reload_character_page()
        finally:
            gate.set()
        await settle(pilot)
        searched = _listed(volume_348, query="Seeded")
        assert list(visible_item_ids(screen)) == searched[:200]
        assert target in searched[200:]
        item_list.focus()
        item_list.highlighted = 195
        await wait_until(
            pilot,
            lambda: target in visible_item_ids(screen),
            what="the later window holding the cancelled load-through's row",
        )
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert highlighted_item_id(screen) == searched[195]


async def test_a_window_asked_for_before_a_server_reloads_blank_never_cancels_that_reload(
    mock_app_instance,
):
    """AC#6: a server reload first blanks the list. B2a took the blank for the
    new list's rows, so a near-end request queued before the blank and
    handled after it started an offset-0 append, and that cancelled the
    reload (one exclusive worker group): the reload's anchor restore, its
    selection mark and a deferred End were lost.

    The request is a synthetic ``WindowWanted`` posted to the list after the
    blank: a real near-end scroll queued before the blank cannot be timed in
    a test. The screen's real handler (``_handle_window_wanted``) takes it,
    after bubbling through the pane as a real one would."""
    records = [{"id": n, "name": f"Remote {n:03d}"} for n in range(1, 151)]
    offsets: list[int] = []
    hold, holding = asyncio.Event(), asyncio.Event()
    armed = False

    async def list_characters(*, mode, limit, offset):
        offsets.append(offset)
        if armed and not hold.is_set():
            holding.set()
            await hold.wait()
        return [dict(record) for record in records[offset : offset + limit]]

    mock_app_instance.runtime_backend = "server"
    mock_app_instance.active_server_id = "server-a"
    mock_app_instance.character_persona_scope_service = SimpleNamespace(
        list_characters=list_characters,
        search_characters=AsyncMock(return_value=[]),
        get_character=AsyncMock(return_value=dict(records[0])),
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=SIZE) as pilot:
        screen = pilot.app.screen
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 100,
            what="the first server window",
        )
        item_list = screen.query_one(RoleplayItemList)
        cursor = visible_item_ids(screen)[50]
        await select_roleplay_item(pilot, screen, cursor, load=False)
        await settle(pilot)
        armed = True
        offsets.clear()
        write = asyncio.create_task(screen.refresh_character_library_list(None))
        try:
            await wait_until(pilot, holding.is_set, what="the reload's read")
            assert visible_item_ids(screen) == ()
            item_list.post_message(
                RoleplayItemList.WindowWanted(item_list, reason="near_end")
            )
            await _pause_without_settling(pilot)
        finally:
            hold.set()
        await write
        await settle(pilot)
        assert offsets == [0]
        assert len(visible_item_ids(screen)) == 100
        assert highlighted_item_id(screen) == cursor


async def test_a_leave_during_the_search_debounce_stops_it_and_the_return_loads_through(
    mock_app_instance, volume_348, monkeypatch
):
    """AC#7, the real suspend path: Roleplay left by ``switch_screen`` while
    its search waits out the debounce. B2a left the timer running, so the
    search read the database while Roleplay was away. Now the leave stops
    the timer, and the return reloads the searched list, which then loads
    through (End reaches its last row)."""
    monkeypatch.setattr(personas_screen_module, "PERSONAS_SEARCH_DEBOUNCE_SECONDS", 1.0)
    reads: list[object] = []
    original = personas_screen_module.get_character_page_for_ui

    def counted(db, **kwargs):
        reads.append(kwargs.get("search_term"))
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(
            personas_screen_module, "get_character_page_for_ui", counted
        )
        screen.query_one("#personas-library-search", Input).value = "Seeded"
        await wait_until(
            pilot,
            lambda: screen.state.search_query == "Seeded",
            what="the keystroke",
        )
        assert screen._search_debounce_timer is not None
        await app.switch_screen("elsewhere")
        await settle(pilot)
        assert screen not in app.screen_stack and screen.is_attached
        assert screen._search_debounce_timer is None
        away = len(reads)
        await pilot.pause(1.5)  # past the debounce: a live timer would have read
        assert len(reads) == away
        await app.switch_screen("roleplay")
        await settle(pilot)
        expected = _listed(volume_348, order_by=screen.state.sort_key, query="Seeded")
        assert list(visible_item_ids(screen)) == expected[:200]
        item_list.focus()
        await pilot.press("end")
        await wait_until(
            pilot,
            lambda: (
                list(visible_item_ids(screen)) == expected
                and highlighted_item_id(screen) == expected[-1]
            ),
            what="End loading through after the return",
        )


async def test_a_leave_inside_a_replaces_paint_reloads_on_the_return(
    mock_app_instance, volume_348, monkeypatch
):
    """AC#7, the real suspend path: Roleplay left while a replace paints its
    rows. A paint yields only where ``update_rows`` swaps the pane's status
    line, that is when the list goes from empty (or recovery) to rows or
    back, so the case built here is the reachable one: a search with no
    match, then the search cleared. The hold sits exactly at that swap and
    only when the status really changes. B2a counted the painting rows as
    landed, so the return reloaded nothing, and the paint the leave
    cancelled never said rows lay past the new list's end: End stopped at
    row 200 for good. Now the return reloads the list, and End loads
    through."""
    painting, release = asyncio.Event(), asyncio.Event()
    armed = False
    sync_status = PersonasLibraryPane._sync_status

    async def held(self, status):
        nonlocal armed
        if armed and status != self._status_key:
            armed = False
            painting.set()
            await release.wait()
        await sync_status(self, status)

    monkeypatch.setattr(PersonasLibraryPane, "_sync_status", held)
    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=volume_348,
        app_class=_ReusedRoleplayApp,
    ) as pilot:
        app = pilot.app
        screen = app.screen
        item_list = screen.query_one(RoleplayItemList)
        screen.state.search_query = "zzzz"
        await screen._reload_character_page()
        await settle(pilot)
        assert visible_item_ids(screen) == () and not item_list.more_available
        assert screen.query("#personas-library-empty")
        armed = True
        screen.state.search_query = ""
        reload = asyncio.create_task(screen._reload_character_page())
        try:
            await wait_until(pilot, painting.is_set, what="the replace's paint")
            await app.switch_screen("elsewhere")
            await settle(pilot)
            assert screen not in app.screen_stack
        finally:
            release.set()
        await asyncio.gather(reload, return_exceptions=True)
        await app.switch_screen("roleplay")
        await settle(pilot)
        item_list.focus()
        await pilot.press("end")
        expected = _db_ids(volume_348)
        await wait_until(
            pilot,
            lambda: (
                list(visible_item_ids(screen)) == expected
                and highlighted_item_id(screen) == expected[-1]
            ),
            what="End loading through after the return",
        )


def _failing_after_row_200(fail: dict):
    """``get_character_page_for_ui`` that raises for any window past row 0 while ``fail["on"]``."""
    original = personas_screen_module.get_character_page_for_ui

    def flaky(db, **kwargs):
        if kwargs.get("offset") and fail["on"]:
            raise RuntimeError("disk I/O error")
        return original(db, **kwargs)

    return flaky


@pytest.mark.parametrize("failure", ["raise", "no_database"])
@pytest.mark.parametrize("move", ["end", "load_through"])
async def test_a_failed_read_leaves_no_move_owed_for_a_later_write_reload(
    mock_app_instance, volume_348, monkeypatch, move, failure
):
    """Fix round 1 (review finding 1): End's or a load-through's read fails
    (the read raises, or no database is there to read), and later an
    unrelated write (a save, an import, a Console or agent write) reloads the
    same list. The move must not come back then. This PR's first push kept
    it owed, so that reload read every remaining row and moved the cursor (to
    the last row, or to the restored row); dev drops it."""
    fail = {"on": True}
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        if failure == "raise":
            monkeypatch.setattr(
                personas_screen_module,
                "get_character_page_for_ui",
                _failing_after_row_200(fail),
            )
        else:
            database = screen._character_db
            screen._character_db = lambda: None if fail["on"] else database()
        item_list.focus()
        await select_roleplay_item(pilot, screen, index=5, load=False)
        before = highlighted_item_id(screen)
        if move == "end":
            await pilot.press("end")
        else:
            await _open_past_row_200(screen, volume_348)
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert len(visible_item_ids(screen)) == 200
        assert highlighted_item_id(screen) == before
        fail["on"] = False
        await screen.refresh_character_library_list(None)
        await settle(pilot)
        await _pause_without_settling(pilot)
        await settle(pilot)
        assert len(visible_item_ids(screen)) == 200
        assert highlighted_item_id(screen) == before


async def test_a_failed_read_lets_the_list_ask_again_near_its_end(
    mock_app_instance, volume_348, monkeypatch
):
    """Fix round 1 (review finding 1): a read that fails answers the request,
    so the next move near the loaded end asks again. On dev and on this PR's
    first push the near-end latch stayed set until another list landed, so no
    window came."""
    fail = {"on": True}
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        monkeypatch.setattr(
            personas_screen_module,
            "get_character_page_for_ui",
            _failing_after_row_200(fail),
        )
        item_list.focus()
        await pilot.press("end")
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert len(visible_item_ids(screen)) == 200
        fail["on"] = False
        item_list.highlighted = 190
        await settle(pilot)
        item_list.highlighted = 195
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) > 200,
            what="the next window after the failed read",
        )


@pytest.mark.parametrize("failure", ["raise", "no_list_call"])
async def test_a_failed_server_read_leaves_no_end_owed_for_a_later_write_reload(
    mock_app_instance, failure
):
    """Fix round 1 (review finding 1), on a server source: End's read fails
    (the call raises, or the service has no list call), then a write reloads
    the list; End must not come back and load every row. This PR's first
    push replayed it; dev drops it."""
    records = [{"id": n, "name": f"Remote {n:03d}"} for n in range(1, 151)]
    fail = {"on": False}

    async def list_characters(*, mode, limit, offset):
        if offset and fail["on"]:
            raise RuntimeError("server unreachable")
        return [dict(record) for record in records[offset : offset + limit]]

    mock_app_instance.runtime_backend = "server"
    mock_app_instance.active_server_id = "server-a"
    mock_app_instance.character_persona_scope_service = SimpleNamespace(
        list_characters=list_characters,
        search_characters=AsyncMock(return_value=[]),
        get_character=AsyncMock(return_value=dict(records[0])),
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=SIZE) as pilot:
        screen = pilot.app.screen
        await wait_until(
            pilot,
            lambda: len(visible_item_ids(screen)) == 100,
            what="the first server window",
        )
        item_list = screen.query_one(RoleplayItemList)
        item_list.focus()
        await select_roleplay_item(pilot, screen, index=5, load=False)
        before = highlighted_item_id(screen)
        service = mock_app_instance.character_persona_scope_service
        if failure == "raise":
            fail["on"] = True
        else:
            service.list_characters = None
        await pilot.press("end")
        await settle(pilot)
        await _pause_without_settling(pilot)
        assert len(visible_item_ids(screen)) == 100
        fail["on"] = False
        service.list_characters = list_characters
        await screen.refresh_character_library_list(None)
        await settle(pilot)
        await _pause_without_settling(pilot)
        await settle(pilot)
        assert len(visible_item_ids(screen)) == 100
        assert highlighted_item_id(screen) == before


async def test_a_background_refresh_of_the_open_card_does_not_cancel_end(
    mock_app_instance, volume_348, monkeypatch
):
    """Fix round 1 (review finding 2): a Console save of the open character
    re-selects it while End reads. That is no user move, so End still lands
    on the last row. This PR's first push took it for a newer selection and
    cancelled End; dev does not."""
    gate, reading = threading.Event(), threading.Event()
    original = personas_screen_module.get_character_page_for_ui

    def gated(db, **kwargs):
        if kwargs.get("offset") and not gate.is_set():
            reading.set()
            gate.wait(10)
        return original(db, **kwargs)

    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        opened = visible_item_ids(screen)[5]
        await select_roleplay_item(pilot, screen, opened)
        await wait_until(
            pilot, lambda: item_list.active_id == opened, what="the opened row"
        )
        await settle(pilot)
        monkeypatch.setattr(personas_screen_module, "get_character_page_for_ui", gated)
        item_list.focus()
        try:
            await pilot.press("end")
            await wait_until(pilot, reading.is_set, what="End's read")
            await screen._on_character_card_changed(
                CharacterCardChanged(int(opened.split(":")[1]))
            )
            await pilot.pause()
        finally:
            gate.set()
        expected = _db_ids(volume_348)
        await wait_until(
            pilot,
            lambda: (
                list(visible_item_ids(screen)) == expected
                and highlighted_item_id(screen) == expected[-1]
            ),
            what="End on the last row",
        )


async def test_reopening_the_open_row_past_the_loaded_window_loads_through_to_it(
    mock_app_instance, volume_348
):
    """Fix round 2: the open character (name row 300) is past the loaded
    window after a reload from row 0 (a search, then the search cleared),
    and is opened again by a user path (here ``_select_character`` with the
    same id, as a Console hand-off or a re-open does). Dev loads through to
    it and puts the cursor on it; fix round 1 left it unloaded, because it
    owed a row only when the open row changed."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        target = await _open_past_row_200(screen, volume_348)
        await wait_until(
            pilot,
            lambda: highlighted_item_id(screen) == target,
            what="the restored row past row 200",
        )
        screen.state.search_query = "Character 00"
        await screen._reload_character_page()
        screen.state.search_query = ""
        await screen._reload_character_page()
        await settle(pilot)
        assert len(visible_item_ids(screen)) == 200
        assert target not in visible_item_ids(screen)
        assert item_list.active_id == target
        record = volume_348.get_character_card_by_id(int(target.split(":")[1]))
        await screen._select_character(str(record["id"]), record["name"])
        await wait_until(
            pilot,
            lambda: (
                target in visible_item_ids(screen)
                and highlighted_item_id(screen) == target
            ),
            what="the re-opened row, loaded through",
        )
        assert item_list.active_id == target


async def test_the_screen_and_its_list_share_one_window_state(
    mock_app_instance, volume_348
):
    """AC#1: one owner for the window state, shared by the screen and the list."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=volume_348
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one(RoleplayItemList)
        window = screen._window_state
        assert isinstance(window, frame_state.ListWindowState)
        assert item_list.window_state is window
        assert item_list.more_available is window.more_available is True
        assert window.rows_current(screen._list_fence())
