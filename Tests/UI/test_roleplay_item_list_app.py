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
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
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
    replaced, and ``set_options`` cleared it, so a second Enter opened nothing."""
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
        await pilot.press("enter")
        release.set()
        await wait_until(
            pilot,
            lambda: highlighted_item_id(screen) == f"character:{target['id']}",
            what="the first match under the cursor",
        )
        assert screen.focused is screen.query_one(RoleplayItemList)


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
            generation = screen._character_page_generation
            assert len(screen._marked_rows) == 3
    finally:
        gate.set()
    assert suspended == []
    # The pane and its list are pruned before the screen's own unmount, so
    # the screen's mark set is what the hook clears here.
    assert screen._marked_rows == ()
    assert screen._character_page_generation == generation + 1
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
