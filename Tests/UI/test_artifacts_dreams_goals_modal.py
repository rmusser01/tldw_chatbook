"""Dreams goals & privacy modal (Dreams Phase 2, Task 1).

Bare-App harness, mirroring ``test_artifacts_dreams_modal.py``: the modal
owns its DB handle via the ``dreams_db_getter`` seam, so no destination
shell (and no ``bootstrap_profile``) is needed. Goal text is user-typed, so
every rendered row is a literal ``rich.text.Text`` -- the assertions read
``.plain`` via the same ``_renderable_text`` helpers.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App
from textual.widgets import Input, ListView, Static

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.UI.Screens.artifacts_dreams_goals_modal import DreamsGoalsModal

pytestmark = pytest.mark.ui


# --- Shared helpers ---------------------------------------------------------


def _region(monkeypatch, region: str = "near Seattle") -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default: (
            region if (section, key) == ("dreams", "region") else default
        ),
    )


def _seed_db(tmp_path) -> DreamsDB:
    db = DreamsDB(tmp_path / "dreams-goals.sqlite", "dreams-goals")
    db.upsert_profile_entry(
        "goal", "visit Japan", weight=1.0, searchable=1, source="user"
    )
    return db


def _renderable_text(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    return str(renderable)


def _visible_text(widget) -> str:
    return "\n".join(
        _renderable_text(item.renderable)
        for item in widget.query(Static)
        if item.display and hasattr(item, "renderable")
    )


def _goal_rows(db: DreamsDB) -> list[dict]:
    return [row for row in db.list_profile() if row.get("facet") == "goal"]


# --- Surface -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_renders_with_searchable_marker_and_region_line(
    tmp_path, monkeypatch
):
    _region(monkeypatch)
    db = _seed_db(tmp_path)
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        text = _visible_text(modal)
        assert "visit Japan" in text
        assert "visit Japan · searchable" in text
        assert "Region (used in queries): near Seattle" in text


@pytest.mark.asyncio
async def test_footer_hints_advertise_exactly_the_four_actions(tmp_path):
    db = _seed_db(tmp_path)
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        hints = _renderable_text(modal.query_one("#dgm-hints", Static).renderable)
        for word in ("a Add", "x Remove", "s Searchable on/off", "q Close"):
            assert word in hints, f"hint must advertise {word!r}"
        for word in ("Keep", "Track", "Export", "Ingest", "Dive"):
            assert word not in hints, f"hint must not advertise {word!r}"


# --- Actions -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_s_toggles_searchable_and_rerenders_marker(tmp_path):
    db = _seed_db(tmp_path)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: changed.append(1)
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("s")
        await pilot.pause()

        (row,) = _goal_rows(db)
        assert row["searchable"] == 0, "s must flip searchable via the real DB"
        assert "visit Japan · private" in _visible_text(modal)
        assert changed == [1], "each mutation fires on_changed exactly once"

        await pilot.press("s")  # toggle back
        await pilot.pause()
        (row,) = _goal_rows(db)
        assert row["searchable"] == 1
        assert "visit Japan · searchable" in _visible_text(modal)


@pytest.mark.asyncio
async def test_a_adds_goal_from_input_as_user_source(tmp_path):
    db = _seed_db(tmp_path)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: changed.append(1)
        )
        await app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#dgm-new-goal", Input).value = "new goal"
        await pilot.press("a")
        await pilot.pause()

        texts = {row["text"]: row for row in _goal_rows(db)}
        assert "new goal" in texts
        assert texts["new goal"]["source"] == "user"
        assert texts["new goal"]["searchable"] == 1, "new goals start searchable"
        assert "new goal · searchable" in _visible_text(modal)
        assert modal.query_one("#dgm-new-goal", Input).value == "", (
            "the input clears after a successful add"
        )
        assert changed == [1]


@pytest.mark.asyncio
async def test_x_removes_selected_goal(tmp_path):
    db = _seed_db(tmp_path)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: changed.append(1)
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("x")
        await pilot.pause()

        assert _goal_rows(db) == [], "x must delete the selected goal"
        assert "visit Japan" not in _visible_text(modal)
        assert changed == [1]


@pytest.mark.asyncio
async def test_q_closes_the_modal(tmp_path):
    db = _seed_db(tmp_path)
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("q")
        await pilot.pause()

        assert app.screen is not modal
        assert _goal_rows(db), "close is not a mutation"


@pytest.mark.asyncio
async def test_missing_dreams_db_degrades_to_notice_without_crash(tmp_path):
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: None, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("s")
        await pilot.press("x")
        await pilot.pause()

        assert app.screen is modal, "a refused action must not dismiss or crash"


@pytest.mark.asyncio
async def test_empty_list_renders_hint_not_a_broken_selection(tmp_path):
    db = DreamsDB(tmp_path / "dreams-empty.sqlite", "dreams-goals")
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert "No goals yet" in _visible_text(modal)
        await pilot.press("s")  # nothing selected: a notice, not a crash
        await pilot.pause()
        assert app.screen is modal
        assert isinstance(modal.query_one("#dgm-goals", ListView), ListView)
