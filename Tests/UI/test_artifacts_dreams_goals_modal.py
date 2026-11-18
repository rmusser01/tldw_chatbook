"""Dreams goals & privacy modal (Dreams Phase 2, Task 1).

Bare-App harness, mirroring ``test_artifacts_dreams_modal.py``: the modal
owns its DB handle via the ``dreams_db_getter`` seam, so no destination
shell (and no ``bootstrap_profile``) is needed. Goal text is user-typed, so
every rendered row is a literal ``rich.text.Text`` -- the assertions read
``.plain`` via the same ``_renderable_text`` helpers.

Import-order admission binding (75ecfdbca9 family): ``Tests/UI/conftest``'s
autouse ``_disable_model_catalog_refresh`` first-imports
``tldw_chatbook.app`` whose module body runs ``load_settings()``; under the
per-test config sandbox that admission fails closed
(``RecoveryRequired("raw_source_selection_changed")``) unless the app was
already imported at COLLECTION time, when the profile is consistent. The
story-modal file gets that binding via its ``app_factory`` import; this
file mounts only a bare App, so it imports the factory purely for the
binding (hence noqa) -- without it the file is order-dependent and errors
standalone.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.bootstrap_profile
from rich.text import Text
from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
from textual.widgets import Input, ListView, Static

import Tests.UI.app_factory  # noqa: F401 - config-participant admission binding (see module docstring)
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
async def test_region_line_shows_no_region_when_the_stored_one_is_ignored(
    tmp_path, monkeypatch
):
    """Review round 3, item 7: the line is labelled "used in queries", so
    it must show the region Dreams actually uses. A stored value that
    fails the text rule is ignored by ``dreams_region()`` -- showing it
    here as "used in queries" was false."""
    _region(monkeypatch, "Seattle\u202e")
    db = _seed_db(tmp_path)
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: None
        )
        await app.push_screen(modal)
        await pilot.pause()

        text = _visible_text(modal)
        assert "Region (used in queries): (not set)" in text
        assert "Seattle" not in text


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
@pytest.mark.parametrize(
    "typed, says",
    [("g" * 201, "longer than 200 characters"),
     ("tickets\u202egnp.exe", "hidden text-direction characters"),
     ("\u200b", "nothing readable")],
    ids=["overlong", "hidden-direction-override", "invisible-only"],
)
async def test_a_refuses_invalid_goal_text_and_keeps_it_for_correction(
    tmp_path, typed, says
):
    """Qodo #8 (PR #2921): goal text reaches query synthesis and outbound
    search, so the add path validates it (shared input validation) instead
    of persisting whatever survived ``strip()``. A refusal writes nothing,
    says why, and leaves the typed text in place to fix."""
    db = _seed_db(tmp_path)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        modal = DreamsGoalsModal(
            dreams_db_getter=lambda: db, on_changed=lambda: changed.append(1)
        )
        await app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#dgm-new-goal", Input).value = typed
        await pilot.press("a")
        await pilot.pause()

        assert [row["text"] for row in _goal_rows(db)] == ["visit Japan"], (
            "a refused goal must not be persisted"
        )
        assert modal.query_one("#dgm-new-goal", Input).value == typed
        assert changed == [], "a refusal is not a mutation"
        notices = [str(n.message) for n in app._notifications]
        # The notice names what was wrong (round 4, C4).
        assert any(
            message.startswith("Goal not added: ") and says in message
            for message in notices), notices


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
