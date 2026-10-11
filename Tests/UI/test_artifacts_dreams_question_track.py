"""Dreams question-track UI entry + tracked-row interaction (task-33164).

Two surfaces, mirroring the story-modal file's harness split: the ``w``
(watch question) action is exercised on a bare ``App`` (the modal owns its
DB handle via the ``dreams_db_getter`` seam), while the Artifacts-screen
Tracked-row interaction (click/Enter opens the origin story, or a summary
notice when the origin is gone) reuses the DestinationHarness harness from
``test_artifacts_dreams_rows.py``.

Import-order admission binding (75ecfdbca9 family, see the goals-modal
file's docstring): the story-modal file gets the binding via its
``app_factory`` import; the bare-App sections here import the factory
purely for the binding (hence noqa) -- without it the file is
order-dependent and errors standalone with RecoveryRequired.
"""

from __future__ import annotations

import pytest
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App
from textual.widgets import Button, Static

import Tests.UI.app_factory  # noqa: F401 - config-participant admission binding (see module docstring)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_artifacts_dreams_rows import _settle_artifacts_refreshes
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.dreams_view import list_recent_dreams
from tldw_chatbook.UI.Screens.artifacts_dreams_modal import DreamsStoryModal
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

pytestmark = pytest.mark.ui

_STORY_TITLE = "Cheap flights to Japan"
_EVENT_STORY_TITLE = "Opera night at Fuji"


# --- Shared helpers ---------------------------------------------------------


def _enable_dreams(monkeypatch) -> None:
    """Flip ``[dreams] enabled`` deterministically (Task 6's seam)."""
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default: True if key == "enabled" else default,
    )


def _dreams_settings_defaults(monkeypatch) -> None:
    """Deterministic Dreams settings for the watch/track chain."""
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default,
    )


def _seed_db(tmp_path) -> DreamsDB:
    """One deal-kind story whose discovery query is watchable."""
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-question-track")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.insert_story(
        collection,
        title=_STORY_TITLE,
        url="https://example.com/flights",
        snippet="Fares from $89",
        body="A story about fares.",
        status="complete",
        source="web",
        kind="deal",
        event_date=None,
        location="Japan",
        matched_topics=["visit japan"],
        query="cheap flights japan",
    )
    return db


def _seed_event_db(tmp_path) -> DreamsDB:
    """One event-kind story pinned to an event date (carry-through check)."""
    db = DreamsDB(tmp_path / "dreams-event.sqlite", "dreams-question-track")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.insert_story(
        collection,
        title=_EVENT_STORY_TITLE,
        url="https://example.com/opera",
        snippet="Tickets on sale soon",
        body="A story about tickets.",
        status="complete",
        source="web",
        kind="event",
        event_date="2026-10-01",
        location="Japan",
        matched_topics=["visit japan"],
        query="opera tickets japan",
    )
    return db


def _seed_failed_cycle(db: DreamsDB) -> None:
    collection = db.create_collection("2026-09-23", "scheduled", "digest")
    db.set_collection_status(collection, "failed")


def _story_row(db: DreamsDB) -> dict:
    rows = list_recent_dreams(db, limit=10)
    return next(row for row in rows if not row.get("synthetic"))


def _synthetic_row(db: DreamsDB) -> dict:
    rows = list_recent_dreams(db, limit=10)
    return next(row for row in rows if row.get("synthetic"))


def _feedback_kinds(db: DreamsDB, story_id: int) -> list[str]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT kind FROM dream_feedback WHERE story_id = ? ORDER BY id",
            (story_id,),
        ).fetchall()
    return [row[0] for row in rows]


def _renderable_text(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_renderable_text(item) for item in renderable.renderables)
    if isinstance(renderable, Markdown):
        return str(renderable.markup)
    return str(renderable)


def _visible_text(widget) -> str:
    return "\n".join(
        _renderable_text(item.renderable)
        for item in widget.query(Static)
        if item.display and hasattr(item, "renderable")
    )


def _button_labels(widget) -> str:
    return " ".join(
        str(button.label)
        for button in widget.query(Button)
        if button.display and button.label is not None
    )


# --- Watch question (w): the modal's question-track entry (task-33164 R1) -----


@pytest.mark.asyncio
async def test_watch_creates_question_item_with_story_query_intent_and_feedback(
    tmp_path, monkeypatch
):
    """``w`` wraps the story's discovery query as a question watch.

    The template is the story's ``query`` verbatim, the intent derives from
    the story's ``kind`` (deal), the origin story is linked, no subscription
    is created, ``tracked`` feedback records, the notice states the watched
    query, and ``on_changed`` fires exactly once.
    """
    _dreams_settings_defaults(monkeypatch)
    db = _seed_db(tmp_path)
    story = _story_row(db)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert "Watch question" in _button_labels(modal), (
            "a story row must offer the watch-question action"
        )
        hints = _renderable_text(modal.query_one("#dsm-hints", Static).renderable)
        assert "w Watch question" in hints, "hints must advertise the w action"

        await pilot.press("w")
        await pilot.pause()

        (tracked,) = db.list_tracked_items()
        assert tracked["mechanism"] == "question"
        assert tracked["query_template"] == "cheap flights japan", (
            "the story's discovery query, verbatim"
        )
        assert tracked["intent"] == "deal", "kind=deal maps to a deal watch"
        assert tracked["origin_story_id"] == story["id"]
        assert tracked["subscription_id"] is None, (
            "a question watch owns no subscription"
        )
        assert tracked["event_date"] is None
        assert _feedback_kinds(db, story["id"]) == ["tracked"]
        assert changed == [1], "on_changed fires exactly once after success"
        assert any(
            "cheap flights japan" in n.message for n in app._notifications
        ), "the confirm notice must state the query being watched"
        assert app.screen is modal, "watch keeps the modal open"


@pytest.mark.asyncio
async def test_watch_on_event_story_carries_event_date_and_intent(
    tmp_path, monkeypatch
):
    """An event-kind story's watch carries its ``event_date`` and intent."""
    _dreams_settings_defaults(monkeypatch)
    db = _seed_event_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("w")
        await pilot.pause()

        (tracked,) = db.list_tracked_items()
        assert tracked["intent"] == "event", "kind=event maps to an event watch"
        assert tracked["event_date"] == "2026-10-01", (
            "the story's event date rides the watch"
        )


@pytest.mark.asyncio
async def test_watch_on_synthetic_row_writes_nothing(tmp_path):
    """``w`` on a failed-cycle row is the synthetic early-return: no write."""
    db = _seed_db(tmp_path)
    _seed_failed_cycle(db)
    synthetic = _synthetic_row(db)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            synthetic,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        labels = _button_labels(modal)
        assert "Watch question" not in labels, (
            "a synthetic row must not offer the watch action"
        )

        await pilot.press("w")
        await pilot.pause()

        assert db.list_tracked_items() == [], "no tracked item for a synthetic row"
        with db.connection() as conn:
            feedback_rows = conn.execute(
                "SELECT COUNT(*) FROM dream_feedback"
            ).fetchone()
        assert int(feedback_rows[0]) == 0, "no feedback for a synthetic row"
        assert changed == []
        assert app.screen is modal, "the refusal must not dismiss or crash"


@pytest.mark.asyncio
async def test_watch_cap_reached_is_a_notice_and_writes_nothing(
    tmp_path, monkeypatch
):
    """A full tracked budget refuses before any write: notice, no item."""
    db = _seed_db(tmp_path)
    story = _story_row(db)
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default: (
            1 if key == "tracked_item_cap" else default
        ),
    )
    db.create_tracked_item(mechanism="question", intent="topic", cadence_seconds=3600)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("w")
        await pilot.pause()

        assert _feedback_kinds(db, story["id"]) == [], (
            "a refused watch records no tracked feedback"
        )
        assert len(db.list_tracked_items()) == 1, "the cap blocked a new item"
        assert changed == [], "on_changed must not fire for a refusal"
        assert any("budget" in n.message.lower() for n in app._notifications), (
            "the cap refusal must be a notice"
        )
        assert app.screen is modal, "a cap refusal must not dismiss or crash"


# --- Tracked-row interaction (task-33164 R2) ----------------------------------


def _seed_resolvable_track_db(tmp_path) -> DreamsDB:
    """One story plus an active question watch whose origin resolves to it."""
    db = _seed_db(tmp_path)
    db.create_tracked_item(
        mechanism="question",
        intent="deal",
        query_template="cheap flights japan",
        origin_story_id=1,
        cadence_seconds=3600,
    )
    return db


def _seed_orphan_track_db(tmp_path) -> DreamsDB:
    """One story plus an active question watch with no origin story."""
    db = _seed_db(tmp_path)
    db.create_tracked_item(
        mechanism="question",
        intent="topic",
        query_template="standalone question",
        cadence_seconds=3600,
    )
    return db


async def _wait_for_dreams(screen, pilot, selector: str, *, attempts: int = 50):
    for _ in range(attempts):
        await pilot.pause(0.05)
        if screen._dreams and screen.query(selector):
            break
    else:
        raise AssertionError(f"dreams refresh never landed {selector!r}")
    await _settle_artifacts_refreshes(screen, pilot)


# The wiring tests below build a real app via ``_build_test_app`` and need
# ``bootstrap_profile`` for the same config-admission reason as the
# story-modal file's wiring section.
@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_tracked_row_click_opens_story_modal_for_resolvable_origin(
    tmp_path, monkeypatch
):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_resolvable_track_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-track-row-1")

        await pilot.click("#artifacts-dream-track-row-1")
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, DreamsStoryModal), (
            "a resolvable tracked row opens its origin story's modal"
        )
        assert _STORY_TITLE in _visible_text(modal)
        assert "· tracked" in _visible_text(modal), (
            "the origin story badges as tracked"
        )

        await pilot.press("q")
        await pilot.pause()
        assert host.screen_stack[-1] is screen, "close returns to the Artifacts screen"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_tracked_row_enter_opens_story_modal(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_resolvable_track_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-track-row-1")

        screen.query_one("#artifacts-dream-track-row-1", Static).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], DreamsStoryModal), (
            "Enter on a focused tracked row opens the story modal"
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_tracked_row_with_unresolvable_origin_shows_summary_notice(
    tmp_path, monkeypatch
):
    """A question watch with no origin story is a summary notice, not a modal."""
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_orphan_track_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-track-row-1")

        await pilot.click("#artifacts-dream-track-row-1")
        await pilot.pause()

        assert host.screen_stack[-1] is screen, (
            "an unresolvable origin must not push a modal"
        )
        # The mounted app here is the harness host, not the wrapped
        # (unmounted) TldwCli, so the toasts land on ``pilot.app``.
        summary = next(
            (
                n.message
                for n in pilot.app._notifications
                if "mechanism: question" in n.message
            ),
            None,
        )
        assert summary is not None, "the summary notice must carry the mechanism"
        assert "status: active" in summary
        assert "standalone question" in summary, "the label (query) rides the summary"
        assert "last run: none yet" in summary, "the last-run state rides the summary"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_track_row_routing_keeps_dream_row_clicks_working(
    tmp_path, monkeypatch
):
    """The widened click routing must not break the existing dream-row path."""
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_resolvable_track_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")

        await pilot.click("#artifacts-dream-row-1")
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, DreamsStoryModal), (
            "a dream story row still opens its own modal"
        )

        await pilot.press("q")
        await pilot.pause()
        assert host.screen_stack[-1] is screen

        await pilot.click("#artifacts-dream-track-row-1")
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], DreamsStoryModal), (
            "and the tracked row still opens the origin story's modal"
        )
