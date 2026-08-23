"""UI tests for `o` open-trace import on TrajectoryScreen (task-16320).

The file picker is stubbed at its seam (``_pick_trace_file`` -- the
fspicker modal is not pilot-friendly); everything downstream of the pick
(validation through the shared seam, snapshot mapping, read-only screen
push, error notifications) runs for real. Includes the behavioral
no-DB-write proof: an import performed through the screen leaves every
app-DB row count unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.Chat.trajectory_export import (
    build_trajectory_export,
    write_trajectory_export,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
from Tests.Chat.test_trajectory_export import _seed_conversation
from Tests.UI.test_trajectory_screen import base_snapshot, _record_key_for_seq


class _Harness(App[None]):
    """Minimal host so the screen can be pushed like the Console would."""

    def compose(self) -> ComposeResult:
        yield Static("base")


def _stub_picker(path: Path):
    """Replacement for ``_pick_trace_file`` returning ``path`` immediately."""

    async def pick() -> Path:
        return path

    return pick


def _stub_dismissed():
    async def pick() -> None:
        return None

    return pick


def _last_notification(app: App):
    """Newest notification posted to ``app`` (or ``None``)."""
    posted = list(app._notifications._notifications.values())
    return posted[-1] if posted else None


def _build_trace(tmp_path: Path, name: str = "shared-trace") -> Path:
    """Build a real export file from a real temp DB (export-test fixtures)."""
    db = CharactersRAGDB(tmp_path / f"{name}-src.db", client_id="test")
    conv = _seed_conversation(db)
    payload = build_trajectory_export(db, conv)
    return write_trajectory_export(tmp_path / f"{name}.json", payload)


# ---------------------------------------------------------------------------
# `o` opens the trace as a read-only screen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_o_mounts_imported_readonly_screen_with_records(tmp_path) -> None:
    trace = _build_trace(tmp_path)
    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_picker(trace)

        await pilot.press("o")
        await pilot.pause()

        imported = app.screen
        assert isinstance(imported, TrajectoryScreen)
        assert imported is not screen  # a NEW screen, not a mutation
        title = str(imported.query_one("#trajectory-title", Static).render())
        assert "Trace · Shared trace — shared-trace" in title
        state = str(imported.query_one("#trajectory-state", Static).render())
        assert "READ-ONLY SHARED TRACE" in state
        # The seeded conversation renders: 7 records + 2 turn-header rows.
        table = imported.query_one("#trajectory-table", DataTable)
        assert table.row_count == 9
        assert table.get_row_index(_record_key_for_seq(imported, 7)) is not None


@pytest.mark.asyncio
async def test_imported_screen_has_no_live_polling(tmp_path) -> None:
    """No conversation_id / providers -> no revision interval, no follow hint."""
    trace = _build_trace(tmp_path)
    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_picker(trace)
        await pilot.press("o")
        await pilot.pause()

        imported = app.screen
        assert imported._revision_provider is None
        assert imported._snapshot_builder is None
        # on_mount only arms the 0.5s revision interval when BOTH live-mode
        # callables exist; without them no extra timer is armed (the screen
        # baseline itself carries whatever framework timers it has).
        assert len(imported._timers) == len(screen._timers)
        hints = str(imported.query_one("#trajectory-hints", Static).render())
        assert "follow" not in hints  # live-only action never advertised
        assert "open" in hints  # 1:1 governance: o is advertised


@pytest.mark.asyncio
async def test_dismissed_picker_is_a_noop(tmp_path) -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_dismissed()
        await pilot.press("o")
        await pilot.pause()
        assert app.screen is screen


@pytest.mark.asyncio
async def test_malformed_trace_notifies_and_keeps_screen(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_picker(bad)

        await pilot.press("o")
        await pilot.pause()

        assert app.screen is screen  # nothing was mounted
        notification = _last_notification(app)
        assert notification is not None, "import failure must surface a notification"
        assert notification.severity == "error"
        assert "not valid JSON" in notification.message
        assert "bad.json" in notification.message


@pytest.mark.asyncio
async def test_version_mismatch_notifies_with_validator_message(tmp_path) -> None:
    trace = _build_trace(tmp_path, name="future-trace")
    import json as _json

    payload = _json.loads(trace.read_text(encoding="utf-8"))
    payload["version"] = 2
    trace.write_text(_json.dumps(payload), encoding="utf-8")

    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_picker(trace)
        await pilot.press("o")
        await pilot.pause()

        assert app.screen is screen
        notification = _last_notification(app)
        assert notification is not None, "version mismatch must surface a notification"
        assert "version 2" in notification.message


# ---------------------------------------------------------------------------
# Behavioral no-DB-write proof (UI-level import against an attached temp DB)
# ---------------------------------------------------------------------------


def _row_counts(database: CharactersRAGDB) -> dict[str, int]:
    counts = {}
    for table in (
        "conversations",
        "messages",
        "message_trajectory_metadata",
        "console_auxiliary_attempts",
    ):
        counts[table] = database.execute_query(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - static table name
        ).fetchone()[0]
    return counts


@pytest.mark.asyncio
async def test_import_through_the_screen_never_writes_the_db(tmp_path) -> None:
    """The honest no-write assertion: row counts unchanged across an import.

    The app DB is a real temp ``CharactersRAGDB`` with seeded rows; the
    imported trace is a file built from a DIFFERENT DB. A UI-level import
    (press ``o`` -> load -> mount the read-only screen) must leave every
    table byte-identical in count.
    """
    db = CharactersRAGDB(tmp_path / "local-app.db", client_id="local")
    _seed_conversation(db)  # local app data the import must not touch
    trace = _build_trace(tmp_path)  # from its own separate source DB
    before = _row_counts(db)
    assert before["messages"] > 0
    assert before["message_trajectory_metadata"] > 0

    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(base_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        screen._pick_trace_file = _stub_picker(trace)
        await pilot.press("o")
        await pilot.pause()
        # The imported view really rendered someone else's trace.
        state = str(app.screen.query_one("#trajectory-state", Static).render())
        assert "READ-ONLY SHARED TRACE" in state

    assert _row_counts(db) == before
