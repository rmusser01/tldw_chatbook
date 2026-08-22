"""Physical root-management surface contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncRootRow,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio


class _Host(App[None]):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, snapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield LibraryNotesSyncRootsCanvas(self.snapshot)

    def on_library_notes_sync_roots_canvas_root_action_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_sync_roots_canvas_page_requested(self, message) -> None:
        self.messages.append(message)


def _snapshot():
    return replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="roots",
        roots=(
            LastingSyncRootRow(
                "root-1",
                "Research [2026]",
                "needs_attention",
                "review_changes",
                "⚠ Needs attention",
                "Review changes",
            ),
        ),
    )


def _frame(app: App[None]) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def test_root_row_renders_literal_name_status_and_contextual_actions() -> None:
    app = _Host(_snapshot())
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        painted = _frame(app)
        assert "Research [2026]" in painted
        assert "Needs attention" in painted
        assert app.query_one("#notes-sync-root-check-0", Button)
        assert app.query_one("#notes-sync-root-review-0", Button)
        assert not app.query("#notes-sync-root-resume-0")


async def test_declared_review_action_is_first_and_visually_primary() -> None:
    app = _Host(_snapshot())
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        row = app.query_one("#notes-sync-root-row-0")
        actions = list(row.query(Button))

        assert actions[0].id == "notes-sync-root-review-0"
        assert actions[0].has_class("console-action-primary")
        assert not actions[1].has_class("console-action-primary")


async def test_migration_review_is_a_physical_primary_action_at_60x20() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="roots",
        roots=(
            LastingSyncRootRow(
                "legacy-root-" + "a" * 40,
                "Sync folder (name unavailable before cutover)",
                "paused",
                "review_migration",
                "Ⅱ Migration review required",
                "Review migration",
            ),
        ),
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        button = app.query_one("#notes-sync-root-migration-0", Button)
        assert button.has_class("console-action-primary")
        assert await pilot.click(button)
        await pilot.pause()

    assert [(message.root_id, message.action) for message in app.messages] == [
        ("legacy-root-" + "a" * 40, "migration")
    ]


async def test_unimplemented_root_management_is_disabled_with_explicit_reason() -> None:
    app = _Host(_snapshot())
    async with app.run_test(size=(80, 24)) as pilot:
        assert await pilot.click("#notes-sync-root-check-0")
        retarget = app.query_one("#notes-sync-root-retarget-0", Button)
        disconnect = app.query_one("#notes-sync-root-disconnect-0", Button)
        for button in (retarget, disconnect):
            button.scroll_visible(immediate=True)
            assert button.disabled is True
            assert "unavailable in this release" in str(button.tooltip)
        await pilot.pause()
        await pilot.click(disconnect)
        await pilot.pause()
        painted = _frame(app)

    assert [(message.root_id, message.action) for message in app.messages] == [
        ("root-1", "check"),
    ]
    assert "Retarget and Disconnect are unavailable in this release" in painted


async def test_root_canvas_is_scrollable_contained_and_focusable_at_60x20() -> None:
    app = _Host(_snapshot())
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        first = app.query_one("#notes-sync-root-check-0", Button)
        first.focus()
        await pilot.pause()
        assert app.focused is first
        canvas = app.query_one(LibraryNotesSyncRootsCanvas)
        assert canvas.region.right <= 60
        assert canvas.region.bottom <= 20


async def test_opaque_root_id_is_message_payload_not_dom_id() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="roots",
        roots=(
            LastingSyncRootRow(
                "remote:profile.v1",
                "Remote profile",
                "paused",
                "resume_sync",
                "Ⅱ Paused",
                "Resume",
            ),
        ),
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        button = app.query_one("#notes-sync-root-resume-0", Button)
        assert ":" not in button.id and "." not in button.id
        assert button.name == "remote:profile.v1"
        assert await pilot.click(button)
        await pilot.pause()

    assert app.messages[0].root_id == "remote:profile.v1"


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        ("offline", "Reconnect the folder"),
        ("passive", "active process"),
    ),
)
async def test_non_authoritative_roots_disable_impossible_manual_check(
    status: str, reason: str
) -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="roots",
        roots=(
            LastingSyncRootRow(
                "root-1",
                "Research",
                status,
                "reconnect_folder" if status == "offline" else "open_active_process",
                f"⚠ {status.title()}",
                "Reconnect folder" if status == "offline" else "Open active process",
            ),
        ),
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        check = app.query_one("#notes-sync-root-check-0", Button)
        assert check.disabled is True
        assert reason in str(check.tooltip)
