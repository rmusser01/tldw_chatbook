"""Trace v2 export preflight and Trace-screen collaboration UX."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from tldw_chatbook.Chat.trajectory_export import (
    TraceExportProfile,
    build_trace_export,
    write_trajectory_export,
)
from tldw_chatbook.Chat.trajectory_import import load_imported_trace
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
from tldw_chatbook.Widgets.Console.trace_export_dialog import TraceExportDialog
from Tests.UI.test_trace_responsive import _TraceHost
from Tests.UI.test_trajectory_screen import base_snapshot


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base")


@pytest.mark.asyncio
async def test_preflight_defaults_to_redacted_diagnostic_and_explains_inventory() -> (
    None
):
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()

        assert dialog.selected_profile is TraceExportProfile.REDACTED_DIAGNOSTIC
        summary = str(dialog.query_one("#trace-export-inventory", Static).render())
        assert "events" in summary
        assert "sensitive" in summary
        assert "redacted" in summary
        policy = str(dialog.query_one("#trace-export-policy", Static).render())
        assert "Credentials are always blocked" in policy


@pytest.mark.asyncio
async def test_export_success_writes_importable_v2_bundle(tmp_path: Path) -> None:
    target = tmp_path / "shared-trace.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)

        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert target.exists()
        imported = load_imported_trace(target)
        assert imported.manifest["profile"] == "redacted_diagnostic"


@pytest.mark.asyncio
async def test_cancel_never_writes(tmp_path: Path) -> None:
    target = tmp_path / "cancelled.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)

        await pilot.click("#trace-export-cancel")
        await pilot.pause()

        assert not target.exists()


@pytest.mark.asyncio
async def test_full_trace_requires_second_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "full.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        await dialog.select_profile(TraceExportProfile.FULL_TRACE)
        dialog.query_one("#trace-export-path", Input).value = str(target)

        async def decline() -> bool:
            return False

        dialog._confirm_full_export = decline
        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert not target.exists()
        assert app.screen is dialog


@pytest.mark.asyncio
async def test_write_failure_preserves_profile_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "failed.json"

    def fail(_path, _payload):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.trace_export_dialog.write_trajectory_export",
        fail,
    )
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)

        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert app.screen is dialog
        assert dialog.selected_profile is TraceExportProfile.REDACTED_DIAGNOSTIC
        assert dialog.query_one("#trace-export-path", Input).value == str(target)
        error = str(dialog.query_one("#trace-export-status", Static).render())
        assert "disk unavailable" in error


@pytest.mark.asyncio
async def test_export_dialog_remains_reachable_at_60_by_18() -> None:
    app = _Harness()
    async with app.run_test(size=(60, 18)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()

        content = dialog.query_one("#trace-export-dialog")
        assert content.region.x >= 0 and content.region.right <= 60
        assert content.region.y >= 0 and content.region.bottom <= 18
        assert dialog.query_one("#trace-export-submit").display


@pytest.mark.parametrize("size", [(60, 18), (80, 24), (100, 30), (120, 35)])
@pytest.mark.asyncio
async def test_export_dialog_composites_with_production_css_at_supported_widths(
    size: tuple[int, int],
) -> None:
    app = _TraceHost()
    async with app.run_test(size=size) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()

        content = dialog.query_one("#trace-export-dialog")
        actions = dialog.query_one("#trace-export-actions")
        assert content.region.x >= 0 and content.region.right <= size[0]
        assert content.region.y >= 0 and content.region.bottom <= size[1]
        assert actions.region.y >= content.region.y
        assert actions.region.bottom <= content.region.bottom
        assert dialog.query_one("#trace-export-submit").region.width >= 10


def test_trace_screen_advertises_write_export_without_displacing_clear_filters() -> (
    None
):
    bindings = {binding.key: binding.action for binding in TrajectoryScreen.BINDINGS}
    assert bindings["w"] == "export_trace"
    assert bindings["x"] == "clear_filters"
    assert ("w", "export trace") in TrajectoryScreen.TRAJECTORY_SHORTCUTS
    assert ("o", "import trace") in TrajectoryScreen.TRAJECTORY_SHORTCUTS


@pytest.mark.asyncio
async def test_w_opens_export_preflight_from_the_trace_screen() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        await app.push_screen(TrajectoryScreen(base_snapshot()))
        await pilot.pause()

        await pilot.press("w")
        await pilot.pause()

        assert isinstance(app.screen, TraceExportDialog)


@pytest.mark.asyncio
async def test_v2_import_surfaces_profile_integrity_privacy_and_import_event(
    tmp_path: Path,
) -> None:
    trace = write_trajectory_export(
        tmp_path / "v2-shared.json",
        build_trace_export(base_snapshot(), exported_at="2026-08-23T09:00:00+00:00"),
    )
    app = _Harness()
    async with app.run_test() as pilot:
        source = TrajectoryScreen(base_snapshot())
        await app.push_screen(source)
        await pilot.pause()

        async def pick() -> Path:
            return trace

        source._pick_trace_file = pick
        await pilot.press("o")
        await pilot.pause()

        imported = app.screen
        state = str(imported.query_one("#trajectory-state", Static).render())
        assert "READ-ONLY SHARED TRACE" in state
        assert "v2 redacted diagnostic" in state
        assert "INTEGRITY VERIFIED" in state
        assert "privacy" in state
        assert any(record.kind == "trace_import" for record in imported._all_records())
