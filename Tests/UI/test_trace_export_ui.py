"""Trace v2 export preflight and Trace-screen collaboration UX."""

from __future__ import annotations

import asyncio
import json
import threading
from html import unescape
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, RadioButton, RadioSet, Static

from tldw_chatbook.Chat.trajectory_export import (
    TraceExportProfile,
    build_trace_export,
    write_trajectory_export,
)
from tldw_chatbook.Chat.trajectory_import import load_imported_trace
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
from tldw_chatbook.Widgets.Console import trace_export_dialog as export_dialog_module
from tldw_chatbook.Widgets.Console.trace_export_dialog import (
    TRACE_EXPORT_PROFILE_COPY,
    TRACE_EXPORT_PROFILE_LABELS,
    TraceExportDialog,
    full_trace_confirmation,
)
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_trace_responsive import _TraceHost
from Tests.UI.test_trajectory_screen import base_snapshot


class _Harness(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def compose(self) -> ComposeResult:
        yield Static("base")


def test_trace_export_publishes_shared_labels_and_full_warning() -> None:
    assert TRACE_EXPORT_PROFILE_LABELS[TraceExportProfile.REDACTED_DIAGNOSTIC] == (
        "Redacted diagnostic (recommended)"
    )
    assert "Credentials remain forbidden" in TRACE_EXPORT_PROFILE_COPY[
        TraceExportProfile.FULL_TRACE
    ]

    confirmation = full_trace_confirmation(noun="Trace")
    assert confirmation.title == "Export full Trace?"
    assert confirmation.confirm_label == "Export full trace"
    assert "injected instructions" in confirmation.message
    assert "Credentials remain structurally blocked" in confirmation.message


@pytest.mark.asyncio
async def test_preflight_defaults_to_redacted_diagnostic_and_explains_inventory() -> (
    None
):
    app = _TraceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()

        assert dialog.selected_profile is TraceExportProfile.REDACTED_DIAGNOSTIC
        summary = str(dialog.query_one("#trace-export-inventory", Static).render())
        assert "events" in summary
        assert "sensitive" in summary
        assert "sensitive fields" in summary
        assert "redacted" in summary
        policy = str(dialog.query_one("#trace-export-policy", Static).render())
        assert "Credentials are always blocked" in policy


@pytest.mark.asyncio
async def test_export_success_writes_importable_v2_bundle(tmp_path: Path) -> None:
    target = tmp_path / "shared-trace.json"
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
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
async def test_export_rejects_destination_that_fails_central_path_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "blocked-trace.json"
    validation_calls: list[str] = []

    def reject_destination(path: str, *, require_exists: bool = False) -> Path:
        validation_calls.append(path)
        assert require_exists is False
        raise ValueError("destination rejected by path policy")

    monkeypatch.setattr(
        export_dialog_module,
        "validate_path_simple",
        reject_destination,
        raising=False,
    )
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)

        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert validation_calls == [str(target)]
        assert not target.exists()
        assert app.screen is dialog
        status = str(dialog.query_one("#trace-export-status", Static).render())
        assert "destination rejected by path policy" in status


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
async def test_existing_destination_requires_replace_confirmation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing.json"
    target.write_text("keep me", encoding="utf-8")
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#trace-export-path", Input).value = str(target)

        async def keep_existing(_destination: Path) -> bool:
            return False

        dialog._confirm_overwrite = keep_existing
        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert target.read_text(encoding="utf-8") == "keep me"
        status = str(dialog.query_one("#trace-export-status", Static).render())
        assert "existing file was kept" in status


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
async def test_radio_selection_and_confirmed_full_trace_write(tmp_path: Path) -> None:
    target = tmp_path / "confirmed-full.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()
        profiles = dialog.query_one("#trace-export-profiles")
        profiles.focus()
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        full = dialog.query_one("#trace-export-profile-full", RadioButton)
        assert full.value
        assert dialog.selected_profile is TraceExportProfile.FULL_TRACE
        dialog.query_one("#trace-export-path", Input).value = str(target)

        async def confirm() -> bool:
            return True

        dialog._confirm_full_export = confirm
        await pilot.click("#trace-export-submit")
        await pilot.pause()

        assert load_imported_trace(target).manifest["profile"] == "full_trace"


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
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
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
        status_region = dialog.query_one("#trace-export-status").region
        assert status_region.bottom <= 18
        painted = unescape(app.export_screenshot(simplify=True)).replace(
            "\N{NO-BREAK SPACE}", " "
        )
        assert "disk unavailable" in painted


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


@pytest.mark.asyncio
async def test_compact_export_keeps_selected_profile_and_focus_visible() -> None:
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
        dialog = TraceExportDialog(base_snapshot())
        await app.push_screen(dialog)
        await pilot.pause()

        summary = dialog.query_one("#trace-export-selection", Static)
        assert "Profile: Redacted diagnostic" in str(summary.render())
        assert summary.region.y >= 0 and summary.region.bottom <= 18
        assert app.focused is dialog.query_one("#trace-export-path", Input)
        assert app.focused.region.y >= 0 and app.focused.region.bottom <= 18
        painted = unescape(app.export_screenshot(simplify=True)).replace(
            "\N{NO-BREAK SPACE}", " "
        )
        assert "Profile: Redacted diagnostic" in painted

        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.focused is dialog.query_one("#trace-export-profiles", RadioSet)
        focused_summary = str(summary.render())
        assert "↑/↓" in focused_summary
        assert "Enter apply" in focused_summary
        assert summary.has_class("is-selector-focused")
        focused_painted = unescape(app.export_screenshot(simplify=True)).replace(
            "\N{NO-BREAK SPACE}", " "
        )
        assert "Enter apply" in focused_painted

        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is dialog.query_one("#trace-export-path", Input)
        assert "Profile: Redacted diagnostic" in str(summary.render())
        assert not summary.has_class("is-selector-focused")


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
        for selector in (
            "#trace-export-inventory",
            "#trace-export-policy",
            "#trace-export-selection",
            "#trace-export-path",
            "#trace-export-status",
        ):
            region = dialog.query_one(selector).region
            assert region.y >= content.region.y
            assert region.bottom <= content.region.bottom


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
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
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
        assert "DIGEST VALID" in state
        assert "SOURCE NOT AUTHENTICATED" in state
        assert "privacy" in state
        for selector in (
            "#trajectory-state",
            "#trajectory-timeline",
            "#trajectory-table",
            "#trajectory-hints",
        ):
            assert imported.query_one(selector).region.bottom <= 18
        painted = unescape(app.export_screenshot(simplify=True)).replace(
            "\N{NO-BREAK SPACE}", " "
        )
        assert "w export" in painted
        assert "o import trace" in painted
        operation = next(
            record for record in imported._all_records() if record.kind == "trace_import"
        )
        assert operation.payload["manifest"] == imported._imported_trace.manifest
        assert operation.payload["integrity"]["verdict"] == "valid"
        assert operation.payload["integrity"]["authenticity"] is False
        assert operation.payload["privacy_inventory"] == (
            imported._imported_trace.privacy_inventory
        )

        table = imported.query_one("#trajectory-table", DataTable)
        table.move_cursor(row=table.row_count - 1)
        imported.action_toggle_inspector()
        await pilot.pause()
        details = str(
            imported.query_one("#trajectory-inspector-content", Static).render()
        )
        assert "source authenticity not established" in details
        assert imported.query_one("#trajectory-inspector", VerticalScroll).display


@pytest.mark.asyncio
async def test_import_validation_runs_off_event_loop_and_shows_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = write_trajectory_export(
        tmp_path / "large-shared.json",
        build_trace_export(base_snapshot(), exported_at="2026-08-23T09:00:00+00:00"),
    )
    started = threading.Event()
    release = threading.Event()
    real_load = load_imported_trace

    def slow_load(path: Path):
        started.set()
        assert release.wait(timeout=2)
        return real_load(path)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.trajectory_screen.load_imported_trace", slow_load
    )
    app = _Harness()
    async with app.run_test() as pilot:
        source = TrajectoryScreen(base_snapshot())
        await app.push_screen(source)
        await pilot.pause()

        async def pick() -> Path:
            return trace

        source._pick_trace_file = pick
        task = asyncio.create_task(source.action_open_trace())
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        state = str(source.query_one("#trajectory-state", Static).render())
        assert "IMPORTING" in state
        ticks = 0
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1
        assert ticks == 5
        release.set()
        await task
        await pilot.pause()
        assert isinstance(app.screen, TrajectoryScreen)
        assert app.screen is not source


@pytest.mark.asyncio
async def test_v2_digest_tamper_is_rejected_through_ui(tmp_path: Path) -> None:
    trace = write_trajectory_export(
        tmp_path / "tampered.json",
        build_trace_export(base_snapshot(), exported_at="2026-08-23T09:00:00+00:00"),
    )
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["events"][0]["status"] = "tampered"
    trace.write_text(json.dumps(payload), encoding="utf-8")
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

        assert app.screen is source
        notifications = list(app._notifications._notifications.values())
        assert notifications
        assert "digest" in notifications[-1].message.lower()
