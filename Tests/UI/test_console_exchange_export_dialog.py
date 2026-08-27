"""Selected-call governed export dialog behavior and compact geometry."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from textual.app import ComposeResult
from textual.widgets import Input, RadioButton, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.trajectory_export import TraceExportProfile
from tldw_chatbook.Widgets.Console.console_exchange_export_dialog import (
    ConsoleExchangeExportDialog,
)


def _capture(detail: CaptureDetail = CaptureDetail.FULL) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag="run",
        seq=1,
        created_at="2026-08-26T00:00:00Z",
        provider="anthropic",
        model="claude-test",
        endpoint="https://example.test/v1",
        request={"messages_payload": [{"role": "user", "content": "hello"}]},
        response={"content": "answer", "tool_calls": []},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=detail,
    )


class _Harness(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.clipboard_items: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("background")

    def copy_to_clipboard(self, text: str) -> None:
        self.clipboard_items.append(text)


@pytest.mark.asyncio
async def test_safe_capture_disables_full_with_visible_reason_at_80x24() -> None:
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(CaptureDetail.SAFE),
            expected_capture_revision=4,
            capture_revision_provider=lambda: 4,
        )
        await app.push_screen(dialog)
        await pilot.pause()

        assert dialog.selected_profile is TraceExportProfile.REDACTED_DIAGNOSTIC
        assert dialog.query_one("#exchange-export-profile-full", RadioButton).disabled
        reason = dialog.query_one("#exchange-export-full-reason", Static)
        assert "captured in Safe mode" in str(reason.render())
        assert reason.region.width > 0 and reason.region.bottom <= 24
        actions = dialog.query_one("#exchange-export-actions")
        assert actions.region.height > 0 and actions.region.bottom <= 24


@pytest.mark.asyncio
async def test_clipboard_revalidates_revision_immediately_before_disclosure() -> None:
    revision = 7
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=7,
            capture_revision_provider=lambda: revision,
        )
        await app.push_screen(dialog)
        await pilot.pause()

        async def stale_after_projection(_profile: TraceExportProfile):
            nonlocal revision
            projection = dialog._project(_profile)
            revision = 8
            return projection

        dialog._project_async = stale_after_projection
        assert await dialog.export_selected() is False

        assert app.clipboard_items == []
        assert dialog._projection is None
        assert "Stored captures changed" in str(
            dialog.query_one("#exchange-export-status", Static).render()
        )


@pytest.mark.asyncio
async def test_every_full_clipboard_action_requires_fresh_confirmation() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        await dialog.select_profile(TraceExportProfile.FULL_TRACE)
        confirmations = 0

        async def confirm() -> bool:
            nonlocal confirmations
            confirmations += 1
            return True

        dialog._confirm_full_export = confirm
        assert await dialog.export_selected() is True
        assert await dialog.export_selected() is True

        assert confirmations == 2
        assert len(app.clipboard_items) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", ["full", "overwrite"])
async def test_revision_change_during_confirmation_blocks_projection(
    confirmation: str, tmp_path: Path
) -> None:
    revision = 1
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: revision,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        if confirmation == "full":
            await dialog.select_profile(TraceExportProfile.FULL_TRACE)

            async def confirm_full() -> bool:
                nonlocal revision
                revision = 2
                return True

            dialog._confirm_full_export = confirm_full
        else:
            target = tmp_path / "existing.json"
            target.write_text("keep", encoding="utf-8")
            await dialog.select_destination("file")
            dialog.query_one("#exchange-export-path", Input).value = str(target)

            async def confirm_overwrite(_path: Path) -> bool:
                nonlocal revision
                revision = 2
                return True

            dialog._confirm_overwrite = confirm_overwrite
        dialog._project_async = AsyncMock()

        assert await dialog.export_selected() is False
        dialog._project_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_file_export_validates_overwrite_and_uses_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "exchange.json"
    target.write_text("keep", encoding="utf-8")
    writes: list[tuple[Path, str, dict[str, object]]] = []
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_exchange_export_dialog.atomic_write_text",
        lambda path, text, **kwargs: writes.append((Path(path), text, kwargs)),
    )
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        await dialog.select_destination("file")
        dialog.query_one("#exchange-export-path", Input).value = str(target)

        async def decline(_path: Path) -> bool:
            return False

        dialog._confirm_overwrite = decline
        assert await dialog.export_selected() is False
        assert writes == []

        async def confirm(_path: Path) -> bool:
            return True

        dialog._confirm_overwrite = confirm
        assert await dialog.export_selected() is True
        assert writes and writes[0][0] == target
        assert writes[0][2]["mode"] == 0o600
        assert writes[0][2]["overwrite"] is True


@pytest.mark.asyncio
async def test_file_appearing_after_validation_is_not_overwritten(
    tmp_path: Path,
) -> None:
    target = tmp_path / "appeared.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        await dialog.select_destination("file")
        dialog.query_one("#exchange-export-path", Input).value = str(target)

        async def project_after_competing_create(profile: TraceExportProfile):
            target.write_text("other writer", encoding="utf-8")
            return dialog._project(profile)

        dialog._project_async = project_after_competing_create

        assert await dialog.export_selected() is False
        assert target.read_text(encoding="utf-8") == "other writer"
        assert "appeared" in str(
            dialog.query_one("#exchange-export-status", Static).render()
        )


@pytest.mark.asyncio
async def test_new_file_export_is_owner_readable_only(tmp_path: Path) -> None:
    target = tmp_path / "private.json"
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        await dialog.select_destination("file")
        dialog.query_one("#exchange-export-path", Input).value = str(target)

        assert await dialog.export_selected() is True

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_escape_cancels_and_returns_to_previous_focus() -> None:
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        background = app.query_one(Static)
        background.can_focus = True
        background.focus()
        dialog = ConsoleExchangeExportDialog(
            _capture(),
            expected_capture_revision=1,
            capture_revision_provider=lambda: 1,
        )
        await app.push_screen(dialog)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert app.screen is app.screen_stack[0]
        assert app.focused is background
