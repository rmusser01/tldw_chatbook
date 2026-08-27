"""Presentation contracts for the Console fork naming modal."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Widgets.Console.console_fork_chat_modal import (
    ConsoleForkChatModal,
    ConsoleForkDialogSummary,
    ConsoleForkSubmitResult,
)


def _summary(*, temporary: bool = False) -> ConsoleForkDialogSummary:
    return ConsoleForkDialogSummary(
        default_title="Forked from Research notes",
        boundary_label="Through Assistant 8",
        boundary_excerpt="The retrieval results suggest a useful answer.",
        message_count=8,
        response_variant="showing response 2 of 3",
        destination=(
            "Temporary chat · Save later to keep it"
            if temporary
            else "Saved chat · Research Workspace"
        ),
        temporary=temporary,
        includes_attachments=True,
        includes_citations=not temporary,
        contains_video=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (100, 30), (80, 24)])
async def test_fork_modal_editing_summary_and_layout_fit(size):
    app = ConsolidatedCSSApp()
    submissions: list[ConsoleForkSubmitResult] = []
    modal = ConsoleForkChatModal(_summary(), on_submit=submissions.append)

    async with app.run_test(size=size) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        title = modal.query_one("#console-fork-chat-title", Input)
        copy = " ".join(
            str(widget.render())
            for widget in modal.query(".console-fork-chat-summary")
            if isinstance(widget, Static)
        )
        panel = modal.query_one("#console-fork-chat-modal")

        assert title.value == "Forked from Research notes"
        assert title.max_length == 60
        assert title.has_focus
        assert modal.state == "editing"
        assert "Through Assistant 8" in copy
        assert "The retrieval results suggest" in copy
        assert "8 messages · showing response 2 of 3" in copy
        assert "Creates: Saved chat · Research Workspace" in copy
        assert "attachments and cited source details" in copy
        assert "new private working files" in copy
        assert "video will appear as unavailable" in copy
        assert panel.region.x >= 0 and panel.region.y >= 0
        assert panel.region.right <= size[0] and panel.region.bottom <= size[1]


@pytest.mark.asyncio
async def test_fork_modal_temporary_disclosure_is_truthful():
    app = ConsolidatedCSSApp()
    modal = ConsoleForkChatModal(_summary(temporary=True), on_submit=lambda _: None)

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        visible = " ".join(
            str(widget.render())
            for widget in modal.query("Static")
            if widget.display
        )
        assert "Temporary chat · Save later to keep it" in visible
        assert "Saving this fork will not save the original chat" in visible
        assert "Citation markers remain" in visible
        assert "source inspector details are not copied" in visible


@pytest.mark.asyncio
async def test_fork_modal_enter_normalizes_once_and_fences_replay():
    app = ConsolidatedCSSApp()
    submissions: list[ConsoleForkSubmitResult] = []
    modal = ConsoleForkChatModal(_summary(), on_submit=submissions.append)

    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        title = modal.query_one("#console-fork-chat-title", Input)
        title.value = "  My   focused\n fork  "

        await pilot.press("enter", "enter")
        modal.query_one("#console-fork-chat-confirm", Button).press()
        await pilot.pause()

        assert submissions == [ConsoleForkSubmitResult(title="My focused fork")]
        assert modal.state == "validating"
        assert modal.query_one("#console-fork-chat-confirm", Button).disabled
        assert "Checking fork" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )


@pytest.mark.asyncio
async def test_fork_modal_blank_title_stays_editing_with_inline_error():
    app = ConsolidatedCSSApp()
    submit = Mock()
    modal = ConsoleForkChatModal(_summary(), on_submit=submit)

    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-fork-chat-title", Input).value = "   "
        await pilot.press("enter")

        assert modal.state == "editing"
        assert "cannot be blank" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )
        submit.assert_not_called()


@pytest.mark.asyncio
async def test_fork_modal_state_contract_and_escape_semantics():
    app = ConsolidatedCSSApp()
    cancellations: list[str] = []
    modal = ConsoleForkChatModal(
        _summary(),
        on_submit=lambda _: None,
        on_cancel=lambda: cancellations.append("cancelled"),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        modal.show_committing()
        await pilot.pause()
        assert modal.state == "committing"
        assert modal.query_one("#console-fork-chat-title", Input).disabled
        assert modal.query_one("#console-fork-chat-cancel", Button).disabled
        await modal.action_request_safe_cancel()
        assert app.screen is modal
        assert "finishing and can no longer be cancelled" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )

        modal.show_precommit_error("Database is busy. Try again.")
        await pilot.pause()
        assert modal.state == "precommit_error"
        assert not modal.query_one("#console-fork-chat-title", Input).disabled
        assert modal.query_one("#console-fork-chat-confirm", Button).label.plain == "Retry"

        modal.show_stale_source()
        await pilot.pause()
        assert modal.state == "stale_source"
        assert "This chat changed. Close and choose Fork again." in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )
        assert modal.query_one("#console-fork-chat-confirm", Button).display is False

        modal.show_created_not_opened(
            title="My focused fork",
            identity="saved chat 12",
            detail="The saved fork exists but could not be opened.",
        )
        await pilot.pause()
        assert modal.state == "created_not_opened"
        assert "saved chat 12" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )
        assert modal.query_one("#console-fork-chat-open", Button).display


@pytest.mark.asyncio
async def test_fork_modal_validation_escape_cancels_and_invalidates_once():
    app = ConsolidatedCSSApp()
    cancellations: list[str] = []
    results: list[object] = []
    modal = ConsoleForkChatModal(
        _summary(),
        on_submit=lambda _: None,
        on_cancel=lambda: cancellations.append("cancelled"),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.press("enter")
        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is not modal
        assert cancellations == ["cancelled"]
        assert results == [None]
