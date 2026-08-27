"""Presentation contracts for the Console fork naming modal."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.events import Key
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


def _assert_inside(inner, outer) -> None:
    assert inner.region.x >= outer.content_region.x
    assert inner.region.y >= outer.content_region.y
    assert inner.region.right <= outer.content_region.right
    assert inner.region.bottom <= outer.content_region.bottom


def _painted_widget_text(modal, widget) -> str:
    strips = modal._compositor.render_strips()
    visible_rows = strips[
        max(0, widget.region.y) : min(len(strips), widget.region.bottom)
    ]
    return "\n".join(
        row.text[max(0, widget.region.x) : widget.region.right] for row in visible_rows
    )


def _painted_button_label(modal, button: Button) -> str:
    return _painted_widget_text(modal, button)


def _assert_actions_clear_visible_content(modal: ConsoleForkChatModal) -> None:
    panel = modal.query_one("#console-fork-chat-modal")
    actions = modal.query_one("#console-fork-chat-actions")
    content = [
        *panel.children,
        modal.query_one("#console-fork-chat-title", Input),
        modal.query_one("#console-fork-chat-status", Static),
    ]
    for child in dict.fromkeys(content):
        if (
            child is actions
            or not child.display
            or not _painted_widget_text(modal, child)
        ):
            continue
        overlaps = (
            child.region.x < actions.region.right
            and child.region.right > actions.region.x
            and child.region.y < actions.region.bottom
            and child.region.bottom > actions.region.y
        )
        assert not overlaps, (child.id, child.region, actions.region)


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
        cancel = modal.query_one("#console-fork-chat-cancel", Button)
        confirm = modal.query_one("#console-fork-chat-confirm", Button)
        _assert_actions_clear_visible_content(modal)
        assert cancel.label.plain in _painted_button_label(modal, cancel)
        assert confirm.label.plain in _painted_button_label(modal, confirm)

        for selector in (
            "#console-fork-chat-title",
            "#console-fork-chat-disclosure",
            "#console-fork-chat-cancel",
            "#console-fork-chat-confirm",
        ):
            child = modal.query_one(selector)
            child.focus()
            child.scroll_visible()
            await pilot.pause()
            _assert_inside(child, panel)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (100, 30), (80, 24)])
async def test_fork_modal_state_copy_stays_visible_above_actions(size):
    app = ConsolidatedCSSApp()
    modal = ConsoleForkChatModal(_summary(), on_submit=lambda _: None)

    async with app.run_test(size=size) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        status = modal.query_one("#console-fork-chat-status", Static)

        modal.show_validating()
        await pilot.pause()
        _assert_actions_clear_visible_content(modal)
        assert "Checking fork…" in _painted_widget_text(modal, status)

        modal.show_committing()
        await pilot.pause()
        _assert_actions_clear_visible_content(modal)
        assert "Forking…" in _painted_widget_text(modal, status)

        await modal.action_request_safe_cancel()
        await pilot.pause()
        _assert_actions_clear_visible_content(modal)
        assert (
            "Fork creation is finishing and can no longer be cancelled."
            in _painted_widget_text(modal, status)
        )


@pytest.mark.asyncio
async def test_fork_modal_temporary_disclosure_is_truthful():
    app = ConsolidatedCSSApp()
    modal = ConsoleForkChatModal(_summary(temporary=True), on_submit=lambda _: None)

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        visible = " ".join(
            str(widget.render()) for widget in modal.query("Static") if widget.display
        )
        assert "Temporary chat · Save later to keep it" in visible
        assert "Saving this fork will not save the original chat" in visible
        assert "Includes sent attachments" in visible
        assert "Citation markers remain" in visible
        assert "source inspector details are not copied" in visible


@pytest.mark.asyncio
async def test_fork_modal_swallows_keys_queued_before_mount():
    app = ConsolidatedCSSApp()
    modal = ConsoleForkChatModal(_summary(), on_submit=lambda _: None)

    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        title = modal.query_one("#console-fork-chat-title", Input)

        stale = Key(key="f", character="f")
        stale.time = 0.0
        title.post_message(stale)
        await pilot.pause()
        await pilot.pause()

        assert title.value == "Forked from Research notes"


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
        assert (
            modal.query_one("#console-fork-chat-confirm", Button).label.plain
            == "Forking…"
        )
        assert all(button.disabled for button in modal.query(Button))
        await modal.action_request_safe_cancel()
        assert app.screen is modal
        assert "finishing and can no longer be cancelled" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )

        modal.show_precommit_error("Database is busy. Try again.")
        await pilot.pause()
        assert modal.state == "precommit_error"
        assert not modal.query_one("#console-fork-chat-title", Input).disabled
        assert (
            modal.query_one("#console-fork-chat-confirm", Button).label.plain == "Retry"
        )

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
@pytest.mark.parametrize("state", ["editing", "validating"])
async def test_fork_modal_backdrop_cancels_even_when_disclosure_is_open(state):
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
        modal.query_one("#console-fork-chat-disclosure", Button).press()
        await pilot.pause()
        if state == "validating":
            modal.query_one("#console-fork-chat-title", Input).focus()
            await pilot.press("enter")
            assert modal.state == "validating"

        await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert app.screen is not modal
        assert cancellations == ["cancelled"]


@pytest.mark.asyncio
async def test_fork_modal_escape_closes_disclosure_before_cancelling():
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
        modal.query_one("#console-fork-chat-disclosure", Button).press()
        await pilot.pause()

        await modal.request_safe_cancel(source="escape")
        await pilot.pause()

        assert app.screen is modal
        assert cancellations == []
        assert modal.query_one("#console-fork-chat-exclusions", Static).display is False


@pytest.mark.asyncio
async def test_fork_modal_committing_backdrop_explains_with_disclosure_open():
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
        modal.query_one("#console-fork-chat-disclosure", Button).press()
        await pilot.pause()
        modal.show_committing()

        await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert app.screen is modal
        assert cancellations == []
        assert modal.query_one("#console-fork-chat-exclusions", Static).display
        assert "finishing and can no longer be cancelled" in str(
            modal.query_one("#console-fork-chat-status", Static).render()
        )


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
