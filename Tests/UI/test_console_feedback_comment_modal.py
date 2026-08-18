"""Console selection feedback comment modal (selection phase 3, task 4).

Covers: Cancel ≡ Escape ≡ backdrop (all dismiss ``None`` = feedback
abandoned), Submit button and Enter both returning the stripped comment,
an empty or whitespace-only comment dismissing ``""`` (feedback sent
WITHOUT a comment — the comment is optional, spec §3), the read-only
quote preview (short quotes verbatim, oversized quotes capped with the
truncation marker), and the action string driving the header copy.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, Static

from tldw_chatbook.Widgets.Console.console_feedback_comment_modal import (
    PREVIEW_QUOTE_CAP,
    PREVIEW_TRUNCATION_MARKER,
    ConsoleFeedbackCommentModal,
)

QUOTE = "the quoted transcript text"


class _CommentModalApp(App[None]):
    CSS = """
    Screen { align: center middle; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


def _modal(
    *, action: str = "comment", quote: str = QUOTE
) -> ConsoleFeedbackCommentModal:
    return ConsoleFeedbackCommentModal(action=action, quote=quote)


def _static_text(modal: ConsoleFeedbackCommentModal, selector: str) -> str:
    return str(modal.query_one(selector, Static).render())


# ---------------------------------------------------------------------------
# Cancellation: Cancel ≡ Escape ≡ backdrop, all returning None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "gesture"),
    [
        ("cancel-button", "click"),
        ("escape", "press"),
        ("backdrop", "click"),
    ],
)
async def test_cancel_sources_dismiss_none(source: str, gesture: str) -> None:
    app = _CommentModalApp()
    modal = _modal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        if source == "cancel-button":
            await pilot.click("#console-feedback-comment-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()
        await pilot.pause()

    assert app.results == [None]


# ---------------------------------------------------------------------------
# Submission: Submit button and Enter return the stripped comment
# ---------------------------------------------------------------------------


async def test_submit_button_returns_comment_text() -> None:
    app = _CommentModalApp()
    modal = _modal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        modal.query_one("#console-feedback-comment-input", Input).value = (
            "tighten the error handling"
        )
        await pilot.click("#console-feedback-comment-submit")
        await pilot.pause()

    assert app.results == ["tighten the error handling"]


async def test_enter_returns_comment_text() -> None:
    app = _CommentModalApp()
    modal = _modal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        comment_input = modal.query_one("#console-feedback-comment-input", Input)
        comment_input.value = "  looks good to me  "
        comment_input.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert app.results == ["looks good to me"]


@pytest.mark.parametrize("value", ["", "   ", "\t"])
async def test_blank_comment_submits_empty_string(value: str) -> None:
    """Submitting without a comment sends feedback anyway (comment optional).

    ``""`` means "submit, no comment"; only Cancel/Escape/backdrop return
    ``None`` and abandon the feedback.
    """
    app = _CommentModalApp()
    modal = _modal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        modal.query_one("#console-feedback-comment-input", Input).value = value
        await pilot.click("#console-feedback-comment-submit")
        await pilot.pause()

    assert app.results == [""]


# ---------------------------------------------------------------------------
# Quote preview: read-only, capped
# ---------------------------------------------------------------------------


async def test_quote_preview_shows_short_quote_verbatim() -> None:
    app = _CommentModalApp()
    modal = _modal(quote=QUOTE)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert _static_text(modal, "#console-feedback-comment-quote") == QUOTE


async def test_quote_preview_caps_oversized_quote() -> None:
    oversized = "x" * (PREVIEW_QUOTE_CAP * 2)
    app = _CommentModalApp()
    modal = _modal(quote=oversized)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        preview = _static_text(modal, "#console-feedback-comment-quote")
        expected = (
            oversized[: PREVIEW_QUOTE_CAP - len(PREVIEW_TRUNCATION_MARKER)]
            + PREVIEW_TRUNCATION_MARKER
        )
        assert preview == expected
        assert len(preview) == PREVIEW_QUOTE_CAP


# ---------------------------------------------------------------------------
# Header copy driven by the action string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected_header"),
    [
        ("request-changes", "Request changes — leave a comment"),
        ("lgm", "LGTM — leave a comment"),
        ("comment", "Comment on selection"),
        ("unknown-action", "Comment on selection"),
    ],
)
async def test_action_drives_header_text(action: str, expected_header: str) -> None:
    app = _CommentModalApp()
    modal = _modal(action=action)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert _static_text(modal, "#console-feedback-comment-header") == (
            expected_header
        )
