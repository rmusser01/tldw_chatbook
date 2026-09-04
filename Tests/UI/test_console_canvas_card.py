"""Focused presentation and interaction tests for native Canvas transcript cards."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Console.console_canvas_card import (
    ConsoleCanvasCard,
    ConsoleCanvasCardOpenRequested,
    ConsoleCanvasCardPresentation,
    canvas_card_signature,
    open_canvas_with_textual,
)


def _card(*, reopenable: bool = True) -> ConsoleCanvasCardPresentation:
    return ConsoleCanvasCardPresentation(
        canvas_id="canvas-a",
        revision_id="revision-7",
        label="Launch plan · revision 7 · updated",
        digest="a" * 64,
        reopenable=reopenable,
        error_code=None if reopenable else "revision_unavailable",
    )


def test_canvas_card_is_source_free_and_reconciles_by_exact_revision():
    card = _card()

    assert canvas_card_signature(card) == (
        "canvas-card",
        "canvas-a",
        "revision-7",
        "Launch plan · revision 7 · updated",
        "a" * 64,
        True,
        None,
    )
    assert "<!doctype" not in repr(card)


def test_canvas_card_messages_distinguish_exact_revision_from_following_head():
    card = _card()

    exact = ConsoleCanvasCardOpenRequested(
        canvas_id=card.canvas_id,
        revision_id=card.revision_id,
        follow_latest=False,
    )
    following = ConsoleCanvasCardOpenRequested(
        canvas_id=card.canvas_id,
        revision_id=None,
        follow_latest=True,
    )

    assert exact.canvas_id == "canvas-a"
    assert exact.revision_id == "revision-7"
    assert exact.follow_latest is False
    assert following.follow_latest is True


@pytest.mark.asyncio
async def test_unavailable_canvas_card_disables_exact_reopen_but_keeps_head_route():
    widget = ConsoleCanvasCard(
        _card(reopenable=False), message_id="assistant-7", card_index=0
    )

    class _CardApp(App[None]):
        def compose(self) -> ComposeResult:
            yield widget

    async with _CardApp().run_test():
        exact = widget.query_one("#canvas-open-revision-assistant-7-0")
        following = widget.query_one("#canvas-follow-latest-assistant-7-0")

        assert exact.disabled is True
        assert "unavailable" in str(exact.tooltip).lower()
        assert following.disabled is False


@pytest.mark.asyncio
async def test_native_open_uses_textual_and_leaves_copyable_url_on_failure():
    class _Gateway:
        async def open_shell(self, scope, *, opener):
            try:
                opener("http://127.0.0.1:43121/canvas/#boot=once")
            except RuntimeError:
                return type(
                    "Launch",
                    (),
                    {
                        "opened": False,
                        "browser_url": "http://127.0.0.1:43121/canvas/#boot=once",
                    },
                )()
            raise AssertionError("the opener should fail")

    class _App:
        def __init__(self):
            self.notices = []

        def open_url(self, url):
            raise RuntimeError("platform opener unavailable")

        def notify(self, copy, *, severity):
            self.notices.append((copy, severity))

    app = _App()
    launch = await open_canvas_with_textual(_Gateway(), object(), app)

    assert launch.opened is False
    assert app.notices == [
        (
            "Could not open a browser — http://127.0.0.1:43121/canvas/#boot=once",
            "error",
        )
    ]
