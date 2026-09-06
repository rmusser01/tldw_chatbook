"""Focused presentation and interaction tests for native Canvas transcript cards."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.message_metadata import (
    CanvasCardMetadata,
    CanvasCardOriginMetadata,
    MessageMetadata,
)
from tldw_chatbook.Widgets.Console.console_canvas_card import (
    ConsoleCanvasCard,
    ConsoleCanvasCardOpenRequested,
    ConsoleCanvasCardPresentation,
    ConsoleCanvasOpenRecoveryCard,
    ConsoleCanvasOpenRetryRequested,
    canvas_card_signature,
    open_canvas_with_textual,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


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
        session_id="session-a",
        canvas_id=card.canvas_id,
        revision_id=card.revision_id,
        follow_latest=False,
    )
    following = ConsoleCanvasCardOpenRequested(
        session_id="session-a",
        canvas_id=card.canvas_id,
        revision_id=None,
        follow_latest=True,
    )

    assert exact.canvas_id == "canvas-a"
    assert exact.session_id == "session-a"
    assert exact.revision_id == "revision-7"
    assert exact.follow_latest is False
    assert following.follow_latest is True


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_canvas_control_styles_keep_geometry_with_targeted_subjects(size):
    card = ConsoleCanvasCard(
        _card(), session_id="session-a", message_id="style", card_index=0
    )
    recovery = ConsoleCanvasOpenRecoveryCard("http://127.0.0.1:43121/canvas/")

    class _StyledCardApp(App[None]):
        CSS = ConsoleCanvasCard.BUNDLED_CSS + ConsoleCanvasOpenRecoveryCard.BUNDLED_CSS

        def compose(self) -> ComposeResult:
            yield card
            yield recovery

    async with _StyledCardApp().run_test(size=size):
        buttons = list(card.query(Button))
        url = recovery.query_one("#console-canvas-recovery-url", Input)
        assert len(buttons) == 2
        for button in buttons:
            assert button.styles.min_width.value == 16
            assert button.styles.height.value == 3
            assert button.styles.margin.right == 1
        assert url.styles.width.value == 100
        assert url.styles.height.value == 3
        # Class-keyed subjects avoid billing every Button/Input in the app.
        assert all(button.has_class("console-canvas-card-action") for button in buttons)
        assert url.has_class("console-canvas-recovery-url")


@pytest.mark.asyncio
async def test_unavailable_canvas_card_disables_exact_reopen_but_keeps_head_route():
    widget = ConsoleCanvasCard(
        _card(reopenable=False),
        session_id="session-a",
        message_id="assistant-7",
        card_index=0,
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
async def test_reconciled_canvas_card_posts_its_current_render_session() -> None:
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Synthetic answer",
        id="assistant-card",
        metadata=MessageMetadata(
            canvas_cards=(
                CanvasCardMetadata(
                    canvas_id="canvas-a",
                    revision_id="revision-7",
                    title="Launch plan",
                    sequence=7,
                    digest="a" * 64,
                    status="updated",
                    origin=CanvasCardOriginMetadata("assistant-card", "run-7"),
                    reopenable=True,
                ),
            )
        ),
    )
    seen: list[ConsoleCanvasCardOpenRequested] = []

    class _CardApp(App[None]):
        def compose(self) -> ComposeResult:
            yield transcript

        def on_console_canvas_card_open_requested(
            self, event: ConsoleCanvasCardOpenRequested
        ) -> None:
            seen.append(event)

    transcript.set_messages([message], session_id="session-a")
    async with _CardApp().run_test() as pilot:
        await pilot.click("#canvas-open-revision-assistant-card-0")
        await pilot.pause()
        assert seen[-1].session_id == "session-a"

        transcript.set_messages([message], session_id="session-b")
        await transcript.refresh_messages()
        await pilot.pause()
        replacement = transcript.query_one(
            "#console-canvas-card-assistant-card-0", ConsoleCanvasCard
        )
        assert replacement.session_id == "session-b"
        await pilot.click("#canvas-open-revision-assistant-card-0")
        await pilot.pause()

        assert [event.session_id for event in seen] == ["session-a", "session-b"]


def test_canvas_row_build_uses_its_captured_session_after_transcript_switch() -> None:
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Synthetic answer",
        id="assistant-delayed-card",
        metadata=MessageMetadata(
            canvas_cards=(
                CanvasCardMetadata(
                    canvas_id="canvas-a",
                    revision_id="revision-7",
                    title="Launch plan",
                    sequence=7,
                    digest="a" * 64,
                    status="updated",
                    origin=CanvasCardOriginMetadata("assistant-delayed-card", "run-7"),
                    reopenable=True,
                ),
            )
        ),
    )

    transcript.set_messages([message], session_id="session-a")
    captured_row = next(
        row for row in transcript._flat_transcript_rows() if row.kind == "canvas-card"
    )
    transcript.set_messages([message], session_id="session-b")

    delayed_widget = transcript._build_row_widget(captured_row, track=False)

    assert isinstance(delayed_widget, ConsoleCanvasCard)
    assert delayed_widget.session_id == "session-a"


@pytest.mark.asyncio
async def test_browser_open_failure_card_keeps_copyable_url_and_retry_action():
    card = ConsoleCanvasOpenRecoveryCard("http://127.0.0.1:43121/canvas/#boot=first")
    seen = []

    class _CardApp(App[None]):
        def compose(self) -> ComposeResult:
            yield card

        def on_console_canvas_open_retry_requested(
            self, event: ConsoleCanvasOpenRetryRequested
        ) -> None:
            seen.append(event)

    async with _CardApp().run_test() as pilot:
        url = card.query_one("#console-canvas-recovery-url")
        assert url.value.endswith("#boot=first")
        card.update_url("http://127.0.0.1:43121/canvas/#boot=fresh")
        assert url.value.endswith("#boot=fresh")
        await pilot.click("#console-canvas-retry-open")
        await pilot.pause()
        assert len(seen) == 1


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
