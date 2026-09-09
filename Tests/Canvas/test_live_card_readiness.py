"""Behavioral coverage for the live Canvas F12 card-readiness adapter."""

import asyncio
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from Tests.Canvas.browser import canvas_live_chatbook_child
from tldw_chatbook.Widgets.Console.console_canvas_card import (
    ConsoleCanvasCard,
    ConsoleCanvasCardOpenRequested,
    ConsoleCanvasCardPresentation,
)


def _card(
    *,
    revision_id: str | None = "revision-target",
    session_id: str | None = "session-current",
    reopenable: bool = True,
    message_id: str = "assistant-card",
) -> ConsoleCanvasCard:
    return ConsoleCanvasCard(
        ConsoleCanvasCardPresentation(
            canvas_id="canvas-a",
            revision_id=revision_id,
            label="Canvas revision",
            digest="a" * 64,
            reopenable=reopenable,
            error_code=None if reopenable else "revision_unavailable",
        ),
        session_id=session_id,
        message_id=message_id,
        card_index=0,
    )


class _CardReadinessApp(App[None]):
    def __init__(self, *cards: ConsoleCanvasCard) -> None:
        super().__init__()
        self.cards = cards
        self.open_events: list[ConsoleCanvasCardOpenRequested] = []

    def compose(self) -> ComposeResult:
        yield Vertical(*self.cards, id="card-region")

    def on_console_canvas_card_open_requested(
        self, event: ConsoleCanvasCardOpenRequested
    ) -> None:
        self.open_events.append(event)


def _set_active_session(app: App[None], session_id: str | None) -> None:
    app.screen._console_chat_store = SimpleNamespace(active_session_id=session_id)


@pytest.mark.asyncio
async def test_wait_returns_the_exact_card_and_button_after_delayed_mount() -> None:
    card = _card()
    app = _CardReadinessApp()

    async with app.run_test():
        _set_active_session(app, "session-current")
        pending = asyncio.create_task(
            canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=20,
                interval=0.01,
            )
        )
        await asyncio.sleep(0)
        assert pending.done() is False

        await app.screen.query_one("#card-region", Vertical).mount(card)
        returned_card, returned_button = await pending

        assert returned_card is card
        assert returned_button is card.query_one("Button", Button)


@pytest.mark.asyncio
async def test_wait_skips_wrong_session_and_dispatches_current_card_once() -> None:
    wrong_session = _card(session_id="session-old", message_id="old")
    current_session = _card(session_id="session-current", message_id="current")
    app = _CardReadinessApp(wrong_session)

    async with app.run_test() as pilot:
        _set_active_session(app, "session-current")
        pending = asyncio.create_task(
            canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=20,
                interval=0.01,
            )
        )
        await asyncio.sleep(0)
        assert pending.done() is False

        await app.screen.query_one("#card-region", Vertical).mount(current_session)
        returned_card, returned_button = await pending
        returned_button.press()
        await pilot.pause()

        assert returned_card is current_session
        assert [(event.session_id, event.revision_id) for event in app.open_events] == [
            ("session-current", "revision-target")
        ]


@pytest.mark.asyncio
async def test_wait_refuses_an_unmounted_exact_card() -> None:
    card = _card()
    app = _CardReadinessApp()

    async with app.run_test():
        _set_active_session(app, "session-current")
        assert card.is_mounted is False

        with pytest.raises(RuntimeError, match=r"^canvas_card_not_ready\("):
            await canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=2,
                interval=0,
            )


@pytest.mark.asyncio
async def test_wait_refuses_a_disabled_exact_button() -> None:
    card = _card(reopenable=False)
    app = _CardReadinessApp(card)

    async with app.run_test():
        _set_active_session(app, "session-current")
        exact_button = card.query_one("Button", Button)
        assert exact_button.disabled is True

        with pytest.raises(RuntimeError, match=r"^canvas_card_not_ready\("):
            await canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=2,
                interval=0,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("active_session", [None, ""])
async def test_wait_requires_a_nonempty_current_session(
    active_session: str | None,
) -> None:
    card = _card(session_id=active_session)
    app = _CardReadinessApp(card)

    async with app.run_test():
        _set_active_session(app, active_session)

        with pytest.raises(RuntimeError, match=r"active_session=false"):
            await canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=2,
                interval=0,
            )


@pytest.mark.asyncio
async def test_wait_refuses_a_missing_target_revision() -> None:
    card = _card(revision_id=None)
    app = _CardReadinessApp(card)

    async with app.run_test():
        _set_active_session(app, "session-current")

        with pytest.raises(RuntimeError, match=r"revision_matches=0"):
            await canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision=None,
                attempts=2,
                interval=0,
            )


@pytest.mark.asyncio
async def test_wait_timeout_has_only_bounded_readiness_diagnostics() -> None:
    wrong_revision = _card(revision_id="revision-other")
    app = _CardReadinessApp(wrong_revision)

    async with app.run_test():
        _set_active_session(app, "session-current")

        with pytest.raises(RuntimeError) as raised:
            await canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-secret-value",
                attempts=3,
                interval=0,
            )

        assert str(raised.value) == (
            "canvas_card_not_ready(cards=1,revision_matches=0,mounted_matches=0,"
            "store_present=true,active_session=true,session_matches=0,buttons=0,"
            "enabled_buttons=0)"
        )
        assert "revision-secret-value" not in str(raised.value)
        assert "session-current" not in str(raised.value)


@pytest.mark.asyncio
async def test_wait_propagates_cancellation() -> None:
    app = _CardReadinessApp()

    async with app.run_test():
        _set_active_session(app, "session-current")
        pending = asyncio.create_task(
            canvas_live_chatbook_child._wait_for_exact_canvas_card(
                app.screen,
                target_revision="revision-target",
                attempts=1_000,
                interval=0.01,
            )
        )
        await asyncio.sleep(0)
        pending.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pending
