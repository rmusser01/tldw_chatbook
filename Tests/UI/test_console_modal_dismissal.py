"""Contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.modal_dismissal import (
    SafeModalDismissMixin,
    is_modal_backdrop_click,
)


@dataclass(frozen=True)
class _FakeRegion:
    contains_point: bool

    def contains(self, _x: int, _y: int) -> bool:
        return self.contains_point


@dataclass(frozen=True)
class _FakeContent:
    region: _FakeRegion


@pytest.mark.parametrize(
    ("button", "known", "descendant", "contains", "expected"),
    [
        pytest.param(1, True, False, False, True, id="primary-outside"),
        pytest.param(1, True, True, False, False, id="primary-descendant"),
        pytest.param(1, True, False, True, False, id="primary-in-region"),
        pytest.param(2, True, False, False, False, id="non-primary-outside"),
        pytest.param(1, False, False, False, False, id="unknown-provenance"),
    ],
)
def test_classifier_identifies_only_known_primary_backdrop_clicks(
    button: int,
    known: bool,
    descendant: bool,
    contains: bool,
    expected: bool,
) -> None:
    content = _FakeContent(_FakeRegion(contains))

    assert (
        is_modal_backdrop_click(
            button=button,
            provenance_known=known,
            target_is_content_or_descendant=descendant,
            point_is_in_content_region=content.region.contains(7, 9),
        )
        is expected
    )


class _HostScreen(Screen[None]):
    """Revealed screen with the same optional focus seam as Console."""

    def __init__(self) -> None:
        super().__init__()
        self.composer_fallback_calls: list[bool] = []

    def compose(self) -> ComposeResult:
        yield Input(id="modal-test-opener")
        yield Input(id="modal-test-other-focus")
        yield Static("host", id="modal-test-host-label")

    def _focus_console_composer_if_needed(self, *, force: bool) -> None:
        self.composer_fallback_calls.append(force)


class _ModalHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.host = _HostScreen()

    async def on_mount(self) -> None:
        await self.push_screen(self.host)


class _NestedModal(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("nested", id="modal-test-nested")


CancelEffect = Callable[[], Awaitable[None]]


class _SafeTestModal(SafeModalDismissMixin, ModalScreen[bool | None]):
    SAFE_MODAL_CONTENT = "#modal-test-content"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    CSS = """
    _SafeTestModal {
        align: center middle;
    }

    #modal-test-content {
        width: 30;
        height: 7;
        background: $surface;
    }
    """

    def __init__(
        self,
        *,
        result: bool | None = False,
        cancel_effect: CancelEffect | None = None,
    ) -> None:
        super().__init__()
        self._cancel_result = result
        self._cancel_effect = cancel_effect

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-test-content"):
            yield Static("safe modal")
            yield Button("Cancel", id="modal-test-cancel")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._cancel_effect is not None:
            await self.run_cancel_effect_once(self._cancel_effect)
        self.dismiss_safe_once(self._cancel_result)

    @on(Button.Pressed, "#modal-test-cancel")
    async def _cancel_from_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")


async def _mount_modal(
    app: _ModalHarness,
    pilot,
    modal: _SafeTestModal,
    results: list[bool | None],
) -> Input:
    opener = app.host.query_one("#modal-test-opener", Input)
    opener.focus()
    await pilot.pause()
    assert app.host.focused is opener
    app.push_screen(modal, results.append)
    await pilot.pause()
    assert app.screen is modal
    return opener


def _outside_click(modal: _SafeTestModal) -> events.Click:
    return events.Click(
        modal,
        0,
        0,
        0,
        0,
        1,
        False,
        False,
        False,
        screen_x=0,
        screen_y=0,
    )


@pytest.mark.asyncio
async def test_single_shot_consumes_repeated_escape_and_backdrop_while_pending():
    entered = asyncio.Event()
    release = asyncio.Event()
    effect_calls = 0

    async def delayed_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        entered.set()
        await release.wait()

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=delayed_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        first_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await entered.wait()
        second_escape = asyncio.create_task(modal.action_request_safe_cancel())
        backdrop = _outside_click(modal)
        backdrop_request = asyncio.create_task(modal.on_click(backdrop))
        try:
            await asyncio.sleep(0)
            assert second_escape.done()
            assert backdrop_request.done()
            assert effect_calls == 1
            assert app.screen is modal
            assert backdrop._stop_propagation
            assert backdrop._no_default_action
        finally:
            release.set()
            await asyncio.gather(
                first_escape,
                second_escape,
                backdrop_request,
                return_exceptions=True,
            )
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_top_screen_check_preserves_nested_modal_and_retry_skips_effect():
    effect_calls = 0
    nested = _NestedModal()

    async def push_nested() -> None:
        nonlocal effect_calls
        effect_calls += 1
        app.push_screen(nested)

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=push_nested)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is nested
        assert modal.is_mounted
        assert results == []
        assert effect_calls == 1

        nested.dismiss(None)
        await pilot.pause()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is app.host
        assert results == [False]
        assert effect_calls == 1


@pytest.mark.asyncio
async def test_single_shot_cancel_effect_commitment_survives_exception():
    effect_calls = 0

    async def failing_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        raise RuntimeError("cancel failed")

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=failing_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        with pytest.raises(RuntimeError, match="cancel failed"):
            await modal.action_request_safe_cancel()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert effect_calls == 1
        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.parametrize("source", ["button", "backdrop"])
@pytest.mark.asyncio
async def test_visible_cancel_and_backdrop_return_exact_typed_value(source: str):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(result=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        if source == "button":
            await pilot.click("#modal-test-cancel")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_restore_focus_returns_to_the_mounted_opener():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        app.host.set_focus(app.host.query_one("#modal-test-other-focus", Input))
        assert app.host.focused is not opener

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.focused is opener
        assert app.host.composer_fallback_calls == []


@pytest.mark.asyncio
async def test_restore_focus_uses_console_fallback_when_opener_was_removed():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        await opener.remove()

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.composer_fallback_calls == [True]
