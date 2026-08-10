"""Focused Pilot coverage for the generated-video capacity choice modal."""

from __future__ import annotations

from typing import cast

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    CapacityAction,
    CapacityReason,
    ConsoleVideoCapacityModal,
)


class _ModalHost(App[None]):
    """Small real Textual host used to exercise modal interaction."""


def _button_label(button: Button) -> str:
    return button.label.plain


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


async def _mount_modal(
    app: _ModalHost,
    *,
    reason: str,
    results: list[CapacityAction],
) -> ConsoleVideoCapacityModal:
    modal = ConsoleVideoCapacityModal(
        reason=cast(CapacityReason, reason),
        size_bytes=3 * 1024 * 1024 + 512 * 1024,
        max_bytes=2 * 1024 * 1024,
    )
    await app.push_screen(modal, callback=results.append)
    return modal


@pytest.mark.asyncio
async def test_over_capacity_modal_has_exact_choices_and_safe_size_copy() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="over_capacity", results=results)
        await pilot.pause()

        assert _button_label(modal.query_one("#video-capacity-keep", Button)) == (
            "Keep here (remove other videos)"
        )
        assert _button_label(modal.query_one("#video-capacity-save", Button)) == (
            "Save to disk"
        )
        assert _button_label(modal.query_one("#video-capacity-discard", Button)) == (
            "Discard"
        )
        copy = " ".join(
            _static_text(widget)
            for widget in modal.query("#video-capacity-summary, #video-capacity-guidance")
            if isinstance(widget, Static)
        )
        assert "3.5 MiB" in copy
        assert "2.0 MiB" in copy
        assert "generated video" in copy.lower()


@pytest.mark.asyncio
async def test_store_failure_modal_has_exact_choices_without_capacity_claim() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="store_failure", results=results)
        await pilot.pause()

        assert _button_label(modal.query_one("#video-capacity-keep", Button)) == "Retry"
        assert _button_label(modal.query_one("#video-capacity-save", Button)) == (
            "Save to disk"
        )
        assert _button_label(modal.query_one("#video-capacity-discard", Button)) == (
            "Discard"
        )
        guidance = _static_text(
            modal.query_one("#video-capacity-guidance", Static)
        ).lower()
        assert "could not be stored" in guidance
        assert "exceeds" not in guidance


@pytest.mark.parametrize(
    ("reason", "button_id", "expected"),
    [
        ("over_capacity", "video-capacity-keep", "keep"),
        ("store_failure", "video-capacity-keep", "keep"),
        ("over_capacity", "video-capacity-save", "save_external"),
        ("store_failure", "video-capacity-discard", "discard"),
    ],
)
@pytest.mark.asyncio
async def test_modal_buttons_dismiss_with_typed_actions(
    reason: str,
    button_id: str,
    expected: CapacityAction,
) -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await _mount_modal(app, reason=reason, results=results)
        await pilot.pause()
        await pilot.click(f"#{button_id}")
        await pilot.pause()

    assert results == [expected]


@pytest.mark.asyncio
async def test_modal_escape_dismisses_as_discard() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await _mount_modal(app, reason="over_capacity", results=results)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == ["discard"]


@pytest.mark.asyncio
async def test_modal_copy_is_plain_and_contains_no_private_sentinels() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="store_failure", results=results)
        await pilot.pause()

        copy_widgets = [
            modal.query_one("#video-capacity-summary", Static),
            modal.query_one("#video-capacity-guidance", Static),
        ]
        rendered = " ".join(_static_text(widget) for widget in copy_widgets)
        assert all(widget._render_markup is False for widget in copy_widgets)
        for sentinel in (
            "/Users/private/generated.mp4",
            "PRIVATE-PATH",
            "make the person identifiable",
            "message-id-123",
            "Traceback",
        ):
            assert sentinel not in rendered


@pytest.mark.asyncio
async def test_modal_buttons_fit_inside_ninety_column_screen() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(90, 32)) as pilot:
        modal = await _mount_modal(app, reason="over_capacity", results=results)
        await pilot.pause()

        dialog = modal.query_one("#video-capacity-dialog")
        assert 0 <= dialog.region.x
        assert dialog.region.right <= modal.size.width
        assert dialog.region.width <= modal.size.width
        for button_id in (
            "video-capacity-keep",
            "video-capacity-save",
            "video-capacity-discard",
        ):
            button = modal.query_one(f"#{button_id}", Button)
            assert dialog.region.x <= button.region.x
            assert button.region.right <= dialog.region.right
            assert button.region.right <= modal.size.width
            assert button.region.width >= len(_button_label(button)) + 2
