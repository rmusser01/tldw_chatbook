"""TASK-1 (Console branching Phase B): the edit modal gains an explicit
"Edit & resend" affordance alongside the existing in-place "Save".

Construction-level tests are the minimum contract (per the task brief); a
mounted `run_test` pilot assertion is added to mirror the existing modal
suite's style (Tests/UI/test_console_edit_modal_keystroke_guard.py).
"""

from pathlib import Path

import pytest
from rich.segment import Segment
from textual.app import App
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal,
    ConsoleEditResult,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_CSS = _REPOSITORY_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


def test_edit_result_dataclass_shape():
    r = ConsoleEditResult(text="hi", resend=True)
    assert (r.text, r.resend) == ("hi", True)


def test_edit_result_is_frozen():
    r = ConsoleEditResult(text="hi", resend=True)
    with pytest.raises(Exception):
        r.text = "changed"  # type: ignore[misc]


def test_modal_accepts_can_resend_kwarg():
    # construction only (no mount) — the resend button is gated on can_resend
    m = ConsoleEditMessageModal(content="orig", can_resend=True)
    assert m._can_resend is True
    m2 = ConsoleEditMessageModal(content="orig")
    assert m2._can_resend is False


class _ModalHost(App):
    pass


class _RealBundleModalHost(_ModalHost):
    """The incumbent modal harness with the generated app stylesheet loaded."""

    CSS_PATH = _BUNDLED_CSS


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _cropped_compositor_region(
    app: App, widget: Button
) -> tuple[list[list[Segment]], str]:
    """Return exact compositor segments and text cropped to ``widget.region``."""
    render_strips = list(app.screen._compositor.render_strips())
    cropped_rows: list[list[Segment]] = []
    for y in range(widget.region.y, widget.region.bottom):
        row: list[Segment] = []
        cursor = 0
        for segment in render_strips[y]:
            next_cursor = cursor + segment.cell_length
            overlap_start = max(widget.region.x, cursor)
            overlap_end = min(widget.region.right, next_cursor)
            if overlap_start < overlap_end:
                _, remainder = segment.split_cells(overlap_start - cursor)
                cropped, _ = remainder.split_cells(overlap_end - overlap_start)
                row.append(cropped)
            cursor = next_cursor
        cropped_rows.append(row)
    text = "\n".join("".join(segment.text for segment in row) for row in cropped_rows)
    return cropped_rows, text


_REAL_BUNDLE_ACTIONS = [
    pytest.param(
        False,
        "#console-edit-message-cancel",
        "Cancel",
        id="without-resend-cancel",
    ),
    pytest.param(
        False,
        "#console-edit-message-save",
        "Save",
        id="without-resend-save",
    ),
    pytest.param(
        True,
        "#console-edit-message-cancel",
        "Cancel",
        id="with-resend-cancel",
    ),
    pytest.param(
        True,
        "#console-edit-message-save",
        "Save",
        id="with-resend-save",
    ),
    pytest.param(
        True,
        "#console-edit-message-resend",
        "Edit & resend",
        id="with-resend-resend",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (235, 52)])
@pytest.mark.parametrize(
    ("can_resend", "selector", "expected_label"), _REAL_BUNDLE_ACTIONS
)
async def test_real_bundle_action_containment(
    size: tuple[int, int],
    can_resend: bool,
    selector: str,
    expected_label: str,
) -> None:
    """Nonzero/display geometry is insufficient: each full action must fit its owners."""
    app = _RealBundleModalHost()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleEditMessageModal(
            content="Synthetic edit body", can_resend=can_resend
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        button = modal.query_one(selector, Button)
        actions = modal.query_one("#console-edit-message-actions")
        root = modal.query_one("#console-edit-message-modal")
        containment = {
            "actions.content_region": actions.content_region.contains_region(
                button.region
            ),
            "modal.content_region": root.content_region.contains_region(button.region),
            "screen.region": app.screen.region.contains_region(button.region),
        }

        assert containment == {
            "actions.content_region": True,
            "modal.content_region": True,
            "screen.region": True,
        }, (
            f"{expected_label!r} must be fully contained at size={size}; "
            f"display={button.display} region={button.region!r} containment={containment}. "
            "A displayed widget with nonzero geometry may still be clipped."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (235, 52)])
@pytest.mark.parametrize(
    ("can_resend", "selector", "expected_label"), _REAL_BUNDLE_ACTIONS
)
async def test_real_bundle_action_hit_test(
    size: tuple[int, int],
    can_resend: bool,
    selector: str,
    expected_label: str,
) -> None:
    """Nonzero/display geometry does not prove an action owns its reported center."""
    app = _RealBundleModalHost()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleEditMessageModal(
            content="Synthetic edit body", can_resend=can_resend
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        button = modal.query_one(selector, Button)
        hit, _ = app.screen.get_widget_at(*button.region.center)

        assert hit is button, (
            f"{expected_label!r} must own its center at size={size}; "
            f"display={button.display} region={button.region!r}, hit={hit!r}. "
            "A displayed widget with nonzero geometry may still be covered or clipped."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (235, 52)])
@pytest.mark.parametrize(
    ("can_resend", "selector", "expected_label"), _REAL_BUNDLE_ACTIONS
)
async def test_real_bundle_action_painted_label(
    size: tuple[int, int],
    can_resend: bool,
    selector: str,
    expected_label: str,
) -> None:
    """Crop each action: whole-frame Save/resend matches can come from USER prose."""
    app = _RealBundleModalHost()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleEditMessageModal(
            content="Synthetic edit body", can_resend=can_resend
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        button = modal.query_one(selector, Button)
        cropped_rows, painted_text = _cropped_compositor_region(app, button)
        compositor = app.screen._compositor

        assert (
            expected_label in painted_text and button in compositor.visible_widgets
        ), (
            f"{expected_label!r} must be painted in its exact region at size={size}; "
            f"display={button.display} region={button.region!r}, "
            f"painted_text={painted_text!r}, cropped_rows={cropped_rows!r}, "
            f"visible={button in compositor.visible_widgets}. Whole-frame matches are "
            "false positives when USER-facing prose itself contains Save/Edit & resend."
        )


@pytest.mark.asyncio
async def test_modal_without_resend_has_no_resend_button():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig")
        app.push_screen(modal)
        await pilot.pause()

        assert len(modal.query("#console-edit-message-resend")) == 0
        save_button = modal.query_one("#console-edit-message-save", Button)
        assert save_button.variant == "primary"


@pytest.mark.asyncio
async def test_modal_with_resend_shows_resend_button_and_demotes_save():
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        app.push_screen(modal)
        await pilot.pause()

        resend_button = modal.query_one("#console-edit-message-resend", Button)
        assert resend_button.variant == "primary"
        save_button = modal.query_one("#console-edit-message-save", Button)
        assert save_button.variant == "default"


@pytest.mark.asyncio
async def test_save_dismisses_edit_result_with_resend_false():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig")
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert result == [ConsoleEditResult(text="edited", resend=False)]


@pytest.mark.asyncio
async def test_resend_dismisses_edit_result_with_resend_true():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited"
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

    assert result == [ConsoleEditResult(text="edited", resend=True)]


@pytest.mark.asyncio
async def test_resend_blank_text_blocked_inline():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "   "
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

        assert result == []
        error = modal.query_one("#console-edit-message-error", Static)
        assert "cannot be blank" in _static_plain_text(error).lower()


@pytest.mark.asyncio
async def test_cancel_dismisses_none_even_with_resend_available():
    app = _ModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

    assert result == [None]


@pytest.mark.asyncio
async def test_context_copy_mentions_resend_only_when_can_resend():
    # can_resend=True: the context Static explains the resend fork option.
    app = _ModalHost()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = ConsoleEditMessageModal(content="orig", can_resend=True)
        app.push_screen(modal)
        await pilot.pause()
        context = _static_plain_text(
            modal.query_one("#console-edit-message-context", Static)
        )
        assert "Edit & resend" in context

    # can_resend=False: the context Static uses the plain in-place copy only.
    app_plain = _ModalHost()
    async with app_plain.run_test(size=(120, 40)) as pilot:
        modal_plain = ConsoleEditMessageModal(content="orig")
        app_plain.push_screen(modal_plain)
        await pilot.pause()
        context_plain = _static_plain_text(
            modal_plain.query_one("#console-edit-message-context", Static)
        )
        assert "Edit & resend" not in context_plain
        assert "will not create a new prompt" in context_plain
