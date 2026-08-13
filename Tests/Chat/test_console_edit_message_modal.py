"""Console edit-message modal behavior and rendered-state contracts.

TASK-1 covers Save/Edit & resend construction and outcomes. TASK-2703 adds
real-bundle compositor evidence for action geometry, paint, contrast, isolated
keyboard focus cues, and Enter activation.
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
) -> tuple[tuple[Segment, ...], ...]:
    """Return exact compositor segments cropped to ``widget.region``."""
    render_strips = list(app.screen._compositor.render_strips())
    cropped_rows: list[tuple[Segment, ...]] = []
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
        cropped_rows.append(tuple(row))
    return tuple(cropped_rows)


def _joined_segment_text(rows: tuple[tuple[Segment, ...], ...]) -> str:
    """Join text only from already-cropped compositor segments."""
    return "\n".join("".join(segment.text for segment in row) for row in rows)


def _relative_luminance(color, *, foreground: bool = True) -> float:
    """Return WCAG relative luminance for a compositor-painted Rich colour."""
    triplet = color.get_truecolor(foreground=foreground)

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast(foreground, background) -> float:
    """Return WCAG contrast for explicit compositor foreground/background."""
    lighter, darker = sorted(
        (
            _relative_luminance(foreground),
            _relative_luminance(background, foreground=False),
        ),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_label_segments(
    rows: tuple[tuple[Segment, ...], ...], expected_label: str
) -> tuple[Segment, ...]:
    """Resolve the exact composited Rich segments that paint one action label."""
    for row in rows:
        painted_text = "".join(segment.text for segment in row)
        label_start = painted_text.find(expected_label)
        if label_start < 0:
            continue

        label_end = label_start + len(expected_label)
        label_segments: list[Segment] = []
        cursor = 0
        for segment in row:
            next_cursor = cursor + segment.cell_length
            overlap_start = max(label_start, cursor)
            overlap_end = min(label_end, next_cursor)
            if overlap_start < overlap_end:
                _, remainder = segment.split_cells(overlap_start - cursor)
                cropped, _ = remainder.split_cells(overlap_end - overlap_start)
                if (
                    cropped.style is None
                    or cropped.style.color is None
                    or cropped.style.bgcolor is None
                ):
                    raise AssertionError(
                        f"label segment lacks explicit foreground/background: {cropped!r}"
                    )
                label_segments.append(cropped)
            cursor = next_cursor

        if "".join(segment.text for segment in label_segments) == expected_label:
            return tuple(label_segments)
    raise AssertionError(f"no exact painted label segments for {expected_label!r}")


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
        cropped_rows = _cropped_compositor_region(app, button)
        painted_text = _joined_segment_text(cropped_rows)
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
@pytest.mark.parametrize("size", [(200, 50), (235, 52)])
@pytest.mark.parametrize(
    ("can_resend", "selector", "expected_label"), _REAL_BUNDLE_ACTIONS
)
async def test_real_bundle_action_ordinary_contrast(
    size: tuple[int, int],
    can_resend: bool,
    selector: str,
    expected_label: str,
) -> None:
    """Every ordinary action label must retain 3:1 in the real app cascade."""
    app = _RealBundleModalHost()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleEditMessageModal(
            content="Synthetic edit body", can_resend=can_resend
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        button = modal.query_one(selector, Button)
        cropped_rows = _cropped_compositor_region(app, button)
        label_segments = _painted_label_segments(cropped_rows, expected_label)
        contrasts = tuple(
            _contrast(segment.style.color, segment.style.bgcolor)
            for segment in label_segments
            if segment.style is not None
        )

        assert contrasts and min(contrasts) >= 3.0, (
            f"ordinary {expected_label!r} must paint at >=3:1 at size={size}; "
            f"region={button.region!r}, contrasts={contrasts!r}, "
            f"label_segments={label_segments!r}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (235, 52)])
@pytest.mark.parametrize(
    ("can_resend", "selector", "expected_label"), _REAL_BUNDLE_ACTIONS
)
async def test_real_bundle_focus(
    size: tuple[int, int],
    can_resend: bool,
    selector: str,
    expected_label: str,
) -> None:
    """Tab focus must preserve each exact label and add a non-colour cue."""
    app = _RealBundleModalHost()
    async with app.run_test(size=size) as pilot:
        modal = ConsoleEditMessageModal(
            content="Synthetic edit body", can_resend=can_resend
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        assert modal.focused is editor

        focus_order = [
            ("#console-edit-message-cancel", "Cancel"),
            ("#console-edit-message-save", "Save"),
        ]
        if can_resend:
            focus_order.append(("#console-edit-message-resend", "Edit & resend"))

        buttons = {
            action_selector: modal.query_one(action_selector, Button)
            for action_selector, _ in focus_order
        }
        ordinary_rows_by_action = {
            action_selector: _cropped_compositor_region(app, buttons[action_selector])
            for action_selector, _ in focus_order
        }
        ordinary_cue_signatures = {}
        for action_selector, action_label in focus_order:
            action_rows = ordinary_rows_by_action[action_selector]
            action_label_segments = _painted_label_segments(action_rows, action_label)
            ordinary_cue_signatures[action_selector] = (
                any(
                    segment.style is not None and segment.style.underline
                    for segment in action_label_segments
                ),
                tuple(
                    tuple((segment.text, segment.style) for segment in edge_row)
                    for edge_row in (action_rows[0], action_rows[-1])
                ),
            )

        button = buttons[selector]
        ordinary_rows = ordinary_rows_by_action[selector]
        ordinary_label_segments = _painted_label_segments(ordinary_rows, expected_label)
        ordinary_styles = tuple(segment.style for segment in ordinary_label_segments)
        target_index = [item[0] for item in focus_order].index(selector)
        for expected_selector, _ in focus_order[: target_index + 1]:
            await pilot.press("tab")
            await pilot.pause()
            await pilot.pause()
            assert modal.focused is buttons[expected_selector], (
                f"Tab order must be {[item[0] for item in focus_order]!r}; "
                f"expected {expected_selector!r}, "
                f"focused={modal.focused!r}"
            )

            for action_selector, action_label in focus_order:
                action_rows = _cropped_compositor_region(app, buttons[action_selector])
                action_label_segments = _painted_label_segments(
                    action_rows, action_label
                )
                cue_signature = (
                    any(
                        segment.style is not None and segment.style.underline
                        for segment in action_label_segments
                    ),
                    tuple(
                        tuple((segment.text, segment.style) for segment in edge_row)
                        for edge_row in (action_rows[0], action_rows[-1])
                    ),
                )
                if action_selector == expected_selector:
                    ordinary_underline, ordinary_edges = ordinary_cue_signatures[
                        action_selector
                    ]
                    focused_underline, focused_edges = cue_signature
                    ordinary_edge_text = tuple(
                        "".join(text for text, _ in edge_row)
                        for edge_row in ordinary_edges
                    )
                    focused_edge_text = tuple(
                        "".join(text for text, _ in edge_row)
                        for edge_row in focused_edges
                    )
                    has_focus_only_non_color_cue = (
                        focused_underline and not ordinary_underline
                    ) or (
                        focused_edge_text != ordinary_edge_text
                        and any(text.strip() for text in focused_edge_text)
                    )
                    assert has_focus_only_non_color_cue, (
                        f"focused {action_label!r} needs a new underline or visible "
                        f"edge glyph at size={size}; ordinary_signature="
                        f"{ordinary_cue_signatures[action_selector]!r}, "
                        f"focused_signature={cue_signature!r}"
                    )
                else:
                    assert cue_signature == ordinary_cue_signatures[action_selector], (
                        f"focusing {expected_selector!r} must not change sibling "
                        f"{action_selector!r} at size={size}; ordinary_signature="
                        f"{ordinary_cue_signatures[action_selector]!r}, "
                        f"current_signature={cue_signature!r}"
                    )
        await pilot.pause()

        focused_rows = _cropped_compositor_region(app, button)
        focused_text = _joined_segment_text(focused_rows)
        focused_label_segments = _painted_label_segments(focused_rows, expected_label)
        focused_styles = tuple(segment.style for segment in focused_label_segments)
        focused_contrasts = tuple(
            _contrast(segment.style.color, segment.style.bgcolor)
            for segment in focused_label_segments
            if segment.style is not None
        )

        assert modal.focused is button
        assert expected_label in focused_text, (
            f"focused {expected_label!r} must survive in its exact region at "
            f"size={size}; focused_text={focused_text!r}, rows={focused_rows!r}"
        )
        assert focused_contrasts and min(focused_contrasts) >= 3.0, (
            f"focused {expected_label!r} must paint at >=3:1 at size={size}; "
            f"contrasts={focused_contrasts!r}, segments={focused_label_segments!r}"
        )
        assert focused_styles != ordinary_styles, (
            f"focused {expected_label!r} styles must differ from ordinary styles; "
            f"ordinary={ordinary_styles!r}, focused={focused_styles!r}"
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
async def test_enter_activates_focused_action():
    """Enter activates Save reached through the modal's real keyboard order."""
    app = _RealBundleModalHost()
    result: list[ConsoleEditResult | None] = []

    async with app.run_test(size=(200, 50)) as pilot:
        modal = ConsoleEditMessageModal(content="orig")
        await app.push_screen(modal, callback=result.append)
        await pilot.pause()

        editor = modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited"
        await pilot.press("tab")
        await pilot.press("tab")
        await pilot.pause()
        assert modal.focused is modal.query_one("#console-edit-message-save", Button)

        await pilot.press("enter")
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
