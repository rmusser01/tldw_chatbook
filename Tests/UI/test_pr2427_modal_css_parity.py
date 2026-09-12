"""Compare consolidation with the original, genuinely registered defaults.

The fixture contains the four exact DEFAULT_CSS values from 875936d66f,
including Recovery's Navigation alias. It is intentionally independent of
the generated selector rewriting exercised by the current arm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.consolidated_css import CSS_DIR, ConsolidatedCSSApp
from Tests.UI.test_css_subject_key_parity import (
    _FullCSSApp,
    _computed_signature,
)
from Tests.UI.test_library_character_repair import CONTEXT, _Service, _controller
from tldw_chatbook.UI.Library_Modules.library_character_repair_controller import (
    LibraryCharacterRepairDialog,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
)
from tldw_chatbook.Widgets.Console.console_appearance_picker_modal import (
    CONSOLE_APPEARANCE_PALETTE,
    ConsoleAppearancePickerModal,
)
from tldw_chatbook.css import build_css


_OWNERS = (
    LibraryCharacterRepairDialog,
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
    ConsoleAppearancePickerModal,
)
_ORIGINAL = json.loads(
    (Path(__file__).parent / "fixtures/pr2427_modal_defaults.json").read_text()
)


class _OriginalDefaultsApp(_FullCSSApp):
    def _get_default_css(self):
        sources = []
        names = "|".join(owner.__name__ for owner in _OWNERS)
        pattern = (
            rf"/\* ===== WIDGET: (?:{names}) \([^\n]*\) ===== \*/"
            r".*?(?=/\* ===== WIDGET:|\Z)"
        )
        for location, css, tie, scope in build_css.widget_defaults_sources(CSS_DIR):
            sources.append((location, re.sub(pattern, "", css, flags=re.S), tie, scope))
        return sources + super(ConsolidatedCSSApp, self)._get_default_css()


@pytest.mark.parametrize("owner", _OWNERS)
def test_modal_defaults_use_existing_consolidation(owner):
    assert "DEFAULT_CSS" not in vars(owner)
    if owner is not RoleplayDraftRecoveryDialog:
        assert "BUNDLED_CSS" in vars(owner)
    else:
        assert not issubclass(owner, RoleplayDraftNavigationDialog)


def test_original_arm_removes_the_migrated_generated_blocks():
    generated = "\n".join(
        css for _, css, _, _ in build_css.widget_defaults_sources(CSS_DIR)
    )
    original = "\n".join(
        css for _, css, _, _ in _OriginalDefaultsApp()._get_default_css()
    )
    for owner in _OWNERS:
        marker = f"/* ===== WIDGET: {owner.__name__} ("
        if owner is not RoleplayDraftRecoveryDialog:
            assert marker in generated
        assert marker not in original


def _modal(owner):
    if owner is LibraryCharacterRepairDialog:
        controller, *_ = _controller(_Service(()))
        return owner(controller, CONTEXT)
    if owner is ConsoleAppearancePickerModal:
        return owner(
            conversation_id="parity",
            conversation_title="Parity",
            icon="🧪",
            color=CONSOLE_APPEARANCE_PALETTE[0][1],
        )
    return owner(("character form",))


def _frame(app):
    """Keep ink, not only text: every moved container/control must agree."""
    return tuple(
        tuple((segment.text, repr(segment.style)) for segment in strip)
        for strip in app.screen._compositor.render_strips()
    )


async def _capture(app, owner, size):
    async with app.run_test(size=size) as pilot:
        modal = _modal(owner)
        await app.push_screen(modal)
        await app.workers.wait_for_complete()
        await pilot.pause()
        first_paint = (
            _frame(app),
            tuple(
                (type(widget).__name__, widget.id, _computed_signature(widget))
                for widget in modal.query("*")
            ),
        )
        if owner is ConsoleAppearancePickerModal:
            assert len(modal.query(".console-appearance-emoji-selected")) == 1
            assert len(modal.query(".console-appearance-swatch-selected")) == 1
        states = []
        buttons = list(modal.query(Button))
        if owner is ConsoleAppearancePickerModal:
            buttons = list(modal.query("#console-appearance-picker-actions Button"))
            assert [button.id for button in buttons] == [
                "console-appearance-picker-apply",
                "console-appearance-picker-clear",
                "console-appearance-picker-cancel",
            ]
        for button in buttons:
            for disabled, focused in ((False, False), (True, False), (False, True)):
                modal.set_focus(None)
                button.disabled = disabled
                if focused:
                    button.focus()
                button.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                await pilot.pause()
                x = button.region.x + button.region.width // 2
                y = button.region.y + button.region.height // 2
                hit = None
                ink = None
                if modal.region.contains(x, y):
                    widget, _ = modal._compositor.get_widget_at(x, y)
                    hit = (type(widget).__name__, widget.id)
                    ink = repr(modal._compositor.get_style_at(x, y))
                states.append(
                    (
                        button.id,
                        _computed_signature(button),
                        repr(button.visual_style),
                        hit,
                        ink,
                        _frame(app),
                    )
                )
        return first_paint, states


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("size", ((52, 20), (120, 50)))
async def test_modal_full_tier_paint_matches_original_defaults(
    monkeypatch, owner, size
):
    import tldw_chatbook.Widgets.Console.console_appearance_picker_modal as appearance

    monkeypatch.setattr(
        appearance,
        "_default_emoji_sequence",
        lambda: [
            {
                "char": "🧪",
                "name": "test tube",
                "category": "objects",
                "aliases": ["lab"],
            }
        ],
    )
    current = await _capture(_FullCSSApp(), owner, size)
    with monkeypatch.context() as original:
        for candidate in _OWNERS:
            original.setattr(candidate, "DEFAULT_CSS", _ORIGINAL[candidate.__name__])
        baseline = await _capture(_OriginalDefaultsApp(), owner, size)
    assert current == baseline
