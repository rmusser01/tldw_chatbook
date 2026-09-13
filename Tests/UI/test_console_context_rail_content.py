"""Context rail content defects found by the 2026-08-29 UX audit.

TASK-23194. Three separate defects, all measured against the running app:

* the no-character empty state rendered TWICE, verbatim, from two different
  widgets stacked on consecutive rows;
* the Agent section mounted focusable widgets at zero size, so keyboard
  focus could land on a text Input that painted nothing;
* controls that cannot act shipped disabled and unexplained.

Each test pins the user-visible outcome rather than the mechanism, so a
later restructure of the Character or Agent sections is free to satisfy
them differently.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.console_rail_section_helpers import open_rail_section
from Tests.UI.test_console_left_rail import make_console_pilot


def _visible_texts(screen, selector: str) -> list[str]:
    """Return the rendered text of every displayed match, in DOM order."""
    texts = []
    for widget in screen.query(selector):
        if not widget.display:
            continue
        renderable = getattr(widget, "renderable", "")
        text = str(renderable).strip()
        if text:
            texts.append(text)
    return texts


@pytest.mark.asyncio
async def test_no_character_empty_state_renders_once() -> None:
    """Two widgets must not both claim there is no character in this chat."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await open_rail_section(screen, pilot, "character")
        await pilot.pause(0.4)

        body = screen.query_one("#console-rail-section-body-character")
        texts = _visible_texts(body, "Static")
        no_character = [t for t in texts if "no character" in t.lower()]

        assert len(no_character) <= 1, (
            f"the no-character empty state is rendered {len(no_character)} "
            f"times: {no_character!r}"
        )


@pytest.mark.asyncio
async def test_character_avatar_placeholder_does_not_echo_the_name_row() -> None:
    """The avatar placeholder must not repeat the name row's empty state."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await open_rail_section(screen, pilot, "character")
        await pilot.pause(0.4)

        try:
            placeholder = screen.query_one("#console-character-avatar-empty", Static)
        except Exception:
            return  # not mounted at all is a valid way to satisfy this
        name = screen.query_one("#console-character-name", Static)

        placeholder_text = str(placeholder.renderable).strip()
        name_text = str(name.renderable).strip()
        if placeholder.display and placeholder_text:
            assert placeholder_text != name_text, (
                "avatar placeholder and name row render identical copy: "
                f"{placeholder_text!r}"
            )


@pytest.mark.asyncio
async def test_character_is_between_conversations_and_model() -> None:
    """Character navigation has a stable peer position in the Context rail."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        ids = [
            widget.id
            for widget in screen.query(".console-rail-section-header")
            if widget.id
        ]
        assert ids.index("console-rail-section-header-conversations") + 1 == ids.index(
            "console-rail-section-header-character"
        )
        assert ids.index("console-rail-section-header-character") + 1 == ids.index(
            "console-rail-section-header-model"
        )
        assert screen.query_one("#console-character-context")


@pytest.mark.asyncio
async def test_no_keyboard_reachable_context_control_paints_nothing() -> None:
    """Every control Tab can reach in the rail must actually be on screen.

    The 2026-08-29 audit reported three zero-size focusable widgets in the
    Agent and Workspace sections -- including a text Input -- and concluded
    keyboard focus could land on a control painting nothing. That
    conclusion was WRONG, and this test records why: the audit used a naive
    ``can_focus and display`` query, but a widget's own ``display`` stays
    True while an ANCESTOR is hidden. Textual's real focus chain already
    excludes those three, so Tab never reaches them.

    The invariant worth pinning is the one the audit was reaching for, and
    it is about ``screen.focus_chain``, not about every mounted widget: if
    a control IS keyboard reachable, it must occupy space.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        for section_id in ("agent", "workspace", "character", "details", "model"):
            await open_rail_section(screen, pilot, section_id)
        await pilot.pause(0.6)

        rail = screen.query_one("#console-left-rail")
        rail_widgets = set(rail.query("*").nodes) | {rail}
        offenders = [
            f"{type(widget).__name__}#{widget.id}"
            for widget in screen.focus_chain
            if widget in rail_widgets
            and (widget.region.width == 0 or widget.region.height == 0)
        ]

        assert not offenders, (
            f"keyboard-reachable Context rail controls painting nothing: {offenders}"
        )
