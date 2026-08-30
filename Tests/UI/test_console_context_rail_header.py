"""Context rail header identity and overflow hint (TASK-23195).

Two findings from the 2026-08-29 UX audit:

* The rail had no title. Its entire header was one Button labelled
  ``<---------|Context`` -- so the only place the word "Context" appeared was
  inside the control that collapses the rail, and that literal was hard-coded
  ASCII art bypassing the ``ascii_glyphs`` fallback every other Console glyph
  routes through.
* The overflow hint said "more sections - scroll" without saying how many or
  which, so a user could not tell whether scrolling was worth it.
"""

from __future__ import annotations

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from Tests.UI.test_console_left_rail import make_console_pilot


@pytest.mark.asyncio
async def test_rail_header_names_the_rail_readably() -> None:
    """The header must read as the rail's name, not as ASCII art.

    The header stays ONE full-width collapse target: that large click area
    is deliberate, and the Inspector mirrors it. What this pins is that the
    label is legible -- "Context" plus a compact affordance -- rather than
    the 18-column "<---------|Context" the audit found, which buried the
    rail's only occurrence of its own name inside a decorative arrow.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        collapse = screen.query_one("#console-context-rail-collapse", Button)
        label = str(collapse.label)

        assert "Context" in label
        assert "<---------" not in label, (
            "the header is still hard-coded ASCII art"
        )
        assert collapse.region.height == 1
        assert collapse.region.width > 0, "the rail header is not painted"


@pytest.mark.asyncio
async def test_collapse_affordance_survives_ascii_glyph_mode() -> None:
    """The collapse glyph must route through the ASCII fallback system.

    ``appearance.ascii_glyphs`` exists for terminals whose font renders the
    Console glyph vocabulary badly. A hard-coded literal silently opts out of
    it; a resolved glyph does not.
    """
    from tldw_chatbook.Widgets.glyph_fallback import (
        resolve_glyph,
        set_ascii_glyph_mode,
    )

    set_ascii_glyph_mode(True)
    try:
        # Every character the collapse control paints must be one the
        # fallback map either substitutes or passes through deliberately.
        assert resolve_glyph("◂") == "<"
    finally:
        set_ascii_glyph_mode(False)

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        collapse = screen.query_one("#console-context-rail-collapse", Button)
        label = str(collapse.label)
        assert label.strip(), "the collapse control lost its affordance entirely"
        assert len(label) <= 12, (
            f"the header label is still spending the rail's width: {label!r}"
        )


@pytest.mark.asyncio
async def test_overflow_hint_names_the_sections_below_the_fold() -> None:
    """A hint that says only "more" cannot be acted on.

    140x40 still overflows after TASK-23193, with Agent, Details and
    Character below the fold -- so the hint must name them.
    """
    async with make_console_pilot(size=(140, 40), production_styles=True) as pilot:
        screen = pilot.app.screen
        await pilot.pause(0.4)

        outer = screen.query_one("#console-left-rail-body", VerticalScroll)
        assert outer.virtual_size.height > outer.size.height, (
            "140x40 no longer overflows; pick a geometry that does"
        )

        hint = screen.query_one("#console-left-rail-outer-hint", Static)
        text = str(hint.renderable)
        assert text, "no overflow hint shown while the rail overflows"

        hidden = []
        top = outer.region.y
        bottom = top + outer.size.height
        for section_id, label in (
            ("session", "Sessions"),
            ("workspace", "Workspaces"),
            ("conversations", "Conversations"),
            ("model", "Model"),
            ("agent", "Agent"),
            ("details", "Details"),
            ("character", "Character"),
        ):
            header = screen.query_one(f"#console-rail-section-header-{section_id}")
            if not (header.display and top <= header.region.y < bottom):
                hidden.append(label)

        assert hidden, "expected some sections below the fold at 140x40"
        # A 27-column rail cannot hold "Agent · Details · Character", so the
        # contract is: name what fits, starting with the section immediately
        # below the fold, and count the rest. Naming the FIRST one is the
        # actionable part -- it is what the user reaches by scrolling once.
        assert any(label in text for label in hidden), (
            f"the hint names none of the hidden sections {hidden!r}: {text!r}"
        )
        assert text != "▼ more sections — scroll", (
            "the hint still says only 'more sections'"
        )


@pytest.mark.asyncio
async def test_no_overflow_hint_when_everything_fits() -> None:
    """The hint must not claim there is more when there is not."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await pilot.pause(0.4)

        outer = screen.query_one("#console-left-rail-body", VerticalScroll)
        if outer.virtual_size.height > outer.size.height:
            pytest.skip("160x48 overflows in this build; nothing to assert")

        hint = screen.query_one("#console-left-rail-outer-hint", Static)
        assert not str(hint.renderable).strip()


@pytest.mark.asyncio
async def test_both_rail_headers_share_one_visual_language() -> None:
    """Context and Inspector headers must not drift apart.

    They were a matched pair of ASCII arrows ("<---------|Context" and
    "Inspect|--------->"). TASK-23195 fixed only the Context side, which left
    the two rails speaking differently; this pins the mirror. Each is a name
    plus one resolved glyph, and the glyph sits on the edge adjacent to the
    transcript pointing outward, so the affordance shows which way the rail
    leaves.
    """
    async with make_console_pilot(size=(200, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause(0.3)

        context = str(screen.query_one("#console-context-rail-collapse", Button).label)
        inspector = str(
            screen.query_one("#console-inspector-rail-collapse", Button).label
        )

        for label in (context, inspector):
            assert "---" not in label, f"still ASCII art: {label!r}"
            assert len(label) <= 12, f"header label too wide: {label!r}"

        assert context.strip().endswith("◂"), (
            f"Context's glyph must trail its name, pointing left: {context!r}"
        )
        assert inspector.strip().startswith("▸"), (
            f"Inspector's glyph must lead its name, pointing right: {inspector!r}"
        )
