"""Console left-rail width budget and content-fit contracts (TASK-2154.3).

Regression coverage for UX-review findings LY-01/LY-04/LY-05/LY-06/LY-07
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

- LY-01/LY-07: the rail rendered at 24-27 columns while its content needed
  ~30 -- `Workspace Defau…`, buttons clipped to `Switc`/`RAG S`/`RA`,
  `New conversati`, and the conversation search box squeezed to `S`/`Sear`
  against its Clear button. The rail's min-width is now 30 (the 3fr share is
  min-bound at 100-160 cols), the action rows no longer carry the spec's
  unaffordable 12-col left indent, and the trays no longer double the
  section-body indent.
- LY-04: the retired grouped browser no longer stacks aggregate empty-state
  copy before any intent; Workspaces uses the native Tree and Conversations
  owns only Default/unassigned rows.
- LY-05: `{title} - Chats` above the collapsible `Chats` group was two list
  metaphors for one thing; the synthetic bucket label left the summary.
- LY-06: `Add another workspace before switching.` rendered as an always-on
  Static before any user intent; it now lives on the disabled Switch
  button's tooltip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _visible_text, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


class StyledConsoleHarness(ConsoleHarness):
    """ConsoleHarness with the shipped stylesheet so app-tier rules apply."""

    CSS_PATH = str(BUNDLE)


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (160, 48)])
async def test_session_rows_fit_inside_the_rail(size: tuple[int, int]) -> None:
    """LY-01/LY-07: every Session-section row fits its container at 140 and
    160 cols -- no mid-word clipping of the Workspace value, and the
    Switch/New/RAG/New conversation buttons render inside the rail."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await _wait_for_selector(console, pilot, "#console-new-workspace-conversation")
        await pilot.pause(0.2)

        tray = console.query_one("#console-workspace-context")

        # Workspace status pair: 13-col label (12 visible characters plus the
        # required gutter) + value with the spec's 10-cell floor, neither
        # overflowing the row.
        pair = console.query_one("#console-active-workspace")
        label = console.query_one("#console-active-workspace-label")
        value = console.query_one("#console-active-workspace-value")
        assert label.region.width == 13
        assert value.region.width >= 10
        assert value.region.right <= pair.content_region.right, (
            f"Workspace value overflows its row at {size}: value right "
            f"{value.region.right} > row content right "
            f"{pair.content_region.right} (the 'Workspace Defau…' clip)"
        )
        # The default workspace name must fit whole, not ellipsize mid-word.
        assert value.region.width >= len("Default")

        # Every Session action button renders inside the tray's content box.
        checks = (
            ("#console-change-workspace", len("Switch")),
            ("#console-new-workspace", len("New")),
            ("#console-workspace-rag-scope-open", len("RAG")),
            ("#console-new-workspace-conversation", len("New conversation")),
        )
        for selector, label_width in checks:
            button = console.query_one(selector, Button)
            assert button.display, f"{selector} must be displayed"
            if selector == "#console-workspace-rag-scope-open":
                assert str(button.label) == "RAG"
                assert str(button.tooltip).startswith("RAG Scope:")
            assert button.region.width >= label_width, (
                f"{selector} is {button.region.width} cells wide at {size} -- "
                f"its {label_width}-cell label clips mid-word"
            )
            assert button.region.right <= tray.content_region.right, (
                f"{selector} renders past the rail's content clip at {size}: "
                f"button right {button.region.right} > tray content right "
                f"{tray.content_region.right} (invisible but clickable)"
            )

        # Switch and New sit side by side on the action row without overlap.
        switch = console.query_one("#console-change-workspace", Button)
        new = console.query_one("#console-new-workspace", Button)
        assert switch.region.y == new.region.y
        assert switch.region.right <= new.region.x


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (160, 48)])
async def test_conversation_search_and_clear_render_without_overlap(
    size: tuple[int, int],
) -> None:
    """LY-01: the conversation search box keeps a usable text area and never
    collides with its Clear button (was `S`/`Sear` jammed against Clear)."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await pilot.pause(0.2)

        row = console.query_one("#console-workspace-conversation-search-row")
        search = console.query_one("#console-workspace-conversation-search")
        clear = console.query_one("#console-workspace-conversation-search-clear")

        assert search.content_region.width >= 8, (
            f"search text area is {search.content_region.width} cells at "
            f"{size} -- the LY-01 `S`/`Sear` clip"
        )
        assert search.region.right <= clear.region.x, (
            f"search overlaps Clear at {size}: search right "
            f"{search.region.right} > clear left {clear.region.x}"
        )
        assert clear.region.width >= len("Clear")
        assert clear.region.right <= row.content_region.right


@pytest.mark.asyncio
async def test_switch_gating_copy_lives_on_the_disabled_button_tooltip() -> None:
    """LY-06: `Add another workspace before switching.` no longer renders as
    an always-on Static before any intent; it is the disabled Switch
    button's tooltip -- visible exactly when the user reaches for it."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await pilot.pause(0.2)

        # The seeded test app has exactly one workspace, so switching is
        # disabled with the "add another workspace" reason (display_state).
        assert not list(console.query("#console-change-workspace-recovery"))
        assert "Add another workspace before switching." not in _visible_text(console)

        switch = console.query_one("#console-change-workspace", Button)
        assert switch.disabled is True
        assert switch.tooltip == "Add another workspace before switching."


@pytest.mark.asyncio
async def test_selected_summary_drops_the_synthetic_chats_suffix() -> None:
    """LY-05: `{title} - Chats` above the collapsible `Chats` group was two
    list metaphors for one thing; the summary shows the title alone."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-selected-conversation"
        )
        await pilot.pause(0.2)

        summary = _static_text(
            console.query_one("#console-workspace-selected-conversation", Static)
        )
        assert " - Chats" not in summary


@pytest.mark.asyncio
async def test_retired_grouped_browser_stays_absent_from_empty_rail() -> None:
    """LY-04: empty rails use native owners, not aggregate group chrome."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await pilot.pause(0.2)

        visible_text = _visible_text(console)
        assert "No starred conversations." not in visible_text
        assert "No workspace conversations." not in visible_text
        assert len(console.query("#console-workspace-tree")) == 1
        assert len(console.query("#console-conversation-browser-section-starred")) == 0
        assert (
            len(console.query("#console-conversation-browser-section-workspaces")) == 0
        )
