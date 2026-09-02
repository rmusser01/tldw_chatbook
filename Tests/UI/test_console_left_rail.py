"""Characterisation test for the Console left rail, written BEFORE it becomes
its own region widget (wave-1 console decomposition, task 3, spec rule 6).

Drives the real ``ChatScreen`` through the real Console harness -- the same
idiom ``test_console_internals_decomposition.py`` and
``test_console_shell_regions.py`` use -- and performs a real ``pilot.click``
on a real rail-section toggle button, asserting the PERSISTED outcome (the
section's open state survives a fresh sync, not just the widget's transient
display flag) after closing it and again after reopening it.

This file must pass against unmodified code before the left rail is
extracted into ``UI/Console_Modules/left_rail.py``, and must stay green and
byte-identical afterwards (task-3 brief, global constraint 3).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.pilot import OutOfBounds
from textual.widgets import Button

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail


@asynccontextmanager
async def make_console_pilot(*, size=(160, 45), production_styles: bool = False):
    """Mount a fresh, send-ready Console (ChatScreen) via the production harness.

    Mirrors ``test_console_shell_regions.py``'s ``make_console_pilot``, plus
    ``_configure_native_ready_console`` (see
    ``test_console_native_chat_flow.py``): rail-click tests need the
    blocking first-run ``ConsoleSetupModal`` dismissed, which requires a
    ready provider, not just a mounted composer. The default size matches
    one of ``test_console_shell_regions.py``'s pinned contract sizes; a
    later section's toggle (Details, Character) can still sit outside
    ``pilot.click``'s visible-area requirement because Session and Model
    both default open and already exceed a ~44-row terminal's rail budget
    (see that module's docstring) -- ``_click_rail_toggle`` below scrolls
    the target into view first, exactly as a real user would.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    if production_styles:

        class ProductionStyledConsoleHarness(ConsoleHarness):
            CSS_PATH = str(BUNDLED_STYLESHEET)

        host = ProductionStyledConsoleHarness(app)
    else:
        host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot


@pytest.mark.asyncio
async def test_context_section_headers_match_inspector_title_band() -> None:
    async with make_console_pilot(size=(160, 45), production_styles=True) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        inspector_heading = screen.query_one("#console-inspector-run-heading")
        section_ids = (
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        )
        for section_id in section_ids:
            header = screen.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
            title = screen.query_one(f"#console-rail-section-title-{section_id}")
            toggle = screen.query_one(
                f"#console-rail-section-toggle-{section_id}", Button
            )

            assert header.styles.background == inspector_heading.styles.background
            assert header.styles.color == inspector_heading.styles.color
            assert header.styles.padding == inspector_heading.styles.padding
            assert title.styles.text_style.bold
            assert title.styles.color == inspector_heading.styles.color
            assert toggle.parent is header
            assert header.region.height == 2
            assert header.region.width == header.parent.scrollable_content_region.width
            assert header.content_region.contains_region(toggle.region)

        sections = list(screen.query("#console-left-rail-body ConsoleBoundedSection"))
        assert [section.max_content_lines for section in sections] == [
            20,
            20,
            15,
            15,
            15,
            35,
        ]


def _section_body_visible(pilot, section_id: str) -> bool:
    body = pilot.app.screen.query_one(f"#console-rail-section-body-{section_id}")
    return bool(body.display) and body.styles.display != "none"


def _section_header_open_flag(pilot, section_id: str) -> bool:
    header = pilot.app.screen.query_one(
        f"#console-rail-section-header-{section_id}",
        DestinationRailSectionHeader,
    )
    return header.open


async def _click_rail_toggle(pilot, section_id: str) -> None:
    """Scroll the toggle button into view, then perform a real pilot click.

    ``#console-left-rail-body`` is a ``VerticalScroll``; a later section's
    toggle (e.g. Details, the 4th of 5) sits outside ``pilot.click``'s
    visible-area requirement even at a generously tall terminal, because
    Session and Model both default open above it. ``scroll_visible()``
    mirrors what a real user scrolling the rail before clicking would do;
    it does not change what gets clicked or what handles the click.
    """
    selector = f"#console-rail-section-toggle-{section_id}"
    before = _section_header_open_flag(pilot, section_id)
    for _ in range(40):
        if _section_header_open_flag(pilot, section_id) is not before:
            return
        try:
            toggle = pilot.app.screen.query_one(selector)
            toggle.scroll_visible(animate=False, force=True)
            await pilot.pause(0.05)
            clicked = await pilot.click(selector)
        except (NoMatches, OutOfBounds):
            clicked = False
        if clicked:
            for _ in range(40):
                if _section_header_open_flag(pilot, section_id) is not before:
                    return
                await pilot.pause(0.05)
            break
        await pilot.pause(0.05)
    raise AssertionError(f"Section {section_id!r} did not toggle from {before!r}")


@pytest.mark.asyncio
async def test_context_header_is_one_full_width_collapse_button() -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        button = screen.query_one("#console-context-rail-collapse", Button)
        header = button.parent

        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert not screen.query("#console-context-rail-title")
        # TASK-23195 replaced the ASCII-art literal with a readable
        # name plus a resolved affordance. The header is still ONE
        # full-width collapse button, which is what this test pins.
        assert "Context" in str(button.label)
        assert "<---------" not in str(button.label)
        assert button.tooltip == "Collapse Console context rail"
        assert header.content_region.contains_region(button.region)
        assert button.region.width == header.content_region.width
        assert header.region.height == 1
        assert button.region.height == 1
        assert button.styles.text_align == "right"
        assert button.styles.content_align_horizontal == "right"


@pytest.mark.asyncio
async def test_clicking_context_header_title_end_collapses_the_rail() -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        button = screen.query_one("#console-context-rail-collapse", Button)
        assert "Context" in str(button.label)
        title_end = (button.region.width - 2, 0)

        assert await pilot.click(button, offset=title_end)
        await pilot.pause(0.2)
        assert screen.query_one("#console-left-rail").display is False
        assert screen.query_one("#console-context-rail-handle").display is True


@pytest.mark.asyncio
async def test_details_section_starts_closed_by_persisted_default():
    """Fresh harness, no stored rail preferences: Details defaults closed.

    Pins ``ConsoleRailPreferences.details_open = False`` (the dataclass
    default in ``Chat/console_rail_state.py``) as observed through the real
    DOM, not just read off the dataclass -- this is the starting point the
    toggle tests below build on.
    """
    async with make_console_pilot() as pilot:
        assert _section_header_open_flag(pilot, "details") is False
        assert _section_body_visible(pilot, "details") is False


@pytest.mark.asyncio
async def test_clicking_details_toggle_opens_then_closes_and_persists():
    """A real click on the Details toggle opens it; a second click closes it.

    "Persisted" here means: re-querying the section body/header after each
    click reflects the new state (this is what ``_toggle_console_rail_
    section`` -> ``_set_console_rail_preference`` -> ``_sync_console_rail_
    visibility`` does today), not merely that the click handler ran without
    raising.
    """
    async with make_console_pilot() as pilot:
        await _click_rail_toggle(pilot, "details")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "details") is True
        assert _section_body_visible(pilot, "details") is True

        await _click_rail_toggle(pilot, "details")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "details") is False
        assert _section_body_visible(pilot, "details") is False


@pytest.mark.asyncio
async def test_clicking_an_open_section_toggle_closes_then_reopens():
    """Pin the close-then-reopen direction from a default-OPEN section.

    This exercised Sessions until TASK-23199 retired it. Conversations is
    now the default-open section, and the toggle path under test is the
    same one -- the point is starting from open, not which section it is.
    """
    async with make_console_pilot() as pilot:
        assert _section_header_open_flag(pilot, "conversations") is True
        assert _section_body_visible(pilot, "conversations") is True

        await _click_rail_toggle(pilot, "conversations")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "conversations") is False
        assert _section_body_visible(pilot, "conversations") is False

        await _click_rail_toggle(pilot, "conversations")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "conversations") is True
        assert _section_body_visible(pilot, "conversations") is True


@pytest.mark.asyncio
async def test_context_sections_are_peer_sections_in_requested_order():
    """Workspaces and Conversations lead the rail as peers.

    TASK-23199 retired Sessions, which used to lead: it showed the active
    chat's name, which Conversations already marks on a selected row.
    """

    async with make_console_pilot() as pilot:
        headers = list(
            pilot.app.screen.query(
                "#console-left-rail-body > .console-rail-section-header"
            )
        )
        assert [header.section_id for header in headers[:2]] == [
            "workspace",
            "conversations",
        ]
        assert [header.title for header in headers[:2]] == [
            "Workspaces",
            "Conversations",
        ]


@pytest.mark.asyncio
async def test_context_direct_bodies_use_six_bounded_wrappers_in_dom_order():
    """Every direct body is bounded; pinned fleet chrome is never one of them."""

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        body = screen.query_one("#console-left-rail-body")
        sections = list(body.query("ConsoleBoundedSection"))
        direct_children = list(body.children)

        assert all(isinstance(section, ConsoleBoundedSection) for section in sections)
        assert [section.section_id for section in sections] == [
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        ]
        assert [section.max_content_lines for section in sections] == [
            20,
            20,
            15,
            15,
            15,
            35,
        ]
        assert [
            section.query_one(".console-rail-section-body").id for section in sections
        ] == [
            "console-rail-section-body-workspace",
            "console-rail-section-body-conversations",
            "console-rail-section-body-model",
            "console-rail-section-body-agent",
            "console-rail-section-body-details",
            "console-rail-section-body-character",
        ]
        assert [
            (
                direct_children[index].section_id,
                direct_children[index + 1].section_id,
            )
            for index in range(0, len(direct_children), 2)
        ] == [
            (section_id, section_id)
            for section_id in [
                                "workspace",
                "conversations",
                "model",
                "agent",
                "details",
                "character",
            ]
        ]
        assert all(
            isinstance(direct_children[index], DestinationRailSectionHeader)
            and isinstance(direct_children[index + 1], ConsoleBoundedSection)
            for index in range(0, len(direct_children), 2)
        )

        fleet = screen.query_one("#console-agent-fleet-summary")
        assert fleet.parent is screen.query_one("#console-left-rail")
        assert not list(fleet.query("ConsoleBoundedSection"))
        assert fleet not in list(body.walk_children())


@pytest.mark.asyncio
async def test_pinned_terminal_action_posts_typed_request_outside_six_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal is always visible without reviving the retired Sessions section."""

    assert issubclass(ConsoleLeftRail.TerminalRequested, Message)
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        calls: list[None] = []
        monkeypatch.setattr(
            screen,
            "action_open_console_terminal",
            lambda: calls.append(None),
        )

        terminal = screen.query_one("#console-terminal-open", Button)
        body = screen.query_one("#console-left-rail-body")
        assert terminal.parent is screen.query_one("#console-left-rail")
        assert terminal not in list(body.walk_children())
        assert len(list(body.query("ConsoleBoundedSection"))) == 6

        assert await pilot.click(terminal)
        await pilot.pause()
        assert calls == [None]


@pytest.mark.asyncio
async def test_character_absence_omits_its_bounded_descriptor_without_phantom_body():
    """The config-gated Character section vanishes from the mounted allocation set."""

    async with make_console_pilot() as pilot:
        rail = pilot.app.screen.query_one("#console-left-rail")
        rail._show_character_section = False
        await rail.recompose()
        await pilot.pause()

        assert [
            section.section_id
            for section in rail.query("#console-left-rail-body ConsoleBoundedSection")
        ] == [
                        "workspace",
            "conversations",
            "model",
            "agent",
            "details",
        ]
        direct_children = list(rail.query_one("#console-left-rail-body").children)
        assert [
            (
                direct_children[index].section_id,
                direct_children[index + 1].section_id,
            )
            for index in range(0, len(direct_children), 2)
        ] == [
            (section_id, section_id)
            for section_id in [
                                "workspace",
                "conversations",
                "model",
                "agent",
                "details",
            ]
        ]
        assert all(
            isinstance(direct_children[index], DestinationRailSectionHeader)
            and isinstance(direct_children[index + 1], ConsoleBoundedSection)
            for index in range(0, len(direct_children), 2)
        )
        assert not rail.query("#console-bounded-section-character")


@pytest.mark.asyncio
async def test_context_section_bodies_do_not_mix_their_controls():
    """Each disclosure body owns one concept instead of nesting all three."""

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        workspace_body = screen.query_one("#console-rail-section-body-workspace")
        conversations_body = screen.query_one(
            "#console-rail-section-body-conversations"
        )

        assert list(workspace_body.query("#console-active-workspace"))
        assert list(workspace_body.query("#console-change-workspace"))
        assert not list(
            workspace_body.query("#console-workspace-selected-conversation")
        )
        assert not list(workspace_body.query("#console-workspace-conversation-search"))

        assert list(conversations_body.query("#console-workspace-conversation-search"))
        assert list(conversations_body.query("#console-workspace-conversations"))
        assert not list(conversations_body.query("#console-active-workspace"))
        assert not list(conversations_body.query("#console-active-scope"))


@pytest.mark.asyncio
async def test_workspace_and_conversation_disclosures_toggle_independently():
    """Closing either new section leaves its peer visible and interactive."""

    async with make_console_pilot() as pilot:
        # TASK-23193 shipped Workspaces closed by default, so establish the
        # both-open precondition this test is actually about rather than
        # relying on it. Conversations remains a default-open section.
        assert _section_body_visible(pilot, "conversations") is True
        if _section_body_visible(pilot, "workspace") is False:
            await _click_rail_toggle(pilot, "workspace")
            await pilot.pause(0.2)
        assert _section_body_visible(pilot, "workspace") is True

        await _click_rail_toggle(pilot, "workspace")
        await pilot.pause(0.2)
        assert _section_body_visible(pilot, "workspace") is False
        assert _section_body_visible(pilot, "conversations") is True

        await _click_rail_toggle(pilot, "conversations")
        await pilot.pause(0.2)
        assert _section_body_visible(pilot, "workspace") is False
        assert _section_body_visible(pilot, "conversations") is False

        await _click_rail_toggle(pilot, "workspace")
        await pilot.pause(0.2)
        assert _section_body_visible(pilot, "workspace") is True
        assert _section_body_visible(pilot, "conversations") is False


@pytest.mark.asyncio
async def test_clicking_a_rail_section_toggle_moves_focus_to_the_toggle_button():
    """A clicked disclosure retains focus after its layout reconciliation."""
    async with make_console_pilot() as pilot:
        composer = pilot.app.screen.query_one("#console-native-composer")
        assert pilot.app.focused is composer or pilot.app.focused in list(
            composer.walk_children()
        )

        await _click_rail_toggle(pilot, "details")
        await pilot.pause(0.2)

        toggle_button = pilot.app.screen.query_one(
            "#console-rail-section-toggle-details"
        )
        for _ in range(40):
            if pilot.app.focused is toggle_button:
                break
            await pilot.pause(0.05)
        assert pilot.app.focused is toggle_button
