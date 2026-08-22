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
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)
from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader


@asynccontextmanager
async def make_console_pilot(*, size=(160, 45)):
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
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot


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
    toggle = pilot.app.screen.query_one(f"#console-rail-section-toggle-{section_id}")
    toggle.scroll_visible(animate=False)
    await pilot.pause(0.2)
    await pilot.click(f"#console-rail-section-toggle-{section_id}")


@pytest.mark.asyncio
async def test_context_header_is_one_full_width_collapse_button() -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        button = screen.query_one("#console-context-rail-collapse", Button)
        header = button.parent

        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert not screen.query("#console-context-rail-title")
        assert str(button.label) == "<---------|Context"
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
        assert str(button.label) == "<---------|Context"
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
async def test_clicking_session_toggle_closes_then_reopens():
    """Session defaults OPEN (unlike Details); pin the opposite direction too.

    Exercises the same toggle path starting from the open persisted default
    (``ConsoleRailPreferences.session_open = True``), closing then reopening.
    """
    async with make_console_pilot() as pilot:
        assert _section_header_open_flag(pilot, "session") is True
        assert _section_body_visible(pilot, "session") is True

        await _click_rail_toggle(pilot, "session")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "session") is False
        assert _section_body_visible(pilot, "session") is False

        await _click_rail_toggle(pilot, "session")
        await pilot.pause(0.2)
        assert _section_header_open_flag(pilot, "session") is True
        assert _section_body_visible(pilot, "session") is True


@pytest.mark.asyncio
async def test_context_sections_are_peer_sections_in_requested_order():
    """Sessions, Workspaces, and Conversations lead the rail as peers."""

    async with make_console_pilot() as pilot:
        headers = list(
            pilot.app.screen.query(
                "#console-left-rail-body > .console-rail-section-header"
            )
        )
        assert [header.section_id for header in headers[:3]] == [
            "session",
            "workspace",
            "conversations",
        ]
        assert [header.title for header in headers[:3]] == [
            "Sessions",
            "Workspaces",
            "Conversations",
        ]


@pytest.mark.asyncio
async def test_context_direct_bodies_use_seven_bounded_wrappers_in_dom_order():
    """Every direct body is bounded; pinned fleet chrome is never one of them."""

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        body = screen.query_one("#console-left-rail-body")
        sections = list(body.query("ConsoleBoundedSection"))
        direct_children = list(body.children)

        assert all(isinstance(section, ConsoleBoundedSection) for section in sections)
        assert [section.section_id for section in sections] == [
            "session",
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        ]
        assert [
            section.query_one(".console-rail-section-body").id for section in sections
        ] == [
            "console-rail-section-body-session",
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
                "session",
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
            "session",
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
                "session",
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
        session_body = screen.query_one("#console-rail-section-body-session")
        workspace_body = screen.query_one("#console-rail-section-body-workspace")
        conversations_body = screen.query_one(
            "#console-rail-section-body-conversations"
        )

        assert list(session_body.query("#console-active-scope"))
        assert not list(session_body.query("#console-active-workspace"))
        assert not list(session_body.query("#console-workspace-conversation-search"))

        assert list(workspace_body.query("#console-active-workspace"))
        assert list(workspace_body.query("#console-change-workspace"))
        assert not list(workspace_body.query("#console-active-scope"))
        assert not list(workspace_body.query("#console-workspace-conversation-search"))

        assert list(conversations_body.query("#console-workspace-conversation-search"))
        assert list(conversations_body.query("#console-workspace-conversations"))
        assert not list(conversations_body.query("#console-active-workspace"))
        assert not list(conversations_body.query("#console-active-scope"))


@pytest.mark.asyncio
async def test_workspace_and_conversation_disclosures_toggle_independently():
    """Closing either new section leaves its peer visible and interactive."""

    async with make_console_pilot() as pilot:
        assert _section_body_visible(pilot, "workspace") is True
        assert _section_body_visible(pilot, "conversations") is True

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
    """Pin today's focus behaviour when a rail section header is pressed.

    The composer holds focus at mount (the harness waits for
    ``#console-native-composer`` before yielding control), and nothing in
    ``_toggle_console_rail_section`` calls ``.focus()`` explicitly -- so
    whatever focus ends up on is whatever Textual's default click-to-focus
    behaviour does for a pressed ``Button``. Pinning the OBSERVED outcome
    here (rather than asserting what "should" happen) is the point of a
    characterisation test.
    """
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
        assert pilot.app.focused is toggle_button
