"""Open a Console Context rail section from a test.

TASK-23193 changed the shipped defaults so only Sessions and Conversations
start open: a 2026-08-29 UX audit measured the previous five-open default at
51 rows against a 32-row viewport at 160x48 and found it overflowed every one
of ten terminal geometries.

Any test that asserts on a section's *rendered geometry* -- widths, avatar
boxes, allocations -- must therefore open that section first. A closed
section body has ``display: none`` and reports ``Size(0, 0)``, which surfaces
as confusing assertions like ``assert 0 < 0`` rather than as "the section is
shut". Prefer this helper over flipping preferences directly so the test
exercises the same disclosure path a user does.
"""

from __future__ import annotations

from textual.css.query import NoMatches
from textual.pilot import OutOfBounds

from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader


def rail_section_is_open(screen, section_id: str) -> bool:
    """Return whether a Context rail section's disclosure is currently open."""
    header = screen.query_one(
        f"#console-rail-section-header-{section_id}",
        DestinationRailSectionHeader,
    )
    return bool(header.open)


async def open_rail_section(screen, pilot, section_id: str) -> None:
    """Ensure a Context rail section is open, scrolling its toggle into view.

    No-op when the section is already open, so callers can use this
    unconditionally regardless of which sections ship open.

    Args:
        screen: The mounted Console (``ChatScreen``) under test.
        pilot: The Textual ``Pilot`` driving it.
        section_id: A stable Context section id, e.g. ``"character"``.

    Raises:
        AssertionError: If the section never reports open.
    """
    if rail_section_is_open(screen, section_id):
        return

    # Press the disclosure button directly rather than via ``pilot.click``.
    # The click path additionally requires the toggle to be inside the outer
    # scroll's visible area, which is exactly the condition a short test
    # terminal fails; ``press()`` still travels the real
    # Button.Pressed -> SectionToggled -> screen handler route.
    selector = f"#console-rail-section-toggle-{section_id}"
    for _ in range(40):
        if rail_section_is_open(screen, section_id):
            return
        try:
            toggle = screen.query_one(selector)
        except NoMatches:
            await pilot.pause(0.05)
            continue
        try:
            toggle.scroll_visible(animate=False, force=True)
        except (NoMatches, OutOfBounds):
            pass
        toggle.press()
        for _ in range(20):
            if rail_section_is_open(screen, section_id):
                return
            await pilot.pause(0.05)

    raise AssertionError(f"Context rail section {section_id!r} would not open")
