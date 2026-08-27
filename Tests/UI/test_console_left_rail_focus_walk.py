"""TASK-22228 item 3: the rail's focus-target walk never uses the selector engine.

``ConsoleLeftRail._focusable_body_controls`` runs once or twice per focus
change inside the rail (``_record_section_focus`` on every
``DescendantFocus``, and again on every focus-recovery attempt). It used to
collect its candidates with ``viewport.query("*")``: Textual builds that
query from ``walk_children(Widget)`` and then runs the parsed universal
selector through ``match()`` for every node it just walked, which measured
74.1 us against 2.2 us for the bare walk on the 16-node Model body (whole
call 87.8 us -> 23.4 us).

The perf arm below pins that no selector query is made at all -- restoring
``query("*")`` fails it. The equivalence arm is what makes that safe: the
returned tuple must still be exactly what the selector-engine walk yields,
in the same order, for every mounted section.
"""

from __future__ import annotations

import pytest
from textual.widget import Widget

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.Widgets.Console.console_bounded_section import ConsoleBoundedSection


def _count_viewport_queries(rail: ConsoleLeftRail) -> dict[str, int]:
    """Count selector-engine calls made on every mounted section viewport."""
    counts = {"query": 0}
    for bounded in rail.query(ConsoleBoundedSection):
        viewport = bounded.viewport
        original = viewport.query

        def counting(*args, _original=original, **kwargs):
            counts["query"] += 1
            return _original(*args, **kwargs)

        viewport.query = counting  # type: ignore[method-assign]
    return counts


@pytest.mark.asyncio
async def test_focus_target_walk_makes_no_selector_query() -> None:
    """Every mounted section's candidate walk bypasses the selector engine."""
    async with make_console_pilot() as pilot:
        rail = pilot.app.screen.query_one(ConsoleLeftRail)
        section_ids = [
            descriptor.section_id for descriptor in rail._mounted_descriptors()
        ]
        assert section_ids, "no Context sections mounted -- probe is vacuous"

        counts = _count_viewport_queries(rail)
        for section_id in section_ids:
            rail._focusable_body_controls(section_id)

        assert counts["query"] == 0, counts


@pytest.mark.asyncio
async def test_focus_target_walk_matches_the_selector_engine_exactly() -> None:
    """The cheap walk yields the same widgets, in the same order, as ``query('*')``."""
    async with make_console_pilot() as pilot:
        rail = pilot.app.screen.query_one(ConsoleLeftRail)
        total_controls = 0
        checked_sections = 0
        for descriptor in rail._mounted_descriptors():
            section_id = descriptor.section_id
            bounded = rail.query_one(
                f"#console-bounded-section-{section_id}", ConsoleBoundedSection
            )
            expected = tuple(
                widget
                for widget in bounded.viewport.query("*")
                if isinstance(widget, Widget) and rail._is_enabled_focus_target(widget)
            )
            if bounded.native_scroll_owner is not None and rail._is_enabled_focus_target(
                bounded.viewport
            ):
                expected = (bounded.viewport, *expected)

            assert rail._focusable_body_controls(section_id) == expected, section_id
            total_controls += len(expected)
            checked_sections += 1

        assert checked_sections >= 5, checked_sections
        # Guard against an all-empty comparison passing vacuously: the rail
        # really does own focusable body controls in this geometry.
        assert total_controls > 0, "no focusable rail controls -- comparison is vacuous"


@pytest.mark.asyncio
async def test_focus_change_in_the_rail_still_records_its_control_set() -> None:
    """Behaviour arm: focusing a rail control still snapshots that section."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        rail = screen.query_one(ConsoleLeftRail)
        target = next(
            (
                widget
                for descriptor in rail._mounted_descriptors()
                for widget in rail._focusable_body_controls(descriptor.section_id)
            ),
            None,
        )
        assert target is not None, "no focusable rail control to drive the arm"

        rail._section_focus_history.clear()
        target.focus()
        await pilot.pause()

        assert rail._section_focus_history, "focus change recorded nothing"
        recorded_target, controls = next(iter(rail._section_focus_history.values()))
        assert recorded_target is target
        assert target in controls
