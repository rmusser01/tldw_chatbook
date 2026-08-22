from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.UI.Console_Modules.rail_section_layout import (
    ContextAllocationResult,
    ContextSectionAllocation,
    ContextSectionDemand,
    allocate_context_sections,
    fallback_active_section,
    local_hint_required,
    outer_hint_required,
)


def demand(
    section_id: str,
    desired_content_rows: int,
    *,
    is_open: bool = True,
) -> ContextSectionDemand:
    return ContextSectionDemand(
        section_id=section_id,
        desired_content_rows=desired_content_rows,
        is_open=is_open,
    )


@pytest.mark.parametrize(
    ("desired", "allocated", "expected"),
    [
        (0, 0, False),
        (1, 0, False),
        (1, 1, False),
        (2, 1, True),
        (20, 20, False),
        (21, 20, True),
        (21, 0, False),
    ],
)
def test_local_hint_is_required_only_for_positive_underallocation(
    desired: int,
    allocated: int,
    expected: bool,
) -> None:
    assert local_hint_required(desired, allocated) is expected


def test_outer_hint_uses_the_viewport_without_its_own_slot() -> None:
    assert outer_hint_required(10, 10) is False
    assert outer_hint_required(11, 10) is True

    # Content shrinking 10 -> 11 -> 10 cannot leave a self-created overflow slot.
    assert [outer_hint_required(rows, 10) for rows in (10, 11, 10)] == [
        False,
        True,
        False,
    ]

    # Terminal growth removes the slot and terminal shrink restores it from truth.
    assert [outer_hint_required(11, rows) for rows in (10, 11, 10)] == [
        True,
        False,
        True,
    ]


def test_policy_records_are_immutable() -> None:
    section_demand = demand("sessions", 3)
    allocation = ContextSectionAllocation("sessions", 2, True, False)
    result = ContextAllocationResult((allocation,), uses_outer_scroll=False)

    with pytest.raises(FrozenInstanceError):
        section_demand.is_open = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        allocation.allocated_content_rows = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.uses_outer_scroll = True  # type: ignore[misc]


def test_normal_allocation_funds_active_first_and_preserves_dom_order() -> None:
    sections = (
        demand("sessions", 30),
        demand("workspaces", 30),
        demand("conversations", 30),
        demand("closed", 5, is_open=False),
        demand("empty", 0),
    )

    result = allocate_context_sections(
        viewport_height=9,
        header_chrome_height=5,
        sections=sections,
        active_section_id="workspaces",
    )

    assert result == ContextAllocationResult(
        allocations=(
            ContextSectionAllocation("sessions", 1, True, False),
            ContextSectionAllocation("workspaces", 1, True, False),
            ContextSectionAllocation("conversations", 0, False, True),
            ContextSectionAllocation("closed", 0, False, False),
            ContextSectionAllocation("empty", 0, False, False),
        ),
        uses_outer_scroll=False,
    )


def test_allocation_rejects_duplicate_section_ids() -> None:
    with pytest.raises(ValueError, match="^duplicate Context section ID: sessions$"):
        allocate_context_sections(
            viewport_height=10,
            header_chrome_height=2,
            sections=(demand("sessions", 2), demand("sessions", 3)),
        )


def test_normal_allocation_breaks_water_fill_ties_in_dom_order() -> None:
    result = allocate_context_sections(
        viewport_height=7,
        header_chrome_height=0,
        sections=(
            demand("sessions", 30),
            demand("workspaces", 30),
            demand("conversations", 30),
        ),
    )

    assert [item.allocated_content_rows for item in result.allocations] == [2, 1, 1]
    assert [item.hint_required for item in result.allocations] == [True, True, True]


def test_normal_allocation_gives_active_section_the_first_water_fill_tie() -> None:
    result = allocate_context_sections(
        viewport_height=7,
        header_chrome_height=0,
        sections=(
            demand("sessions", 30),
            demand("workspaces", 30),
            demand("conversations", 30),
        ),
        active_section_id="conversations",
    )

    assert [item.allocated_content_rows for item in result.allocations] == [1, 1, 2]


def test_normal_allocation_water_fills_unused_rows_up_to_twenty() -> None:
    result = allocate_context_sections(
        viewport_height=24,
        header_chrome_height=0,
        sections=(demand("short", 3), demand("long", 30)),
        active_section_id="long",
    )

    assert result.allocations == (
        ContextSectionAllocation("short", 3, False, False),
        ContextSectionAllocation("long", 20, True, False),
    )


def test_normal_allocation_redistributes_released_hint_rows_until_stable() -> None:
    result = allocate_context_sections(
        viewport_height=7,
        header_chrome_height=0,
        sections=(
            demand("first-short", 2),
            demand("second-short", 2),
            demand("long", 30),
        ),
    )

    assert result.allocations == (
        ContextSectionAllocation("first-short", 2, False, False),
        ContextSectionAllocation("second-short", 2, False, False),
        ContextSectionAllocation("long", 2, True, False),
    )
    assert (
        sum(
            item.allocated_content_rows + int(item.hint_required)
            for item in result.allocations
        )
        == 7
    )


def test_short_height_allocation_gives_honest_bases_and_expands_only_active() -> None:
    sections = (
        demand("one-row", 1),
        demand("inactive-long", 8),
        demand("active-long", 50),
        demand("closed", 8, is_open=False),
        demand("empty", 0),
    )

    result = allocate_context_sections(
        viewport_height=8,
        header_chrome_height=9,
        sections=sections,
        active_section_id="active-long",
    )

    assert result == ContextAllocationResult(
        allocations=(
            ContextSectionAllocation("one-row", 1, False, False),
            ContextSectionAllocation("inactive-long", 1, True, False),
            ContextSectionAllocation("active-long", 5, True, False),
            ContextSectionAllocation("closed", 0, False, False),
            ContextSectionAllocation("empty", 0, False, False),
        ),
        uses_outer_scroll=True,
    )
    assert sections[3].is_open is False


def test_short_height_active_allocation_hugs_demand_and_twenty_row_ceiling() -> None:
    for desired, expected in ((2, 2), (20, 20), (21, 20)):
        result = allocate_context_sections(
            viewport_height=30,
            header_chrome_height=31,
            sections=(demand("active", desired),),
            active_section_id="active",
        )
        assert result.allocations[0].allocated_content_rows == expected
        assert result.allocations[0].hint_required is (desired > expected)


def test_fallback_active_section_prefers_nearest_preceding_then_first_following() -> (
    None
):
    sections = (
        demand("sessions", 2),
        demand("workspaces", 0),
        demand("conversations", 2, is_open=False),
        demand("model", 0),
        demand("agent", 2),
        demand("details", 2),
    )

    assert fallback_active_section(sections, "model") == "sessions"
    assert (
        fallback_active_section(
            (demand("sessions", 0), demand("agent", 2), demand("details", 2)),
            "sessions",
        )
        == "agent"
    )
    assert (
        fallback_active_section(
            (demand("closed", 2, is_open=False), demand("empty", 0)),
            "closed",
        )
        is None
    )


def test_fallback_active_section_retains_valid_active_and_handles_no_known_active() -> (
    None
):
    sections = (
        demand("sessions", 0),
        demand("workspaces", 2),
        demand("conversations", 2),
    )

    assert fallback_active_section(sections, "conversations") == "conversations"
    assert fallback_active_section(sections, "missing") == "workspaces"
    assert fallback_active_section(sections, None) == "workspaces"
