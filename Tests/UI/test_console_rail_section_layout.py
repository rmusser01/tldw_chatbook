from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.UI.Console_Modules.rail_section_layout import (
    ContextSectionDemand,
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
        (15, 15, False),
        (16, 15, True),
        (20, 20, False),
        (21, 20, True),
        (35, 35, False),
        (36, 35, True),
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

    assert [outer_hint_required(rows, 10) for rows in (10, 11, 10)] == [
        False,
        True,
        False,
    ]
    assert [outer_hint_required(11, rows) for rows in (10, 11, 10)] == [
        True,
        False,
        True,
    ]


def test_context_demand_record_is_immutable() -> None:
    section_demand = demand("sessions", 3)

    with pytest.raises(FrozenInstanceError):
        section_demand.is_open = False  # type: ignore[misc]


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
