"""Pure height-allocation policies for bounded Console Context sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_MAX_CONTENT_ROWS = 20


@dataclass(frozen=True, slots=True)
class ContextSectionDemand:
    """Measured content demand for one Context section in DOM order."""

    section_id: str
    desired_content_rows: int
    is_open: bool


@dataclass(frozen=True, slots=True)
class ContextSectionAllocation:
    """Stable content and hint allocation for one Context section."""

    section_id: str
    allocated_content_rows: int
    hint_required: bool
    no_room: bool


@dataclass(frozen=True, slots=True)
class ContextAllocationResult:
    """Complete allocation snapshot for atomic application by the Context rail."""

    allocations: tuple[ContextSectionAllocation, ...]
    uses_outer_scroll: bool


def local_hint_required(desired_content_rows: int, allocated_content_rows: int) -> bool:
    """Return whether a positive content allocation has hidden rows."""

    return desired_content_rows > allocated_content_rows > 0


def outer_hint_required(
    desired_outer_rows: int,
    viewport_rows_without_hint: int,
) -> bool:
    """Derive outer overflow without letting the hint slot affect its own existence."""

    return desired_outer_rows > viewport_rows_without_hint


def allocate_context_sections(
    *,
    viewport_height: int,
    header_chrome_height: int,
    sections: Sequence[ContextSectionDemand],
    active_section_id: str | None = None,
) -> ContextAllocationResult:
    """Allocate Context content rows while preserving the input section order."""

    uses_outer_scroll = header_chrome_height > viewport_height
    if uses_outer_scroll:
        allocated_rows = _allocate_short_height(
            viewport_height,
            sections,
            active_section_id,
        )
    else:
        allocated_rows = _allocate_header_fit(
            max(0, viewport_height - header_chrome_height),
            sections,
            active_section_id,
        )

    allocations = tuple(
        ContextSectionAllocation(
            section_id=section.section_id,
            allocated_content_rows=allocated,
            hint_required=local_hint_required(
                section.desired_content_rows,
                allocated,
            ),
            no_room=(
                not uses_outer_scroll
                and section.is_open
                and section.desired_content_rows > 0
                and allocated == 0
            ),
        )
        for section, allocated in zip(sections, allocated_rows)
    )
    return ContextAllocationResult(allocations, uses_outer_scroll)


def fallback_active_section(
    sections: Sequence[ContextSectionDemand],
    active_section_id: str | None,
) -> str | None:
    """Choose the closest valid predecessor, then the first valid successor."""

    def valid(section: ContextSectionDemand) -> bool:
        return section.is_open and section.desired_content_rows > 0

    active_index = next(
        (
            index
            for index, section in enumerate(sections)
            if section.section_id == active_section_id
        ),
        None,
    )
    if active_index is None:
        return next(
            (section.section_id for section in sections if valid(section)),
            None,
        )
    if valid(sections[active_index]):
        return sections[active_index].section_id

    for section in reversed(sections[:active_index]):
        if valid(section):
            return section.section_id
    return next(
        (
            section.section_id
            for section in sections[active_index + 1 :]
            if valid(section)
        ),
        None,
    )


def _allocate_short_height(
    viewport_height: int,
    sections: Sequence[ContextSectionDemand],
    active_section_id: str | None,
) -> list[int]:
    allocated = [
        1 if section.is_open and section.desired_content_rows > 0 else 0
        for section in sections
    ]
    for index, section in enumerate(sections):
        if section.section_id != active_section_id or allocated[index] == 0:
            continue
        allocated[index] = min(
            section.desired_content_rows,
            _MAX_CONTENT_ROWS,
            max(1, viewport_height - 3),
        )
        break
    return allocated


def _allocate_header_fit(
    content_budget: int,
    sections: Sequence[ContextSectionDemand],
    active_section_id: str | None,
) -> list[int]:
    allocated = [0] * len(sections)
    eligible = [
        index
        for index, section in enumerate(sections)
        if section.is_open and section.desired_content_rows > 0
    ]
    priority = sorted(
        eligible,
        key=lambda index: sections[index].section_id != active_section_id,
    )

    remaining = content_budget
    for index in priority:
        base_cost = 1 + int(sections[index].desired_content_rows > 1)
        if base_cost <= remaining:
            allocated[index] = 1
            remaining -= base_cost

    priority_rank = {index: rank for rank, index in enumerate(priority)}
    while True:
        candidates = [
            index
            for index in eligible
            if allocated[index] > 0
            and allocated[index]
            < min(sections[index].desired_content_rows, _MAX_CONTENT_ROWS)
        ]
        candidates.sort(key=lambda index: (allocated[index], priority_rank[index]))

        for index in candidates:
            desired = sections[index].desired_content_rows
            current = allocated[index]
            next_allocation = current + 1
            row_cost = (
                1
                + int(local_hint_required(desired, next_allocation))
                - int(local_hint_required(desired, current))
            )
            if row_cost <= remaining:
                allocated[index] = next_allocation
                remaining -= row_cost
                break
        else:
            return allocated
