"""Pure overflow and active-fallback policies for Console Context sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ContextSectionDemand:
    """Measured content demand for one Context section in DOM order.

    Attributes:
        section_id: Stable section identifier.
        desired_content_rows: Uncapped rendered content demand.
        is_open: Whether the section body is currently open.
    """

    section_id: str
    desired_content_rows: int
    is_open: bool


def local_hint_required(desired_content_rows: int, allocated_content_rows: int) -> bool:
    """Return whether a positive content allocation has hidden rows."""

    return desired_content_rows > allocated_content_rows > 0


def outer_hint_required(
    desired_outer_rows: int,
    viewport_rows_without_hint: int,
) -> bool:
    """Derive outer overflow without making the hint self-sustaining."""

    return desired_outer_rows > viewport_rows_without_hint


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
