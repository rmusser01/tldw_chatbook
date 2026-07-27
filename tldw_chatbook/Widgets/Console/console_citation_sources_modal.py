"""Pure Console citation-source transformations.

The Sources modal UI and governed payload hydration are added in the next
delivery slice. Footer discovery intentionally reads only immutable trace
metadata from this module.
"""

from __future__ import annotations

from tldw_chatbook.Chat.citation_trace_models import (
    CitationTrace,
    StructuralValidationState,
)


def selected_valid_evidence_ordinals(trace: CitationTrace) -> tuple[int, ...]:
    """Return selected-attempt valid evidence ordinals in first-citation order."""

    selected_attempt = next(
        (
            attempt
            for attempt in trace.answer_attempts
            if attempt.attempt_id == trace.selected_attempt_id
        ),
        None,
    )
    if selected_attempt is None:
        return ()

    seen: set[int] = set()
    ordinals: list[int] = []
    for occurrence in selected_attempt.occurrences:
        evidence_ordinal = occurrence.evidence_ordinal
        if (
            occurrence.structural_state is not StructuralValidationState.VALID
            or type(evidence_ordinal) is not int
            or evidence_ordinal in seen
        ):
            continue
        seen.add(evidence_ordinal)
        ordinals.append(evidence_ordinal)
    return tuple(ordinals)


__all__ = ["selected_valid_evidence_ordinals"]
